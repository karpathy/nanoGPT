from dataclasses import asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te
from pydantic.dataclasses import dataclass


@dataclass
class MistralConfig:
    n_layers: int    # L
    n_heads: int     # H
    n_kv_heads: int  # J
    d_embd: int      # E
    d_hid: int       # K
    vocab_size: int  # V
    max_seq_len: int # T
    window_size: int # W
    rope_base: float
    norm_eps: float
    arch_name: str = 'mistral'

    def estimate_flops_per_token(self, model, bsz, rank=0):
        head_dim = self.d_embd // self.n_heads
        N = sum(p.numel() for p in model.parameters())         # get param count
        density = self.window_size / self.max_seq_len  # sliding window attention mask
        self.flops_per_token = 6 * N + 12 * self.n_layers * self.n_heads * head_dim * self.max_seq_len * density
        if rank == 0:
            print(f"Number of parameters: {N/1e9:.2f}B")    # print number of billion parameters


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_embd, n_heads, n_kv_heads, **kwargs):
        super().__init__()
        self.d_embd = d_embd
        self.d_head = d_embd // n_heads  # D
        self.d_kv_embd = n_kv_heads * self.d_head

        self.attn_proj = nn.Linear(d_embd, d_embd+2*self.d_kv_embd, bias=False)
        self.out_proj = nn.Linear(d_embd, d_embd, bias=False)
        self.sdpa = disable_torch_compile_if_amd(F.scaled_dot_product_attention)

    def forward(self, x_BTE, attn_mask, freq_cis_TF):
        qkv = self.attn_proj(x_BTE).split([self.d_embd, self.d_kv_embd, self.d_kv_embd], -1)
        split_attn_head = lambda z: z.unflatten(-1, [-1, self.d_head]).transpose(1, 2)
        q_BHTD, k_BJTD, v_BJTD = map(split_attn_head, qkv)

        q_BHTD = apply_rotary_embd(q_BHTD, freq_cis_TF)
        k_BJTD = apply_rotary_embd(k_BJTD, freq_cis_TF)

        k_BHTD = k_BJTD.repeat_interleave(self.d_embd//self.d_kv_embd, 1)
        v_BHTD = v_BJTD.repeat_interleave(self.d_embd//self.d_kv_embd, 1)

        o_BHTD = self.sdpa(q_BHTD, k_BHTD, v_BHTD, attn_mask=attn_mask)
        y_BTE = self.out_proj(o_BHTD.transpose(1, 2).flatten(-2))

        return y_BTE


def disable_torch_compile_if_amd(fn):
    # Define a wrapper that applies the torch.compiler.disable decorator conditionally
    if torch.cuda.is_available() and "MI300X" in torch.cuda.get_device_name():
        return torch.compiler.disable()(fn)
    else:
        return fn


def apply_rotary_embd(x_BXTD, freq_cis_TFC):
    x_BXTFC = x_BXTD.unflatten(-1, [-1, 2])  # C: Complex number dimension
    freq_cis_BXTFC = freq_cis_TFC.expand_as(x_BXTFC)

    out_BXTDC = torch.stack([
        x_BXTFC[..., 0] * freq_cis_BXTFC[..., 0] - x_BXTFC[..., 1] * freq_cis_BXTFC[..., 1],
        x_BXTFC[..., 1] * freq_cis_BXTFC[..., 0] + x_BXTFC[..., 0] * freq_cis_BXTFC[..., 1],
    ], dim=-1)
    out_BXTD = out_BXTDC.flatten(-2)

    return out_BXTD.type_as(x_BXTD)


class SwiGLU(nn.Module):
    def __init__(self, d_embd, d_hid, **kwargs):
        super().__init__()
        self.up_proj = nn.Linear(d_embd, d_hid, bias=False)
        self.gate_proj = nn.Linear(d_embd, d_hid, bias=False)
        self.down_proj = nn.Linear(d_hid, d_embd, bias=False)

    def forward(self, x_BTE):
        h_BTK = F.silu(self.gate_proj(x_BTE)) * self.up_proj(x_BTE)
        out_BTE = self.down_proj(h_BTK)
        return out_BTE


class RMSNorm(nn.Module):
    def __init__(self, d_embd, norm_eps, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_embd))
        self.eps = norm_eps

    def forward(self, x_BTE):
        x_BTE_fp32 = x_BTE.to(torch.float32)
        r_rms = torch.rsqrt(x_BTE_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        y_BTE = (x_BTE_fp32 * r_rms).to(x_BTE) * self.weight
        return y_BTE


class MistralBlock(nn.Module):
    def __init__(self, d_embd, **kwargs):
        super().__init__()
        self.attn_norm = RMSNorm(d_embd, **kwargs)
        self.attn = GroupedQueryAttention(d_embd, **kwargs)
        self.ffn_norm = RMSNorm(d_embd, **kwargs)
        self.ffn = SwiGLU(d_embd, **kwargs)

    def forward(self, x_BTE, attn_mask, freq_cis_TFC):
        h_BTE = x_BTE + self.attn(self.attn_norm(x_BTE), attn_mask, freq_cis_TFC)
        out_BTE = h_BTE + self.ffn(self.ffn_norm(h_BTE))
        return out_BTE


class Mistral(nn.Module):
    def __init__(self, vocab_size, d_embd, n_layers, n_heads, window_size, **kwargs):
        super().__init__()
        self.tok_embd = nn.Embedding(vocab_size, d_embd)
        self.tsfmr_blks = nn.ModuleList(MistralBlock(d_embd, n_heads=n_heads, **kwargs) for _ in range(n_layers))
        self.norm = RMSNorm(d_embd, **kwargs)
        self.lm_head = nn.Linear(d_embd, vocab_size, bias=False)
        self.register_buffer('freq_cis_TFC', precompute_freq_cis(d_embd//n_heads, **kwargs).to(self.lm_head.weight.dtype))
        self.register_buffer('swa_mask', create_sliding_window_attention_mask(window_size, kwargs['max_seq_len']))

    def forward(self, idx_BT, **kwargs):
        x_BTE = self.tok_embd(idx_BT)
        for tsfmr_blk in self.tsfmr_blks:
            x_BTE = tsfmr_blk(x_BTE, self.swa_mask, self.freq_cis_TFC)
        logits_BTV = self.lm_head(self.norm(x_BTE))
        return logits_BTV


def precompute_freq_cis(dim, rope_base, max_seq_len, **kwargs):
    assert dim % 2 == 0
    theta_F = 1 / (rope_base ** (torch.arange(0, dim, 2).float() / dim))  # F = dim // 2
    pos_idx_T = torch.arange(max_seq_len)
    freq_TF = pos_idx_T.unsqueeze(1) * theta_F.unsqueeze(0)
    freq_cis_TF = torch.polar(torch.ones_like(freq_TF), freq_TF)
    freq_cis_TFC = torch.stack([freq_cis_TF.real, freq_cis_TF.imag], dim=-1)
    return freq_cis_TFC


def create_sliding_window_attention_mask(window_size, max_seq_len):
    rows = torch.arange(max_seq_len).unsqueeze(1)  # [T, 1]
    cols = torch.arange(max_seq_len).unsqueeze(0)  # [1, T]
    l1_dist = rows - cols
    swa_mask = (l1_dist >= 0) & (l1_dist < window_size)
    return swa_mask


if __name__ == '__main__':
    import json
    from pydantic import RootModel

    mistral_v01_proxy = MistralConfig(
        n_layers=1,
        n_heads=32,
        n_kv_heads=8,
        d_embd=4096,
        d_hid=14336,
        vocab_size=32000,
        max_seq_len=32768,
        window_size=4096,
        rope_base=1e4,
        norm_eps=1e-5
    )
    with open('configs/mistral-v0.1-proxy.json', 'w') as f:
        json.dump(RootModel[MistralConfig](mistral_v01_proxy).model_dump(), f, indent=2)
