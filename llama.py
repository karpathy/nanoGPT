from dataclasses import asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te

from pydantic.dataclasses import dataclass


def disable_torch_compile_if_amd(func):
    # Define a wrapper that applies the torch.compiler.disable decorator conditionally
    if torch.cuda.is_available() and "MI300X" in torch.cuda.get_device_name():
        return torch.compiler.disable()(func)
    else:
        return func


@disable_torch_compile_if_amd
def scaled_dot_product_attention_wrapper(q_BHTD, k_BHTD, v_BHTD, dropout_p=0.0, is_causal=True):
    # with torch.nn.attention.sdpa_kernel(
    #     enable_math=True,
    #     enable_flash=False,
    #     enable_mem_efficient=False
    # ):
    o_BHTD = F.scaled_dot_product_attention(q_BHTD, k_BHTD, v_BHTD, dropout_p=dropout_p, is_causal=is_causal)
    return o_BHTD


@dataclass
class LLaMAConfig:
    n_layers: int    # L
    n_heads: int     # H
    n_kv_heads: int  # J
    d_embd: int      # E
    max_seq_len: int # T
    vocab_size: int  # V
    ffn_mult: float
    ffn_factor: int
    rope_base: float
    norm_eps: float
    d_hid: int = Optional[int] # K
    arch_name: str = 'llama'

    def estimate_flops_per_token(self, model, bsz, rank=0):
        head_dim = self.d_embd // self.n_heads
        N = sum(p.numel() for p in model.parameters())  # get param count

        if rank == 0:
            print(f"Number of parameters: {N/1e9:.2f}B")    # print number of billion parameters 

        self.flops_per_token = 6 * N + 12 * self.n_layers * self.n_heads * head_dim * self.max_seq_len

    def __post_init__(self):
        assert self.d_embd % self.n_heads == 0, 'd_embd must be a multiple of n_heads.'
        assert self.d_embd % self.n_kv_heads == 0, 'd_embd must be a multiple of n_kv_heads.'
        assert self.n_kv_heads <= self.n_heads, 'n_kv_heads must not be larger than n_heads.'

        # FFN hidden dimension
        d_hid = int((4 * self.d_embd) * 2 / 3)
        d_hid = int(d_hid * self.ffn_mult)
        self.d_hid = self.ffn_factor * ((d_hid + self.ffn_factor - 1) // self.ffn_factor)


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_embd, n_heads, n_kv_heads, **kwargs):
        super().__init__()
        self.d_head = d_embd // n_heads  # D
        self.d_embd = d_embd
        self.d_kv_embd = n_kv_heads * self.d_head

        self.attn_proj = nn.Linear(d_embd, d_embd+2*self.d_kv_embd, bias=False)
        self.out_proj = nn.Linear(d_embd, d_embd, bias=False)

    def forward(self, x_BTE, freq_cis_TF):
        qkv = self.attn_proj(x_BTE).split([self.d_embd, self.d_kv_embd, self.d_kv_embd], -1)
        split_attn_head = lambda z: z.unflatten(-1, [-1, self.d_head]).transpose(1, 2)
        q_BHTD, k_BJTD, v_BJTD = map(split_attn_head, qkv)

        q_BHTD = apply_rotary_embd(q_BHTD, freq_cis_TF)
        k_BJTD = apply_rotary_embd(k_BJTD, freq_cis_TF)

        k_BHTD = k_BJTD.repeat_interleave(self.d_embd//self.d_kv_embd, 1)
        v_BHTD = v_BJTD.repeat_interleave(self.d_embd//self.d_kv_embd, 1)

        o_BHTD = scaled_dot_product_attention_wrapper(q_BHTD, k_BHTD, v_BHTD, dropout_p=0.0, is_causal=True)
        y_BTE = self.out_proj(o_BHTD.transpose(1, 2).flatten(-2))

        return y_BTE

    def load_ref_weights(self, ref):
        self.attn_proj.load_state_dict(ref.wqkv.state_dict())
        self.out_proj.load_state_dict(ref.wo.state_dict())


def apply_rotary_embd(x_BXTD, freq_cis_TFC):
    x_BXTFC = x_BXTD.unflatten(-1, [-1, 2])  # C: Complex number dimension
    freq_cis_BXTFC = freq_cis_TFC.expand_as(x_BXTFC)

    out_BXTDC = torch.stack([
        x_BXTFC[..., 0] * freq_cis_BXTFC[..., 0] - x_BXTFC[..., 1] * freq_cis_BXTFC[..., 1],
        x_BXTFC[..., 1] * freq_cis_BXTFC[..., 0] + x_BXTFC[..., 0] * freq_cis_BXTFC[..., 1],
    ], dim=-1)
    out_BXTD = out_BXTDC.flatten(-2)

    return out_BXTD.type_as(x_BXTD)


class FeedForwardNet(nn.Module):
    def __init__(self, d_embd, d_hid, **kwargs):
        super().__init__()
        self.up_proj = nn.Linear(d_embd, d_hid, bias=False)
        self.gate_proj = nn.Linear(d_embd, d_hid, bias=False)
        self.down_proj = nn.Linear(d_hid, d_embd, bias=False)

    def forward(self, x_BTE):
        h_BTK = F.silu(self.gate_proj(x_BTE)) * self.up_proj(x_BTE)
        out_BTE = self.down_proj(h_BTK)
        return out_BTE

    def load_ref_weights(self, ref):
        self.up_proj.load_state_dict(ref.w3.state_dict())
        self.gate_proj.load_state_dict(ref.w1.state_dict())
        self.down_proj.load_state_dict(ref.w2.state_dict())


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

    def load_ref_weights(self, ref):
        self.load_state_dict(ref.state_dict())  # Parameter is a Tensor subclass
        self.eps = ref.eps


class LLaMABlock(nn.Module):
    def __init__(self, d_embd, **kwargs):
        super().__init__()
        self.attn_norm = RMSNorm(d_embd, **kwargs)
        self.attn = GroupedQueryAttention(d_embd, **kwargs)
        self.ffn_norm = RMSNorm(d_embd, **kwargs)
        self.ffn = FeedForwardNet(d_embd, **kwargs)

    def forward(self, x_BTE, freq_cis_TFC):
        h_BTE = x_BTE + self.attn(self.attn_norm(x_BTE), freq_cis_TFC)
        out_BTE = h_BTE + self.ffn(self.ffn_norm(h_BTE))
        return out_BTE

    def load_ref_weights(self, ref):
        self.attn_norm.load_ref_weights(ref.attention_norm)
        self.attn.load_ref_weights(ref.attention)
        self.ffn_norm.load_ref_weights(ref.ffn_norm)
        self.ffn.load_ref_weights(ref.feed_forward)


class LLaMA(nn.Module):
    def __init__(self, vocab_size, d_embd, n_layers, n_heads, **kwargs):
        super().__init__()
        self.tok_embd = nn.Embedding(vocab_size, d_embd)
        self.tsfmr_blks = nn.ModuleList(LLaMABlock(d_embd, n_heads=n_heads, **kwargs) for _ in range(n_layers))
        self.norm = RMSNorm(d_embd, **kwargs)
        self.lm_head = nn.Linear(d_embd, vocab_size, bias=False)
        self.register_buffer('freq_cis_TFC', precompute_freq_cis(d_embd//n_heads, **kwargs).to(self.lm_head.weight.dtype))

    def forward(self, idx_BT, **kwargs):
        x_BTE = self.tok_embd(idx_BT)
        for tsfmr_blk in self.tsfmr_blks:
            x_BTE = tsfmr_blk(x_BTE, self.freq_cis_TFC)
        logits_BTV = self.lm_head(self.norm(x_BTE))
        return logits_BTV

    def load_ref_weights(self, ref):
        self.tok_embd.load_state_dict(ref.tok_embeddings.state_dict())

        for tsfmr_blk, tsfmr_blk_ref in zip(self.tsfmr_blks, ref.layers):
            tsfmr_blk.load_ref_weights(tsfmr_blk_ref)

        self.norm.load_ref_weights(ref.norm)
        self.lm_head.load_state_dict(ref.output.state_dict())


def precompute_freq_cis(dim, rope_base, max_seq_len, **kwargs):
    assert dim % 2 == 0
    theta_F = 1 / (rope_base ** (torch.arange(0, dim, 2).float() / dim))  # F = dim // 2
    pos_idx_T = torch.arange(max_seq_len)
    freq_TF = pos_idx_T.unsqueeze(1) * theta_F.unsqueeze(0)
    freq_cis_TF = torch.polar(torch.ones_like(freq_TF), freq_TF)
    freq_cis_TFC = torch.stack([freq_cis_TF.real, freq_cis_TF.imag], dim=-1)
    return freq_cis_TFC


class Fp8LLaMA(nn.Module):
    def __init__(self, vocab_size, d_embd, n_layers, n_heads, **kwargs):
        super().__init__()
        self.tok_embd = nn.Embedding(vocab_size, d_embd)
        self.tsfmr_blks = nn.ModuleList(
            Fp8LLaMABlock(d_embd, n_heads=n_heads, **kwargs) for _ in range(n_layers)
        )
        self.norm_lm_head = te.LayerNormLinear(
            d_embd, vocab_size, bias=False,
            normalization='RMSNorm', eps=kwargs['norm_eps']
        )

        # Reference: https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/config.json
        freq_cis_TE = te.attention.RotaryPositionEmbedding(d_embd//n_heads)(max_seq_len=131072)
        self.register_buffer('freq_cis_TE', freq_cis_TE.to(torch.bfloat16))

    def forward(self, idx_BT, is_first_microbatch):
        x_BTE = self.tok_embd(idx_BT)
        for tsfmr_blk in self.tsfmr_blks:
            x_BTE = tsfmr_blk(x_BTE, rotary_pos_emb=self.freq_cis_TE, is_first_microbatch=is_first_microbatch)
        logits_BTV = self.norm_lm_head(x_BTE)
        return logits_BTV


class Fp8LLaMABlock(te.TransformerLayer):
    ''' Reference Implementation:
    https://github.com/NVIDIA/TransformerEngine/blob/55dcbb4b02f560d52dc1215a9de348b37487ee3d/docs/examples/te_llama/te_llama.py#L42
    '''
    def __init__(self, d_embd, d_hid, n_heads, n_kv_heads, norm_eps, **kwargs):
        super().__init__(
            hidden_size=d_embd,
            num_attention_heads=n_heads,
            num_gqa_groups=n_heads//n_kv_heads,
            fuse_qkv_params=True,
            attn_input_format='bshd',
            attention_dropout=0.0,
            normalization='RMSNorm',
            layernorm_epsilon=norm_eps,
            ffn_hidden_size=d_hid,
            bias=False,
            activation='swiglu',
            hidden_dropout=0.0
        )


if __name__ == '__main__':
    import json
    from pydantic import RootModel

    ll31_8b = LLaMAConfig(
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        d_embd=4096,
        max_seq_len=4096,
        vocab_size=128256,
        ffn_mult=1.3,
        ffn_factor=1024,
        rope_base=5e5,
        norm_eps=1e-5
    )
    assert ll31_8b.d_hid == 14336, f'Expected d_hid=14336, got {ll31_8b.d_hid}'
    with open('configs/llama-3.1-8b.json', 'w') as f:
        json.dump(RootModel[LLaMAConfig](ll31_8b).model_dump(), f, indent=2)

    ll31_8b_proxy = LLaMAConfig(
        n_layers=1,
        n_heads=32,
        n_kv_heads=8,
        d_embd=4096,
        max_seq_len=4096,
        vocab_size=128256,
        ffn_mult=1.3,
        ffn_factor=1024,
        rope_base=5e5,
        norm_eps=1e-5
    )
    with open('configs/llama-3.1-8b-proxy.json', 'w') as f:
        json.dump(RootModel[LLaMAConfig](ll31_8b_proxy).model_dump(), f, indent=2)

    ll31_70b = LLaMAConfig(
        n_layers=80,
        n_heads=64,
        n_kv_heads=8,
        d_embd=8192,
        ffn_mult=1.3,
        ffn_factor=1024,
        rope_base=5e5,
        max_seq_len=4096,
        vocab_size=128256,
        norm_eps=1e-5
    )
    assert ll31_70b.d_hid == 28672, f'Expected d_hid=28672, got {ll31_70b.d_hid}'
    with open('configs/llama-3.1-70b.json', 'w') as f:
        json.dump(RootModel[LLaMAConfig](ll31_70b).model_dump(), f, indent=2)

    ll31_70b_proxy = LLaMAConfig(
        n_layers=1,
        n_heads=64,
        n_kv_heads=8,
        d_embd=8192,
        max_seq_len=4096,
        vocab_size=128256,
        ffn_mult=1.3,
        ffn_factor=1024,
        rope_base=5e5,
        norm_eps=1e-5
    )
    with open('configs/llama-3.1-70b-proxy.json', 'w') as f:
        json.dump(RootModel[LLaMAConfig](ll31_70b_proxy).model_dump(), f, indent=2)

    ll2_7b = LLaMAConfig(
        n_layers=32,
        n_heads=32,
        n_kv_heads=32,
        d_embd=4096,
        max_seq_len=4096,
        vocab_size=3200,
        ffn_mult=1.0,
        ffn_factor=256,
        rope_base=1e5,
        norm_eps=1e-5,
        arch_name='llama'
    )
    with open('configs/llama-2-7b.json', 'w') as f:
        json.dump(RootModel[LLaMAConfig](ll2_7b).model_dump(), f, indent=2)

    ll2_7b_proxy = LLaMAConfig(
        n_layers=1,
        n_heads=32,
        n_kv_heads=32,
        d_embd=4096,
        max_seq_len=4096,
        vocab_size=3200,
        ffn_mult=1.0,
        ffn_factor=256,
        rope_base=1e5,
        norm_eps=1e-5,
        arch_name='llama'
    )
    with open('configs/llama-2-7b-proxy.json', 'w') as f:
        json.dump(RootModel[LLaMAConfig](ll2_7b_proxy).model_dump(), f, indent=2)
