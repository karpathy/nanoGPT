from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from llama import *

''' Reference implemenation taken from Horace's GPT-fast. '''


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)

@dataclass
class ModelArgs:
    block_size: int = 2048
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    rope_scaling: Optional[dict] = None

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head

cfg = ModelArgs(n_local_heads=8)


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        y = self.wo(y)
        return y


def ref_precompute_freqs_cis(
    seq_len: int, n_elem: int, base: int = 10000,
    dtype: torch.dtype = torch.bfloat16,
    rope_scaling: Optional[dict] = None,
) -> Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    if rope_scaling is not None:
        freqs = apply_rope_scaling(freqs, rope_scaling)
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RefRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class RefTransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: Tensor) -> Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(RefTransformerBlock(config) for _ in range(config.n_layer))
        self.norm = RefRMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        freqs_cis = ref_precompute_freqs_cis(
            self.config.block_size, self.config.dim // self.config.n_head, self.config.rope_base, self.output.weight.dtype
        )
        self.register_buffer('freqs_cis', freqs_cis)

        causal_mask = torch.tril(torch.ones(self.config.block_size, self.config.block_size, dtype=torch.bool))
        self.register_buffer('causal_mask', causal_mask)

    def forward(self, idx: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        x = self.tok_embeddings(idx)

        for i, layer in enumerate(self.layers):
            x = layer(x, input_pos, self.freqs_cis, self.causal_mask)
        x = self.norm(x)
        logits = self.output(x)
        return logits


# ========= Tests =========

torch.set_grad_enabled(False)


def test_rope_freq_cis():
    freq_cis_ref = ref_precompute_freqs_cis(
        cfg.block_size,
        cfg.dim//cfg.n_head,
        cfg.rope_base,
        torch.float16
    )
    freq_cis = precompute_freq_cis(cfg.dim//cfg.n_head, cfg.rope_base, cfg.block_size)
    assert torch.allclose(freq_cis.to(freq_cis_ref), freq_cis_ref)


def test_apply_rope():
    freq_cis_ref = ref_precompute_freqs_cis(
        cfg.block_size,
        cfg.dim//cfg.n_head,
        cfg.rope_base,
        torch.float16
    ).to('cuda')
    freq_cis = precompute_freq_cis(cfg.dim//cfg.n_head, cfg.rope_base, cfg.block_size).to(freq_cis_ref)

    x_BHTD = torch.rand([2, cfg.n_head, cfg.block_size, cfg.dim//cfg.n_head], device='cuda')
    y_BHTD_ref = apply_rotary_emb(x_BHTD.transpose(1, 2), freq_cis_ref).transpose(1, 2)
    y_BHTD = apply_rotary_embd(x_BHTD, freq_cis)

    assert torch.allclose(y_BHTD, y_BHTD_ref)


def test_attn():
    attn_ref = Attention(cfg).to('cuda')
    freq_cis_ref = ref_precompute_freqs_cis(
        cfg.block_size,
        cfg.dim//cfg.n_head,
        cfg.rope_base,
        torch.float16
    ).to('cuda')

    attn = GroupedQueryAttention(cfg.dim, cfg.n_head, cfg.n_local_heads).to('cuda')
    freq_cis = precompute_freq_cis(cfg.dim//cfg.n_head, cfg.rope_base, cfg.block_size).to(freq_cis_ref)
    attn.load_ref_weights(attn_ref)

    x_BTE = torch.rand([2, cfg.block_size, cfg.dim], device='cuda')
    mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool, device='cuda'))

    y_BTE_ref = attn_ref(x_BTE, freq_cis_ref, mask)
    y_BTE = attn(x_BTE, freq_cis)

    assert torch.allclose(y_BTE, y_BTE_ref)


def test_ffn():
    ffn_ref = FeedForward(cfg).to('cuda')
    ffn = FeedForwardNet(cfg.dim, cfg.intermediate_size).to('cuda')
    ffn.load_ref_weights(ffn_ref)

    x_BTE = torch.rand([2, cfg.block_size, cfg.dim], device='cuda')
    y_BTE_ref = ffn_ref(x_BTE)
    y_BTE = ffn(x_BTE)

    assert torch.allclose(y_BTE, y_BTE_ref)


def test_rms_norm():
    norm_ref = RefRMSNorm(cfg.dim, cfg.norm_eps).to('cuda')
    norm = RMSNorm(cfg.dim, cfg.norm_eps).to('cuda')
    norm.load_ref_weights(norm_ref)

    x_BTE = torch.rand([2, cfg.block_size, cfg.dim], device='cuda')
    y_BTE_ref = norm_ref(x_BTE)
    y_BTE = norm(x_BTE)

    assert torch.allclose(y_BTE, y_BTE_ref)


def test_llama_block():
    tsfmr_blk_ref = RefTransformerBlock(cfg).to('cuda')
    freq_cis_ref = ref_precompute_freqs_cis(
        cfg.block_size,
        cfg.dim//cfg.n_head,
        cfg.rope_base,
        torch.float16
    ).to('cuda')
    mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool, device='cuda'))

    tsfmr_blk = LLaMABlock(
        cfg.dim,
        norm_eps=cfg.norm_eps,
        n_heads=cfg.n_head,
        n_kv_heads=cfg.n_local_heads,
        d_hid=cfg.intermediate_size
    ).to('cuda')
    freq_cis = precompute_freq_cis(cfg.dim//cfg.n_head, cfg.rope_base, cfg.block_size).to(freq_cis_ref)
    tsfmr_blk.load_ref_weights(tsfmr_blk_ref)

    x_BTE = torch.rand([2, cfg.block_size, cfg.dim], device='cuda')
    y_BTE_ref = tsfmr_blk_ref(x_BTE, None, freq_cis_ref, mask)
    y_BTE = tsfmr_blk(x_BTE, freq_cis)

    assert torch.allclose(y_BTE, y_BTE_ref)


def test_llama():
    cfg.n_layer = 1  # Just test proxy

    llama_ref = Transformer(cfg).to('cuda')
    llama = LLaMA(
        cfg.vocab_size, cfg.dim, cfg.n_layer, cfg.n_head,
        norm_eps=cfg.norm_eps,
        n_kv_heads=cfg.n_local_heads,
        d_hid=cfg.intermediate_size,
        rope_base=cfg.rope_base,
        max_seq_len=cfg.block_size
    ).to('cuda')
    llama.load_ref_weights(llama_ref)

    idx_BT = torch.randint(cfg.vocab_size, [2, cfg.block_size], dtype=torch.int64, device='cuda')
    logits_ref = llama_ref(idx_BT)
    logits = llama(idx_BT)

    assert torch.allclose(logits, logits_ref)
