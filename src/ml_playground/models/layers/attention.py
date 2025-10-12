from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F

from ml_playground.models.core.config import GPTConfig


class CausalSelfAttention(nn.Module):
    """Causal self-attention block shared across GPT variants."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.n_head <= 0:
            raise ValueError("n_head must be a positive integer")
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(
            batch_size, seq_len, self.n_head, embed_dim // self.n_head
        ).transpose(1, 2)
        q = q.view(
            batch_size, seq_len, self.n_head, embed_dim // self.n_head
        ).transpose(1, 2)
        v = v.view(
            batch_size, seq_len, self.n_head, embed_dim // self.n_head
        ).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        y = self.c_proj(y)
        return y


__all__ = ["CausalSelfAttention"]
