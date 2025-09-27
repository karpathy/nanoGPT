from __future__ import annotations

import torch
import torch.nn as nn

from ml_playground.models.core.config import GPTConfig
from ml_playground.models.layers.attention import CausalSelfAttention
from ml_playground.models.layers.mlp import MLP
from ml_playground.models.layers.normalization import LayerNorm


class Block(nn.Module):
    """Single transformer block (attention + MLP with residuals)."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


__all__ = ["Block"]
