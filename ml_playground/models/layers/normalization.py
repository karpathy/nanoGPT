from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """Layer normalization with optional bias, mirroring the original GPT implementation."""

    def __init__(self, ndim: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


__all__ = ["LayerNorm"]
