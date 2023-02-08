# from awgu repo
# https://github.com/awgu/nanoGPT/blob/fsdp/common_model.py

import math
from dataclasses import dataclass
from typing import Type

import torch
import torch.nn as nn


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1


def new_gelu(x):
    """
    Computes GeLU as in the Google BERT repo (identical to OpenAI GPT).
    See the Gaussian Error Linear Units (GELU) paper:
    https://arxiv.org/abs/1606.08415
    """
    return (
        0.5
        * x
        * (
            1.0
            + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        )
    )


class BlockBase(nn.Module):
    """
    Base class for the GPT block shared by the tensor parallel implementation
    and the normal non-distributed implementation.
    """

    def __init__(
        self,
        config: GPTConfig,
        causal_self_attn_class: Type[nn.Module],
        mlp_class: Type[nn.Module],
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = causal_self_attn_class(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = mlp_class(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
