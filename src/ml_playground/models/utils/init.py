from __future__ import annotations

import torch.nn as nn


def init_transformer_weights(module: nn.Module) -> None:
    """Initialize transformer weights following the nanoGPT defaults."""

    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


__all__ = ["init_transformer_weights"]
