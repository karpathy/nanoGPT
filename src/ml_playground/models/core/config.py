from __future__ import annotations

from ml_playground.configuration.models import ModelConfig


class GPTConfig:
    """Minimal configuration container for GPT models."""

    def __init__(
        self,
        block_size: int,
        vocab_size: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias


def build_gpt_config(cfg: ModelConfig) -> GPTConfig:
    if cfg.vocab_size is None:
        raise ValueError("ModelConfig.vocab_size must be set before building GPTConfig")
    return GPTConfig(
        block_size=cfg.block_size,
        vocab_size=cfg.vocab_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
        bias=cfg.bias,
    )


__all__ = ["GPTConfig", "build_gpt_config"]
