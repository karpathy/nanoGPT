from __future__ import annotations

import pytest
import torch

from ml_playground.models.layers.attention import CausalSelfAttention
from ml_playground.models.core.config import GPTConfig


def _make_config(
    *, n_embd: int = 64, n_head: int = 8, dropout: float = 0.1
) -> GPTConfig:
    return GPTConfig(
        block_size=16,
        vocab_size=128,
        n_layer=2,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        bias=True,
    )


def test_causal_self_attention_init_with_valid_params() -> None:
    """CausalSelfAttention should initialize successfully with valid parameters."""
    config = _make_config()
    attention = CausalSelfAttention(config)
    assert attention is not None


def test_causal_self_attention_init_raises_on_invalid_n_embd() -> None:
    """CausalSelfAttention should raise ValueError when n_embd not divisible by n_head."""
    config = _make_config(n_embd=62, n_head=7)
    with pytest.raises(ValueError, match="n_embd must be divisible"):
        CausalSelfAttention(config)


def test_causal_self_attention_init_raises_on_invalid_n_head() -> None:
    """CausalSelfAttention should raise ValueError when n_head <= 0."""
    config = _make_config(n_head=0)
    with pytest.raises(ValueError, match="n_head must be a positive integer"):
        CausalSelfAttention(config)


def test_causal_self_attention_forward_produces_correct_shape() -> None:
    """CausalSelfAttention forward should produce output with correct shape."""
    config = _make_config()
    batch_size, seq_len = 2, 10

    attention = CausalSelfAttention(config)
    x = torch.randn(batch_size, seq_len, config.n_embd)

    output = attention(x)

    # Output should have same shape as input
    assert output.shape == (batch_size, seq_len, config.n_embd)
