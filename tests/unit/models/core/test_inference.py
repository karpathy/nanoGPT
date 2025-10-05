from __future__ import annotations

import logging
import torch
import pytest

from ml_playground.configuration.models import ModelConfig
from ml_playground.models.core.model import GPT
from ml_playground.models.core.inference import estimate_model_mfu, generate_tokens


def _make_model() -> GPT:
    """Create a minimal GPT model for testing."""
    cfg = ModelConfig(
        n_layer=1,
        n_head=1,
        n_embd=4,
        block_size=4,
        dropout=0.0,
        vocab_size=50,
    )
    logger = logging.getLogger(__name__)
    return GPT(cfg, logger)


def test_estimate_model_mfu_calculates_flops() -> None:
    """estimate_model_mfu should calculate model flops utilization."""
    model = _make_model()

    # Simulate some iteration timing
    fwdbwd_per_iter = 1
    dt = 0.1  # 100ms

    mfu = estimate_model_mfu(model, fwdbwd_per_iter, dt)

    # MFU should be a positive float
    assert isinstance(mfu, float)
    assert mfu > 0.0


def test_estimate_model_mfu_raises_without_config() -> None:
    """estimate_model_mfu should raise AttributeError if model has no config."""
    model = torch.nn.Linear(10, 10)

    with pytest.raises(AttributeError):
        estimate_model_mfu(model, 1, 0.1)


def test_generate_tokens_greedy_decoding() -> None:
    """generate_tokens should perform greedy decoding when temperature=0."""
    model = _make_model()
    model.eval()

    # Start with a single token
    idx = torch.tensor([[1]], dtype=torch.long)

    with torch.no_grad():
        result = generate_tokens(model, idx, max_new_tokens=2, temperature=0.0)

    # Should have generated 2 additional tokens
    assert result.shape == (1, 3)
    assert result[0, 0] == 1  # Original token preserved


def test_generate_tokens_with_temperature() -> None:
    """generate_tokens should sample with temperature > 0."""
    model = _make_model()
    model.eval()

    idx = torch.tensor([[1]], dtype=torch.long)

    with torch.no_grad():
        result = generate_tokens(model, idx, max_new_tokens=3, temperature=1.0)

    # Should have generated 3 additional tokens
    assert result.shape == (1, 4)


def test_generate_tokens_with_top_k() -> None:
    """generate_tokens should apply top-k filtering."""
    model = _make_model()
    model.eval()

    idx = torch.tensor([[1]], dtype=torch.long)

    with torch.no_grad():
        result = generate_tokens(model, idx, max_new_tokens=2, temperature=1.0, top_k=5)

    # Should have generated 2 additional tokens with top-k filtering
    assert result.shape == (1, 3)


def test_generate_tokens_raises_on_negative_temperature() -> None:
    """generate_tokens should raise ValueError for negative temperature."""
    model = _make_model()
    idx = torch.tensor([[1]], dtype=torch.long)

    with pytest.raises(ValueError, match="temperature must be >= 0.0"):
        generate_tokens(model, idx, max_new_tokens=1, temperature=-1.0)


def test_generate_tokens_raises_without_config() -> None:
    """generate_tokens should raise AttributeError if model has no config."""
    model = torch.nn.Linear(10, 10)
    idx = torch.tensor([[1]], dtype=torch.long)

    with pytest.raises(AttributeError):
        generate_tokens(model, idx, max_new_tokens=1)


def test_generate_tokens_clamps_out_of_vocab_indices() -> None:
    """generate_tokens should clamp indices that exceed vocab_size."""
    model = _make_model()
    model.eval()

    # Start with an out-of-vocab token (vocab_size=50, so 100 is invalid)
    idx = torch.tensor([[100]], dtype=torch.long)

    with torch.no_grad():
        result = generate_tokens(model, idx, max_new_tokens=1, temperature=0.0)

    # Should have clamped the initial token and generated one more
    assert result.shape == (1, 2)
    # First token should be clamped to max_vocab_idx (49)
    assert result[0, 0] <= 49


def test_generate_tokens_handles_long_context() -> None:
    """generate_tokens should truncate context longer than block_size."""
    model = _make_model()
    model.eval()

    # Create a context longer than block_size (4)
    idx = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)

    with torch.no_grad():
        result = generate_tokens(model, idx, max_new_tokens=1, temperature=0.0)

    # Should have generated 1 additional token
    assert result.shape == (1, 7)
