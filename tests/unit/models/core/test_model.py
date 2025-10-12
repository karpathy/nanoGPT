from __future__ import annotations

import logging

import pytest
import torch

from ml_playground.configuration.models import (
    ModelConfig,
)
from ml_playground.models.core.model import GPT


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


def test_gpt_init_creates_model_with_correct_config() -> None:
    """GPT should initialize with correct configuration."""
    model = _make_model()
    assert model.config.n_layer == 1
    assert model.config.n_head == 1
    assert model.config.n_embd == 4
    assert model.config.vocab_size == 50


def test_gpt_forward_produces_correct_output_shape() -> None:
    """GPT forward should produce output with correct shape."""
    model = _make_model()
    model.eval()

    batch_size, seq_len = 2, 3
    idx = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        # Test with targets (training mode) - should return full sequence logits
        targets = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
        logits, loss = model(idx, targets)
        assert logits.shape == (batch_size, seq_len, model.config.vocab_size)
        assert loss is not None

        # Test without targets (inference mode) - should return only last token logits
        logits, loss = model(idx)
        assert logits.shape == (batch_size, 1, model.config.vocab_size)
        assert loss is None


def test_forward_without_targets_returns_none_loss() -> None:
    """forward should return None loss when targets are not provided."""
    model = _make_model()
    model.eval()

    idx = torch.randint(0, model.config.vocab_size, (1, 3))

    with torch.no_grad():
        logits, loss = model(idx)

    assert logits.shape == (1, 1, model.config.vocab_size)  # Only last token logits
    assert loss is None


def test_crop_block_size_raises_on_increase() -> None:
    """crop_block_size should raise ValueError when trying to increase block_size."""
    model = _make_model()
    original_block_size = model.config.block_size

    with pytest.raises(ValueError, match="block_size cannot be increased dynamically"):
        model.crop_block_size(original_block_size + 1)


def test_crop_block_size_updates_config_and_embeddings() -> None:
    """crop_block_size should update config and recreate embeddings when decreasing."""
    model = _make_model()
    original_block_size = model.config.block_size
    new_block_size = original_block_size - 1

    model.crop_block_size(new_block_size)

    assert model.config.block_size == new_block_size
    assert model.position_embeddings.num_embeddings == new_block_size


def test_from_pretrained_raises_not_implemented() -> None:
    """from_pretrained should raise NotImplementedError."""
    with pytest.raises(NotImplementedError, match="from_pretrained is not supported"):
        GPT.from_pretrained()


def test_configure_optimizers_calls_helper() -> None:
    """configure_optimizers should call the helper function."""
    model = _make_model()

    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=1e-3,
        betas=(0.9, 0.999),
        device_type="cpu",
    )

    assert isinstance(optimizer, torch.optim.Optimizer)


def test_estimate_mfu_delegates_to_function() -> None:
    """estimate_mfu should delegate to the standalone function."""
    model = _make_model()

    mfu = model.estimate_mfu(fwdbwd_per_iter=1, dt=0.1)

    assert isinstance(mfu, float)
    assert mfu > 0.0


def test_generate_delegates_to_function() -> None:
    """generate should delegate to the standalone function."""
    model = _make_model()
    model.eval()

    idx = torch.tensor([[1]], dtype=torch.long)

    with torch.no_grad():
        result = model.generate(idx, max_new_tokens=2, temperature=0.0)

    assert result.shape == (1, 3)  # Original + 2 new tokens
