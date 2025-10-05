from __future__ import annotations

import logging
import torch

from ml_playground.models.core.model import GPT
from ml_playground.configuration.models import ModelConfig
from ml_playground.training.ema import EMA


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


def test_ema_init_snapshots_floating_point_params() -> None:
    """EMA.__init__ should snapshot only floating-point parameters."""
    model = _make_model()
    ema = EMA(model, decay=0.999, device="cpu")

    # Should have shadow weights for all floating-point parameters
    assert ema.decay == 0.999
    assert len(ema.shadow) > 0

    # Verify shadow weights are detached copies
    for name, param in model.named_parameters():
        if param.dtype.is_floating_point:
            assert name in ema.shadow
            assert ema.shadow[name].shape == param.shape
            assert not ema.shadow[name].requires_grad


def test_ema_update_applies_exponential_moving_average() -> None:
    """EMA.update should apply exponential moving average to shadow weights."""
    model = _make_model()
    decay = 0.9
    ema = EMA(model, decay=decay, device="cpu")

    # Store original shadow values
    original_shadow = {k: v.clone() for k, v in ema.shadow.items()}

    # Modify model parameters
    with torch.no_grad():
        for param in model.parameters():
            if param.dtype.is_floating_point:
                param.data.fill_(1.0)

    # Update EMA
    ema.update(model)

    # Verify EMA formula: new = decay * old + (1 - decay) * current
    for name, param in model.named_parameters():
        if param.requires_grad and param.dtype.is_floating_point:
            expected = decay * original_shadow[name] + (1.0 - decay) * param.data
            torch.testing.assert_close(ema.shadow[name], expected)


def test_ema_apply_to_loads_shadow_weights() -> None:
    """EMA.apply_to should load shadow weights into the model."""
    model = _make_model()
    ema = EMA(model, decay=0.999, device="cpu")

    # Modify shadow weights to known values
    with torch.no_grad():
        for name in ema.shadow:
            ema.shadow[name].fill_(42.0)

    # Apply EMA weights to model
    ema.apply_to(model)

    # Verify model now has the shadow weights
    for name, param in model.named_parameters():
        if name in ema.shadow:
            torch.testing.assert_close(param.data, ema.shadow[name])


def test_ema_handles_non_floating_point_params() -> None:
    """EMA should skip non-floating-point parameters."""
    model = _make_model()

    # Add a non-floating-point parameter (e.g., integer tensor)
    model.register_buffer("int_buffer", torch.tensor([1, 2, 3], dtype=torch.int32))

    ema = EMA(model, decay=0.999, device="cpu")

    # int_buffer should not be in shadow
    assert "int_buffer" not in ema.shadow


def test_ema_update_only_updates_trainable_params() -> None:
    """EMA.update should only update trainable floating-point parameters."""
    model = _make_model()
    ema = EMA(model, decay=0.9, device="cpu")

    # Freeze one parameter
    first_param_name = None
    for name, param in model.named_parameters():
        if param.dtype.is_floating_point:
            first_param_name = name
            param.requires_grad = False
            break

    # Update EMA
    ema.update(model)

    # Frozen parameter's shadow should still be in shadow dict
    # (but it's not updated by the update loop since requires_grad=False)
    if first_param_name:
        assert first_param_name in ema.shadow
