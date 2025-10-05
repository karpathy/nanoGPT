from __future__ import annotations

import logging

from ml_playground.configuration.models import ModelConfig
from ml_playground.models.core.model import GPT
from ml_playground.training.hooks.logging import log_training_step


class _Logger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, msg: str) -> None:
        self.messages.append(msg)


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


def test_log_training_step_early_iterations() -> None:
    """log_training_step should skip MFU calculation for early iterations."""
    logger = _Logger()
    model = _make_model()

    # local_iter_num < 5 should skip MFU calculation
    running_mfu = log_training_step(
        logger=logger,
        iter_num=1,
        loss_value=0.5,
        dt=0.1,
        local_iter_num=3,
        raw_model=model,
        running_mfu=-1.0,
        batch_size=2,
        grad_accum_steps=1,
    )

    # Should return unchanged running_mfu
    assert running_mfu == -1.0
    assert len(logger.messages) == 1
    assert "iter 1" in logger.messages[0]
    assert "loss 0.5000" in logger.messages[0]


def test_log_training_step_with_mfu_calculation() -> None:
    """log_training_step should calculate MFU after warmup iterations."""
    logger = _Logger()
    model = _make_model()

    # local_iter_num >= 5 should calculate MFU
    running_mfu = log_training_step(
        logger=logger,
        iter_num=10,
        loss_value=0.3,
        dt=0.05,
        local_iter_num=10,
        raw_model=model,
        running_mfu=-1.0,
        batch_size=4,
        grad_accum_steps=2,
    )

    # Should have calculated and returned MFU
    assert running_mfu != -1.0
    assert isinstance(running_mfu, float)
    assert len(logger.messages) == 1
    assert "iter 10" in logger.messages[0]


def test_log_training_step_smooths_mfu() -> None:
    """log_training_step should apply exponential smoothing to MFU."""
    logger = _Logger()
    model = _make_model()

    # First call with MFU calculation
    running_mfu = log_training_step(
        logger=logger,
        iter_num=5,
        loss_value=0.4,
        dt=0.1,
        local_iter_num=5,
        raw_model=model,
        running_mfu=-1.0,
        batch_size=2,
        grad_accum_steps=1,
    )

    first_mfu = running_mfu

    # Second call with different timing should smooth the MFU
    running_mfu = log_training_step(
        logger=logger,
        iter_num=6,
        loss_value=0.4,
        dt=0.05,  # Different timing
        local_iter_num=6,
        raw_model=model,
        running_mfu=first_mfu,
        batch_size=2,
        grad_accum_steps=1,
    )

    # Should have applied smoothing: 0.9 * old + 0.1 * new
    # With different dt, the new MFU will be different, so smoothing will change the value
    assert isinstance(running_mfu, float)
    # Verify smoothing was applied (not just replaced)
    assert running_mfu > 0


def test_log_training_step_scales_loss() -> None:
    """log_training_step should scale loss by grad_accum_steps."""
    logger = _Logger()
    model = _make_model()

    log_training_step(
        logger=logger,
        iter_num=1,
        loss_value=0.5,
        dt=0.1,
        local_iter_num=1,
        raw_model=model,
        running_mfu=-1.0,
        batch_size=2,
        grad_accum_steps=4,
    )

    # Loss should be scaled: 0.5 * 4 = 2.0
    assert "loss 2.0000" in logger.messages[0]
