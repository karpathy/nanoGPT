"""Logging helpers for the training loop."""

from __future__ import annotations

from ml_playground.models.core.model import GPT


__all__ = ["log_training_step"]


def log_training_step(
    logger,
    iter_num: int,
    loss_value: float,
    dt: float,
    local_iter_num: int,
    raw_model: GPT,
    running_mfu: float,
    batch_size: int,
    grad_accum_steps: int,
) -> float:
    """Log training progress and compute updated model FLOPS utilization."""
    scaled_loss = loss_value * grad_accum_steps
    if local_iter_num >= 5:
        mfu = raw_model.estimate_mfu(batch_size * grad_accum_steps, dt)
        running_mfu = (
            mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * float(mfu)
        )

    mfu_pct = max(0.0, min(float(running_mfu), 100.0))
    logger.info(
        f"iter {iter_num}: loss {scaled_loss:.4f}, time {dt * 1000:.2f}ms, mfu {mfu_pct:.2f}%"
    )
    return running_mfu
