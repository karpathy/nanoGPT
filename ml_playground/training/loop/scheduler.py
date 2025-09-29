"""Learning rate scheduling utilities for the training loop."""

from __future__ import annotations

from ml_playground import lr_scheduler
from ml_playground.configuration import LRSchedule, OptimConfig


__all__ = ["get_lr"]


def get_lr(it: int, schedule: LRSchedule, optim: OptimConfig) -> float:
    """Compute the learning rate for the current iteration."""
    if not schedule.decay_lr:
        return optim.learning_rate
    return lr_scheduler.get_lr(
        it,
        warmup=schedule.warmup_iters,
        decay_iters=schedule.lr_decay_iters,
        min_lr=schedule.min_lr,
        base_lr=optim.learning_rate,
    )
