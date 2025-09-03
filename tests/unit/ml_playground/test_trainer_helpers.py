from __future__ import annotations

import pytest

from ml_playground.trainer import get_lr
from ml_playground.config import LRSchedule, OptimConfig


def test_get_lr_no_decay():
    """If decay_lr is False, LR should be constant."""
    schedule = LRSchedule(decay_lr=False)
    optim = OptimConfig(learning_rate=0.1)
    lr = get_lr(0, schedule, optim)
    assert lr == 0.1


def test_get_lr_warmup_phase():
    """During warmup, LR should increase linearly."""
    schedule = LRSchedule(warmup_iters=10, lr_decay_iters=20, min_lr=0.01)
    optim = OptimConfig(learning_rate=0.1)
    # At half of warmup, LR should be half of learning_rate
    lr = get_lr(5, schedule, optim)
    assert lr == pytest.approx(0.05)

    # At the end of warmup, LR should be learning_rate
    lr = get_lr(10, schedule, optim)
    assert lr == 0.1


def test_get_lr_decay_phase():
    """After warmup, LR should decay towards min_lr."""
    schedule = LRSchedule(warmup_iters=10, lr_decay_iters=20, min_lr=0.01)
    optim = OptimConfig(learning_rate=0.1)
    # After warmup, at half of decay, LR should be halfway between learning_rate and min_lr
    lr = get_lr(15, schedule, optim)
    assert lr > 0.01 and lr < 0.1

    # At the end of decay, LR should be min_lr
    lr = get_lr(20, schedule, optim)
    assert lr == 0.01


def test_get_lr_past_decay():
    """After decay phase, LR should remain at min_lr."""
    schedule = LRSchedule(warmup_iters=10, lr_decay_iters=20, min_lr=0.01)
    optim = OptimConfig(learning_rate=0.1)
    lr = get_lr(25, schedule, optim)
    assert lr == 0.01


def test_get_lr_warmup_decay_edges():
    """Test edge cases for LR calculation."""
    schedule = LRSchedule(warmup_iters=10, lr_decay_iters=20, min_lr=0.01)
    optim = OptimConfig(learning_rate=0.1)
    # it=0 -> 0
    assert get_lr(0, schedule, optim) == 0.0
    # it=warmup_iters -> learning_rate
    assert get_lr(10, schedule, optim) == 0.1
    # it=lr_decay_iters -> min_lr
    assert get_lr(20, schedule, optim) == 0.01
    # it > lr_decay_iters -> min_lr
    assert get_lr(100, schedule, optim) == 0.01
