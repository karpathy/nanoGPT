from __future__ import annotations

import pytest

from ml_playground.lr_scheduler import get_lr


def test_warmup_phase_linear_scaling() -> None:
    # Warmup: linear from 0 -> base_lr over `warmup` steps
    base_lr = 1.0
    min_lr = 0.1
    warmup = 10
    decay_iters = 110

    assert get_lr(
        0, warmup=warmup, decay_iters=decay_iters, min_lr=min_lr, base_lr=base_lr
    ) == pytest.approx(0.0)
    assert get_lr(
        5, warmup=warmup, decay_iters=decay_iters, min_lr=min_lr, base_lr=base_lr
    ) == pytest.approx(0.5)
    # At warmup boundary, cosine decay starts; ratio becomes 0 -> full base_lr
    assert get_lr(
        10, warmup=warmup, decay_iters=decay_iters, min_lr=min_lr, base_lr=base_lr
    ) == pytest.approx(1.0)


def test_cosine_decay_midpoint() -> None:
    base_lr = 1.0
    min_lr = 0.1
    warmup = 10
    decay_iters = 110
    # Midpoint of decay (ratio=0.5) -> coeff=0.5, so halfway between base and min
    it = 60  # ratio = (60-10)/(110-10) = 0.5
    expected = min_lr + 0.5 * (base_lr - min_lr)  # 0.1 + 0.5*0.9 = 0.55
    assert get_lr(
        it, warmup=warmup, decay_iters=decay_iters, min_lr=min_lr, base_lr=base_lr
    ) == pytest.approx(expected)


def test_end_of_decay_and_beyond() -> None:
    base_lr = 1.0
    min_lr = 0.1
    warmup = 10
    decay_iters = 110

    # At decay_iters we should be at min_lr (ratio=1 -> coeff=0)
    assert get_lr(
        110, warmup=warmup, decay_iters=decay_iters, min_lr=min_lr, base_lr=base_lr
    ) == pytest.approx(min_lr)
    # Beyond decay_iters we clamp to min_lr
    assert get_lr(
        200, warmup=warmup, decay_iters=decay_iters, min_lr=min_lr, base_lr=base_lr
    ) == pytest.approx(min_lr)


def test_nontrivial_params_precision() -> None:
    # Use nontrivial parameters and validate a couple of points with approx
    base_lr = 2.5
    min_lr = 0.25
    warmup = 8
    decay_iters = 108

    # Warmup scale
    assert get_lr(
        4, warmup=warmup, decay_iters=decay_iters, min_lr=min_lr, base_lr=base_lr
    ) == pytest.approx(base_lr * 4 / warmup)

    # Some interior point
    it = 58  # ratio = (58-8)/(108-8) = 50/100 = 0.5 -> coeff=0.5
    expected = min_lr + 0.5 * (base_lr - min_lr)
    assert get_lr(
        it, warmup=warmup, decay_iters=decay_iters, min_lr=min_lr, base_lr=base_lr
    ) == pytest.approx(expected, rel=1e-6, abs=1e-9)
