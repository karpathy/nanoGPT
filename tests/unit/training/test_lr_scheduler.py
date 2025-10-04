from __future__ import annotations

import pytest

from ml_playground.training.optim.lr_scheduler import get_lr


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


def test_cosine_decay_quarter_and_three_quarters() -> None:
    base_lr = 1.0
    min_lr = 0.1
    warmup = 10
    decay_iters = 110
    # Quarter point (ratio=0.25)
    it_q = 35  # (35-10)/(110-10) = 25/100 = 0.25
    expected_q = min_lr + 0.5 * (1.0 + (-0.0 + 1.0)) * (
        base_lr - min_lr
    )  # simplified below
    # More explicitly via cosine: coeff = 0.5 * (1 + cos(pi*0.25))
    import math as _math

    coeff_q = 0.5 * (1.0 + _math.cos(_math.pi * 0.25))
    expected_q = min_lr + coeff_q * (base_lr - min_lr)
    assert get_lr(
        it_q, warmup=warmup, decay_iters=decay_iters, min_lr=min_lr, base_lr=base_lr
    ) == pytest.approx(expected_q, rel=1e-6, abs=1e-9)

    # Three-quarters (ratio=0.75)
    it_tq = 85  # (85-10)/(110-10) = 75/100 = 0.75
    coeff_tq = 0.5 * (1.0 + _math.cos(_math.pi * 0.75))
    expected_tq = min_lr + coeff_tq * (base_lr - min_lr)
    assert get_lr(
        it_tq, warmup=warmup, decay_iters=decay_iters, min_lr=min_lr, base_lr=base_lr
    ) == pytest.approx(expected_tq, rel=1e-6, abs=1e-9)


def test_warmup_additional_points() -> None:
    base_lr = 1.0
    min_lr = 0.0
    warmup = 8
    decay_iters = 108
    # Additional warmup checks to catch numeric literal mutations
    for it in (1, 2, 3, 7):
        assert get_lr(
            it, warmup=warmup, decay_iters=decay_iters, min_lr=min_lr, base_lr=base_lr
        ) == pytest.approx(base_lr * it / warmup)


def test_cosine_shape_properties_many_points() -> None:
    base_lr = 1.0
    min_lr = 0.1
    warmup = 10
    decay_iters = 110
    # Values must be within [min_lr, base_lr] during cosine decay and after
    for it in range(warmup, decay_iters + 1, 5):
        lr = get_lr(
            it, warmup=warmup, decay_iters=decay_iters, min_lr=min_lr, base_lr=base_lr
        )
        assert min_lr <= lr <= base_lr
    # Strict interior bounds: immediately after warmup and before decay end
    lr_after_warmup = get_lr(
        warmup + 1,
        warmup=warmup,
        decay_iters=decay_iters,
        min_lr=min_lr,
        base_lr=base_lr,
    )
    assert lr_after_warmup < base_lr
    lr_before_end = get_lr(
        decay_iters - 1,
        warmup=warmup,
        decay_iters=decay_iters,
        min_lr=min_lr,
        base_lr=base_lr,
    )
    assert lr_before_end > min_lr
    # Non-increasing after warmup (strict, no epsilon)
    prev = get_lr(
        warmup, warmup=warmup, decay_iters=decay_iters, min_lr=min_lr, base_lr=base_lr
    )
    for it in range(warmup + 1, decay_iters + 1, 3):
        lr = get_lr(
            it, warmup=warmup, decay_iters=decay_iters, min_lr=min_lr, base_lr=base_lr
        )
        assert lr <= prev
        prev = lr
    # Additional ratio checks close to edges
    import math as _math

    for r, it in [(0.1, 20), (0.9, 100)]:
        coeff = 0.5 * (1.0 + _math.cos(_math.pi * r))
        expected = min_lr + coeff * (base_lr - min_lr)
        lr = get_lr(
            it, warmup=warmup, decay_iters=decay_iters, min_lr=min_lr, base_lr=base_lr
        )
        assert lr == pytest.approx(expected, rel=1e-6, abs=1e-9)

    # Cosine symmetry: lr(r) + lr(1-r) == base_lr + min_lr
    for r in [0.2, 0.3, 0.4]:
        it1 = int(warmup + r * (decay_iters - warmup))
        it2 = int(warmup + (1 - r) * (decay_iters - warmup))
        lr1 = get_lr(
            it1, warmup=warmup, decay_iters=decay_iters, min_lr=min_lr, base_lr=base_lr
        )
        lr2 = get_lr(
            it2, warmup=warmup, decay_iters=decay_iters, min_lr=min_lr, base_lr=base_lr
        )
        assert (lr1 + lr2) == pytest.approx(base_lr + min_lr, rel=1e-6, abs=1e-9)

    # Dense ratio sweep to validate cosine constants are correct
    # Map ratios to specific it values: it = warmup + r*(decay_iters - warmup)
    for r in [i / 10.0 for i in range(1, 10)]:
        it = int(warmup + r * (decay_iters - warmup))
        coeff = 0.5 * (1.0 + _math.cos(_math.pi * r))
        expected = min_lr + coeff * (base_lr - min_lr)
        lr = get_lr(
            it, warmup=warmup, decay_iters=decay_iters, min_lr=min_lr, base_lr=base_lr
        )
        assert lr == pytest.approx(expected, rel=1e-6, abs=1e-9)


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
