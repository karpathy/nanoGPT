from __future__ import annotations

import math


__all__ = ["get_lr"]


def get_lr(
    it: int, *, warmup: int, decay_iters: int, min_lr: float, base_lr: float
) -> float:
    """Cosine decay learning rate scheduler with linear warmup.

    - Warmup: linearly scales from 0 to base_lr over `warmup` steps.
    - Cosine decay: decays from base_lr to min_lr over (decay_iters - warmup) steps.
    - After decay_iters: clamps to min_lr.
    """
    # 1) linear warmup for warmup steps: 0 -> base_lr (ignores min_lr during warmup)
    if it < warmup:
        return base_lr * (it / warmup)
    # 2) at or beyond decay_iters, return min learning rate
    if it >= decay_iters:
        return min_lr
    # 3) cosine decay between warmup and decay_iters
    decay_ratio = (it - warmup) / (decay_iters - warmup)
    assert 0.0 <= decay_ratio <= 1.0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff in [0, 1]
    return min_lr + coeff * (base_lr - min_lr)
