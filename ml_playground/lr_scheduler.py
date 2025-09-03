from __future__ import annotations

import math


def get_lr(
    it: int, *, warmup: int, decay_iters: int, min_lr: float, base_lr: float
) -> float:
    """Learning rate decay scheduler (cosine with warmup)."""
    # 1) linear warmup for warmup_iters steps
    if it < warmup:
        return base_lr * it / warmup
    # 2) if it > lr_decay_iters, return min learning rate
    if it > decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup) / (decay_iters - warmup)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (base_lr - min_lr)
