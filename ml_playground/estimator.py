from __future__ import annotations

from typing import Any, Dict

import torch

from ml_playground.data import SimpleBatches
from ml_playground.model import GPT


@torch.no_grad()
def estimate_loss(
    model: GPT, batches: SimpleBatches, eval_iters: int, ctx: Any
) -> Dict[str, float]:
    """Estimate loss on train/val splits."""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = batches.get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out
