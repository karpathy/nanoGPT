from __future__ import annotations

from typing import Any, Dict, Literal, Tuple, cast

import torch

from ml_playground.data_pipeline.sampling.batches import SimpleBatches
from ml_playground.models.core.model import GPT


__all__ = ["estimate_loss"]


@torch.no_grad()
def estimate_loss(
    model: GPT, batches: SimpleBatches, eval_iters: int, ctx: Any
) -> Dict[str, float]:
    """Estimate loss on train/val splits."""
    out: Dict[str, float] = {}
    model.eval()
    splits: Tuple[Literal["train"], Literal["val"]] = ("train", "val")
    for split in splits:
        losses = torch.zeros(eval_iters, dtype=torch.float32)
        for k in range(eval_iters):
            X, Y = batches.get_batch(cast(Literal["train", "val"], split))
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out
