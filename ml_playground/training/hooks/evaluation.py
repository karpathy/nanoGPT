"""Evaluation helpers for the training loop."""

from __future__ import annotations

from typing import Any, Callable, Dict

from torch.utils.tensorboard import SummaryWriter

from ml_playground.configuration.models import TrainerConfig
from ml_playground.data_pipeline.sampling.batches import SimpleBatches
from ml_playground.estimator import estimate_loss
from ml_playground.models.core.model import GPT


__all__ = ["run_evaluation"]


EstimateLossFn = Callable[[GPT, SimpleBatches, int, Any], Dict[str, float]]


def run_evaluation(
    cfg: TrainerConfig,
    *,
    logger,
    iter_num: int,
    lr: float,
    raw_model: GPT,
    batches: SimpleBatches,
    ctx,
    writer: SummaryWriter | None,
    estimate_loss_fn: EstimateLossFn | None = None,
) -> dict[str, float]:
    """Run validation, log metrics, and optionally record TensorBoard scalars."""
    loss_fn = estimate_loss_fn or estimate_loss
    losses = loss_fn(raw_model, batches, cfg.runtime.eval_iters, ctx)
    logger.info(
        f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
    )

    if writer:
        writer.add_scalar("Loss/train", losses["train"], iter_num)
        writer.add_scalar("Loss/val", losses["val"], iter_num)
        writer.add_scalar("LR", lr, iter_num)

    return losses
