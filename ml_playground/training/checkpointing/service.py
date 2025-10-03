"""Checkpoint management helpers for the training loop."""

from __future__ import annotations

import shutil
from typing import Optional

from ml_playground.checkpoint import Checkpoint, CheckpointManager
from ml_playground.configuration.models import (
    TrainerConfig,
    SharedConfig,
    READ_POLICY_BEST,
)
from ml_playground.core.error_handling import CheckpointError
from ml_playground.models.core.model import GPT


__all__ = [
    "create_manager",
    "load_checkpoint",
    "apply_checkpoint",
    "save_checkpoint",
    "propagate_metadata",
]


def create_manager(cfg: TrainerConfig, shared: SharedConfig) -> CheckpointManager:
    """Construct a checkpoint manager respecting the retention policy."""
    return CheckpointManager(
        out_dir=shared.train_out_dir,
        atomic=cfg.runtime.ckpt_atomic,
        keep_last=cfg.runtime.checkpointing.keep.last,
        keep_best=cfg.runtime.checkpointing.keep.best,
    )


def load_checkpoint(
    manager: CheckpointManager,
    cfg: TrainerConfig,
    *,
    logger,
) -> Optional[Checkpoint]:
    """Load the latest or best checkpoint according to the read policy."""
    # DI override if provided
    if cfg.checkpoint_load_fn is not None:
        try:
            return cfg.checkpoint_load_fn(manager=manager, cfg=cfg, logger=logger)
        except (
            CheckpointError,
            RuntimeError,
        ) as exc:  # pragma: no cover - DI override path is user-supplied
            logger.warning(f"checkpoint_load_fn failed: {exc}")
            return None

    if not manager.out_dir.exists():
        return None

    try:
        if cfg.runtime.checkpointing.read_policy == READ_POLICY_BEST:
            return manager.load_best_checkpoint(
                device=cfg.runtime.device, logger=logger
            )
        return manager.load_latest_checkpoint(device=cfg.runtime.device, logger=logger)
    except CheckpointError as exc:
        logger.warning(
            f"Could not load checkpoint ({cfg.runtime.checkpointing.read_policy}): {exc}"
        )
        return None


def apply_checkpoint(
    checkpoint: Checkpoint,
    *,
    model: GPT,
    optimizer,
    ema,
) -> tuple[int, float]:
    """Apply checkpoint state to the model/optimizer and return iteration metrics."""
    model.load_state_dict(checkpoint.model, strict=False)
    optimizer.load_state_dict(checkpoint.optimizer)
    iter_num = checkpoint.iter_num
    best_val_loss = checkpoint.best_val_loss
    if ema and checkpoint.ema:
        ema.shadow = checkpoint.ema
    return iter_num, best_val_loss


def save_checkpoint(
    manager: CheckpointManager,
    cfg: TrainerConfig,
    *,
    model: GPT,
    optimizer,
    ema,
    iter_num: int,
    best_val_loss: float,
    logger,
    is_best: bool,
) -> None:
    """Persist the current training state via the checkpoint manager."""
    checkpoint = Checkpoint(
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        model_args=cfg.model.model_dump(),
        iter_num=iter_num,
        best_val_loss=best_val_loss,
        config=cfg.model_dump(),
        ema=ema.shadow if ema else None,
    )
    # DI override if provided
    if cfg.checkpoint_save_fn is not None:
        try:
            cfg.checkpoint_save_fn(
                manager=manager,
                cfg=cfg,
                checkpoint=checkpoint,
                is_best=is_best,
                logger=logger,
            )
            return
        except (
            CheckpointError,
            RuntimeError,
        ) as exc:  # pragma: no cover - DI override path is user-supplied
            logger.warning(
                f"checkpoint_save_fn failed, falling back to default save: {exc}"
            )

    base_filename = "ckpt_best.pt" if is_best else "ckpt_last.pt"
    manager.save_checkpoint(
        checkpoint,
        base_filename=base_filename,
        metric=best_val_loss,
        iter_num=iter_num,
        logger=logger,
        is_best=is_best,
    )


def propagate_metadata(cfg: TrainerConfig, shared: SharedConfig, *, logger) -> None:
    """Copy dataset metadata into train and sample output directories when available."""
    try:
        meta_src = cfg.data.meta_path(shared.dataset_dir)
    except (
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
    ) as exc:  # pragma: no cover - defensive
        if logger:
            logger.warning(f"Failed to resolve meta source path: {exc}")
        return

    if not meta_src or not meta_src.exists():
        return

    destinations = [shared.train_out_dir]
    if shared.sample_out_dir not in destinations:
        destinations.append(shared.sample_out_dir)

    for dst_dir in destinations:
        try:
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(meta_src, dst_dir / meta_src.name)
        except (OSError, IOError) as exc:
            if logger:
                logger.warning(f"Failed to copy meta file to {dst_dir}: {exc}")
