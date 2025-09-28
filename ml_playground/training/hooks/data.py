"""Data loading helpers for the training loop."""

from __future__ import annotations

from pathlib import Path

from ml_playground.config import TrainerConfig, SharedConfig
from ml_playground.data import SimpleBatches


__all__ = ["initialize_batches"]


def initialize_batches(cfg: TrainerConfig, shared: SharedConfig) -> SimpleBatches:
    """Create a `SimpleBatches` iterator bound to the resolved dataset directory."""
    dataset_dir: Path = shared.dataset_dir
    return SimpleBatches(
        data=cfg.data, device=cfg.runtime.device, dataset_dir=dataset_dir
    )
