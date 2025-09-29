"""Model initialization helpers used by the training loop."""

from __future__ import annotations

from typing import Tuple

import torch

from ml_playground.configuration import TrainerConfig
from ml_playground.models.core.model import GPT


__all__ = ["initialize_model"]


def initialize_model(cfg: TrainerConfig, logger) -> Tuple[GPT, torch.optim.Optimizer]:
    """Materialize the GPT model and optimizer with configured hyperparameters."""
    logger.info("Initializing model and optimizer")
    model = GPT(cfg.model, logger=logger)
    model.to(cfg.runtime.device)
    optimizer = model.configure_optimizers(
        cfg.optim.weight_decay,
        cfg.optim.learning_rate,
        (cfg.optim.beta1, cfg.optim.beta2),
        cfg.runtime.device,
    )
    return model, optimizer
