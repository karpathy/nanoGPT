"""Optional component setup utilities for training."""

from __future__ import annotations

from typing import Callable, Optional, Tuple, cast

import torch
from torch.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter

from ml_playground.configuration.models import TrainerConfig
from ml_playground.training.ema import EMA
from ml_playground.models.core.model import GPT
from ml_playground.training.hooks.runtime import RuntimeContext


__all__ = ["initialize_components"]


def initialize_components(
    model: GPT,
    cfg: TrainerConfig,
    runtime: RuntimeContext,
    *,
    log_dir: str,
    compile_fn: Optional[Callable[[GPT], GPT]] = None,
) -> Tuple[GPT, GradScaler, Optional[EMA], Optional[SummaryWriter]]:
    """Compile model, create scaler/EMA, and initialize TensorBoard writer."""
    compiled_model = model
    if cfg.runtime.compile:
        compiler = compile_fn
        if compiler is None:
            compiler = getattr(torch, "compile", None)
        if compiler is None:
            raise RuntimeError("torch.compile requested but unavailable")
        compiled_model = cast(GPT, compiler(model))

    scaler = GradScaler(
        enabled=(runtime.device_type == "cuda" and cfg.runtime.dtype == "float16")
    )

    ema: Optional[EMA] = None
    if cfg.runtime.ema_decay > 0.0:
        ema = EMA(compiled_model, cfg.runtime.ema_decay, cfg.runtime.device)

    writer: Optional[SummaryWriter] = None
    if cfg.runtime.tensorboard_enabled:
        writer = SummaryWriter(log_dir=log_dir)

    return compiled_model, scaler, ema, writer
