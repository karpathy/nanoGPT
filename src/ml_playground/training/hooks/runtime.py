"""Runtime initialization helpers for training."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import ContextManager, Any, Callable

import torch
from torch import autocast

from ml_playground.configuration.models import TrainerConfig


__all__ = ["RuntimeContext", "setup_runtime"]


@dataclass(slots=True)
class RuntimeContext:
    """Runtime artifacts required by the training loop."""

    device_type: str
    autocast_context: ContextManager[Any]


_PT_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def setup_runtime(
    cfg: TrainerConfig,
    *,
    cuda_available_func: Callable[[], bool] | None = None,
    cuda_seed_func: Callable[[int], None] | None = None,
    autocast_func: Callable[[str, torch.dtype], ContextManager[Any]] | None = None,
) -> RuntimeContext:
    """Seed torch RNGs and configure autocast context based on runtime settings.

    Optional callables allow injecting test doubles for CUDA availability, seeding, and autocast creation.
    """
    torch.manual_seed(cfg.runtime.seed)
    try:
        cuda_available = (
            cuda_available_func()
            if cuda_available_func is not None
            else torch.cuda.is_available()
        )
        if cuda_available:
            (
                cuda_seed_func(cfg.runtime.seed)
                if cuda_seed_func is not None
                else torch.cuda.manual_seed(cfg.runtime.seed)
            )
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    except (RuntimeError, AssertionError, AttributeError):
        pass

    device_type = "cuda" if "cuda" in cfg.runtime.device else "cpu"
    dtype = _PT_DTYPES[cfg.runtime.dtype]
    ctx: ContextManager[Any] = (
        nullcontext()
        if device_type == "cpu"
        else (
            autocast_func(device_type, dtype)
            if autocast_func is not None
            else autocast(device_type=device_type, dtype=dtype)
        )
    )
    return RuntimeContext(device_type=device_type, autocast_context=ctx)
