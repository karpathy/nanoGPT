"""Runtime initialization helpers for training."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import ContextManager, Any

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


def setup_runtime(cfg: TrainerConfig) -> RuntimeContext:
    """Seed torch RNGs and configure autocast context based on runtime settings."""
    torch.manual_seed(cfg.runtime.seed)
    try:
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cfg.runtime.seed)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    except (RuntimeError, AssertionError, AttributeError):
        pass

    device_type = "cuda" if "cuda" in cfg.runtime.device else "cpu"
    dtype = _PT_DTYPES[cfg.runtime.dtype]
    ctx: ContextManager[Any] = (
        nullcontext()
        if device_type == "cpu"
        else autocast(device_type=device_type, dtype=dtype)
    )
    return RuntimeContext(device_type=device_type, autocast_context=ctx)
