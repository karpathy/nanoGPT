from __future__ import annotations

# Backwards-compatibility shim
# Canonical implementation lives at ml_playground/training/optim/lr_scheduler.py
from ml_playground.training.optim.lr_scheduler import get_lr  # re-export

__all__ = ["get_lr"]
