from __future__ import annotations

# Backwards-compatibility shim
# Canonical implementation lives at ml_playground/training/ema.py
from ml_playground.training.ema import EMA  # re-export

__all__ = ["EMA"]
