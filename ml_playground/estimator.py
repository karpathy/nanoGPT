from __future__ import annotations

# Backwards-compatibility shim
# Canonical implementation lives at ml_playground/models/utils/estimator.py
from ml_playground.models.utils.estimator import estimate_loss  # re-export

__all__ = ["estimate_loss"]
