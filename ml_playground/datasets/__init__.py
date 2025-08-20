from __future__ import annotations

from typing import Callable, Dict

# Legacy compatibility registry for dataset preparers.
# By default, alias to the experiments registry so existing experiment
# registrations are visible when the legacy package is present.
try:
    from ml_playground.experiments import PREPARERS as _EXP_PREPARERS  # type: ignore
except Exception:  # pragma: no cover - best effort
    _EXP_PREPARERS = {}  # type: ignore

PREPARERS: Dict[str, Callable[[], None]] = _EXP_PREPARERS  # alias by default
