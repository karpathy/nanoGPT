from __future__ import annotations
from typing import Callable, Dict

# Registry of dataset preparers used by CLI and tests.
# This module must remain cheap to import and side-effect free.
PREPARERS: Dict[str, Callable[[], None]] = {}
# Keep a reference to the original dict object to detect monkeypatching in tests/CLI
DEFAULT_PREPARERS_REF = PREPARERS

def load_preparers() -> None:
    """Populate PREPARERS by importing experiment prepare modules on demand.

    Delegates discovery to ml_playground.experiments.load_preparers and mirrors
    its registry to avoid hardcoded experiment references here.
    """
    if PREPARERS:
        return
    # Local import to avoid import-time side effects
    from ml_playground import experiments as _experiments

    _experiments.load_preparers()
    # Mirror experiments registry into datasets registry
    PREPARERS.update(_experiments.PREPARERS)
