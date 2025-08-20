from __future__ import annotations
from typing import Callable, Dict

# Registry of dataset preparers used by CLI and tests.
# This module must remain cheap to import and side-effect free.
PREPARERS: Dict[str, Callable[[], None]] = {}


def load_preparers() -> None:
    """Populate PREPARERS by importing experiment prepare modules on demand.

    Lazy import: avoids side effects and heavy imports at module import time,
    complying with Import Guidelines (rules 7 and 11).
    """
    if PREPARERS:
        return
    # Local imports justified as plugin loading entry point
    from ml_playground.experiments.shakespeare.prepare import main as _shakespeare  # noqa: F401
    from ml_playground.experiments.bundestag_char.prepare import main as _bundestag_char  # noqa: F401
    from ml_playground.experiments.bundestag_tiktoken.prepare import (
        main as _bundestag_tiktoken,
    )  # noqa: F401

    PREPARERS.update(
        {
            "shakespeare": _shakespeare,
            "bundestag_char": _bundestag_char,
            "bundestag_tiktoken": _bundestag_tiktoken,
        }
    )
