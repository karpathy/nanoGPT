from __future__ import annotations
from typing import Callable, Dict

# Registry for experiment-level dataset preparers (mirrors previous datasets.PREPARERS)
PREPARERS: Dict[str, Callable[[], None]] = {}


def register(name: str):
    def _wrap(fn: Callable[[], None]) -> Callable[[], None]:
        PREPARERS[name] = fn
        return fn

    return _wrap


def load_preparers() -> None:
    """Plugin loader: import experiment prepare modules to populate PREPARERS.

    Lazy import: defers importing experiment submodules (and any heavy deps they
    may transitively import) until explicitly requested by callers. This avoids
    side effects at import time and follows the Import Guidelines (rules 7 and 11).
    """
    # Local imports justified as plugin loading entry point
    from ml_playground.experiments.shakespeare import prepare as _prep_shakespeare  # noqa: F401
    from ml_playground.experiments.bundestag_char import prepare as _prep_bundestag_char  # noqa: F401
    from ml_playground.experiments.bundestag_tiktoken import (
        prepare as _prep_bundestag_tiktoken,
    )  # noqa: F401
    from ml_playground.experiments.speakger import prepare as _prep_speakger  # noqa: F401

    # Mark imports as used to satisfy linter; registration occurs on import
    _ = (
        _prep_shakespeare,
        _prep_bundestag_char,
        _prep_bundestag_tiktoken,
        _prep_speakger,
    )
