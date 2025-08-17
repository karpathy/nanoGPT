from __future__ import annotations
from typing import Callable, Dict

# Each preparer is a callable main() that performs dataset preparation
PREPARERS: Dict[str, Callable[[], None]] = {}


def register(name: str):
    def _wrap(fn: Callable[[], None]) -> Callable[[], None]:
        PREPARERS[name] = fn
        return fn

    return _wrap


# Import submodules to populate PREPARERS via the @register decorator
from ml_playground.datasets import shakespeare  # noqa: F401, E402
from ml_playground.datasets import bundestag_char  # noqa: F401, E402
