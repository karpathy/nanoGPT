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
from . import shakespeare  # noqa: F401
from . import bundestag_char  # noqa: F401
