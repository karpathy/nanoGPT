from __future__ import annotations
from typing import Callable, Dict

# Registry for experiment-level dataset preparers (mirrors previous datasets.PREPARERS)
PREPARERS: Dict[str, Callable[[], None]] = {}


def register(name: str):
    def _wrap(fn: Callable[[], None]) -> Callable[[], None]:
        PREPARERS[name] = fn
        return fn

    return _wrap


# Import experiment prepare modules so they can register their main() preparers
# Keep these imports at the end to avoid circular import issues.
from ml_playground.experiments.shakespeare import prepare as _prep_shakespeare  # noqa: F401,E402
from ml_playground.experiments.bundestag_char import prepare as _prep_bundestag_char  # noqa: F401,E402
from ml_playground.experiments.bundestag_tiktoken import (
    prepare as _prep_bundestag_tiktoken,
)  # noqa: F401,E402
from ml_playground.experiments.speakger import prepare as _prep_speakger  # noqa: F401,E402
