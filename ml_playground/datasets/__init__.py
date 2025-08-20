from __future__ import annotations
from typing import Callable, Dict

# Compatibility shim for legacy tests and CLI imports.
# Delegate to experiment-level preparers so real runs use the new experiment logic.
from ml_playground.experiments.shakespeare.prepare import main as _exp_shakespeare  # noqa: E402
from ml_playground.experiments.bundestag_char.prepare import main as _exp_bundestag_char  # noqa: E402
from ml_playground.experiments.bundestag_tiktoken.prepare import (
    main as _exp_bundestag_tiktoken,
)  # noqa: E402

PREPARERS: Dict[str, Callable[[], None]] = {
    "shakespeare": _exp_shakespeare,
    "bundestag_char": _exp_bundestag_char,
    "bundestag_tiktoken": _exp_bundestag_tiktoken,
}
