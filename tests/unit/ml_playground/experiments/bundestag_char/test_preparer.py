from __future__ import annotations

import pytest

from ml_playground.error_handling import DataError
from ml_playground.experiments.bundestag_char.preparer import (
    BundestagCharPreparer,
    ensure_modern_ngram,
)
from ml_playground.prepare import PreparerConfig


def test_preparer_rejects_legacy_ngram_size() -> None:
    """ngram_size extras greater than 1 must raise a DataError."""
    preparer = BundestagCharPreparer()
    cfg = PreparerConfig(extras={"ngram_size": 3})

    with pytest.raises(DataError, match="Legacy n-gram preparation has been removed"):
        preparer.prepare(cfg)


def test_preparer_rejects_non_numeric_ngram(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-numeric ngram_size extras are refused before any filesystem work."""
    preparer = BundestagCharPreparer()
    cfg = PreparerConfig(extras={"ngram_size": "three"})

    with pytest.raises(DataError, match="Remove the 'ngram_size' extra"):
        preparer.prepare(cfg)


def test_ensure_modern_ngram_accepts_one() -> None:
    """Passing an explicit legacy value of 1 is treated as modern and allowed."""
    ensure_modern_ngram({"ngram_size": 1})
