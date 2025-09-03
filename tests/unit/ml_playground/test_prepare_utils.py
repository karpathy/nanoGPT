from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

import ml_playground.prepare as prep
from ml_playground.tokenizer import CharTokenizer


def test_split_train_val_ratio():
    train, val = prep.split_train_val("abcdefghij", split=0.7)
    assert train == "abcdefg"
    assert val == "hij"


def test_create_standardized_metadata_basic():
    tok = CharTokenizer(vocab={"a": 1, "b": 2})
    meta = prep.create_standardized_metadata(tok, train_tokens=5, val_tokens=3)
    assert meta["meta_version"] == 1
    assert meta["vocab_size"] == 2
    assert meta["train_tokens"] == 5
    assert meta["val_tokens"] == 3
    assert meta["tokenizer"] == "char"
    assert meta["has_encode"] is True
    assert meta["has_decode"] is True


def test_prepare_with_tokenizer_arrays_and_meta():
    tok = CharTokenizer(vocab={"a": 1, "b": 2})
    train, val, meta = prep.prepare_with_tokenizer("abba", tok, split=0.5)
    assert isinstance(train, np.ndarray) and train.dtype == np.uint16
    assert isinstance(val, np.ndarray) and val.dtype == np.uint16
    # With split=0.5, first half "ab" then "ba"
    assert train.tolist() == [1, 2]
    assert val.tolist() == [2, 1]
    assert meta["tokenizer"] == "char"


def test_write_bin_and_meta_creates_and_is_idempotent(tmp_path: Path):
    ds = tmp_path / "dataset"
    tok = CharTokenizer(vocab={"a": 1})
    train = np.array([1, 1, 1], dtype=np.uint16)
    val = np.array([1], dtype=np.uint16)
    meta = prep.create_standardized_metadata(tok, 3, 1)

    # First write creates files
    prep.write_bin_and_meta(ds, train, val, meta)
    assert (ds / "train.bin").exists()
    assert (ds / "val.bin").exists()
    assert (ds / "meta.pkl").exists()

    # Second write should detect valid meta and skip rewriting
    before_mtime = (ds / "meta.pkl").stat().st_mtime
    prep.write_bin_and_meta(ds, train, val, meta)
    after_mtime = (ds / "meta.pkl").stat().st_mtime
    assert before_mtime == after_mtime


def test_snapshot_and_diff_helpers(tmp_path: Path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    paths = [a, b]
    before = prep.snapshot_files(paths)
    # create one file
    a.write_text("x", encoding="utf-8")
    created, updated, skipped = prep.diff_files(paths, before)
    assert a in created
    # If b was absent before and remains absent after, implementation may omit it from all sets.
    assert b not in created and b not in updated

    # touch a to update
    before2 = prep.snapshot_files(paths)
    a.write_text("xy", encoding="utf-8")
    c2, u2, s2 = prep.diff_files(paths, before2)
    assert a in u2


def test_make_preparer_runs_and_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ds = tmp_path / "ds"

    # Provide raw text via extras to avoid file IO
    cfg = prep.Preparer.Config(  # type: ignore[attr-defined]
        dataset_dir=ds,
        extras={
            "tokenizer_type": "char",
            "raw_text": "abbaabba",
        },
    )

    # Use a deterministic tokenizer via create_tokenizer
    # make_preparer should construct _PreparerInstance and write files
    p = prep.make_preparer(cfg)
    p()

    # Verify artifacts
    assert (ds / "train.bin").exists()
    assert (ds / "val.bin").exists()
    meta_path = ds / "meta.pkl"
    assert meta_path.exists()
    meta = pickle.loads(meta_path.read_bytes())
    # tokenization name recorded
    assert meta["tokenizer"] in ("char", "word", "tiktoken")
