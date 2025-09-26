from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Mapping

import numpy as np
import pytest

import ml_playground.prepare as prep
from ml_playground.prepare import (
    split_train_val,
    write_bin_and_meta,
    create_standardized_metadata,
    prepare_with_tokenizer,
    snapshot_file_states,
    diff_file_states,
)
from ml_playground.tokenizer import CharTokenizer
from ml_playground.error_handling import DataError


"""Logging helpers are provided via fixtures in conftest.py (list_logger, list_logger_factory)."""


class DummyTok:
    def __init__(self) -> None:
        self._name = "dummy"
        # Provide a minimal vocab mapping to satisfy the protocol
        self._vocab: dict[str, int] = {"a": 1, "b": 2}

    @property
    def name(self) -> str:  # noqa: D401
        return self._name

    @property
    def vocab_size(self) -> int:  # noqa: D401
        # Return a fixed size to match test expectations
        return 123

    @property
    def vocab(self) -> Mapping[str, int]:  # noqa: D401
        return self._vocab

    def encode(self, text: str) -> List[int]:  # noqa: D401
        return [self._vocab.get(ch, 0) for ch in text]

    def decode(self, token_ids: List[int]) -> str:  # noqa: D401
        inv = {v: k for k, v in self._vocab.items()}
        return "".join(inv.get(i, "") for i in token_ids)


def _mk_arrays(n: int = 4):
    return (
        np.arange(n, dtype=np.uint16),
        np.arange(n, dtype=np.uint16),
        {"meta_version": 1},
    )


# ---- split helpers ----


def test_split_train_val_ratio() -> None:
    train, val = prep.split_train_val("abcdefghij", split=0.7)
    assert train == "abcdefg"
    assert val == "hij"


def test_split_train_val_edges() -> None:
    text = "abcdef"
    # split=0 -> all val
    train, val = split_train_val(text, split=0.0)
    assert train == ""
    assert val == text
    # split=1 -> all train
    train, val = split_train_val(text, split=1.0)
    assert train == text
    assert val == ""


# ---- metadata helpers ----


def test_create_standardized_metadata_basic() -> None:
    tok = CharTokenizer(vocab={"a": 1, "b": 2})
    meta = prep.create_standardized_metadata(tok, train_tokens=5, val_tokens=3)
    assert meta["meta_version"] == 1
    assert meta["vocab_size"] == 2
    assert meta["train_tokens"] == 5
    assert meta["val_tokens"] == 3
    assert meta["tokenizer"] == "char"
    assert meta["has_encode"] is True
    assert meta["has_decode"] is True


def test_create_standardized_metadata_sets_flags_and_extras() -> None:
    tok = DummyTok()
    meta = create_standardized_metadata(
        tok, train_tokens=10, val_tokens=4, extras={"x": 1}
    )
    assert meta["meta_version"] == 1
    assert meta["vocab_size"] == 123
    assert meta["tokenizer"] == "dummy"
    assert meta["has_encode"] and meta["has_decode"]
    assert meta["x"] == 1


# ---- prepare_with_tokenizer ----


def test_prepare_with_tokenizer_arrays_and_meta() -> None:
    tok = CharTokenizer(vocab={"a": 1, "b": 2})
    train, val, meta, tokenizer = prep.prepare_with_tokenizer("abba", tok, split=0.5)
    assert isinstance(train, np.ndarray) and train.dtype == np.uint16
    assert isinstance(val, np.ndarray) and val.dtype == np.uint16
    # With split=0.5, first half "ab" then "ba"
    # Function rebuilds vocab from text: {'a': 0, 'b': 1}
    assert train.tolist() == [0, 1]  # "ab" -> [0, 1]
    assert val.tolist() == [1, 0]  # "ba" -> [1, 0]
    assert meta["tokenizer"] == "char"
    assert tokenizer is not None
    assert tokenizer.stoi == {"a": 0, "b": 1}  # Rebuilt vocab


def test_prepare_with_tokenizer_splits_and_encodes() -> None:
    tok = DummyTok()
    text = "abcdefghij"  # len 10 -> split 9/1 by default
    train_arr, val_arr, meta, tokenizer = prepare_with_tokenizer(text, tok)

    assert isinstance(train_arr, np.ndarray) and isinstance(val_arr, np.ndarray)
    assert train_arr.dtype == np.uint16 and val_arr.dtype == np.uint16
    assert meta["train_tokens"] == 9 and meta["val_tokens"] == 1
    assert tokenizer is not None


# ---- write_bin_and_meta (public) ----


def test_write_bin_and_meta_creates_and_is_idempotent(
    tmp_path: Path, list_logger
) -> None:
    ds = tmp_path / "dataset"
    tok = CharTokenizer(vocab={"a": 1})
    train = np.array([1, 1, 1], dtype=np.uint16)
    val = np.array([1], dtype=np.uint16)
    meta = prep.create_standardized_metadata(tok, 3, 1)

    # First write creates files
    prep.write_bin_and_meta(ds, train, val, meta, logger=list_logger)
    assert (ds / "train.bin").exists()
    assert (ds / "val.bin").exists()
    assert (ds / "meta.pkl").exists()

    # Second write should detect valid meta and skip rewriting
    before_mtime = (ds / "meta.pkl").stat().st_mtime
    prep.write_bin_and_meta(ds, train, val, meta, logger=list_logger)
    after_mtime = (ds / "meta.pkl").stat().st_mtime
    assert before_mtime == after_mtime


def test_write_bin_and_meta_logs_skipped_on_idempotent_run(
    tmp_path: Path, list_logger, list_logger_factory
) -> None:
    ds = tmp_path / "dataset2"
    tok = CharTokenizer(vocab={"a": 1})
    train = np.array([1, 1, 1], dtype=np.uint16)
    val = np.array([1], dtype=np.uint16)
    meta = prep.create_standardized_metadata(tok, 3, 1)

    # First run: create files
    write_bin_and_meta(ds, train, val, meta, logger=list_logger)
    assert (
        (ds / "train.bin").exists()
        and (ds / "val.bin").exists()
        and (ds / "meta.pkl").exists()
    )

    # Second run: nothing to change, should log Skipped entries
    logger2 = list_logger_factory()
    write_bin_and_meta(ds, train, val, meta, logger=logger2)
    infos = "\n".join(logger2.infos)
    assert "Created:" in infos
    assert "Skipped:" in infos


def test_write_bin_and_meta_regenerates_on_invalid_meta(
    tmp_path: Path, list_logger
) -> None:
    ds = tmp_path / "ds"
    ds.mkdir()
    # Create initial valid files
    (ds / "train.bin").write_bytes(np.array([1], dtype=np.uint16).tobytes())
    (ds / "val.bin").write_bytes(np.array([2], dtype=np.uint16).tobytes())
    with (ds / "meta.pkl").open("wb") as f:
        pickle.dump({"not_meta_version": True}, f)

    # Now call with new arrays and valid meta; strict policy should raise on invalid existing meta
    train = np.array([3, 4], dtype=np.uint16)
    val = np.array([5], dtype=np.uint16)
    meta = {
        "meta_version": 1,
        "vocab_size": 2,
        "train_tokens": 2,
        "val_tokens": 1,
        "tokenizer": "char",
        "has_encode": True,
        "has_decode": True,
    }

    with pytest.raises(DataError):
        write_bin_and_meta(ds, train, val, meta, logger=list_logger)


# ---- snapshot/diff helpers ----


def test_snapshot_and_diff_helpers(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    paths = [a, b]
    before = snapshot_file_states(paths)
    # create one file
    a.write_text("x", encoding="utf-8")
    created, updated, skipped = diff_file_states(paths, before)
    assert a in created
    # If b was absent before and remains absent after, implementation may omit it from all sets.
    assert b not in created and b not in updated

    # touch a to update
    before2 = snapshot_file_states(paths)
    a.write_text("xy", encoding="utf-8")
    c2, u2, s2 = diff_file_states(paths, before2)
    assert a in u2


def test_diff_files_updated_and_skipped(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    # create both files
    a.write_text("one", encoding="utf-8")
    b.write_text("keep", encoding="utf-8")

    before = snapshot_file_states([a, b])

    # update a, leave b unchanged
    a.write_text("two", encoding="utf-8")

    created, updated, skipped = diff_file_states([a, b], before)
    assert a in updated
    assert b in skipped


# ---- write_bin_and_meta internal behaviors via public API ----


def test_write_bin_and_meta_raises_on_invalid_existing_meta(
    tmp_path: Path, list_logger
) -> None:
    ds = tmp_path / "ds"
    ds.mkdir()
    # Pre-create files with invalid meta content (not a dict with meta_version)
    (ds / "train.bin").write_bytes(b"x")
    (ds / "val.bin").write_bytes(b"y")
    with (ds / "meta.pkl").open("wb") as f:
        pickle.dump([1, 2, 3], f)  # invalid

    tr, va, meta = _mk_arrays(8)
    with pytest.raises(DataError):
        write_bin_and_meta(ds, tr, va, meta, logger=list_logger)


def test_write_bin_and_meta_raises_on_unreadable_meta(
    tmp_path: Path, list_logger
) -> None:
    ds = tmp_path / "ds2"
    ds.mkdir()
    # Pre-create files with unreadable meta (corrupted bytes)
    (ds / "train.bin").write_bytes(b"x")
    (ds / "val.bin").write_bytes(b"y")
    (ds / "meta.pkl").write_bytes(b"not-a-pickle")

    tr, va, meta = _mk_arrays(6)
    with pytest.raises(DataError):
        write_bin_and_meta(ds, tr, va, meta, logger=list_logger)


def test_write_bin_and_meta_info_logs_created_then_raises_on_invalid_second_run(
    tmp_path: Path, list_logger, list_logger_factory
) -> None:
    ds = tmp_path / "ds3"

    # First write -> should log Created entries
    tr, va, meta = _mk_arrays(5)
    write_bin_and_meta(ds, tr, va, meta, logger=list_logger)
    assert any("Created:" in i for i in list_logger.infos)

    # Prepare pre-existing invalid meta and ensure second run raises
    with (ds / "meta.pkl").open("wb") as f:
        pickle.dump(["bad"], f)

    logger2 = list_logger_factory()
    tr2, va2, meta2 = _mk_arrays(7)
    with pytest.raises(DataError):
        write_bin_and_meta(ds, tr2, va2, meta2, logger=logger2)


# ---- seed file helpers ----


def test_seed_text_file_copies_first_existing_candidate(tmp_path: Path) -> None:
    src1 = tmp_path / "a.txt"
    src2 = tmp_path / "b.txt"
    dst = tmp_path / "out" / "seed.txt"

    # Only src2 exists
    src2.write_text("hello", encoding="utf-8")

    prep.seed_text_file(dst, [src1, src2])
    assert dst.exists()
    assert dst.read_text(encoding="utf-8") == "hello"


def test_seed_text_file_noop_if_dst_exists(tmp_path: Path) -> None:
    src = tmp_path / "in.txt"
    dst = tmp_path / "dst.txt"
    src.write_text("hello", encoding="utf-8")
    dst.write_text("old", encoding="utf-8")

    prep.seed_text_file(dst, [src])
    # Should not overwrite existing dst
    assert dst.read_text(encoding="utf-8") == "old"


def test_seed_text_file_raises_when_no_candidates_exist(tmp_path: Path) -> None:
    dst = tmp_path / "dst.txt"
    with pytest.raises(FileNotFoundError):
        prep.seed_text_file(dst, [tmp_path / "missing1.txt", tmp_path / "missing2.txt"])


# ---- integrative make_preparer ----


def test_make_preparer_runs_and_writes(tmp_path: Path) -> None:
    ds = tmp_path / "ds"

    # Provide raw text via extras to avoid file IO
    cfg = prep.PreparerConfig(
        extras={
            "tokenizer_type": "char",
            "raw_text": "abbaabba",
        },
    )

    # Use a deterministic tokenizer via create_tokenizer
    p = prep.Preparer(cfg)
    from ml_playground.config import SharedConfig

    shared = SharedConfig(
        experiment="unit",
        config_path=tmp_path / "cfg.toml",
        project_home=tmp_path,
        dataset_dir=ds,
        train_out_dir=tmp_path / "train_out",
        sample_out_dir=tmp_path / "sample_out",
    )
    p(shared)

    # Verify artifacts
    assert (ds / "train.bin").exists()
    assert (ds / "val.bin").exists()
    meta_path = ds / "meta.pkl"
    assert meta_path.exists()
    meta = pickle.loads(meta_path.read_bytes())
    # tokenization name recorded
    assert meta["tokenizer"] in ("char", "word", "tiktoken")
