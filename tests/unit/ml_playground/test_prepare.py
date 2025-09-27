from __future__ import annotations

from pathlib import Path
import pickle
from typing import List, Mapping

import numpy as np
import pytest

import ml_playground.prepare as prep
from ml_playground.prepare import (
    split_train_val,
    write_bin_and_meta,
    create_standardized_metadata,
    prepare_with_tokenizer,
)
from ml_playground.tokenizer import CharTokenizer, WordTokenizer


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
        return "".join(inv.get(tid, "?") for tid in token_ids)


# ---- small helpers ----


def _mk_arrays(n: int) -> tuple[np.ndarray, np.ndarray, dict]:
    train = np.arange(n, dtype=np.uint16)
    val = np.arange(n, dtype=np.uint16)
    meta = {"meta_version": 1}
    return train, val, meta


# Additional fake tokenizer for tiktoken-like metadata enrichment tests
class _FakeTiktoken:
    def __init__(self) -> None:
        self.name = "tiktoken"
        self.encoding_name = "gpt2"
        self._vocab_size = 1000

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, s: str) -> list[int]:
        return list(range(len(s)))

    def decode(self, ids: list[int]) -> str:
        return "".join("x" for _ in ids)


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


def test_create_standardized_metadata_tiktoken_enrichment() -> None:
    tok = _FakeTiktoken()
    meta = create_standardized_metadata(tok, train_tokens=3, val_tokens=2)
    assert meta["tokenizer_type"] == "tiktoken"
    assert meta["encoding_name"] == "gpt2"
    assert meta["vocab_size"] == 1000
    assert meta["has_encode"] is True and meta["has_decode"] is True


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

    # Additional logging assertions can be added here if a logger fixture is introduced.


def test_prepare_with_tokenizer_word_vocab_rebuild() -> None:
    # Include punctuation to exercise regex tokenization path
    text = "Hello, world! Hello"
    tok = WordTokenizer()
    train_arr, val_arr, meta, rebuilt = prepare_with_tokenizer(text, tok, split=0.5)
    assert meta["tokenizer_type"] == "word"
    # Rebuilt tokenizer should produce uint16 arrays
    assert train_arr.dtype == np.uint16 and val_arr.dtype == np.uint16
    assert hasattr(rebuilt, "encode") and hasattr(rebuilt, "decode")


def test_write_bin_and_meta_logging_exception_is_ignored(tmp_path: Path) -> None:
    class RaisingLogger:
        def info(self, _message: str) -> None:
            raise ValueError("fail")

    train, val, meta = _mk_arrays(3)
    write_bin_and_meta(tmp_path, train, val, meta, logger=RaisingLogger())


def test_write_bin_and_meta_already_exists_logs(tmp_path: Path) -> None:
    ds = tmp_path / "ds"
    ds.mkdir(parents=True, exist_ok=True)
    # Pre-create valid artifacts to trigger early-return logging path
    (ds / "train.bin").write_bytes(np.arange(4, dtype=np.uint16).tobytes())
    (ds / "val.bin").write_bytes(np.arange(4, dtype=np.uint16).tobytes())
    with (ds / "meta.pkl").open("wb") as f:
        pickle.dump({"meta_version": 1}, f)

    class ListLogger:
        def __init__(self) -> None:
            self.infos: list[str] = []

        def info(self, msg: str) -> None:
            self.infos.append(str(msg))

    logger = ListLogger()
    train = np.arange(2, dtype=np.uint16)
    val = np.arange(2, dtype=np.uint16)
    meta = {"meta_version": 1}

    write_bin_and_meta(ds, train, val, meta, logger=logger, data_cfg=None)

    # Ensure the logging branch executed (we do not assert exact content to avoid brittleness)
    assert any("[prepare] Created" in m for m in logger.infos)
    assert any("[prepare] Skipped" in m for m in logger.infos)


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
