from __future__ import annotations

import numpy as np

from ml_playground.prepare import create_standardized_metadata, prepare_with_tokenizer


class DummyTok:
    def __init__(self):
        self.vocab_size = 123
        self.name = "dummy"

    def encode(self, s: str):  # noqa: D401
        # simple ordinal encoding
        return [ord(c) % 256 for c in s]

    def decode(self, ids):  # noqa: D401
        return "".join(chr(i) for i in ids)


def test_create_standardized_metadata_sets_flags_and_extras():
    tok = DummyTok()
    meta = create_standardized_metadata(
        tok, train_tokens=10, val_tokens=4, extras={"x": 1}
    )
    assert meta["meta_version"] == 1
    assert meta["vocab_size"] == 123
    assert meta["tokenizer"] == "dummy"
    assert meta["has_encode"] and meta["has_decode"]
    assert meta["x"] == 1


def test_prepare_with_tokenizer_splits_and_encodes():
    tok = DummyTok()
    text = "abcdefghij"  # len 10 -> split 9/1 by default
    train_arr, val_arr, meta = prepare_with_tokenizer(text, tok)

    # Check lengths (numpy arrays) and meta counts
    assert isinstance(train_arr, np.ndarray) and isinstance(val_arr, np.ndarray)
    assert train_arr.dtype == np.uint16 and val_arr.dtype == np.uint16
    assert meta["train_tokens"] == 9 and meta["val_tokens"] == 1
