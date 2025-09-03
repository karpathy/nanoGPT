from __future__ import annotations

import pickle
from pathlib import Path

import pytest

import ml_playground.sampler as sampler


def _write_word_meta(meta_path: Path) -> None:
    meta = {
        "meta_version": 1,
        "kind": "word",
        "dtype": "uint16",
        "stoi": {"hello": 1, "world": 2},
    }
    meta_path.write_bytes(pickle.dumps(meta))


def test_validate_and_create_codec_auto_word_meta(tmp_path: Path):
    meta_path = tmp_path / "meta.pkl"
    _write_word_meta(meta_path)
    enc, dec = sampler.validate_and_create_codec(meta_path, tokenizer_type="auto")
    ids = enc("hello world")
    assert isinstance(ids, list)
    assert dec(ids)


def test_validate_and_create_codec_explicit_word_missing_vocab_raises():
    with pytest.raises(sampler.DataError, match="Word tokenizer requires 'vocab'"):
        sampler.validate_and_create_codec(None, tokenizer_type="word")
