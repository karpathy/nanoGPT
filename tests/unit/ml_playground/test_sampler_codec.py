from __future__ import annotations

import builtins
import pytest

import ml_playground.sampler as sampler


def test_create_codec_char_missing_vocab_raises():
    with pytest.raises(sampler.DataError, match="requires 'vocab'"):
        sampler.create_codec_from_tokenizer_type("char")


def test_create_codec_tiktoken_missing_dep_raises(monkeypatch: pytest.MonkeyPatch):
    real_import = builtins.__import__

    def _no_tiktoken(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "tiktoken":
            raise ImportError("forced for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_tiktoken)
    with pytest.raises(
        sampler.DataError,
        match="Required dependency for tiktoken tokenizer is not installed|Failed to create tiktoken tokenizer|tiktoken",
    ):
        sampler.create_codec_from_tokenizer_type("tiktoken", encoding_name="gpt2")
