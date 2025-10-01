"""Tokenization utilities shared across data preparation flows."""

from __future__ import annotations

from typing import Any, Literal, cast
import re

import numpy as np

from ml_playground.core.error_handling import DataError
from ml_playground.core.tokenizer import CharTokenizer, WordTokenizer, create_tokenizer
from ml_playground.core.tokenizer_protocol import Tokenizer

__all__ = [
    "TokenizerKind",
    "coerce_tokenizer_type",
    "split_train_val",
    "prepare_with_tokenizer",
    "create_standardized_metadata",
]

TokenizerKind = Literal["char", "word", "tiktoken"]


def coerce_tokenizer_type(value: str) -> TokenizerKind:
    if value not in {"char", "word", "tiktoken"}:
        raise DataError(
            "Unsupported tokenizer type. Expected one of {'char', 'word', 'tiktoken'}"
        )
    return cast(TokenizerKind, value)


def split_train_val(text: str, split: float = 0.9) -> tuple[str, str]:
    n = len(text)
    train_end = int(n * split)
    return text[:train_end], text[train_end:]


def prepare_with_tokenizer(
    text: str, tokenizer: Tokenizer, split: float = 0.9
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], Tokenizer]:
    train_text, val_text = split_train_val(text, split)

    if isinstance(tokenizer, (CharTokenizer, WordTokenizer)):
        all_text = train_text + val_text
        if isinstance(tokenizer, CharTokenizer):
            chars = sorted(set(all_text))
            vocab = {ch: i for i, ch in enumerate(chars)}
            tokenizer = create_tokenizer("char", vocab=vocab)
        elif isinstance(tokenizer, WordTokenizer):
            words = re.findall(r"\w+|[^\w\s]", all_text)
            unique_words = sorted(set(words))
            vocab = {word: i for i, word in enumerate(unique_words)}
            tokenizer = create_tokenizer("word", vocab=vocab)

    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)

    train_arr = np.array(train_ids, dtype=np.uint16)
    val_arr = np.array(val_ids, dtype=np.uint16)

    meta = create_standardized_metadata(tokenizer, len(train_ids), len(val_ids))
    return train_arr, val_arr, meta, tokenizer


def create_standardized_metadata(
    tokenizer: Tokenizer, train_tokens: int, val_tokens: int, extras: dict | None = None
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "meta_version": 1,
        "tokenizer_type": getattr(tokenizer, "name", None) or "unknown",
        "vocab_size": tokenizer.vocab_size,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "has_encode": hasattr(tokenizer, "encode"),
        "has_decode": hasattr(tokenizer, "decode"),
    }

    meta["tokenizer"] = meta["tokenizer_type"]

    try:
        if meta["tokenizer_type"] in ("char", "word"):
            vocab = getattr(tokenizer, "stoi", None)
            if isinstance(vocab, dict) and vocab:
                meta["stoi"] = vocab
        elif meta["tokenizer_type"] == "tiktoken":
            encoding_name = getattr(tokenizer, "encoding_name", None)
            if isinstance(encoding_name, str):
                meta["encoding_name"] = encoding_name
    except (AttributeError, TypeError, ValueError):
        pass

    if extras:
        meta.update(extras)

    return meta
