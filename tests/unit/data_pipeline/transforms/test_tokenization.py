from __future__ import annotations

import pytest

from ml_playground.core.error_handling import DataError
from ml_playground.core.tokenizer import CharTokenizer, WordTokenizer, TiktokenTokenizer
from ml_playground.data_pipeline.transforms.tokenization import (
    coerce_tokenizer_type,
    prepare_with_tokenizer,
    create_standardized_metadata,
)


def test_coerce_tokenizer_type_raises_on_invalid() -> None:
    """coerce_tokenizer_type should raise DataError for invalid types."""
    with pytest.raises(DataError, match="Unsupported tokenizer type"):
        coerce_tokenizer_type("invalid")


def test_prepare_with_tokenizer_word() -> None:
    """prepare_with_tokenizer should handle WordTokenizer."""
    text = "Hello world. This is a test."
    tokenizer = WordTokenizer()  # Empty vocab, will be built

    train_arr, val_arr, meta, updated_tokenizer = prepare_with_tokenizer(
        text, tokenizer
    )

    # Should have built vocab and tokenized
    assert len(train_arr) > 0
    assert len(val_arr) > 0
    assert meta["tokenizer_type"] == "word"
    assert updated_tokenizer.vocab_size > 0


def test_create_standardized_metadata_with_char_tokenizer() -> None:
    """create_standardized_metadata should include stoi for char tokenizer."""
    vocab = {"a": 0, "b": 1, "c": 2}
    tokenizer = CharTokenizer(vocab=vocab)

    meta = create_standardized_metadata(tokenizer, 100, 20)

    assert meta["tokenizer_type"] == "char"
    assert "stoi" in meta
    assert meta["stoi"] == vocab


def test_create_standardized_metadata_with_word_tokenizer() -> None:
    """create_standardized_metadata should include stoi for word tokenizer."""
    vocab = {"hello": 0, "world": 1}
    tokenizer = WordTokenizer(vocab=vocab)

    meta = create_standardized_metadata(tokenizer, 100, 20)

    assert meta["tokenizer_type"] == "word"
    assert "stoi" in meta
    assert meta["stoi"] == vocab


def test_create_standardized_metadata_with_tiktoken() -> None:
    """create_standardized_metadata should include encoding_name for tiktoken."""
    tokenizer = TiktokenTokenizer(encoding_name="gpt2")

    meta = create_standardized_metadata(tokenizer, 100, 20)

    assert meta["tokenizer_type"] == "tiktoken"
    assert "encoding_name" in meta
    assert meta["encoding_name"] == "gpt2"


def test_create_standardized_metadata_with_extras() -> None:
    """create_standardized_metadata should merge extras into metadata."""
    tokenizer = CharTokenizer(vocab={"a": 0})
    extras = {"custom_field": "value", "another": 123}

    meta = create_standardized_metadata(tokenizer, 100, 20, extras=extras)

    assert meta["custom_field"] == "value"
    assert meta["another"] == 123
