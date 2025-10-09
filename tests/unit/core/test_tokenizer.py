from typing import Any
from collections.abc import Mapping
from types import MappingProxyType
import pytest

from ml_playground.core.tokenizer import (
    CharTokenizer,
    WordTokenizer,
    TiktokenTokenizer,
    create_tokenizer,
)


def test_char_tokenizer_roundtrip_proto() -> None:
    """Round-trips a small char vocab including spaces."""
    vocab = {"a": 1, "b": 2, " ": 3}
    tk = CharTokenizer(vocab=vocab)
    assert tk.name == "char"
    assert tk.vocab_size == 3
    text = "ab a"
    ids = tk.encode(text)
    assert ids == [1, 2, 3, 1]
    back = tk.decode(ids)
    assert back == text
    # vocab mapping should be read-only
    v = tk.vocab
    assert isinstance(v, MappingProxyType)
    with pytest.raises(TypeError):
        v["c"] = 3


def test_char_tokenizer_decode_rebuilds_lookup_array() -> None:
    """Rebuilds lookup array when vocabulary grows dynamically."""
    tok = CharTokenizer({"a": 1})
    assert tok.decode([1]) == "a"
    tok.itos[2] = "b"
    assert tok.decode([2]) == "b"
    assert tok.decode([-1, 3]) == ""


def test_word_tokenizer_decode_strips_invalid_ids_and_exposes_vocab_proxy() -> None:
    """Drops out-of-range ids and enforces read-only vocab mapping."""
    tok = WordTokenizer({"hello": 1, "world": 2})
    assert tok.decode([1, 2]) == "hello world"
    assert tok.decode([-5, 999]) == ""
    vocab_proxy = tok.vocab
    assert isinstance(vocab_proxy, MappingProxyType)
    with pytest.raises(TypeError):
        vocab_proxy["new"] = 3


def test_word_tokenizer_roundtrip() -> None:
    """Ensures simple word tokenizer encode/decode symmetry."""
    tok = WordTokenizer({"hello": 1, "world": 2})
    ids = tok.encode("hello world!")
    assert ids == [1, 2, 0]


def test_word_tokenizer_roundtrip_proto() -> None:
    """Exercises protocol-based round trip plus metadata checks."""
    vocab = {"Hello": 1, ",": 2, "world": 3, "!": 4}
    tk = WordTokenizer(vocab=vocab)
    assert tk.name == "word"
    # tokenization preserves punctuation as separate tokens
    ids = tk.encode("Hello, world!")
    # decode joins with spaces by design
    assert tk.decode(ids) == "Hello , world !"
    # additional property checks to catch decorator removals
    v = tk.vocab
    assert isinstance(v, Mapping)


@pytest.mark.parametrize(
    ("tok_type", "kwargs", "expected_cls"),
    [
        ("char", {"vocab": {"x": 1}}, CharTokenizer),
        ("word", {"vocab": {"x": 1}}, WordTokenizer),
    ],
)
def test_create_tokenizer_factory_char_word_proto(
    tok_type: str, kwargs: dict[str, Any], expected_cls: type
) -> None:
    """Factory returns expected tokenizer subclass for char/word."""
    tk = create_tokenizer(tok_type, **kwargs)
    assert isinstance(tk, expected_cls)


def test_create_tokenizer_factory_unknown_proto() -> None:
    """Factory raises ValueError for unknown tokenizer names."""
    with pytest.raises(ValueError):
        create_tokenizer("nope")


def test_tiktoken_tokenizer_properties_with_fake_module() -> None:
    """Provide a fake tiktoken module to validate TiktokenTokenizer properties without installing tiktoken."""

    class FakeEncoder:
        def __init__(self):
            self.n_vocab = 3
            self._mergeable_ranks = {"a": 1, "b": 2, "c": 3}

        def encode(self, text, allowed_special=None):
            return [1, 2]

        def decode(self, ids):
            return "ab"

    class FakeTiktokenModule:
        @staticmethod
        def get_encoding(name):
            return FakeEncoder()

    tk = TiktokenTokenizer(loader=lambda: FakeTiktokenModule)
    assert tk.name == "tiktoken"
    assert tk.vocab_size == 3
    assert tk.decode(tk.encode("hi")) == "ab"
    v = tk.vocab
    # Mapping with expected keys
    assert hasattr(v, "__getitem__") and "a" in v and v["a"] == 1


@pytest.mark.parametrize(
    "bad", ["charz", "wordz", "tiktokenz"]
)  # avoid real tiktoken import
def test_create_tokenizer_lexicographic_non_matches_raise(bad) -> None:
    """Ensure strings that are lexicographically >= but not equal still raise, killing Eq->GtE mutants."""
    with pytest.raises(ValueError):
        create_tokenizer(bad)
