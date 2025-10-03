from collections.abc import Mapping
import pytest

from ml_playground.core.tokenizer import (
    CharTokenizer,
    WordTokenizer,
    TiktokenTokenizer,
    create_tokenizer,
)


def test_char_tokenizer():
    # Test character tokenizer
    text = "hello world"
    tokenizer = CharTokenizer()

    # Since we didn't provide a vocab, it will use the default empty vocab
    # Let's create a vocab for testing
    vocab = {ch: i for i, ch in enumerate(sorted(set(text)))}
    tokenizer = CharTokenizer(vocab)

    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    print(f"Original: {text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Vocab size: {tokenizer.vocab_size}")

    assert decoded == text, f"Expected '{text}', got '{decoded}'"
    print("CharTokenizer test passed!")


def test_word_tokenizer():
    # Test word tokenizer
    text = "hello world hello"
    tokenizer = WordTokenizer()

    # Since we didn't provide a vocab, it will use the default empty vocab
    # Let's create a vocab for testing
    import re

    words = sorted(set(re.findall(r"\w+|[^\w\s]", text)))
    vocab = {word: i for i, word in enumerate(words)}
    tokenizer = WordTokenizer(vocab)

    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    print(f"Original: {text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Vocab size: {tokenizer.vocab_size}")

    # Note: Word tokenizer adds spaces between words during decoding
    # So we can't directly compare with the original text
    print("WordTokenizer test completed!")


def test_tiktoken_tokenizer():
    # Test tiktoken tokenizer
    try:
        tokenizer = TiktokenTokenizer()
        text = "hello world"

        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)

        print(f"Original: {text}")
        print(f"Encoded: {encoded}")
        print(f"Decoded: {decoded}")
        print(f"Vocab size: {tokenizer.vocab_size}")

        assert decoded == text, f"Expected '{text}', got '{decoded}'"
        print("TiktokenTokenizer test passed!")
    except ImportError:
        print("TiktokenTokenizer test skipped (tiktoken not installed)")


def test_factory_function():
    # Test factory function
    char_tokenizer = create_tokenizer("char")
    word_tokenizer = create_tokenizer("word")

    assert isinstance(char_tokenizer, CharTokenizer)
    assert isinstance(word_tokenizer, WordTokenizer)

    try:
        tiktoken_tokenizer = create_tokenizer("tiktoken")
        assert isinstance(tiktoken_tokenizer, TiktokenTokenizer)
    except ImportError:
        print("Tiktoken tokenizer creation skipped (tiktoken not installed)")

    print("Factory function test passed!")


# ---------------------------------------------------------------------------
# Consolidated precise protocol tests from test_tokenizer_protocol.py
# ---------------------------------------------------------------------------


def test_char_tokenizer_roundtrip_proto() -> None:
    vocab = {"a": 1, "b": 2, " ": 3}
    tk = CharTokenizer(vocab=vocab)
    assert tk.name == "char"
    assert tk.vocab_size == 3
    text = "ab a"
    ids = tk.encode(text)
    # unknown maps to 0
    assert ids == [1, 2, 3, 1]
    back = tk.decode(ids)
    assert back == text
    # vocab mapping should be read-only
    v = tk.vocab
    assert isinstance(v, Mapping)
    with pytest.raises(TypeError):
        v["z"] = 9  # type: ignore[index]  # MappingProxyType is immutable


def test_word_tokenizer_roundtrip_proto() -> None:
    vocab = {"Hello": 1, ",": 2, "world": 3, "!": 4}
    tk = WordTokenizer(vocab=vocab)
    assert tk.name == "word"
    # tokenization preserves punctuation as separate tokens
    ids = tk.encode("Hello, world!")
    assert ids == [1, 2, 3, 4]
    # decode joins with spaces by design
    assert tk.decode(ids) == "Hello , world !"
    # additional property checks to catch decorator removals
    assert tk.vocab_size == len(vocab)
    v = tk.vocab
    assert isinstance(v, Mapping)


@pytest.mark.parametrize(
    "tok_type,kwargs,cls",
    [
        ("char", {"vocab": {"x": 1}}, CharTokenizer),
        ("word", {"vocab": {"x": 1}}, WordTokenizer),
    ],
)
def test_create_tokenizer_factory_char_word_proto(tok_type, kwargs, cls) -> None:
    tk = create_tokenizer(tok_type, **kwargs)
    assert isinstance(tk, cls)


def test_create_tokenizer_factory_unknown_proto() -> None:
    with pytest.raises(ValueError):
        create_tokenizer("nope")


def test_tiktoken_tokenizer_import_error_proto() -> None:
    def failing_loader():
        raise ImportError("boom")

    with pytest.raises(ImportError):
        TiktokenTokenizer(loader=failing_loader)


def test_tiktoken_tokenizer_properties_with_fake_module() -> None:
    """Provide a fake tiktoken module to validate TiktokenTokenizer properties without installing tiktoken.
    This helps kill decorator removal mutants on TiktokenTokenizer.* properties.
    """

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


def main():
    print("Testing tokenizer implementations...")
    test_char_tokenizer()
    test_word_tokenizer()
    test_tiktoken_tokenizer()
    test_factory_function()
    print("All tests passed!")


if __name__ == "__main__":
    main()
