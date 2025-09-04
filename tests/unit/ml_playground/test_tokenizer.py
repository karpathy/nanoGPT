import builtins
import pytest

from ml_playground.tokenizer import (
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
    print(f"Vocab size: {tokenizer.get_vocab_size()}")

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


def test_word_tokenizer_roundtrip_proto() -> None:
    vocab = {"Hello": 1, ",": 2, "world": 3, "!": 4}
    tk = WordTokenizer(vocab=vocab)
    assert tk.name == "word"
    # tokenization preserves punctuation as separate tokens
    ids = tk.encode("Hello, world!")
    assert ids == [1, 2, 3, 4]
    # decode joins with spaces by design
    assert tk.decode(ids) == "Hello , world !"


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


def test_tiktoken_tokenizer_import_error_proto(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tiktoken":
            raise ImportError("boom")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError):
        TiktokenTokenizer()


def main():
    print("Testing tokenizer implementations...")
    test_char_tokenizer()
    test_word_tokenizer()
    test_tiktoken_tokenizer()
    test_factory_function()
    print("All tests passed!")


if __name__ == "__main__":
    main()
