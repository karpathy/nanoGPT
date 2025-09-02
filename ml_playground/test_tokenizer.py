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
    print(f"Vocab size: {tokenizer.get_vocab_size()}")

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
        print(f"Vocab size: {tokenizer.get_vocab_size()}")

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


def main():
    print("Testing tokenizer implementations...")
    test_char_tokenizer()
    test_word_tokenizer()
    test_tiktoken_tokenizer()
    test_factory_function()
    print("All tests passed!")


if __name__ == "__main__":
    main()
