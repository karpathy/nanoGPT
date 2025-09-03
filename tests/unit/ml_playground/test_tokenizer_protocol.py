import builtins
import pytest

from ml_playground.tokenizer import (
    CharTokenizer,
    WordTokenizer,
    TiktokenTokenizer,
    create_tokenizer,
)


def test_char_tokenizer_encode_decode_roundtrip():
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


def test_word_tokenizer_encode_decode_roundtrip():
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
def test_create_tokenizer_factory_char_word(tok_type, kwargs, cls):
    tk = create_tokenizer(tok_type, **kwargs)
    assert isinstance(tk, cls)


def test_create_tokenizer_factory_unknown():
    with pytest.raises(ValueError):
        create_tokenizer("nope")


def test_tiktoken_tokenizer_import_error(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tiktoken":
            raise ImportError("boom")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError):
        TiktokenTokenizer()
