from __future__ import annotations

import pytest

from ml_playground.sampler import CodecManager, DataError


def test_codec_manager_raises_if_not_initialized():
    cm = CodecManager()
    with pytest.raises(DataError, match="Codec not initialized"):
        cm.encode("hi")
    with pytest.raises(DataError, match="Codec not initialized"):
        cm.decode([1, 2])


def test_codec_manager_initialize_and_use_word_vocab():
    cm = CodecManager()
    cm.initialize_codec(tokenizer_type="word", vocab={"hello": 1, "world": 2})
    ids = cm.encode("hello world")
    assert isinstance(ids, list)
    text = cm.decode(ids)
    assert "hello" in text
