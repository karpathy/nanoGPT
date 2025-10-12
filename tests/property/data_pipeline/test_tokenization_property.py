from __future__ import annotations

import re

import hypothesis.strategies as st
from hypothesis import given, settings

from ml_playground.core.tokenizer import CharTokenizer
from ml_playground.data_pipeline.transforms.tokenization import prepare_with_tokenizer
from ml_playground.core.tokenizer import WordTokenizer


class TestTokenizationProperties:
    """Property-based tests for tokenization transforms."""

    @given(text=st.text(min_size=0, max_size=100))
    @settings(max_examples=50, deadline=None, derandomize=True)
    def test_prepare_with_empty_text_word_tokenizer(self, text: str) -> None:
        """Test word tokenizer rebuilds vocab correctly for any text input, covering lines 56-62."""
        import numpy as np

        tokenizer = WordTokenizer({})  # Empty vocab initially
        train_arr, val_arr, meta, updated_tokenizer = prepare_with_tokenizer(
            text, tokenizer
        )
        words = re.findall(r"\w+|[^\w\s]", text)
        if words:
            assert len(updated_tokenizer.vocab) > 0
            assert meta["vocab_size"] == len(updated_tokenizer.vocab)
        else:
            assert updated_tokenizer.vocab == {}
            assert meta["vocab_size"] == 0
        assert isinstance(train_arr, np.ndarray)
        assert isinstance(val_arr, np.ndarray)

    @given(text=st.text(min_size=0, max_size=100))
    @settings(max_examples=50, deadline=None, derandomize=True)
    def test_prepare_with_empty_text_char_tokenizer(self, text: str) -> None:
        """Test char tokenizer rebuilds vocab correctly for any text input, covering lines 50-56."""
        import numpy as np

        tokenizer = CharTokenizer({})  # Empty vocab initially
        train_arr, val_arr, meta, updated_tokenizer = prepare_with_tokenizer(
            text, tokenizer
        )
        if text:
            assert len(updated_tokenizer.vocab) > 0
            assert meta["vocab_size"] == len(updated_tokenizer.vocab)
        else:
            assert updated_tokenizer.vocab == {}
            assert meta["vocab_size"] == 0
        assert isinstance(train_arr, np.ndarray)
        assert isinstance(val_arr, np.ndarray)
