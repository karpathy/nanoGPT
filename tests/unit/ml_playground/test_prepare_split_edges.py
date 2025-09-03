from __future__ import annotations

from ml_playground.prepare import split_train_val


def test_split_train_val_edges():
    text = "abcdef"
    # split=0 -> all val
    train, val = split_train_val(text, split=0.0)
    assert train == ""
    assert val == text
    # split=1 -> all train
    train, val = split_train_val(text, split=1.0)
    assert train == text
    assert val == ""
