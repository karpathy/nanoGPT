from __future__ import annotations

from pathlib import Path
import pickle

import pytest

import ml_playground.trainer as tr


def test_validate_checkpoint_missing_keys_raises():
    with pytest.raises(tr.CheckpointError, match="missing required keys"):
        tr.validate_checkpoint({"a": 1}, {"model", "optimizer"})


def test_get_lr_warmup_decay_edges():
    # warmup region
    assert tr._get_lr(0, warmup=10, decay_iters=100, min_lr=1e-4, base_lr=1e-3) == 0.0
    assert tr._get_lr(
        5, warmup=10, decay_iters=100, min_lr=1e-4, base_lr=1e-3
    ) == pytest.approx(0.0005)
    # after decay_iters -> min_lr
    assert tr._get_lr(
        150, warmup=10, decay_iters=100, min_lr=1e-4, base_lr=1e-3
    ) == pytest.approx(1e-4)
    # at boundaries should be within range [min_lr, base_lr]
    lr_at_warmup = tr._get_lr(10, warmup=10, decay_iters=100, min_lr=1e-4, base_lr=1e-3)
    assert 1e-4 <= lr_at_warmup <= 1e-3


def test_load_meta_vocab_size_variants(tmp_path: Path):
    p = tmp_path / "meta.pkl"
    # valid dict
    with p.open("wb") as f:
        pickle.dump({"vocab_size": 42}, f)
    assert tr._load_meta_vocab_size(p) == 42
    # non-dict content
    with p.open("wb") as f:
        pickle.dump([1, 2, 3], f)
    assert tr._load_meta_vocab_size(p) is None
    # corrupted/unreadable
    p.write_bytes(b"not a pickle")
    assert tr._load_meta_vocab_size(p) is None
