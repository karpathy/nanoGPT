from __future__ import annotations

import pickle
from pathlib import Path
import pytest

import numpy as np

from ml_playground.prepare import write_bin_and_meta
from ml_playground.error_handling import DataError


def test_write_bin_and_meta_regenerates_on_invalid_meta(tmp_path: Path):
    ds = tmp_path / "ds"
    ds.mkdir()
    # Create initial valid files
    (ds / "train.bin").write_bytes(np.array([1], dtype=np.uint16).tobytes())
    (ds / "val.bin").write_bytes(np.array([2], dtype=np.uint16).tobytes())
    with (ds / "meta.pkl").open("wb") as f:
        pickle.dump({"not_meta_version": True}, f)

    # Now call with new arrays and valid meta; strict policy should raise on invalid existing meta
    train = np.array([3, 4], dtype=np.uint16)
    val = np.array([5], dtype=np.uint16)
    meta = {
        "meta_version": 1,
        "vocab_size": 2,
        "train_tokens": 2,
        "val_tokens": 1,
        "tokenizer": "char",
        "has_encode": True,
        "has_decode": True,
    }

    with pytest.raises(DataError):
        write_bin_and_meta(ds, train, val, meta)
