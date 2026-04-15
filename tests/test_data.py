import pickle
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "shakespeare_char"


@pytest.fixture(scope="module")
def prepared_data():
    train_bin = DATA_DIR / "train.bin"
    val_bin = DATA_DIR / "val.bin"
    meta_pkl = DATA_DIR / "meta.pkl"
    if not (train_bin.exists() and val_bin.exists() and meta_pkl.exists()):
        pytest.skip(
            "shakespeare_char not prepared; run `python data/shakespeare_char/prepare.py`"
        )
    return train_bin, val_bin, meta_pkl


def test_bin_files_are_uint16(prepared_data):
    train_bin, val_bin, _ = prepared_data
    train = np.memmap(train_bin, dtype=np.uint16, mode="r")
    val = np.memmap(val_bin, dtype=np.uint16, mode="r")
    assert train.dtype == np.uint16
    assert val.dtype == np.uint16
    assert len(train) > len(val)


def test_meta_pkl_has_required_keys(prepared_data):
    _, _, meta_pkl = prepared_data
    with open(meta_pkl, "rb") as f:
        meta = pickle.load(f)
    assert set(meta.keys()) >= {"vocab_size", "itos", "stoi"}
    assert meta["vocab_size"] == len(meta["itos"]) == len(meta["stoi"])
    assert isinstance(meta["vocab_size"], int)


def test_token_ids_are_within_vocab(prepared_data):
    train_bin, val_bin, meta_pkl = prepared_data
    with open(meta_pkl, "rb") as f:
        vocab_size = pickle.load(f)["vocab_size"]
    train = np.memmap(train_bin, dtype=np.uint16, mode="r")
    assert int(train[:10000].max()) < vocab_size
    val = np.memmap(val_bin, dtype=np.uint16, mode="r")
    assert int(val[:10000].max()) < vocab_size


def test_stoi_itos_roundtrip(prepared_data):
    _, _, meta_pkl = prepared_data
    with open(meta_pkl, "rb") as f:
        meta = pickle.load(f)
    for ch, i in meta["stoi"].items():
        assert meta["itos"][i] == ch
