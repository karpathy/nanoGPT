from __future__ import annotations
from pathlib import Path
import torch

from ml_playground.trainer import _sha256_of_file, _atomic_save


def test_sha256_of_file(tmp_path: Path):
    p = tmp_path / "f.txt"
    p.write_text("hello", encoding="utf-8")
    h1 = _sha256_of_file(p)
    # Deterministic and changes with content
    p.write_text("hello!", encoding="utf-8")
    h2 = _sha256_of_file(p)
    assert h1 != h2
    # Known value for simple content
    p.write_text("abc", encoding="utf-8")
    assert (
        _sha256_of_file(p)
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_atomic_save_and_readback(tmp_path: Path):
    # atomic save should create only final file, no lingering .tmp
    p = tmp_path / "ckpt.pt"
    obj = {"a": 1, "b": [1, 2, 3]}
    _atomic_save(obj, p, atomic=True)
    assert p.exists()
    assert not (tmp_path / "ckpt.pt.tmp").exists()
    loaded = torch.load(p, map_location="cpu")
    assert loaded == obj


def test_non_atomic_save(tmp_path: Path):
    p = tmp_path / "ckpt2.pt"
    obj = {"x": 42}
    _atomic_save(obj, p, atomic=False)
    assert p.exists()
    loaded = torch.load(p, map_location="cpu")
    assert loaded == obj
