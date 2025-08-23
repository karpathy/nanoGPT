from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import torch

from ml_playground.config import DataConfig
from ml_playground.data import SimpleBatches


def _write_bin(path: Path, arr: np.ndarray) -> None:
    path.write_bytes(arr.tobytes())


def _prepare_dataset(tmp_path: Path, L: int, dtype: str = "uint16") -> Path:
    ddir = tmp_path / "ds"
    ddir.mkdir(parents=True, exist_ok=True)
    arr = (np.arange(L) % np.iinfo(np.uint16).max).astype(dtype)
    _write_bin(ddir / "train.bin", arr)
    _write_bin(ddir / "val.bin", arr)
    return ddir


def _make_batches(
    ddir: Path,
    *,
    batch_size: int,
    block_size: int,
    sampler: str,
) -> SimpleBatches:
    cfg = DataConfig(
        dataset_dir=ddir,
        batch_size=batch_size,
        block_size=block_size,
        grad_accum_steps=1,
        sampler=sampler,  # type: ignore[arg-type]
    )
    return SimpleBatches(cfg, device="cpu")


def test_random_mode_basic(tmp_path: Path) -> None:
    ddir = _prepare_dataset(tmp_path, L=100)
    batches = _make_batches(ddir, batch_size=4, block_size=8, sampler="random")
    x, y = batches.get_batch("train")
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    # For contiguous windows, y is x shifted by 1 with one next token appended
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_sequential_progression_basic(tmp_path: Path) -> None:
    L, T, B = 20, 5, 2
    ddir = _prepare_dataset(tmp_path, L=L)
    batches = _make_batches(ddir, batch_size=B, block_size=T, sampler="sequential")
    # First call
    x1, y1 = batches.get_batch("train")
    # Expected sequences: starts at 0 and 5
    exp_x0 = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=torch.long)
    exp_y0 = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=torch.long)
    assert torch.equal(x1.cpu(), exp_x0)
    assert torch.equal(y1.cpu(), exp_y0)

    # Second call: cursor logic advances; first sample at 10..14, second wraps
    x2, y2 = batches.get_batch("train")
    exp_x1 = torch.tensor([[10, 11, 12, 13, 14], [16, 17, 18, 19, 0]], dtype=torch.long)
    exp_y1 = torch.tensor([[11, 12, 13, 14, 15], [17, 18, 19, 0, 1]], dtype=torch.long)
    assert torch.equal(x2.cpu(), exp_x1)
    assert torch.equal(y2.cpu(), exp_y1)


def test_sequential_wrap_small_L_leq_T(tmp_path: Path) -> None:
    # L <= T path must wrap within a single sequence
    L, T, B = 4, 6, 1
    ddir = _prepare_dataset(tmp_path, L=L)
    batches = _make_batches(ddir, batch_size=B, block_size=T, sampler="sequential")
    x1, y1 = batches.get_batch("train")
    exp_x1 = torch.tensor([[0, 1, 2, 3, 0, 1]], dtype=torch.long)
    exp_y1 = torch.tensor([[1, 2, 3, 0, 1, 2]], dtype=torch.long)
    assert torch.equal(x1.cpu(), exp_x1)
    assert torch.equal(y1.cpu(), exp_y1)
    # Next call starts from cursor advanced by T mod L
    x2, y2 = batches.get_batch("train")
    exp_x2 = torch.tensor([[2, 3, 0, 1, 2, 3]], dtype=torch.long)
    exp_y2 = torch.tensor([[3, 0, 1, 2, 3, 0]], dtype=torch.long)
    assert torch.equal(x2.cpu(), exp_x2)
    assert torch.equal(y2.cpu(), exp_y2)
