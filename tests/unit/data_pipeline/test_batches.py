"""Targeted unit tests for `ml_playground.data_pipeline.sampling.batches`."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ml_playground.configuration.models import DataConfig
from ml_playground.data_pipeline.sampling.batches import SimpleBatches, sample_batch


def _writer(path: Path, arr: np.ndarray) -> None:
    path.write_bytes(arr.tobytes())


def _dataset_dir(
    tmp_path: Path, *, train: np.ndarray, val: np.ndarray, meta: dict | None = None
) -> Path:
    ddir = tmp_path / "dataset"
    ddir.mkdir()
    _writer(ddir / "train.bin", train)
    _writer(ddir / "val.bin", val)
    if meta is not None:
        import pickle

        (ddir / "meta.pkl").write_bytes(pickle.dumps(meta))
    return ddir


def test_sample_batch_empty_reader_raises_value_error() -> None:
    """`sample_batch` should reject readers with zero length."""

    reader = SimpleNamespace(arr=np.array([], dtype=np.uint16), length=0)
    with pytest.raises(ValueError):
        sample_batch(reader, batch_size=1, block_size=1, device="cpu")


def test_sample_batch_wraps_when_length_leq_block() -> None:
    """When `L <= block_size`, `sample_batch` should wrap around the dataset."""

    arr = np.arange(3, dtype=np.uint16)
    reader = SimpleNamespace(arr=arr, length=arr.shape[0])
    np.random.seed(0)
    x, y = sample_batch(reader, batch_size=2, block_size=5, device="cpu")

    np.random.seed(0)
    idx = np.random.randint(0, reader.length, size=(2,), dtype=np.int64)
    steps = np.arange(5, dtype=np.int64)
    expected_x = arr[(idx[:, None] + steps) % reader.length]
    expected_y = arr[(idx[:, None] + 1 + steps) % reader.length]

    assert torch.equal(x.cpu(), torch.from_numpy(expected_x.astype(np.int64)))
    assert torch.equal(y.cpu(), torch.from_numpy(expected_y.astype(np.int64)))


def test_sample_batch_sliding_window_path_extracts_contiguous_windows() -> None:
    """When `L > block_size`, the sliding window path should be used."""

    arr = np.arange(12, dtype=np.uint16)
    reader = SimpleNamespace(arr=arr, length=arr.shape[0])
    np.random.seed(1)
    x, y = sample_batch(reader, batch_size=3, block_size=4, device="cpu")

    np.random.seed(1)
    idx = np.random.randint(0, reader.length - 4, size=(3,), dtype=np.int64)
    expected_x = np.stack([arr[i : i + 4] for i in idx], axis=0)
    expected_y = np.stack([arr[i + 1 : i + 5] for i in idx], axis=0)

    assert torch.equal(x.cpu(), torch.from_numpy(expected_x.astype(np.int64)))
    assert torch.equal(y.cpu(), torch.from_numpy(expected_y.astype(np.int64)))


def test_simple_batches_defaults_to_uint16_without_meta(tmp_path: Path) -> None:
    """`SimpleBatches` should default to `uint16` when `meta.pkl` is absent."""

    arr = np.arange(16, dtype=np.uint16)
    ddir = _dataset_dir(tmp_path, train=arr, val=arr)
    cfg = DataConfig(batch_size=2, block_size=4, sampler="random")
    batches = SimpleBatches(cfg, device="cpu", dataset_dir=ddir)
    assert batches.train.arr.dtype == np.uint16


def test_simple_batches_respects_meta_dtype_uint32(tmp_path: Path) -> None:
    """`SimpleBatches` should honor `dtype` specified in `meta.pkl`."""

    arr = np.arange(16, dtype=np.uint32)
    ddir = _dataset_dir(tmp_path, train=arr, val=arr, meta={"dtype": "uint32"})
    cfg = DataConfig(batch_size=2, block_size=4, sampler="random")
    batches = SimpleBatches(cfg, device="cpu", dataset_dir=ddir)
    assert batches.train.arr.dtype == np.uint32


def test_simple_batches_unknown_sampler_raises(tmp_path: Path) -> None:
    """`SimpleBatches.get_batch` should reject unsupported sampler names."""

    arr = np.arange(16, dtype=np.uint16)
    ddir = _dataset_dir(tmp_path, train=arr, val=arr)
    cfg = DataConfig(batch_size=2, block_size=4, sampler="random")
    batches = SimpleBatches(cfg, device="cpu", dataset_dir=ddir)
    # mutate sampler after construction to simulate unexpected configuration value
    object.__setattr__(batches.data, "sampler", "bogus")
    with pytest.raises(ValueError):
        batches.get_batch("train")


def test_simple_batches_sequential_empty_dataset_raises(tmp_path: Path) -> None:
    """Sequential batches should raise when underlying reader is empty."""

    arr = np.arange(4, dtype=np.uint16)
    ddir = _dataset_dir(tmp_path, train=arr, val=arr)
    cfg = DataConfig(batch_size=1, block_size=1, sampler="sequential")
    batches = SimpleBatches(cfg, device="cpu", dataset_dir=ddir)
    # replace reader with empty dataset to trigger guard without memmap creating empty files
    empty_reader = SimpleNamespace(arr=np.array([], dtype=np.uint16), length=0)
    batches.train = empty_reader  # type: ignore[assignment]
    with pytest.raises(ValueError):
        batches.get_batch("train")


def test_simple_batches_corrupt_meta_falls_back_to_uint16(tmp_path: Path) -> None:
    """Corrupt meta files should be ignored, defaulting to uint16."""

    arr = np.arange(8, dtype=np.uint16)
    ddir = _dataset_dir(tmp_path, train=arr, val=arr)
    (ddir / "meta.pkl").write_text("not-a-pickle", encoding="utf-8")

    cfg = DataConfig(batch_size=1, block_size=4, sampler="random")
    batches = SimpleBatches(cfg, device="cpu", dataset_dir=ddir)

    assert batches.train.arr.dtype == np.uint16


def test_simple_batches_meta_uint16_keeps_dtype(tmp_path: Path) -> None:
    """Meta files declaring uint16 should preserve the default dtype."""

    arr = np.arange(8, dtype=np.uint16)
    ddir = _dataset_dir(tmp_path, train=arr, val=arr, meta={"dtype": "uint16"})

    cfg = DataConfig(batch_size=1, block_size=4, sampler="random")
    batches = SimpleBatches(cfg, device="cpu", dataset_dir=ddir)

    assert batches.train.arr.dtype == np.uint16


def test_simple_batches_sequential_val_split_wraps(tmp_path: Path) -> None:
    """Sequential batches should advance the cursor separately for the val split."""

    arr = np.arange(10, dtype=np.uint16)
    ddir = _dataset_dir(tmp_path, train=arr, val=arr)
    cfg = DataConfig(batch_size=2, block_size=3, sampler="sequential")
    batches = SimpleBatches(cfg, device="cpu", dataset_dir=ddir)

    # Advance train cursor so val path is isolated
    batches.get_batch("train")

    x_val, y_val = batches.get_batch("val")
    assert x_val.shape == (2, 3)
    assert y_val.shape == (2, 3)

    # Cursor update should wrap around independently for val split
    x_next, _ = batches.get_batch("val")
    assert not torch.equal(x_val, x_next)
