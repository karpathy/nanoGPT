from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Any
import numpy as np
import torch
import pickle
from ml_playground.config import DataConfig


@dataclass
class _MemmapReader:
    arr: np.memmap
    length: int

    @classmethod
    def open(cls, path: Path, *, dtype: np.dtype) -> "_MemmapReader":
        arr = np.memmap(path, dtype=dtype, mode="r")
        return cls(arr=arr, length=int(arr.shape[0]))


def _sample_batch(
    reader: _MemmapReader, batch_size: int, block_size: int, device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    L = int(reader.length)
    if L == 0:
        raise ValueError(
            "Dataset is empty: no tokens available. Ensure the dataset preparation wrote non-empty train/val bins."
        )

    if L <= block_size:
        # dataset shorter than (or equal to) block_size: wrap around to build sequences
        idx = np.random.randint(0, L, size=(batch_size,), dtype=np.int64)
        base = np.asarray(reader.arr)

        def take_seq(start: int, length: int) -> np.ndarray:
            offs = (start + np.arange(length, dtype=np.int64)) % L
            return base[offs]

        x_np = np.stack(
            [take_seq(int(i), block_size).astype(np.int64, copy=False) for i in idx]
        )
        y_np = np.stack(
            [
                take_seq(int((i + 1) % L), block_size).astype(np.int64, copy=False)
                for i in idx
            ]
        )
    else:
        # normal path: sample contiguous blocks without wrapping
        idx = np.random.randint(0, L - block_size, size=(batch_size,), dtype=np.int64)
        x_np = np.stack(
            [reader.arr[i : i + block_size].astype(np.int64, copy=False) for i in idx]
        )
        y_np = np.stack(
            [
                reader.arr[i + 1 : i + 1 + block_size].astype(np.int64, copy=False)
                for i in idx
            ]
        )

    x = torch.from_numpy(x_np).to(device)
    y = torch.from_numpy(y_np).to(device)
    return x, y


class SimpleBatches:
    def __init__(self, data: DataConfig, device: str) -> None:
        self.data = data
        self.device = device
        train_path = data.dataset_dir / data.train_bin
        val_path = data.dataset_dir / data.val_bin
        if not train_path.exists() or not val_path.exists():
            raise FileNotFoundError(
                f"Training data not found at {train_path} and/or {val_path}"
            )
        # Determine dtype from meta.pkl if available; default to uint16
        dtype: np.dtype[Any] = np.dtype(np.uint16)
        try:
            if data.meta_pkl is not None:
                meta_path = data.dataset_dir / data.meta_pkl
                if meta_path.exists():
                    with meta_path.open("rb") as f:
                        meta = pickle.load(f)
                    dts = meta.get("dtype")
                    if dts == "uint32":
                        dtype = np.dtype(np.uint32)
                    elif dts == "uint16":
                        dtype = np.dtype(np.uint16)
        except Exception:
            # If meta cannot be read, default remains uint16
            pass
        self.train = _MemmapReader.open(train_path, dtype=dtype)
        self.val = _MemmapReader.open(val_path, dtype=dtype)

    def get_batch(self, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        reader = self.train if split == "train" else self.val
        return _sample_batch(
            reader, self.data.batch_size, self.data.block_size, self.device
        )
