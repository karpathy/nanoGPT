from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
from .config import DataConfig


@dataclass
class _MemmapReader:
    arr: np.memmap
    length: int

    @classmethod
    def open(cls, path: Path) -> "_MemmapReader":
        arr = np.memmap(path, dtype=np.uint16, mode="r")
        return cls(arr=arr, length=int(arr.shape[0]))


def _sample_batch(reader: _MemmapReader, batch_size: int, block_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    # choose random start positions
    idx = np.random.randint(0, reader.length - block_size, size=(batch_size,), dtype=np.int64)
    x_np = np.stack([reader.arr[i:i + block_size].astype(np.int64, copy=False) for i in idx])
    y_np = np.stack([reader.arr[i + 1:i + 1 + block_size].astype(np.int64, copy=False) for i in idx])
    x = torch.from_numpy(x_np).to(device)
    y = torch.from_numpy(y_np).to(device)
    return x, y


class SimpleBatches:
    def __init__(self, data: DataConfig, device: str) -> None:
        self.data = data
        self.device = device
        self.train = _MemmapReader.open(data.dataset_dir / data.train_bin)
        self.val = _MemmapReader.open(data.dataset_dir / data.val_bin)

    def get_batch(self, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        reader = self.train if split == "train" else self.val
        return _sample_batch(reader, self.data.batch_size, self.data.block_size, self.device)
