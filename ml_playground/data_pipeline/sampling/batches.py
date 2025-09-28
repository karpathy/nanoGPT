"""Batch sampling utilities backed by memory-mapped datasets."""

from __future__ import annotations

from typing import Any, Literal, cast

import pickle
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch

from ml_playground.config import DataConfig, DeviceKind
from ml_playground.data_pipeline.sources.memmap import MemmapReader

__all__ = ["sample_batch", "SimpleBatches"]


def sample_batch(
    reader: MemmapReader,
    batch_size: int,
    block_size: int,
    device: DeviceKind,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of token sequences from the given memmap reader."""
    L = int(reader.length)
    if L == 0:
        raise ValueError(
            "Dataset is empty: no tokens available. Ensure the dataset preparation wrote non-empty train/val bins."
        )

    if L <= block_size:
        idx = np.random.randint(0, L, size=(batch_size,), dtype=np.int64)
        base = np.asarray(reader.arr)

        def _take_seq(start: int, length: int) -> np.ndarray:
            offs: npt.NDArray[np.int64] = (
                start + np.arange(length, dtype=np.int64)
            ) % L
            return base[offs]

        x_np = np.stack(
            [_take_seq(int(i), block_size).astype(np.int64, copy=False) for i in idx]
        )
        y_np = np.stack(
            [
                _take_seq(int((i + 1) % L), block_size).astype(np.int64, copy=True)
                for i in idx
            ]
        )
    else:
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
    """Iterate over memory-mapped training and validation datasets."""

    def __init__(
        self,
        data: DataConfig,
        device: DeviceKind,
        dataset_dir: Path,
    ) -> None:
        self.data = data
        self.device: DeviceKind = device
        self._dataset_dir = dataset_dir
        train_path = data.train_path(dataset_dir)
        val_path = data.val_path(dataset_dir)
        if not train_path.exists() or not val_path.exists():
            raise FileNotFoundError(
                f"Training data not found at {train_path} and/or {val_path}"
            )

        dtype: np.dtype[Any] = np.dtype(np.uint16)
        meta_path = data.meta_path(dataset_dir)
        if meta_path.exists():
            try:
                with meta_path.open("rb") as f:
                    meta = pickle.load(f)
            except (OSError, pickle.UnpicklingError, EOFError):
                meta = None
            if isinstance(meta, dict):
                dts = meta.get("dtype")
                if dts == "uint32":
                    dtype = np.dtype(np.uint32)
                elif dts == "uint16":
                    dtype = np.dtype(np.uint16)
        self.train = MemmapReader.open(train_path, dtype=dtype)
        self.val = MemmapReader.open(val_path, dtype=dtype)
        self._cursor: dict[Literal["train", "val"], int] = {"train": 0, "val": 0}

    def get_batch(
        self, split: Literal["train", "val"]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reader = self.train if split == "train" else self.val
        sampler = cast(
            Literal["random", "sequential"],
            getattr(self.data, "sampler", "random"),
        )
        if sampler == "sequential":
            return self._get_sequential_batch(split, reader)
        if sampler == "random":
            return sample_batch(
                reader,
                self.data.batch_size,
                self.data.block_size,
                self.device,
            )
        raise ValueError(f"Unknown sampler '{sampler}'")

    def _get_sequential_batch(
        self, split: Literal["train", "val"], reader: MemmapReader
    ) -> tuple[torch.Tensor, torch.Tensor]:
        L = int(reader.length)
        if L == 0:
            raise ValueError(
                "Dataset is empty: no tokens available. Ensure the dataset preparation wrote non-empty train/val bins."
            )
        bsz = int(self.data.batch_size)
        T = int(self.data.block_size)
        cur = self._cursor[split]
        base = np.asarray(reader.arr)
        x_list: list[np.ndarray] = []
        y_list: list[np.ndarray] = []
        for _ in range(bsz):
            s = cur
            if L <= T:
                offs: npt.NDArray[np.int64] = (s + np.arange(T, dtype=np.int64)) % L
                x_seq = base[offs].astype(np.int64, copy=False)
                offs_y: npt.NDArray[np.int64] = (
                    (s + 1) + np.arange(T, dtype=np.int64)
                ) % L
                y_seq = base[offs_y].astype(np.int64, copy=False)
                cur = (cur + T) % L
            else:
                si = int(s)
                if si + T <= L:
                    x_seq = base[si : si + T].astype(np.int64, copy=False)
                    if si + 1 + T <= L:
                        y_seq = base[si + 1 : si + 1 + T].astype(np.int64, copy=False)
                    else:
                        y_first = base[si + 1 : L].astype(np.int64, copy=False)
                        y_rem = T - int(y_first.shape[0])
                        rem = base[0:y_rem].astype(np.int64, copy=False)
                        y_seq = np.concatenate((y_first, rem))
                else:
                    offs_x: npt.NDArray[np.int64] = (
                        si + np.arange(T, dtype=np.int64)
                    ) % L
                    x_seq = base[offs_x].astype(np.int64, copy=False)
                    offs_y = (si + 1 + np.arange(T, dtype=np.int64)) % L
                    y_seq = base[offs_y].astype(np.int64, copy=False)
                cur = (cur + T) % L
            x_list.append(x_seq)
            y_list.append(y_seq)

        self._cursor[split] = cur
        x = torch.from_numpy(np.stack(x_list)).to(self.device)
        y = torch.from_numpy(np.stack(y_list)).to(self.device)
        return x, y
