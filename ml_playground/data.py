from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Any, Dict
import numpy as np
import numpy.typing as npt
import torch
import pickle
from ml_playground.config import DataConfig


@dataclass
class _MemmapReader:
    arr: npt.NDArray[Any]
    length: int

    @classmethod
    def open(cls, path: Path, *, dtype: np.dtype[Any]) -> "_MemmapReader":
        arr: npt.NDArray[Any] = np.memmap(path, dtype=dtype, mode="r")
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
            offs: npt.NDArray[np.int64] = (
                start + np.arange(length, dtype=np.int64)
            ) % L
            return base[offs]

        x_np = np.stack(
            [take_seq(int(i), block_size).astype(np.int64, copy=False) for i in idx]
        )
        y_np = np.stack(
            [
                take_seq(int((i + 1) % L), block_size).astype(np.int64, copy=True)
                for i in idx
            ]
        )
    else:
        # normal path: sample contiguous blocks without wrapping
        # choose valid start indices so that slicing i:i+block_size stays within bounds
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
    def __init__(self, data: DataConfig, device: str, dataset_dir: Path) -> None:
        self.data = data
        self.device = device
        self._dataset_dir = dataset_dir
        train_path = data.train_path(dataset_dir)
        val_path = data.val_path(dataset_dir)
        if not train_path.exists() or not val_path.exists():
            raise FileNotFoundError(
                f"Training data not found at {train_path} and/or {val_path}"
            )
        # Determine dtype from meta.pkl if available; default to uint16
        dtype: np.dtype[Any] = np.dtype(np.uint16)
        try:
            # Read dtype from meta.pkl when present
            meta_path = data.meta_path(dataset_dir)
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
        # Maintain per-split cursors for sequential sampling
        self._cursor: Dict[str, int] = {"train": 0, "val": 0}

    def get_batch(self, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        reader = self.train if split == "train" else self.val
        if getattr(self.data, "sampler", "random") == "sequential":
            # Deterministic, strided coverage with wrap-around
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
                    # wrap-around sequence
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
                        # straight slice without wrap
                        x_seq = base[si : si + T].astype(np.int64, copy=False)
                        y_seq = base[si + 1 : si + 1 + T].astype(np.int64, copy=False)
                        cur = si + T
                        if cur >= L - T:
                            # stride by 1 at epoch boundary to avoid repeating last token window
                            cur = (cur + 1) % L
                    else:
                        # need to wrap for last few tokens
                        x_first = base[si:L].astype(np.int64, copy=False)
                        x_rem = T - int(x_first.shape[0])
                        if x_rem > 0:
                            x_wrap = base[:x_rem].astype(np.int64, copy=False)
                            x_seq = np.concatenate([x_first, x_wrap], axis=0)
                        else:
                            x_seq = x_first
                        y_first = base[si + 1 : L].astype(np.int64, copy=False)
                        y_rem = T - int(y_first.shape[0])
                        if y_rem > 0:
                            y_wrap = base[:y_rem].astype(np.int64, copy=False)
                            y_seq = np.concatenate([y_first, y_wrap], axis=0)
                        else:
                            y_seq = y_first
                        cur = (si + T) % L
                x_list.append(x_seq)
                y_list.append(y_seq)
            self._cursor[split] = cur
            x_np = np.stack(x_list)
            y_np = np.stack(y_list)
            x = torch.from_numpy(x_np).to(self.device)
            y = torch.from_numpy(y_np).to(self.device)
            return x, y
        else:
            return _sample_batch(
                reader, self.data.batch_size, self.data.block_size, self.device
            )
