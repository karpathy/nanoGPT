from __future__ import annotations

from typing import Tuple, Protocol, Iterable
from pathlib import Path
import pickle
import numpy as np


class Encoder(Protocol):
    def encode_ordinary(self, text: str) -> list[int]: ...


def split_train_val(text: str, split: float = 0.9) -> tuple[str, str]:
    """Deterministically split text into train/val by ratio."""
    if not 0.0 < split < 1.0:
        raise ValueError(f"split must be in (0,1), got {split}")
    n = len(text)
    return text[: int(n * split)], text[int(n * split) :]


def encode_split_with_encoder(
    train_data: str, val_data: str, enc: Encoder
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode provided train/val splits using a provided encoder.

    Returns two numpy arrays of dtype uint16.
    """
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)
    train_arr = np.array(train_ids, dtype=np.uint16)
    val_arr = np.array(val_ids, dtype=np.uint16)
    return train_arr, val_arr


def prepare_with_encoder(text: str, enc: Encoder) -> Tuple[np.ndarray, np.ndarray]:
    """Split text 90/10 and encode with a provided encoder."""
    train_data, val_data = split_train_val(text, 0.9)
    return encode_split_with_encoder(train_data, val_data, enc)


def seed_text_file(dest: Path, candidates: Iterable[Path]) -> Path:
    """Copy the first existing candidate file into dest; fail fast if none found."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    for p in candidates:
        if p.exists():
            dest.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"Seeded {dest} from resource {p}.")
            return dest
    raise SystemExit(f"Expected input text at {dest}, and no resource was found.")


def write_bin_and_meta(ds_dir: Path, train: np.ndarray, val: np.ndarray, meta: dict) -> None:
    """Write train.bin, val.bin, and meta.pkl; fail if arrays have unexpected dtypes/shapes."""
    if train.dtype != val.dtype:
        raise ValueError("train and val arrays must have the same dtype")
    ds_dir.mkdir(parents=True, exist_ok=True)
    (ds_dir / "train.bin").write_bytes(train.tobytes())
    (ds_dir / "val.bin").write_bytes(val.tobytes())
    with (ds_dir / "meta.pkl").open("wb") as f:
        pickle.dump(meta, f)
    print("Wrote:", ds_dir / "train.bin", ds_dir / "val.bin", ds_dir / "meta.pkl")
