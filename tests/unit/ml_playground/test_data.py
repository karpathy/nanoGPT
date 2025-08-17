from __future__ import annotations
from pathlib import Path
import numpy as np
from ml_playground.data import SimpleBatches
from ml_playground.config import DataConfig


def test_batches(tmp_path: Path) -> None:
    ddir = tmp_path
    arr = (np.arange(200) % 256).astype("uint16")
    (ddir / "train.bin").write_bytes(arr.tobytes())
    (ddir / "val.bin").write_bytes(arr.tobytes())

    cfg = DataConfig(dataset_dir=ddir, batch_size=2, block_size=8, grad_accum_steps=1)
    batches = SimpleBatches(cfg, device="cpu")
    x, y = batches.get_batch("train")
    assert x.shape == (2, 8)
    assert y.shape == (2, 8)
