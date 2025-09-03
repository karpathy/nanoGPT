from __future__ import annotations

import pickle
from pathlib import Path
import pytest

import numpy as np

from ml_playground.prepare import _PreparerInstance, PreparerConfig
from ml_playground.error_handling import DataError


class Logger:
    def __init__(self):
        self.infos: list[str] = []
        self.warnings: list[str] = []

    def info(self, msg: str):  # noqa: D401
        self.infos.append(str(msg))

    def warning(self, msg: str):  # noqa: D401
        self.warnings.append(str(msg))


def _mk_arrays(n=4):
    return (
        np.arange(n, dtype=np.uint16),
        np.arange(n, dtype=np.uint16),
        {"meta_version": 1},
    )


def test_write_bin_and_meta_raises_on_invalid_existing_meta(tmp_path: Path):
    ds = tmp_path / "ds"
    ds.mkdir()
    # Pre-create files with invalid meta content (not a dict with meta_version)
    (ds / "train.bin").write_bytes(b"x")
    (ds / "val.bin").write_bytes(b"y")
    with (ds / "meta.pkl").open("wb") as f:
        pickle.dump([1, 2, 3], f)  # invalid

    logger = Logger()
    cfg = PreparerConfig(dataset_dir=ds, logger=logger)
    inst = _PreparerInstance(cfg)
    tr, va, meta = _mk_arrays(8)
    with pytest.raises(DataError):
        inst._write_bin_and_meta(ds, tr, va, meta)


def test_write_bin_and_meta_raises_on_unreadable_meta(tmp_path: Path):
    ds = tmp_path / "ds2"
    ds.mkdir()
    # Pre-create files with unreadable meta (corrupted bytes)
    (ds / "train.bin").write_bytes(b"x")
    (ds / "val.bin").write_bytes(b"y")
    (ds / "meta.pkl").write_bytes(b"not-a-pickle")

    cfg = PreparerConfig(dataset_dir=ds, logger=None)
    inst = _PreparerInstance(cfg)
    tr, va, meta = _mk_arrays(6)
    with pytest.raises(DataError):
        inst._write_bin_and_meta(ds, tr, va, meta)


def test_write_bin_and_meta_info_logs_created_then_raises_on_invalid_second_run(
    tmp_path: Path,
):
    ds = tmp_path / "ds3"
    logger = Logger()
    cfg = PreparerConfig(dataset_dir=ds, logger=logger)
    inst = _PreparerInstance(cfg)

    # First write -> should log Created entries
    tr, va, meta = _mk_arrays(5)
    inst._write_bin_and_meta(ds, tr, va, meta)
    assert any("Created: [" in i for i in logger.infos)

    # Prepare pre-existing invalid meta and ensure second run raises
    with (ds / "meta.pkl").open("wb") as f:
        pickle.dump(["bad"], f)

    logger2 = Logger()
    inst2 = _PreparerInstance(PreparerConfig(dataset_dir=ds, logger=logger2))
    tr2, va2, meta2 = _mk_arrays(7)
    with pytest.raises(DataError):
        inst2._write_bin_and_meta(ds, tr2, va2, meta2)
