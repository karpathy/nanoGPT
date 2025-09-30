from __future__ import annotations

from pathlib import Path

from ml_playground.configuration import SharedConfig
from pydantic import ValidationError
import pytest


def test_shared_paths_resolve_relative_string_values(tmp_path: Path) -> None:
    cfg_path = tmp_path / "exp" / "cfg.toml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text("")

    data = {
        "experiment": "unit",
        "config_path": cfg_path,
        "project_home": "..",  # relative string
        "dataset_dir": "../data",  # relative string
        "train_out_dir": "../out/train",  # relative string
        "sample_out_dir": "../out/sample",  # relative string
    }

    shared = SharedConfig(**data)

    assert shared.project_home.is_absolute()
    assert shared.dataset_dir.is_absolute()
    assert shared.train_out_dir.is_absolute()
    assert shared.sample_out_dir.is_absolute()

    # Resolved relative to cfg directory (cfg_dir = tmp_path/exp)
    assert shared.project_home == cfg_path.parent.parent.resolve()
    assert shared.dataset_dir == (cfg_path.parent.parent / "data").resolve()
    assert shared.train_out_dir == (cfg_path.parent.parent / "out" / "train").resolve()
    assert (
        shared.sample_out_dir == (cfg_path.parent.parent / "out" / "sample").resolve()
    )


def test_shared_paths_preserve_absolute_values(tmp_path: Path) -> None:
    cfg_path = tmp_path / "exp" / "cfg.toml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text("")

    abs_home = tmp_path / "home"
    abs_ds = tmp_path / "ds"
    abs_train = tmp_path / "runs" / "train"
    abs_sample = tmp_path / "runs" / "sample"

    data = {
        "experiment": "unit",
        "config_path": cfg_path,
        "project_home": abs_home,
        "dataset_dir": abs_ds,
        "train_out_dir": abs_train,
        "sample_out_dir": abs_sample,
    }

    shared = SharedConfig(**data)

    assert shared.project_home == abs_home
    assert shared.dataset_dir == abs_ds
    assert shared.train_out_dir == abs_train
    assert shared.sample_out_dir == abs_sample


def test_shared_paths_missing_config_path_raises() -> None:
    # SharedConfig is strict: config_path is required
    data = {
        "experiment": "unit",
        # no config_path provided
        "project_home": Path(".."),
        "dataset_dir": Path("data"),
        "train_out_dir": Path("out/train"),
        "sample_out_dir": Path("out/sample"),
    }

    with pytest.raises(ValidationError):
        SharedConfig(**data)
