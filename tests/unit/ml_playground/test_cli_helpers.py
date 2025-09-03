from __future__ import annotations

from pathlib import Path

import pytest

import ml_playground.cli as cli
from ml_playground.config import (
    TrainerConfig,
    SamplerConfig,
    ModelConfig,
    DataConfig,
    OptimConfig,
    LRSchedule,
    RuntimeConfig,
    SampleConfig,
)


def test_apply_train_overrides_env_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    base = TrainerConfig(
        model=ModelConfig(),
        data=DataConfig(dataset_dir=tmp_path),
        optim=OptimConfig(),
        schedule=LRSchedule(),
        runtime=RuntimeConfig(out_dir=tmp_path),
    )
    monkeypatch.setenv(
        "ML_PLAYGROUND_TRAIN_OVERRIDES",
        '{"runtime": {"log_interval": 123}, "data": {"batch_size": 99}}',
    )
    out = cli._apply_train_overrides(base)
    assert out.runtime.log_interval == 123
    assert out.data.batch_size == 99


def test_apply_sample_overrides_env_roundtrip(monkeypatch: pytest.MonkeyPatch):
    base = SamplerConfig(
        runtime=RuntimeConfig(out_dir=Path("/tmp")),
        sample=SampleConfig(),
    )
    monkeypatch.setenv(
        "ML_PLAYGROUND_SAMPLE_OVERRIDES",
        '{"sample": {"num_samples": 7, "top_k": 5}}',
    )
    out = cli._apply_sample_overrides(base)
    assert out.sample.num_samples == 7 and out.sample.top_k == 5


def test_get_cfg_path_with_and_without_override(tmp_path: Path):
    # When exp_config provided, it should win
    override = tmp_path / "x.toml"
    override.write_text("[train]\n")
    p = cli.get_cfg_path("bundestag_char", override)
    assert p == override

    # Else should point to experiments/<experiment>/config.toml
    p2 = cli.get_cfg_path("bundestag_char", None)
    assert p2.name == "config.toml" and p2.parts[-3:-1] == (
        "experiments",
        "bundestag_char",
    )


def test_read_toml_dict_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        cli._read_toml_dict(tmp_path / "missing.toml")


def test_apply_overrides_invalid_json_is_ignored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    base = TrainerConfig(
        model=ModelConfig(),
        data=DataConfig(dataset_dir=tmp_path),
        optim=OptimConfig(),
        schedule=LRSchedule(),
        runtime=RuntimeConfig(out_dir=tmp_path),
    )
    monkeypatch.setenv("ML_PLAYGROUND_TRAIN_OVERRIDES", "{not: json}")
    out = cli._apply_train_overrides(base)
    # unchanged
    assert out == base
