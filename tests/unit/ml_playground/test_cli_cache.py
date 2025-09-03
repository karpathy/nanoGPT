from __future__ import annotations

from pathlib import Path

import pytest

import ml_playground.cli as cli
from ml_playground.config import (
    AppConfig,
    PreparerConfig,
    TrainerConfig,
    ModelConfig,
    DataConfig,
    OptimConfig,
    LRSchedule,
    RuntimeConfig,
)


class DummyCtx:
    def __init__(self):
        self.obj = {}

    def ensure_object(self, typ):  # noqa: D401
        # match Typer Context API used in ensure_loaded
        if not isinstance(self.obj, dict):
            self.obj = {}
        return self.obj


def test_ensure_loaded_caches_and_propagates_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # Prepare fake return values
    cfg_path = tmp_path / "exp" / "config.toml"
    app = AppConfig(train=None, sample=None)
    prep = PreparerConfig()

    calls = {"n": 0}

    def fake_load_app_config(experiment: str, exp_config: Path | None):
        calls["n"] += 1
        # Record errors for this cfg_path so ensure_loaded can surface them
        cli._last_load_errors[cfg_path] = {"train": "TERR", "sample": "SERR"}
        return cfg_path, app, prep

    monkeypatch.setattr(cli, "load_app_config", fake_load_app_config)

    ctx = DummyCtx()
    out = cli.ensure_loaded(ctx, "bundestag_char")
    assert out == (cfg_path, app, prep)
    # First call hit loader
    assert calls["n"] == 1
    # Errors cached on ctx
    assert ctx.obj["loaded_errors"]["train"] == "TERR"
    assert ctx.obj["loaded_errors"]["sample"] == "SERR"

    # Second call should return from cache and not call loader again
    out2 = cli.ensure_loaded(ctx, "bundestag_char")
    assert out2 == (cfg_path, app, prep)
    assert calls["n"] == 1


def test_apply_overrides_non_dict_is_ignored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # Use train-specific wrapper to hit generic path
    base = TrainerConfig(
        model=ModelConfig(),
        data=DataConfig(dataset_dir=tmp_path),
        optim=OptimConfig(),
        schedule=LRSchedule(),
        runtime=RuntimeConfig(out_dir=tmp_path),
    )
    # Set env to a valid JSON that is not a dict
    monkeypatch.setenv("ML_PLAYGROUND_TRAIN_OVERRIDES", "[1, 2, 3]")
    out = cli._apply_train_overrides(base)
    assert out == base
