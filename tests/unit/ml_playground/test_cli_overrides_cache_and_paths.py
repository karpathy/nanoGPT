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


class _Ctx:
    def __init__(self):
        self.obj: dict | None = None

    def ensure_object(self, typ):  # noqa: D401
        if not isinstance(self.obj, dict):
            self.obj = {}
        return self.obj


def _mk_train_cfg(tmp_path: Path) -> TrainerConfig:
    return TrainerConfig(
        model=ModelConfig(),
        data=DataConfig(dataset_dir=tmp_path),
        optim=OptimConfig(),
        schedule=LRSchedule(),
        runtime=RuntimeConfig(out_dir=tmp_path),
    )


def _mk_sample_cfg(tmp_path: Path) -> SamplerConfig:
    return SamplerConfig(runtime=RuntimeConfig(out_dir=tmp_path), sample=SampleConfig())


def test_apply_overrides_malformed_and_non_dict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    base_t = _mk_train_cfg(tmp_path)
    base_s = _mk_sample_cfg(tmp_path)

    # Malformed JSON -> keep original
    monkeypatch.setenv("ML_PLAYGROUND_TRAIN_OVERRIDES", "{not json}")
    assert cli._apply_train_overrides(base_t) == base_t

    # Non-dict JSON -> keep original
    monkeypatch.setenv("ML_PLAYGROUND_TRAIN_OVERRIDES", "[1,2,3]")
    assert cli._apply_train_overrides(base_t) == base_t

    # Same checks for sample overrides
    monkeypatch.setenv("ML_PLAYGROUND_SAMPLE_OVERRIDES", "null")
    assert cli._apply_sample_overrides(base_s) == base_s
    monkeypatch.setenv("ML_PLAYGROUND_SAMPLE_OVERRIDES", '"str"')
    assert cli._apply_sample_overrides(base_s) == base_s


def test_ensure_loaded_uses_cache_and_ignores_non_path_exp_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    ctx = _Ctx()
    ctx.ensure_object(dict)
    # Put a non-Path exp_config to exercise type filtering (becomes None)
    ctx.obj["exp_config"] = "not a path"

    # Prepare cached values
    cfg_path = tmp_path / "cfg.toml"
    app = cli.AppConfig(train=None, sample=None)
    prep = cli.PreparerConfig()
    cache_key = ("exp1", None)
    ctx.obj["loaded_cache"] = {
        "key": cache_key,
        "cfg_path": cfg_path,
        "app": app,
        "prep": prep,
    }

    # If load_app_config were called, raise to detect it; but cache should short-circuit.
    def _should_not_load(*_a, **_k):  # noqa: D401
        raise RuntimeError("should not load")

    monkeypatch.setattr(cli, "load_app_config", _should_not_load)

    out = cli.ensure_loaded(ctx, "exp1")
    assert out == (cfg_path, app, prep)


def test_get_cfg_path_explicit_and_default(tmp_path: Path):
    explicit = tmp_path / "x.toml"
    # Explicit path returned as-is
    p = cli.get_cfg_path("some_exp", explicit)
    assert p == explicit
    # Default path under experiments root
    d = cli.get_cfg_path("bundestag_char", None)
    assert d.as_posix().endswith("ml_playground/experiments/bundestag_char/config.toml")


def test_resolve_and_load_configs_error_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # 1) Non-str experiment -> syntax error exit
    with pytest.raises(SystemExit) as ei1:
        cli._resolve_and_load_configs(123)  # type: ignore[arg-type]
    assert "Syntax error" in str(ei1.value)

    # 2) Missing config path -> exit
    with pytest.raises(SystemExit) as ei2:
        cli._resolve_and_load_configs("__no_such_experiment__")
    assert "Config not found" in str(ei2.value)

    # 3) Defaults invalid -> exit mentioning defaults path
    cfg = tmp_path / "ok.toml"
    cfg.write_text(
        "[train]\n[train.runtime]\n out_dir='.'\n[train.model]\n[train.data]\n[train.optim]\n[train.schedule]\n"
    )

    defaults_name = "default_config.toml"

    def fake_read_toml(path: Path):  # noqa: D401
        if path.name == defaults_name:
            raise ValueError("bad defaults")
        # Minimal dict for experiment
        return {
            "train": {
                "runtime": {"out_dir": "."},
                "model": {},
                "data": {},
                "optim": {},
                "schedule": {},
            }
        }

    monkeypatch.setattr(cli, "_read_toml_dict", fake_read_toml)
    with pytest.raises(SystemExit) as ei3:
        cli._resolve_and_load_configs("irrelevant", cfg)
    assert "Default config invalid" in str(ei3.value)

    # 4) Experiment config invalid -> exit mentioning cfg path
    def fake_read_toml_2(path: Path):  # noqa: D401
        if path == cfg:
            raise ValueError("bad exp")
        return {}

    monkeypatch.setattr(cli, "_read_toml_dict", fake_read_toml_2)
    with pytest.raises(SystemExit) as ei4:
        cli._resolve_and_load_configs("irrelevant", cfg)
    assert "Experiment config invalid" in str(ei4.value)
