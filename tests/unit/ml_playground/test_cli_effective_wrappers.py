from __future__ import annotations

from pathlib import Path

import pytest
import typer

import ml_playground.cli as cli
from ml_playground.config import (
    PreparerConfig,
    TrainerConfig,
    SamplerConfig,
    ModelConfig,
    DataConfig,
    OptimConfig,
    LRSchedule,
    RuntimeConfig,
    SampleConfig,
)


def test_get_cfg_path_uses_exp_config_and_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # When exp_config provided, it should be returned as-is
    p = tmp_path / "custom.toml"
    assert cli.get_cfg_path("exp", p) is p

    # When not provided, should be experiments/<experiment>/config.toml
    base = tmp_path / "experiments_root"
    monkeypatch.setattr(cli, "_experiments_root", lambda: base)
    got = cli.get_cfg_path("foo", None)
    assert got == base / "foo" / "config.toml"


def test_load_effective_train_raises_exit_on_validation_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
):
    cfg_path = tmp_path / "exp" / "config.toml"
    monkeypatch.setattr(
        cli,
        "_resolve_and_load_configs",
        lambda experiment, exp_config: (cfg_path, {"train": {}}, {"train": {}}),
    )

    # Validation from raw raises
    def _raise_bad(*_args, **_kwargs):  # noqa: D401
        raise ValueError("bad")

    monkeypatch.setattr(cli, "_load_train_config_from_raw", _raise_bad)

    with pytest.raises(typer.Exit) as ei:
        cli.load_effective_train("exp", None)
    assert ei.value.exit_code == 2
    assert "bad" in capsys.readouterr().out


def test_load_effective_train_success_applies_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    cfg_path = tmp_path / "exp" / "config.toml"
    monkeypatch.setattr(
        cli,
        "_resolve_and_load_configs",
        lambda experiment, exp_config: (cfg_path, {"train": {}}, {"train": {}}),
    )
    # Strict validation succeeds (returns ignored value)
    monkeypatch.setattr(
        cli, "_load_train_config_from_raw", lambda *_args, **_kwargs: object()
    )

    loaded_cfg = object()
    monkeypatch.setattr(cli, "load_train_config", lambda _path: loaded_cfg)

    sentinel = object()
    monkeypatch.setattr(cli, "_apply_train_overrides", lambda cfg: sentinel)

    got_path, got_cfg = cli.load_effective_train("exp", None)
    assert got_path == cfg_path
    assert got_cfg is sentinel


class FakeRuntime:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir

    def model_copy(self, update=None):  # noqa: D401
        update = update or {}
        return FakeRuntime(update.get("out_dir", self.out_dir))


class FakeSampleCfg:
    def __init__(self, runtime: FakeRuntime | None):
        self.runtime = runtime

    def model_copy(self, update=None):  # noqa: D401
        update = update or {}
        return FakeSampleCfg(update.get("runtime", self.runtime))


def test_load_effective_sample_resolves_relative_out_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    cfg_path = tmp_path / "exp" / "config.toml"
    monkeypatch.setattr(
        cli,
        "_resolve_and_load_configs",
        lambda experiment, exp_config: (cfg_path, {"sample": {}}, {"sample": {}}),
    )

    rel_runtime = FakeRuntime(Path("out"))
    sample_cfg0 = FakeSampleCfg(rel_runtime)
    monkeypatch.setattr(
        cli, "_load_sample_config_from_raw", lambda *_args, **_kwargs: sample_cfg0
    )

    # Identity override to observe out_dir resolution
    monkeypatch.setattr(cli, "_apply_sample_overrides", lambda cfg: cfg)

    got_path, got_cfg = cli.load_effective_sample("exp", None)
    assert got_path == cfg_path
    # out_dir should be resolved relative to config path's parent
    assert got_cfg.runtime.out_dir == (cfg_path.parent / "out").resolve()


def test_load_effective_sample_bubbles_validation_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
):
    cfg_path = tmp_path / "exp" / "config.toml"
    monkeypatch.setattr(
        cli,
        "_resolve_and_load_configs",
        lambda experiment, exp_config: (cfg_path, {"sample": {}}, {"sample": {}}),
    )
    monkeypatch.setattr(
        cli,
        "_load_sample_config_from_raw",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("oops")),
    )

    with pytest.raises(typer.Exit) as ei:
        cli.load_effective_sample("exp", None)
    assert ei.value.exit_code == 2
    assert "oops" in capsys.readouterr().out


def test_load_effective_train_bubbles_validation_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
):
    # Fake resolve_and_load_configs to return some paths and raw dicts
    p = tmp_path / "exp" / "config.toml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("[dummy]\n")

    monkeypatch.setattr(
        cli,
        "_resolve_and_load_configs",
        lambda experiment, exp_config: (p, {"train": {}}, {"train": {}}),
    )

    # Strict raw loader raises
    monkeypatch.setattr(
        cli,
        "_load_train_config_from_raw",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("oops-train")),
    )

    with pytest.raises(typer.Exit) as ei:
        cli.load_effective_train("exp", None)
    assert ei.value.exit_code == 2
    assert "oops-train" in capsys.readouterr().out


def test_load_train_config_resolves_relative_paths(tmp_path: Path):
    # Create an experiment TOML with only relative paths; rely on defaults for the rest
    p = tmp_path / "exp" / "config.toml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        """
        [train]
        [train.data]
        dataset_dir = "data_rel"
        [train.runtime]
        out_dir = "out_rel"
        """
    )

    cfg = cli.load_train_config(p)
    # Both paths should be resolved relative to p.parent
    assert cfg.data.dataset_dir.is_absolute()
    assert cfg.runtime.out_dir.is_absolute()
    assert str(cfg.data.dataset_dir).startswith(str(p.parent))
    assert str(cfg.runtime.out_dir).startswith(str(p.parent))


def test_load_sample_config_resolves_relative_out_dir_with_runtime_ref(tmp_path: Path):
    # Use runtime_ref to defaults' train.runtime and override out_dir relatively
    p = tmp_path / "exp" / "config.toml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        """
        [sample]
        runtime_ref = "train.runtime"
        [sample.runtime]
        out_dir = "out_rel"
        [sample.sample]
        # rely on defaults for actual fields
        """
    )

    cfg = cli.load_sample_config(p)
    assert cfg.runtime is not None
    assert cfg.runtime.out_dir.is_absolute()
    assert str(cfg.runtime.out_dir).startswith(str(p.parent))


def test_load_app_config_resolves_relative_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # Fake resolve_and_load_configs to control cfg path and raw dicts
    cfg_path = tmp_path / "exp" / "config.toml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text("[dummy]\n")

    monkeypatch.setattr(
        cli,
        "_resolve_and_load_configs",
        lambda experiment, exp_config: (
            cfg_path,
            {"prepare": {}, "train": {}, "sample": {}},
            {"prepare": {}, "train": {}, "sample": {}},
        ),
    )

    # Return configs with relative paths
    monkeypatch.setattr(
        cli,
        "_load_prepare_config_from_raw",
        lambda *_a, **_k: PreparerConfig(dataset_dir=Path("data"), raw_dir=Path("raw")),
    )

    monkeypatch.setattr(
        cli,
        "_load_train_config_from_raw",
        lambda *_a, **_k: TrainerConfig(
            model=ModelConfig(),
            data=DataConfig(dataset_dir=Path("ds")),
            optim=OptimConfig(),
            schedule=LRSchedule(),
            runtime=RuntimeConfig(out_dir=Path("out")),
        ),
    )

    monkeypatch.setattr(
        cli,
        "_load_sample_config_from_raw",
        lambda *_a, **_k: SamplerConfig(
            runtime=RuntimeConfig(out_dir=Path("sout")), sample=SampleConfig()
        ),
    )

    # Ensure overrides are no-ops
    monkeypatch.setattr(cli, "_apply_train_overrides", lambda cfg: cfg)
    monkeypatch.setattr(cli, "_apply_sample_overrides", lambda cfg: cfg)

    got_cfg_path, app, prep = cli.load_app_config("irrelevant", None)
    assert got_cfg_path == cfg_path

    # Preparer paths resolved
    assert prep.dataset_dir == (cfg_path.parent / "data").resolve()
    assert prep.raw_dir == (cfg_path.parent / "raw").resolve()

    # Train paths resolved
    assert app.train is not None
    assert app.train.data.dataset_dir == (cfg_path.parent / "ds").resolve()
    assert app.train.runtime.out_dir == (cfg_path.parent / "out").resolve()

    # Sample out_dir resolved
    assert app.sample is not None
    assert app.sample.runtime is not None
    assert app.sample.runtime.out_dir == (cfg_path.parent / "sout").resolve()
