from __future__ import annotations

from pathlib import Path
import logging

import pytest
import typer

import ml_playground.cli as cli


class _Ctx:
    def __init__(self):
        self.obj = {}

    def ensure_object(self, typ):  # noqa: D401
        if not isinstance(self.obj, dict):
            self.obj = {}
        return self.obj


def test_global_options_missing_exp_config_exits(tmp_path: Path):
    ctx = _Ctx()
    missing = tmp_path / "does_not_exist.toml"
    assert not missing.exists()
    with pytest.raises(typer.Exit) as ei:
        cli.global_options(ctx, missing)
    assert ei.value.exit_code == 2


def test_ensure_loaded_loader_exception_maps_to_exit(
    monkeypatch: pytest.MonkeyPatch, capsys
):
    # Make load_app_config raise an unexpected exception
    monkeypatch.setattr(
        cli,
        "load_app_config",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("kaboom")),
    )

    ctx = _Ctx()
    with pytest.raises(typer.Exit) as ei:
        cli.ensure_loaded(ctx, "some_exp")
    assert ei.value.exit_code == 2
    assert "kaboom" in capsys.readouterr().out


def test_global_options_with_existing_handler_keeps_logger(tmp_path: Path):
    # Ensure root logger has a handler before calling global_options
    root = logging.getLogger()
    # Save and restore handlers
    old_handlers = list(root.handlers)
    try:
        h = logging.StreamHandler()
        root.addHandler(h)
        ctx = _Ctx()
        # Should not crash and should not reset handlers
        cli.global_options(ctx, None)
        assert isinstance(ctx.obj, dict)
        assert h in logging.getLogger().handlers
    finally:
        # restore
        root.handlers = old_handlers


def test_global_options_without_handlers_sets_basic_config(tmp_path: Path):
    root = logging.getLogger()
    # Save and clear handlers
    old_handlers = list(root.handlers)
    try:
        root.handlers = []
        ctx = _Ctx()
        cli.global_options(ctx, None)
        # Should have at least one handler configured now
        assert logging.getLogger().handlers, "Expected basicConfig to add a handler"
    finally:
        root.handlers = old_handlers


def test_log_command_status_covers_paths(tmp_path: Path):
    # Create existing dataset_dir and out_dir, and a missing path scenario is covered by non-existent files
    from ml_playground.config import (
        TrainerConfig,
        ModelConfig,
        DataConfig,
        OptimConfig,
        LRSchedule,
        RuntimeConfig,
        SamplerConfig,
        SampleConfig,
    )

    ds = tmp_path / "dataset"
    ds.mkdir(parents=True, exist_ok=True)
    (ds / "a.txt").write_text("x")
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)

    train_cfg = TrainerConfig(
        model=ModelConfig(),
        data=DataConfig(dataset_dir=ds),
        optim=OptimConfig(),
        schedule=LRSchedule(),
        runtime=RuntimeConfig(out_dir=out),
    )
    # Should not raise
    cli._log_command_status("train", train_cfg)

    # Sampler with runtime
    samp_cfg = SamplerConfig(runtime=RuntimeConfig(out_dir=out), sample=SampleConfig())
    cli._log_command_status("sample", samp_cfg)


def test_log_command_status_missing_paths(tmp_path: Path):
    from ml_playground.config import (
        TrainerConfig,
        ModelConfig,
        DataConfig,
        OptimConfig,
        LRSchedule,
        RuntimeConfig,
    )

    # None of these paths exist
    ds = tmp_path / "missing_ds"
    out = tmp_path / "missing_out"
    cfg = TrainerConfig(
        model=ModelConfig(),
        data=DataConfig(dataset_dir=ds),
        optim=OptimConfig(),
        schedule=LRSchedule(),
        runtime=RuntimeConfig(out_dir=out),
    )
    # Should not raise
    cli._log_command_status("train", cfg)


def test_extract_exp_config_edge_cases():
    class Ctx1:
        def __init__(self):  # noqa: D401
            self.obj = {"exp_config": Path("/x/y.toml")}

    assert cli._extract_exp_config(Ctx1()) == Path("/x/y.toml")

    class Ctx2:
        def __init__(self):  # noqa: D401
            self.obj = None  # malformed

    assert cli._extract_exp_config(Ctx2()) is None

    class Ctx3:
        pass  # no obj attribute

    assert cli._extract_exp_config(Ctx3()) is None
