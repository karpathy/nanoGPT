from __future__ import annotations

from pathlib import Path

import pytest
import typer

import ml_playground.cli as cli
from ml_playground.config import (
    TrainerConfig,
    ModelConfig,
    DataConfig,
    OptimConfig,
    LRSchedule,
    RuntimeConfig,
)


def test_run_or_exit_keyboard_interrupt(capsys):
    def fn():
        raise KeyboardInterrupt

    # Should not raise typer.Exit; should swallow or just return None
    out = cli.run_or_exit(fn, keyboard_interrupt_msg="bye")
    captured = capsys.readouterr()
    # Message printed
    assert "bye" in captured.out
    assert out is None


def test_run_or_exit_passthrough_exit():
    def fn():
        raise typer.Exit(code=0)

    with pytest.raises(typer.Exit) as ei:
        cli.run_or_exit(fn)
    assert ei.value.exit_code == 0


def test_run_or_exit_general_exception_to_exit(capsys):
    def fn():
        raise RuntimeError("boom")

    with pytest.raises(typer.Exit) as ei:
        cli.run_or_exit(fn)
    captured = capsys.readouterr()
    assert "boom" in captured.out
    assert ei.value.exit_code == 2


class _BadCtx:
    def __init__(self):
        class Obj:
            def get(self, *_args, **_kwargs):
                raise RuntimeError("bad obj")

        self.obj = Obj()


def test_extract_exp_config_ok_and_bad():
    class Ctx:
        def __init__(self):
            self.obj = {"exp_config": Path("/tmp/x.toml")}

    assert cli._extract_exp_config(Ctx()) == Path("/tmp/x.toml")
    # bad ctx should return None
    assert cli._extract_exp_config(_BadCtx()) is None


class _Ctx:
    def __init__(self):
        self.obj = None
        self._ensure_raises = False

    def ensure_object(self, typ):
        if self._ensure_raises:
            raise RuntimeError("boom")
        self.obj = {}
        return self.obj


def test_global_options_sets_ctx_and_handles_ensure_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    # Case 1: normal path sets ctx.obj and exp_config
    ctx = _Ctx()
    cli.global_options(ctx, None)
    assert isinstance(ctx.obj, dict)
    assert ctx.obj.get("exp_config") is None

    # Case 2: ensure_object failure path is caught and returns
    ctx2 = _Ctx()
    ctx2._ensure_raises = True
    # Ensure this doesn't raise
    cli.global_options(ctx2, None)


def test_read_toml_dict_success(tmp_path: Path):
    p = tmp_path / "ok.toml"
    p.write_text("[train]\n[train.runtime]\n out_dir='out'\n")
    out = cli._read_toml_dict(p)
    assert isinstance(out, dict) and "train" in out


def test_apply_overrides_validation_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    base = TrainerConfig(
        model=ModelConfig(),
        data=DataConfig(dataset_dir=tmp_path),
        optim=OptimConfig(),
        schedule=LRSchedule(),
        runtime=RuntimeConfig(out_dir=tmp_path),
    )

    # Force model_validate to raise to exercise fallback to original config
    orig_validate = TrainerConfig.model_validate

    def bad_validate(_):  # noqa: D401
        raise ValueError("nope")

    monkeypatch.setenv(
        "ML_PLAYGROUND_TRAIN_OVERRIDES", '{"runtime": {"log_interval": 1}}'
    )
    try:
        TrainerConfig.model_validate = staticmethod(bad_validate)  # type: ignore[assignment]
        out = cli._apply_train_overrides(base)
        assert out == base
    finally:
        # restore
        TrainerConfig.model_validate = orig_validate  # type: ignore[assignment]
