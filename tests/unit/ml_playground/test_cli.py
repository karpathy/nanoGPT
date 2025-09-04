from __future__ import annotations
import pytest
from pathlib import Path
from pytest_mock import MockerFixture
from typer.testing import CliRunner

# Additional imports for consolidated CLI tests
import sys
import subprocess
import logging
import typer
from typing import cast

import ml_playground.cli as cli
from ml_playground.cli import app
from ml_playground import config_loader
from ml_playground.config import (
    TrainerConfig,
    SamplerConfig,
    RuntimeConfig,
    SampleConfig,
)
from ml_playground.prepare import PreparerConfig

runner = CliRunner()


def test_main_prepare_shakespeare_success(mocker: MockerFixture) -> None:
    """Test prepare command with shakespeare dataset succeeds."""
    mock_run = mocker.patch("ml_playground.cli._run_prepare")
    mocker.patch(
        "ml_playground.cli.def_load_effective_prepare",
        return_value=(Path("cfg.toml"), PreparerConfig()),
    )
    result = runner.invoke(app, ["prepare", "shakespeare"])
    assert result.exit_code == 0
    mock_run.assert_called_once()


def test_main_prepare_bundestag_char_success(mocker: MockerFixture) -> None:
    """Test prepare command with bundestag_char dataset succeeds."""
    mock_run = mocker.patch("ml_playground.cli._run_prepare")
    mocker.patch(
        "ml_playground.cli.def_load_effective_prepare",
        return_value=(Path("cfg.toml"), PreparerConfig()),
    )
    result = runner.invoke(app, ["prepare", "bundestag_char"])
    assert result.exit_code == 0
    mock_run.assert_called_once()


def test_main_prepare_unknown_dataset_fails(
    mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """Unknown experiment should surface as a CLI error exit."""
    mocker.patch(
        "ml_playground.cli.def_load_effective_prepare",
        side_effect=FileNotFoundError("Config not found"),
    )
    result = runner.invoke(app, ["prepare", "unknown"])
    assert result.exit_code == 1
    assert "Config not found" in result.stdout


def test_main_train_success(tmp_path: Path, mocker: MockerFixture) -> None:
    """Test train command auto-resolves config for experiment and calls train (strict loader)."""
    mock_train_cfg = mocker.Mock(spec=TrainerConfig)
    mock_run = mocker.patch("ml_playground.cli._run_train")
    mocker.patch(
        "ml_playground.cli.def_load_effective_train",
        return_value=(Path("cfg.toml"), mock_train_cfg),
    )
    result = runner.invoke(app, ["train", "shakespeare"])
    assert result.exit_code == 0
    mock_run.assert_called_once()


def test_main_train_no_train_block_fails(
    tmp_path: Path, mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test train command fails when strict loader raises."""
    mocker.patch(
        "ml_playground.cli.def_load_effective_train",
        side_effect=ValueError("Missing train config"),
    )
    result = runner.invoke(app, ["train", "shakespeare"])
    assert result.exit_code == 1
    assert "Missing train config" in result.stdout


def test_main_sample_success(tmp_path: Path, mocker: MockerFixture) -> None:
    """Test sample command auto-resolves config and calls sample function (strict loader)."""
    mock_sample_cfg = SamplerConfig(
        runtime=RuntimeConfig(out_dir=Path("out")),
        sample=SampleConfig(start="x"),
    )
    mock_run = mocker.patch("ml_playground.cli._run_sample")
    mocker.patch(
        "ml_playground.cli.def_load_effective_sample",
        return_value=(Path("cfg.toml"), mock_sample_cfg),
    )
    result = runner.invoke(app, ["sample", "shakespeare"])
    assert result.exit_code == 0
    mock_run.assert_called_once()


def test_main_sample_no_sample_block_fails(
    tmp_path: Path, mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test sample command fails when strict loader raises."""
    mocker.patch(
        "ml_playground.cli.def_load_effective_sample",
        side_effect=ValueError("Missing sample config"),
    )
    result = runner.invoke(app, ["sample", "shakespeare"])
    assert result.exit_code == 1
    assert "Missing sample config" in result.stdout


def test_main_loop_success(tmp_path: Path, mocker: MockerFixture) -> None:
    """Test loop command executes via _run_loop with loaded configs."""
    mock_train_config = mocker.Mock(spec=TrainerConfig)
    mock_sample_config = SamplerConfig(
        runtime=RuntimeConfig(out_dir=Path("out")),
        sample=SampleConfig(start="x"),
    )
    mock_run = mocker.patch("ml_playground.cli._run_loop")
    mocker.patch(
        "ml_playground.cli.def_load_effective_prepare",
        return_value=(Path("cfg.toml"), PreparerConfig()),
    )
    mocker.patch(
        "ml_playground.cli.def_load_effective_train",
        return_value=(Path("cfg.toml"), mock_train_config),
    )
    mocker.patch(
        "ml_playground.cli.def_load_effective_sample",
        return_value=(Path("cfg.toml"), mock_sample_config),
    )

    result = runner.invoke(app, ["loop", "shakespeare"])
    assert result.exit_code == 0
    mock_run.assert_called_once()


def test_main_loop_unknown_dataset_fails(tmp_path: Path, mocker: MockerFixture) -> None:
    """Unknown experiment should bubble up as CLI error exit."""
    mocker.patch(
        "ml_playground.cli.def_load_effective_prepare",
        side_effect=FileNotFoundError("Config not found"),
    )
    result = runner.invoke(app, ["loop", "shakespeare"])
    assert result.exit_code == 1
    assert "Config not found" in result.stdout


def test_main_loop_missing_train_block_fails(
    tmp_path: Path, mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test loop command fails when strict train loader raises."""
    mocker.patch(
        "ml_playground.cli.def_load_effective_prepare",
        return_value=(Path("cfg.toml"), PreparerConfig()),
    )
    mocker.patch(
        "ml_playground.cli.def_load_effective_train",
        side_effect=ValueError("Missing train config"),
    )
    result = runner.invoke(app, ["loop", "shakespeare"])
    assert result.exit_code == 1
    assert "Missing train config" in result.stdout


def test_main_loop_missing_sample_block_fails(
    tmp_path: Path, mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test loop command fails when strict sample loader raises."""
    mocker.patch(
        "ml_playground.cli.def_load_effective_prepare",
        return_value=(Path("cfg.toml"), PreparerConfig()),
    )
    mocker.patch(
        "ml_playground.cli.def_load_effective_train",
        return_value=(Path("cfg.toml"), mocker.Mock(spec=TrainerConfig)),
    )
    mocker.patch(
        "ml_playground.cli.def_load_effective_sample",
        side_effect=ValueError("Missing sample config"),
    )
    result = runner.invoke(app, ["loop", "shakespeare"])
    assert result.exit_code == 1
    assert "Missing sample config" in result.stdout


# Tests for removed functionality (legacy registry usage, direct train/sample calls)
# have been updated to reflect the new CLI architecture.


# ---------------------------------------------------------------------------
# Consolidated from test_cli_cache.py
# ---------------------------------------------------------------------------


def test_strict_mode_has_no_override_functions() -> None:
    # Strict mode: configuration is TOML-only, no override helpers are exposed
    assert not hasattr(config_loader, "apply_train_overrides")
    assert not hasattr(config_loader, "apply_sample_overrides")


# ---------------------------------------------------------------------------
# Consolidated from test_cli_device_setup.py
# ---------------------------------------------------------------------------


def test_global_setup_sets_seed_and_is_deterministic() -> None:
    import torch

    # First call with seed=123
    cli._global_device_setup("cpu", "float32", seed=123)
    a = torch.rand(1).item()
    # Different seed -> different number likely
    cli._global_device_setup("cpu", "float32", seed=124)
    b = torch.rand(1).item()
    # Same seed again -> reproducible
    cli._global_device_setup("cpu", "float32", seed=123)
    c = torch.rand(1).item()
    assert a != b
    assert abs(a - c) < 1e-8


def test_global_setup_enables_tf32_when_cuda_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    # Simulate CUDA availability; ensure flags become True after setup
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True, raising=True)
    # Reset flags before
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass
    cli._global_device_setup("cuda", "bfloat16", seed=1)
    # Flags should be enabled
    assert getattr(torch.backends.cuda.matmul, "allow_tf32", True) is True
    assert getattr(torch.backends.cudnn, "allow_tf32", True) is True


def test_global_setup_no_crash_without_cuda() -> None:
    # Should not raise even if CUDA unavailable
    cli._global_device_setup("cuda", "float16", seed=42)


# ---------------------------------------------------------------------------
# Consolidated from test_cli_effective_wrappers.py
# ---------------------------------------------------------------------------


def test_load_train_config_resolves_relative_paths(tmp_path: Path):
    # Create a dummy default config in the parent of the experiments dir
    default_config_path = tmp_path / "default_config.toml"
    default_config_path.write_text(
        """
[train.model]
 n_layer = 1
 n_head = 1
 n_embd = 32
 block_size = 16
[train.optim]
 learning_rate = 0.001
[train.schedule]
"""
    )

    exp_dir = tmp_path / "experiments" / "exp"
    exp_dir.mkdir(parents=True)
    config_path = exp_dir / "config.toml"
    toml_text = """
[train.data]
 dataset_dir = "data_rel"

[train.runtime]
 out_dir = "out_rel"
"""
    config_path.write_text(toml_text)

    # Use strict loader with explicit path so we don't depend on package experiments root
    _, cfg = cli.def_load_effective_train("exp", config_path)

    assert str(cfg.data.dataset_dir).startswith(str(exp_dir))
    assert str(cfg.runtime.out_dir).startswith(str(exp_dir))


def test_load_sample_config_resolves_relative_out_dir_with_runtime_ref(tmp_path: Path):
    # Use runtime_ref to defaults' train.runtime and override out_dir relatively
    p = tmp_path / "experiments" / "exp" / "config.toml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        """[sample]
runtime_ref = "train.runtime"

[train]
[train.runtime]
device = "cpu"
"""
    )
    default_config_p = tmp_path / "default_config.toml"
    default_config_p.write_text(
        """[train]
[train.runtime]
out_dir = "out_rel"
"""
    )

    # Use strict loader with explicit path; will resolve runtime_ref and make out_dir absolute
    _, cfg = cli.def_load_effective_sample("exp", p)
    # Help static type checkers: runtime is required by runtime_ref resolution
    r = cast(RuntimeConfig, cfg.runtime)
    assert r is not None
    assert r.out_dir.is_absolute()
    assert str(r.out_dir).startswith(str(p.parent))


# ---------------------------------------------------------------------------
# Consolidated from test_cli_logger_and_ensure.py
# ---------------------------------------------------------------------------


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
        cli.global_options(cast(typer.Context, ctx), missing)
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
        cli.ensure_loaded(cast(typer.Context, ctx), "some_exp")
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
        cli.global_options(cast(typer.Context, ctx), None)
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
        cli.global_options(cast(typer.Context, ctx), None)
        # Should have at least one handler configured now
        assert logging.getLogger().handlers, "Expected basicConfig to add a handler"
    finally:
        root.handlers = old_handlers


def test_log_command_status_covers_paths(tmp_path: Path):
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

    assert cli._extract_exp_config(cast(typer.Context, Ctx1())) == Path("/x/y.toml")

    class Ctx2:
        def __init__(self):  # noqa: D401
            self.obj = None  # malformed

    assert cli._extract_exp_config(cast(typer.Context, Ctx2())) is None

    class Ctx3:
        pass  # no obj attribute

    assert cli._extract_exp_config(cast(typer.Context, Ctx3())) is None


# ---------------------------------------------------------------------------
# Consolidated from test_cli_more_helpers_and_loader.py
# (selected tests not already covered above)
# ---------------------------------------------------------------------------


def test_ensure_loaded_cache_key_mismatch_calls_loader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    class _Ctx2:
        def __init__(self):
            self.obj = {}

        def ensure_object(self, typ):
            if not isinstance(self.obj, dict):
                self.obj = {}
            return self.obj

    ctx = _Ctx2()
    ctx.ensure_object(dict)
    # Seed cache with different experiment
    ctx.obj["loaded_cache"] = {
        "key": ("expX", None),
        "cfg_path": tmp_path,
        "app": cli.AppConfig(train=None, sample=None),
        "prep": PreparerConfig(),
    }

    called = {"n": 0}

    def fake_loader(exp: str, exp_config: Path | None):  # noqa: D401
        called["n"] += 1
        return (
            tmp_path / "cfg.toml",
            cli.AppConfig(train=None, sample=None),
            PreparerConfig(),
        )

    monkeypatch.setattr(cli, "load_app_config", fake_loader)

    _ = cli.ensure_loaded(cast(typer.Context, ctx), "expY")
    assert called["n"] == 1


def test_run_or_exit_custom_exit_code(capsys):
    def boom():  # noqa: D401
        raise RuntimeError("bad")

    with pytest.raises(typer.Exit) as ei:  # type: ignore[name-defined]
        cli.run_or_exit(boom, exception_exit_code=5)
    assert ei.value.exit_code == 5
    assert "bad" in capsys.readouterr().out


def test_run_or_exit_keyboard_interrupt_with_message(capsys):
    def boom():  # noqa: D401
        raise KeyboardInterrupt()

    # Should print provided message and not re-raise as typer.Exit
    cli.run_or_exit(
        boom, keyboard_interrupt_msg="\nInterrupted!", exception_exit_code=9
    )
    out = capsys.readouterr().out
    assert "Interrupted!" in out


def test_run_or_exit_keyboard_interrupt_no_message(capsys):
    def boom():  # noqa: D401
        raise KeyboardInterrupt()

    # No message provided: should not print anything and not raise
    cli.run_or_exit(boom, keyboard_interrupt_msg=None, exception_exit_code=9)
    out = capsys.readouterr().out
    assert out == ""


def test_cmd_train_uses_loaded_error_and_exits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
):
    # Ensure ensure_loaded returns app with train=None
    def fake_ensure_loaded(ctx, experiment):  # noqa: D401
        return (
            tmp_path / "cfg.toml",
            cli.AppConfig(train=None, sample=None),
            PreparerConfig(),
        )

    monkeypatch.setattr(cli, "ensure_loaded", fake_ensure_loaded)

    # Build ctx with loaded_errors matching the key and custom train error
    class Ctx:
        def __init__(self):  # noqa: D401
            self.obj = {
                "exp_config": None,
                "loaded_errors": {
                    "key": ("expZ", None),
                    "train": "custom-train-error",
                    "sample": None,
                },
            }

        def ensure_object(self, t):  # noqa: D401
            return self.obj

    ctx = Ctx()
    with pytest.raises(typer.Exit) as ei:
        cli.cmd_train(cast(typer.Context, ctx), "expZ")
    assert ei.value.exit_code == 2
    out = capsys.readouterr().out
    assert "custom-train-error" in out


def test_cmd_sample_uses_loaded_error_and_exits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
):
    # ensure_loaded returns app with sample=None
    def fake_ensure_loaded(ctx, experiment):  # noqa: D401
        return (
            tmp_path / "cfg.toml",
            cli.AppConfig(train=None, sample=None),
            PreparerConfig(),
        )

    monkeypatch.setattr(cli, "ensure_loaded", fake_ensure_loaded)

    class Ctx:
        def __init__(self):  # noqa: D401
            self.obj = {
                "exp_config": None,
                "loaded_errors": {
                    "key": ("expY", None),
                    "train": None,
                    "sample": "custom-sample-error",
                },
            }

        def ensure_object(self, t):  # noqa: D401
            return self.obj

    ctx = Ctx()
    with pytest.raises(typer.Exit) as ei:
        cli.cmd_sample(cast(typer.Context, ctx), "expY")
    assert ei.value.exit_code == 2
    out = capsys.readouterr().out
    assert "custom-sample-error" in out


def test_cmd_sample_speakger_uses_unified_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # Ensure ensure_loaded returns a non-None sample config so cmd_sample proceeds
    expected_cfg = tmp_path / "cfg.toml"

    from types import SimpleNamespace

    def fake_ensure_loaded(ctx, experiment):  # noqa: D401
        # Return a lightweight object with the required attributes to avoid Pydantic validation
        return (
            expected_cfg,
            SimpleNamespace(train=None, sample=SimpleNamespace()),
            PreparerConfig(),
        )

    called = {"n": 0, "args": None}

    def fake_run_sample(experiment, sample_cfg, cfg_path):  # noqa: D401
        called["n"] += 1
        called["args"] = (experiment, sample_cfg, cfg_path)

    class Ctx:
        def __init__(self):  # noqa: D401
            self.obj = {"exp_config": None}

        def ensure_object(self, t):  # noqa: D401
            return self.obj

    monkeypatch.setattr(cli, "ensure_loaded", fake_ensure_loaded)
    monkeypatch.setattr(cli, "_run_sample", fake_run_sample)

    ctx = Ctx()
    cli.cmd_sample(cast(typer.Context, ctx), "speakger")
    assert (
        called["n"] == 1
        and called["args"][0] == "speakger"
        and called["args"][2] == expected_cfg
    )


def test_run_analyze_unsupported_experiment():
    with pytest.raises(RuntimeError) as ei:
        cli._run_analyze("other_exp", host="127.0.0.1", port=9999, open_browser=False)
    assert "only 'bundestag_char'" in str(ei.value)


def test_cmd_convert_only_bundestag_char(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
):
    # Non-supported experiment prints and exits code 2
    class Ctx:
        def __init__(self):  # noqa: D401
            self.obj = {"exp_config": None}

        def ensure_object(self, t):  # noqa: D401
            return self.obj

    ctx = Ctx()
    with pytest.raises(typer.Exit) as ei:
        cli.cmd_convert(cast(typer.Context, ctx), "not_supported")
    assert ei.value.exit_code == 2
    assert "supports only 'bundestag_char'" in capsys.readouterr().out


def test_cmd_convert_success_and_error_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
):
    # ensure_loaded should be called but we only care it returns (cfg_path, _, _)
    def fake_ensure_loaded(ctx, experiment):  # noqa: D401
        return (
            tmp_path / "cfg.toml",
            cli.AppConfig(train=None, sample=None),
            PreparerConfig(),
        )

    monkeypatch.setattr(cli, "ensure_loaded", fake_ensure_loaded)

    called = {"ok": 0, "args": None}

    class _ExportCfg:
        def __init__(
            self,
            enabled,
            export_dir,
            model_name,
            quant,
            template=None,
            convert_bin=None,
            quant_bin=None,
        ):  # noqa: D401
            self.enabled = enabled
            self.export_dir = export_dir
            self.model_name = model_name
            self.quant = quant
            self.template = template
            self.convert_bin = convert_bin
            self.quant_bin = quant_bin

    class OkMod:
        OllamaExportConfig = _ExportCfg

        def convert(self, export_cfg, out_dir, read_policy):  # noqa: D401
            called["ok"] += 1
            called["args"] = (export_cfg, out_dir, read_policy)

    # Success path
    from types import SimpleNamespace

    monkeypatch.setattr(
        cli, "importlib", SimpleNamespace(import_module=lambda _n: OkMod())
    )

    class Ctx:
        def __init__(self):  # noqa: D401
            self.obj = {"exp_config": None}

        def ensure_object(self, t):  # noqa: D401
            return self.obj

    ctx = Ctx()
    cli.cmd_convert(cast(typer.Context, ctx), "bundestag_char")
    assert called["ok"] == 1
    args_obj = called["args"]
    assert isinstance(args_obj, tuple) and len(args_obj) >= 2
    assert isinstance(args_obj[1], Path)

    # SystemExit inside converter is mapped to typer.Exit with same code and message echoed
    class ExitMod:
        OllamaExportConfig = _ExportCfg

        def convert(self, export_cfg, out_dir, read_policy):  # noqa: D401
            raise SystemExit("inner-msg")

    monkeypatch.setattr(
        cli, "importlib", SimpleNamespace(import_module=lambda _n: ExitMod())
    )
    with pytest.raises(typer.Exit) as ei:
        cli.cmd_convert(cast(typer.Context, ctx), "bundestag_char")
    # Current implementation forwards SystemExit.code even if it is a string
    assert ei.value.exit_code == "inner-msg"
    assert "inner-msg" in capsys.readouterr().out

    # Generic exception maps to exit code 1 and echoes message
    class ErrMod:
        OllamaExportConfig = _ExportCfg

        def convert(self, export_cfg, out_dir, read_policy):  # noqa: D401
            raise RuntimeError("boom")

    monkeypatch.setattr(
        cli, "importlib", SimpleNamespace(import_module=lambda _n: ErrMod())
    )
    with pytest.raises(typer.Exit) as ei2:
        cli.cmd_convert(cast(typer.Context, ctx), "bundestag_char")
    assert ei2.value.exit_code == 1
    assert "boom" in capsys.readouterr().out


def test_cmd_prepare_happy_calls_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # ensure_loaded returns preparer config in position 3
    pcfg = PreparerConfig()

    def fake_ensure_loaded(ctx, experiment):  # noqa: D401
        return tmp_path / "cfg.toml", cli.AppConfig(train=None, sample=None), pcfg

    called = {"n": 0, "arg": None}

    def fake_run_prepare(experiment, prep_cfg, cfg_path):  # noqa: D401
        called["n"] += 1
        called["arg"] = prep_cfg

    monkeypatch.setattr(cli, "ensure_loaded", fake_ensure_loaded)
    monkeypatch.setattr(cli, "_run_prepare", fake_run_prepare)

    class Ctx:
        def __init__(self):  # noqa: D401
            self.obj = {"exp_config": None}

        def ensure_object(self, t):  # noqa: D401
            return self.obj

    ctx = Ctx()
    cli.cmd_prepare(cast(typer.Context, ctx), "expX")
    assert called["n"] == 1 and called["arg"] is pcfg


def test_cmd_train_happy_calls_runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from ml_playground.config import (
        ModelConfig,
        DataConfig,
        OptimConfig,
        LRSchedule,
    )

    tcfg = TrainerConfig(
        model=ModelConfig(),
        data=DataConfig(dataset_dir=tmp_path),
        optim=OptimConfig(),
        schedule=LRSchedule(),
        runtime=RuntimeConfig(out_dir=tmp_path),
    )

    def fake_ensure_loaded(ctx, experiment):  # noqa: D401
        return tmp_path / "cfg.toml", cli.AppConfig(train=tcfg, sample=None), None

    called = {"n": 0, "arg": None}

    def fake_run_train(experiment, train_cfg, cfg_path):  # noqa: D401
        called["n"] += 1
        called["arg"] = train_cfg

    monkeypatch.setattr(cli, "ensure_loaded", fake_ensure_loaded)
    monkeypatch.setattr(cli, "_run_train", fake_run_train)

    class Ctx:
        def __init__(self):  # noqa: D401
            self.obj = {
                "exp_config": None,
                "loaded_errors": {"key": ("expA", None), "train": None, "sample": None},
            }

        def ensure_object(self, t):  # noqa: D401
            return self.obj

    ctx = Ctx()
    cli.cmd_train(cast(typer.Context, ctx), "expA")
    assert called["n"] == 1 and called["arg"] is tcfg


def test_cmd_sample_happy_calls_runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from ml_playground.config import SampleConfig as _SampleCfg

    scfg = SamplerConfig(runtime=RuntimeConfig(out_dir=tmp_path), sample=_SampleCfg())

    def fake_ensure_loaded(ctx, experiment):  # noqa: D401
        return tmp_path / "cfg.toml", cli.AppConfig(train=None, sample=scfg), None

    called = {"n": 0, "arg": None}

    def fake_run_sample(experiment, sample_cfg, cfg_path):  # noqa: D401
        called["n"] += 1
        called["arg"] = sample_cfg

    monkeypatch.setattr(cli, "ensure_loaded", fake_ensure_loaded)
    monkeypatch.setattr(cli, "_run_sample", fake_run_sample)

    class Ctx:
        def __init__(self):  # noqa: D401
            self.obj = {
                "exp_config": None,
                "loaded_errors": {"key": ("expB", None), "train": None, "sample": None},
            }

        def ensure_object(self, t):  # noqa: D401
            return self.obj

    ctx = Ctx()
    cli.cmd_sample(cast(typer.Context, ctx), "expB")
    assert called["n"] == 1 and called["arg"] is scfg


def test__load_sample_config_missing_sample_block(tmp_path: Path):
    # Config with no [sample] block at all
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("[train]\n")

    with pytest.raises(ValueError, match=r"must contain a \[sample\] section"):
        config_loader.load_sample_config(cfg_path)


def test__load_sample_config_unknown_top_key(tmp_path: Path):
    p = tmp_path / "exp" / "config.toml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        """
    [sample]
    bad = 1
    [sample.sample]
    """
    )
    with pytest.raises(ValueError) as ei:
        cli._load_sample_config(p)
    assert "Unknown key(s) in [sample]" in str(ei.value)


def test__load_sample_config_missing_sample_sample(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # Disable defaults so missing [sample.sample] is not filled from defaults
    orig_load = cli.tomllib.load

    def _no_defaults_load(f):  # noqa: D401
        name = getattr(f, "name", "")
        if isinstance(name, str) and name.endswith("default_config.toml"):
            return {}
        return orig_load(f)

    monkeypatch.setattr(cli.tomllib, "load", _no_defaults_load)
    p = tmp_path / "exp" / "config.toml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        """
    [sample]
    runtime_ref = "train.runtime"
    """
    )
    with pytest.raises(ValueError) as ei:
        cli._load_sample_config(p)
    assert "Missing required section [sample]" in str(ei.value)


def test__load_sample_config_requires_runtime_or_ref(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # Disable defaults so runtime/runtime_ref isn't provided by defaults
    orig_load = cli.tomllib.load

    def _no_defaults_load(f):  # noqa: D401
        name = getattr(f, "name", "")
        if isinstance(name, str) and name.endswith("default_config.toml"):
            return {}
        return orig_load(f)

    monkeypatch.setattr(cli.tomllib, "load", _no_defaults_load)
    p = tmp_path / "exp" / "config.toml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        """
    [sample]
    [sample.sample]
    """
    )
    with pytest.raises(ValueError) as ei:
        cli._load_sample_config(p)
    assert "requires either [sample.runtime] or sample.runtime_ref" in str(ei.value)


def test__load_sample_config_runtime_ref_unsupported_value(tmp_path: Path):
    p = tmp_path / "exp" / "config.toml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        """
    [sample]
    runtime_ref = "other.ref"
    [sample.sample]
    """
    )
    with pytest.raises(ValueError) as ei:
        cli._load_sample_config(p)
    assert "Unsupported sample.runtime_ref" in str(ei.value)


def test__load_sample_config_runtime_ref_missing_train_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # runtime_ref points to train.runtime, but no train.runtime provided in exp nor defaults
    orig_load = cli.tomllib.load

    def _no_defaults_load(f):  # noqa: D401
        name = getattr(f, "name", "")
        if isinstance(name, str) and name.endswith("default_config.toml"):
            return {}
        return orig_load(f)

    monkeypatch.setattr(cli.tomllib, "load", _no_defaults_load)
    p = tmp_path / "exp" / "config.toml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        """
    [sample]
    runtime_ref = "train.runtime"
    [sample.sample]
    """
    )
    with pytest.raises(ValueError) as ei:
        cli._load_sample_config(p)
    assert "points to 'train.runtime'" in str(ei.value)


# ---------------------------------------------------------------------------
# Consolidated from test_cli_overrides_cache_and_paths.py
# ---------------------------------------------------------------------------


def test_get_cfg_path_explicit_and_default(tmp_path: Path):
    explicit = tmp_path / "x.toml"
    # Explicit path returned as-is
    p = cli._cfg_path_for("some_exp", explicit)
    assert p == explicit
    # Default path under experiments root
    d = cli._cfg_path_for("bundestag_char", None)
    assert d.as_posix().endswith("ml_playground/experiments/bundestag_char/config.toml")


def test_load_config_error_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # 1) Missing config path -> exit
    with pytest.raises(FileNotFoundError):
        cli.def_load_effective_train("exp", Path("/__no_such_file__"))

    # Create experiments structure: tmp/experiments/exp/config.toml
    exp_dir = tmp_path / "experiments" / "exp"
    exp_dir.mkdir(parents=True)
    cfg = exp_dir / "config.toml"
    cfg.write_text(
        "[train]\n[train.runtime]\nout_dir='.'\n[train.model]\n[train.data]\ndataset_dir='.'\n[train.optim]\n[train.schedule]"
    )

    # 2) Defaults invalid -> exit mentioning defaults path sibling to experiments/
    defaults_path = tmp_path / "default_config.toml"
    defaults_path.write_text("this is not valid toml")

    with pytest.raises(Exception) as ei3:
        cli.def_load_effective_train("exp", cfg)
    assert "default_config.toml" in str(ei3.value).lower()

    # 3) Experiment config invalid -> exit mentioning cfg path
    bad_cfg = exp_dir / "bad.toml"
    bad_cfg.write_text("this is not valid toml")
    with pytest.raises(Exception) as ei4:
        cli.def_load_effective_train("exp", bad_cfg)
    assert "bad.toml" in str(ei4.value).lower()


# ---------------------------------------------------------------------------
# Consolidated from test_cli_parse.py
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args, expect",
    [
        (["--help"], "usage:"),
        (["prepare", "--help"], "prepare"),
        (["train", "--help"], "train"),
        (["sample", "--help"], "sample"),
        (["loop", "--help"], "loop"),
        (["analyze", "--help"], "analyze"),
    ],
)
def test_cli_help_and_version(args, expect):
    # Run via python -m to exercise entry-point wiring without performing heavy work
    cmd = [sys.executable, "-m", "ml_playground.cli"] + args
    cp = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    out = cp.stdout
    if args == ["--help"]:
        assert "usage" in out.lower()
    else:
        assert expect in out.lower()
    assert cp.returncode == 0


def test_cli_global_exp_config_missing_exits(tmp_path):
    # Point to a definitely missing path
    missing = tmp_path / "nope.toml"
    # Use a real subcommand to ensure the callback runs (help short-circuits)
    cmd = [
        sys.executable,
        "-m",
        "ml_playground.cli",
        "--exp-config",
        str(missing),
        "prepare",
        "shakespeare",
    ]
    cp = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    # Typer exits with code 2 for callback-triggered validation failures
    assert cp.returncode == 2
    assert "config file not found" in cp.stdout.lower()


# ---------------------------------------------------------------------------
# Consolidated from test_cli_run_train_and_loop.py
# ---------------------------------------------------------------------------


def test_run_loop_calls_in_order_and_handles_print_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    calls: list[str] = []

    # Monkeypatch public module entrypoints called by CLI runners
    import ml_playground.prepare as prepare_mod
    import ml_playground.trainer as trainer_mod
    import ml_playground.sampler as sampler_mod

    def fake_preparer():
        calls.append("prepare")

    monkeypatch.setattr(prepare_mod, "make_preparer", lambda cfg: fake_preparer)
    monkeypatch.setattr(trainer_mod, "train", lambda cfg: calls.append("train"))
    monkeypatch.setattr(sampler_mod, "sample", lambda cfg: calls.append("sample"))

    # Configs
    from ml_playground.config import (
        ModelConfig,
        DataConfig,
        OptimConfig,
        LRSchedule,
    )

    prep_cfg = PreparerConfig()
    tcfg = TrainerConfig(
        model=ModelConfig(),
        data=DataConfig(dataset_dir=tmp_path),
        optim=OptimConfig(),
        schedule=LRSchedule(),
        runtime=RuntimeConfig(out_dir=tmp_path),
    )
    scfg = SamplerConfig(runtime=RuntimeConfig(out_dir=tmp_path), sample=SampleConfig())

    # Should not raise; ensure order prepare -> train -> sample
    cli._run_loop("exp", tmp_path / "cfg.toml", prep_cfg, tcfg, scfg)

    assert calls == ["prepare", "train", "sample"]


# ---------------------------------------------------------------------------
# Consolidated from test_cli_speakger.py
# ---------------------------------------------------------------------------


def test_sample_routes_to_injected_sampler(
    monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    """CLI should dispatch 'sample speakger' to the unified injected-config sampler path (_run_sample)."""
    # Patch the unified run path; we don't want to import heavy experiment deps
    run_sample = mocker.patch("ml_playground.cli._run_sample")

    # Also ensure legacy entrypoint is NOT consulted anymore
    called = {"count": 0}

    def _fake_sample_from_toml(path):  # type: ignore[no-untyped-def]
        called["count"] += 1

    monkeypatch.setattr(
        "ml_playground.experiments.speakger.sampler.sample_from_toml",
        _fake_sample_from_toml,
        raising=False,
    )

    # Act: run CLI with experiment auto-resolved config (no SystemExit on success)
    cli.main(["sample", "speakger"])

    # Assert: unified path called once; legacy path not invoked
    run_sample.assert_called_once()
    assert called["count"] == 0
