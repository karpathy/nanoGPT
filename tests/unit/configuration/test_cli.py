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
from ml_playground.configuration import cli as config_cli
from ml_playground.configuration import loading as config_loading
from ml_playground.configuration import (
    TrainerConfig,
    SamplerConfig,
    RuntimeConfig,
    SampleConfig,
    ExperimentConfig,
    SharedConfig,
    ModelConfig,
    DataConfig,
    OptimConfig,
    LRSchedule,
)
from ml_playground import config_loader
from ml_playground.configuration import PreparerConfig

runner = CliRunner()


def _make_shared(
    experiment: str,
    *,
    dataset_dir: Path | None = None,
    out_dir: Path | None = None,
) -> SharedConfig:
    return SharedConfig(
        experiment=experiment,
        config_path=Path("config.toml"),
        project_home=Path("."),
        dataset_dir=dataset_dir or Path("."),
        train_out_dir=out_dir or Path("."),
        sample_out_dir=out_dir or Path("."),
    )


def _make_full_experiment(shared: SharedConfig) -> ExperimentConfig:
    return ExperimentConfig(
        prepare=PreparerConfig(),
        train=TrainerConfig(
            model=ModelConfig(),
            data=DataConfig(),
            optim=OptimConfig(),
            schedule=LRSchedule(),
            runtime=RuntimeConfig(out_dir=shared.train_out_dir),
        ),
        sample=SamplerConfig(
            runtime=RuntimeConfig(out_dir=shared.sample_out_dir),
            sample=SampleConfig(),
        ),
        shared=shared,
    )


def test_main_prepare_shakespeare_success(mocker: MockerFixture) -> None:
    """Test prepare command with shakespeare dataset succeeds."""
    mock_run = mocker.patch("ml_playground.cli._run_prepare")
    shared = _make_shared("shakespeare")
    mocker.patch(
        "ml_playground.configuration.cli.load_experiment",
        return_value=_make_full_experiment(shared),
    )
    result = runner.invoke(app, ["prepare", "shakespeare"])
    assert result.exit_code == 0
    mock_run.assert_called_once()


def test_main_sample_missing_meta_fails(
    tmp_path: Path, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
) -> None:
    """Sample should fail fast when neither train meta nor runtime meta exist."""
    shared = _make_shared("shakespeare", out_dir=Path("out"))
    mocker.patch(
        "ml_playground.configuration.cli.load_experiment",
        return_value=_make_full_experiment(shared),
    )
    # Prepare command does not require additional prerequisite helpers
    mocker.patch(
        "ml_playground.cli.config_cli.ensure_sample_prerequisites",
        side_effect=ValueError("Missing required meta file for sampling"),
    )
    result = runner.invoke(app, ["sample", "shakespeare"])
    assert result.exit_code == 1
    assert "Missing required meta file for sampling" in caplog.messages[-1]


def test_main_prepare_bundestag_char_success(mocker: MockerFixture) -> None:
    """Test prepare command with bundestag_char dataset succeeds."""
    mock_run = mocker.patch("ml_playground.cli._run_prepare")
    shared = _make_shared("bundestag_char")
    mocker.patch(
        "ml_playground.configuration.cli.load_experiment",
        return_value=_make_full_experiment(shared),
    )
    result = runner.invoke(app, ["prepare", "bundestag_char"])
    assert result.exit_code == 0
    mock_run.assert_called_once()


def test_main_prepare_unknown_dataset_fails(
    mocker: MockerFixture, caplog: pytest.LogCaptureFixture
) -> None:
    """Unknown experiment should surface as a CLI error exit."""
    mocker.patch(
        "ml_playground.configuration.cli.load_experiment",
        side_effect=FileNotFoundError("Config not found"),
    )
    result = runner.invoke(app, ["prepare", "unknown"])
    assert result.exit_code == 1
    assert "Config not found" in caplog.messages[-1]


def test_main_train_success(tmp_path: Path, mocker: MockerFixture) -> None:
    """Train command loads config and invokes trainer once prerequisites pass."""
    mock_run = mocker.patch("ml_playground.cli._run_train")
    shared = _make_shared("shakespeare")
    mocker.patch(
        "ml_playground.configuration.cli.load_experiment",
        return_value=_make_full_experiment(shared),
    )
    mocker.patch(
        "ml_playground.cli.config_cli.ensure_train_prerequisites",
        return_value=Path("meta.pkl"),
    )
    result = runner.invoke(app, ["train", "shakespeare"])
    assert result.exit_code == 0
    mock_run.assert_called_once()


def test_main_train_no_train_block_fails(
    tmp_path: Path, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
) -> None:
    """Train command surfaces loader errors as CLI exits."""
    mocker.patch(
        "ml_playground.configuration.cli.load_experiment",
        side_effect=ValueError("Missing train config"),
    )
    result = runner.invoke(app, ["train", "shakespeare"])
    assert result.exit_code == 1
    assert "Missing train config" in caplog.messages[-1]


def test_main_sample_success(tmp_path: Path, mocker: MockerFixture) -> None:
    """Sample command loads config and invokes sampler once prerequisites pass."""
    mock_sample_cfg = SamplerConfig(
        runtime=RuntimeConfig(out_dir=Path("out")),
        sample=SampleConfig(start="x"),
    )
    mock_run = mocker.patch("ml_playground.cli._run_sample")
    shared = _make_shared("shakespeare")
    exp = _make_full_experiment(shared).model_copy(update={"sample": mock_sample_cfg})
    mocker.patch(
        "ml_playground.configuration.cli.load_experiment",
        return_value=exp,
    )
    mocker.patch(
        "ml_playground.cli.config_cli.ensure_sample_prerequisites",
        return_value=(Path("meta.pkl"), Path("meta.pkl")),
    )
    result = runner.invoke(app, ["sample", "shakespeare"])
    assert result.exit_code == 0
    mock_run.assert_called_once()


def test_main_sample_no_sample_block_fails(
    tmp_path: Path, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
) -> None:
    """Test sample command fails when strict loader raises."""
    mocker.patch(
        "ml_playground.configuration.cli.load_experiment",
        side_effect=ValueError("Missing sample config"),
    )
    result = runner.invoke(app, ["sample", "shakespeare"])
    assert result.exit_code == 1
    assert "Missing sample config" in caplog.messages[-1]


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

[train.runtime]
 out_dir = "out_rel"
"""
    config_path.write_text(toml_text)

    # Use strict partial loader with explicit path so we don't depend on package experiments root
    cfg = config_loader.load_train_config(config_path)
    # Manually attach a shared config for this partial-load test
    shared = SharedConfig(
        experiment="exp",
        config_path=config_path,
        project_home=tmp_path,
        dataset_dir=exp_dir / "data_rel",
        train_out_dir=exp_dir / "out_rel",
        sample_out_dir=exp_dir / "out_rel",
    )
    cfg = cfg.model_copy(update={"shared": shared})
    # With Shared-only paths, partial loader resolves runtime.out_dir; dataset_dir lives in Shared
    assert str(cfg.runtime.out_dir).startswith(str(exp_dir))


def test_load_sample_config_resolves_relative_out_dir_strict(tmp_path: Path):
    # Provide explicit [sample.runtime] with relative out_dir and expect absolute resolution
    p = tmp_path / "experiments" / "exp" / "config.toml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        """
[sample]
[sample.runtime]
out_dir = "out_rel"
device = "cpu"
dtype = "float32"

[sample.sample]
max_new_tokens = 1
"""
    )
    default_config_p = tmp_path / "default_config.toml"
    default_config_p.write_text("")

    # Use strict partial loader with explicit path; will resolve relative out_dir and make it absolute
    cfg = config_loader.load_sample_config(p)
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
    ds = tmp_path / "dataset"
    ds.mkdir(parents=True, exist_ok=True)
    (ds / "a.txt").write_text("x")
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)

    shared = SharedConfig(
        experiment="test",
        config_path=tmp_path / "config.toml",
        project_home=tmp_path,
        dataset_dir=ds,
        train_out_dir=out,
        sample_out_dir=out,
    )
    # Should not raise
    cli._log_command_status("train", shared, out, logging.getLogger(__name__))

    # Sampler with runtime
    shared = SharedConfig(
        experiment="test",
        config_path=tmp_path / "config.toml",
        project_home=tmp_path,
        dataset_dir=ds,
        train_out_dir=out,
        sample_out_dir=out,
    )
    cli._log_command_status("sample", shared, out, logging.getLogger(__name__))


def test_log_command_status_missing_paths(tmp_path: Path):
    # None of these paths exist
    out = tmp_path / "missing_out"
    shared = SharedConfig(
        experiment="test",
        config_path=tmp_path / "config.toml",
        project_home=tmp_path,
        dataset_dir=tmp_path / "missing_ds",
        train_out_dir=out,
        sample_out_dir=out,
    )
    # Should not raise
    cli._log_command_status("train", shared, out, logging.getLogger(__name__))


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


# Removed legacy cache/loader tests tied to ensure_loaded() (no longer present)
def test_run_or_exit_keyboard_interrupt_with_message(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO, logger="ml_playground.cli")

    def boom():  # noqa: D401
        raise KeyboardInterrupt()

    # Should log provided message and not re-raise as typer.Exit
    cli.run_or_exit(
        boom, keyboard_interrupt_msg="\nInterrupted!", exception_exit_code=9
    )
    assert "Interrupted!" in caplog.messages[-1]


def test_run_or_exit_keyboard_interrupt_no_message(capsys):
    def boom():  # noqa: D401
        raise KeyboardInterrupt()

    # No message provided: should not print anything and not raise
    cli.run_or_exit(boom, keyboard_interrupt_msg=None, exception_exit_code=9)
    out = capsys.readouterr().out
    assert out == ""


# Removed legacy cmd_train error propagation test (relied on ensure_loaded + cmd_train())


# Removed legacy cmd_sample error propagation test (relied on ensure_loaded + cmd_sample())


# Removed legacy direct cmd_sample runner test (cmd_* helpers removed)


def test_run_analyze_unsupported_experiment():
    with pytest.raises(RuntimeError) as ei:
        cli._run_analyze("other_exp", host="127.0.0.1", port=9999, open_browser=False)
    assert "only 'bundestag_char'" in str(ei.value)


# Removed legacy convert command direct-call test (cmd_convert removed)


# Removed legacy convert success/error path tests (cmd_convert removed)


# Removed legacy direct cmd_prepare runner test (cmd_* helpers removed)


# Removed legacy direct cmd_train runner test (cmd_* helpers removed)


# Removed legacy direct cmd_sample runner test (cmd_* helpers removed)


def test__load_sample_config_missing_sample_block(tmp_path: Path):
    # Config with no [sample] block at all
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("[train]\n")

    with pytest.raises(ValueError, match=r"must contain a \[sample\] section"):
        config_loader.load_sample_config(cfg_path)


# Removed legacy _load_sample_config top-level key validation test


# Removed legacy _load_sample_config nested [sample.sample] presence test


# Removed legacy _load_sample_config runtime/ref requirement test


# Removed legacy _load_sample_config runtime_ref unsupported value test


# Removed legacy _load_sample_config missing train.runtime test


# ---------------------------------------------------------------------------
# Consolidated from test_cli_overrides_cache_and_paths.py
# ---------------------------------------------------------------------------


def test_get_cfg_path_explicit_and_default(tmp_path: Path):
    explicit = tmp_path / "x.toml"
    # Explicit path returned as-is
    p = config_cli.cfg_path_for("some_exp", explicit)
    assert p == explicit
    # Default path under experiments root
    d = config_cli.cfg_path_for("bundestag_char", None)
    assert d.as_posix().endswith("ml_playground/experiments/bundestag_char/config.toml")


def test_load_config_error_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # 1) Missing config path -> exit
    with pytest.raises(FileNotFoundError):
        config_loader.load_train_config(Path("/__no_such_file__"))

    # Create experiments structure: tmp/experiments/exp/config.toml
    exp_dir = tmp_path / "experiments" / "exp"
    exp_dir.mkdir(parents=True)
    cfg = exp_dir / "config.toml"
    cfg.write_text(
        "[train]\n[train.runtime]\nout_dir='.'\n[train.model]\n[train.data]\n[train.optim]\n[train.schedule]"
    )

    # 2) Defaults invalid -> exit mentioning defaults path
    # Mock the defaults path to point to our test location
    test_defaults_path = tmp_path / "default_config.toml"
    test_defaults_path.write_text("this is not valid toml")

    def mock_default_config_path_from_root(project_root: Path) -> Path:
        return test_defaults_path

    monkeypatch.setattr(
        config_loading,
        "_default_config_path_from_root",
        mock_default_config_path_from_root,
    )

    with pytest.raises(Exception) as ei3:
        config_loader.load_train_config(cfg)
    assert "default_config.toml" in str(ei3.value).lower()

    # 3) Experiment config invalid -> exit mentioning cfg path
    bad_cfg = exp_dir / "bad.toml"
    bad_cfg.write_text("this is not valid toml")
    with pytest.raises(Exception) as ei4:
        config_loader.load_train_config(bad_cfg)
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
        (["analyze", "--help"], "analyze"),
    ],
)
def test_cli_help_and_version(args, expect):
    # Run via python -m to exercise entry-point wiring without performing heavy work
    cmd = [sys.executable, "-m", "ml_playground.cli"] + args
    cp = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
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
# Consolidated from test_cli_speakger.py
# ---------------------------------------------------------------------------


def test_sample_routes_to_injected_sampler(
    monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    """CLI should dispatch 'sample speakger' to the unified injected-config sampler path (_run_sample)."""
    # Patch the unified run path; we don't want to import heavy experiment deps
    run_sample = mocker.patch("ml_playground.cli._run_sample")
    # Ensure legacy sampler entry point is not consulted anymore
    called = {"count": 0}

    def _legacy_sample_from_toml(path):  # type: ignore[no-untyped-def]
        called["count"] += 1

    monkeypatch.setattr(
        "ml_playground.experiments.speakger.sampler.sample_from_toml",
        _legacy_sample_from_toml,
        raising=False,
    )

    shared = _make_shared("speakger")
    mocker.patch(
        "ml_playground.configuration.cli.load_experiment",
        return_value=_make_full_experiment(shared),
    )
    mocker.patch(
        "ml_playground.cli.config_cli.ensure_sample_prerequisites",
        return_value=(Path("meta.pkl"), Path("meta.pkl")),
    )

    cli.main(["sample", "speakger"])

    run_sample.assert_called_once()
    assert called["count"] == 0
