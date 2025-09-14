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
    ExperimentConfig,
    SharedConfig,
    ModelConfig,
    DataConfig,
    OptimConfig,
    LRSchedule,
)
from ml_playground.prepare import PreparerConfig

runner = CliRunner()


def test_main_prepare_shakespeare_success(mocker: MockerFixture) -> None:
    """Test prepare command with shakespeare dataset succeeds."""
    mock_run = mocker.patch("ml_playground.cli._run_prepare")
    # Mock canonical loader to return minimal full experiment config
    shared = SharedConfig(
        experiment="shakespeare",
        config_path=Path("config.toml"),
        project_home=Path("."),
        dataset_dir=Path("."),
        train_out_dir=Path("."),
        sample_out_dir=Path("."),
    )
    mocker.patch(
        "ml_playground.config_loader.load_full_experiment_config",
        return_value=ExperimentConfig(
            prepare=PreparerConfig(),
            train=TrainerConfig(
                model=ModelConfig(),
                data=DataConfig(),
                optim=OptimConfig(),
                schedule=LRSchedule(),
                runtime=RuntimeConfig(out_dir=Path(".")),
            ),
            sample=SamplerConfig(
                runtime=RuntimeConfig(out_dir=Path(".")), sample=SampleConfig()
            ),
            shared=shared,
        ),
    )
    result = runner.invoke(app, ["prepare", "shakespeare"])
    assert result.exit_code == 0
    mock_run.assert_called_once()


def test_main_sample_missing_meta_fails(tmp_path: Path, mocker: MockerFixture) -> None:
    """Sample should fail fast when neither train meta nor runtime meta exist."""
    mocker.patch("ml_playground.config_loader.fs_path_exists", return_value=False)
    shared = SharedConfig(
        experiment="shakespeare",
        config_path=Path("config.toml"),
        project_home=Path("."),
        dataset_dir=Path("."),
        train_out_dir=Path("."),
        sample_out_dir=Path("out"),
    )
    mocker.patch(
        "ml_playground.config_loader.load_full_experiment_config",
        return_value=ExperimentConfig(
            prepare=PreparerConfig(),
            train=TrainerConfig(
                model=ModelConfig(),
                data=DataConfig(),
                optim=OptimConfig(),
                schedule=LRSchedule(),
                runtime=RuntimeConfig(out_dir=Path(".")),
            ),
            sample=SamplerConfig(
                runtime=RuntimeConfig(out_dir=Path("out")), sample=SampleConfig()
            ),
            shared=shared,
        ),
    )
    result = runner.invoke(app, ["sample", "shakespeare"])
    assert result.exit_code == 1
    assert "Missing required meta file for sampling" in result.stdout


def test_main_prepare_bundestag_char_success(mocker: MockerFixture) -> None:
    """Test prepare command with bundestag_char dataset succeeds."""
    mock_run = mocker.patch("ml_playground.cli._run_prepare")
    shared = SharedConfig(
        experiment="bundestag_char",
        config_path=Path("config.toml"),
        project_home=Path("."),
        dataset_dir=Path("."),
        train_out_dir=Path("."),
        sample_out_dir=Path("."),
    )
    mocker.patch(
        "ml_playground.config_loader.load_full_experiment_config",
        return_value=ExperimentConfig(
            prepare=PreparerConfig(),
            train=TrainerConfig(
                model=ModelConfig(),
                data=DataConfig(),
                optim=OptimConfig(),
                schedule=LRSchedule(),
                runtime=RuntimeConfig(out_dir=Path(".")),
            ),
            sample=SamplerConfig(
                runtime=RuntimeConfig(out_dir=Path(".")), sample=SampleConfig()
            ),
            shared=shared,
        ),
    )
    result = runner.invoke(app, ["prepare", "bundestag_char"])
    assert result.exit_code == 0
    mock_run.assert_called_once()


def test_main_prepare_unknown_dataset_fails(
    mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """Unknown experiment should surface as a CLI error exit."""
    mocker.patch(
        "ml_playground.config_loader.load_full_experiment_config",
        side_effect=FileNotFoundError("Config not found"),
    )
    result = runner.invoke(app, ["prepare", "unknown"])
    assert result.exit_code == 1
    assert "Config not found" in result.stdout


def test_main_train_success(tmp_path: Path, mocker: MockerFixture) -> None:
    """Test train command auto-resolves config for experiment and calls train (strict loader)."""
    # E1.2: mock meta existence
    mocker.patch("ml_playground.config_loader.fs_path_exists", return_value=True)
    mock_run = mocker.patch("ml_playground.cli._run_train")
    shared = SharedConfig(
        experiment="shakespeare",
        config_path=Path("config.toml"),
        project_home=Path("."),
        dataset_dir=Path("."),
        train_out_dir=Path("."),
        sample_out_dir=Path("."),
    )
    mocker.patch(
        "ml_playground.config_loader.load_full_experiment_config",
        return_value=ExperimentConfig(
            prepare=PreparerConfig(),
            train=TrainerConfig(
                model=ModelConfig(),
                data=DataConfig(),
                optim=OptimConfig(),
                schedule=LRSchedule(),
                runtime=RuntimeConfig(out_dir=Path(".")),
            ),
            sample=SamplerConfig(
                runtime=RuntimeConfig(out_dir=Path(".")), sample=SampleConfig()
            ),
            shared=shared,
        ),
    )
    result = runner.invoke(app, ["train", "shakespeare"])
    assert result.exit_code == 0
    mock_run.assert_called_once()


def test_main_train_no_train_block_fails(
    tmp_path: Path, mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test train command fails when strict loader raises."""
    mocker.patch(
        "ml_playground.config_loader.load_full_experiment_config",
        side_effect=ValueError("Missing train config"),
    )
    result = runner.invoke(app, ["train", "shakespeare"])
    assert result.exit_code == 1
    assert "Missing train config" in result.stdout


def test_main_sample_success(tmp_path: Path, mocker: MockerFixture) -> None:
    """Test sample command auto-resolves config and calls sample function (strict loader)."""
    # E1.2: mock meta discovery success
    mocker.patch("ml_playground.config_loader.fs_path_exists", return_value=True)
    mock_sample_cfg = SamplerConfig(
        runtime=RuntimeConfig(out_dir=Path("out")),
        sample=SampleConfig(start="x"),
    )
    mock_run = mocker.patch("ml_playground.cli._run_sample")
    shared = SharedConfig(
        experiment="shakespeare",
        config_path=Path("config.toml"),
        project_home=Path("."),
        dataset_dir=Path("."),
        train_out_dir=Path("."),
        sample_out_dir=Path("."),
    )
    mocker.patch(
        "ml_playground.config_loader.load_full_experiment_config",
        return_value=ExperimentConfig(
            prepare=PreparerConfig(),
            train=TrainerConfig(
                model=ModelConfig(),
                data=DataConfig(),
                optim=OptimConfig(),
                schedule=LRSchedule(),
                runtime=RuntimeConfig(out_dir=Path(".")),
            ),
            sample=mock_sample_cfg,
            shared=shared,
        ),
    )
    result = runner.invoke(app, ["sample", "shakespeare"])
    assert result.exit_code == 0
    mock_run.assert_called_once()


def test_main_sample_no_sample_block_fails(
    tmp_path: Path, mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test sample command fails when strict loader raises."""
    mocker.patch(
        "ml_playground.config_loader.load_full_experiment_config",
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
    shared = SharedConfig(
        experiment="shakespeare",
        config_path=Path("config.toml"),
        project_home=Path("."),
        dataset_dir=Path("."),
        train_out_dir=Path("."),
        sample_out_dir=Path("."),
    )
    mocker.patch(
        "ml_playground.config_loader.load_full_experiment_config",
        return_value=ExperimentConfig(
            prepare=PreparerConfig(),
            train=mock_train_config,  # type: ignore[arg-type]
            sample=mock_sample_config,
            shared=shared,
        ),
    )

    result = runner.invoke(app, ["loop", "shakespeare"])
    assert result.exit_code == 0
    mock_run.assert_called_once()


def test_main_loop_unknown_dataset_fails(tmp_path: Path, mocker: MockerFixture) -> None:
    """Unknown experiment should bubble up as CLI error exit."""
    mocker.patch(
        "ml_playground.config_loader.load_full_experiment_config",
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
        "ml_playground.config_loader.load_full_experiment_config",
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
        "ml_playground.config_loader.load_full_experiment_config",
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
        data=DataConfig(),
        optim=OptimConfig(),
        schedule=LRSchedule(),
        runtime=RuntimeConfig(out_dir=out),
    )
    # Should not raise
    cli._log_command_status("train", train_cfg.runtime)

    # Sampler with runtime
    samp_cfg = SamplerConfig(runtime=RuntimeConfig(out_dir=out), sample=SampleConfig())
    cli._log_command_status("sample", samp_cfg.runtime)


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
    out = tmp_path / "missing_out"
    cfg = TrainerConfig(
        model=ModelConfig(),
        data=DataConfig(),
        optim=OptimConfig(),
        schedule=LRSchedule(),
        runtime=RuntimeConfig(out_dir=out),
    )
    # Should not raise
    cli._log_command_status("train", cfg.runtime)


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
    p = cli._cfg_path_for("some_exp", explicit)
    assert p == explicit
    # Default path under experiments root
    d = cli._cfg_path_for("bundestag_char", None)
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
    
    monkeypatch.setattr(config_loader, "_default_config_path_from_root", mock_default_config_path_from_root)
    
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

    def fake_preparer(shared):
        calls.append("prepare")

    monkeypatch.setattr(prepare_mod, "make_preparer", lambda cfg: fake_preparer)
    monkeypatch.setattr(trainer_mod, "train", lambda cfg, shared: calls.append("train"))
    monkeypatch.setattr(
        sampler_mod, "sample", lambda cfg, shared: calls.append("sample")
    )

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
        data=DataConfig(),
        optim=OptimConfig(),
        schedule=LRSchedule(),
        runtime=RuntimeConfig(out_dir=tmp_path),
    )
    scfg = SamplerConfig(runtime=RuntimeConfig(out_dir=tmp_path), sample=SampleConfig())
    shared = SharedConfig(
        experiment="exp",
        config_path=tmp_path / "cfg.toml",
        project_home=tmp_path,
        dataset_dir=tmp_path,
        train_out_dir=tmp_path,
        sample_out_dir=tmp_path,
    )

    # Should not raise; ensure order prepare -> train -> sample
    cli._run_loop("exp", tmp_path / "cfg.toml", prep_cfg, tcfg, scfg, shared)

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
    # E1.2: mock meta discovery to avoid filesystem dependency
    mocker.patch("ml_playground.config_loader.fs_path_exists", return_value=True)

    # Also ensure legacy entrypoint is NOT consulted anymore
    called = {"count": 0}

    def _fake_sample_from_toml(path):  # type: ignore[no-untyped-def]
        called["count"] += 1

    monkeypatch.setattr(
        "ml_playground.experiments.speakger.sampler.sample_from_toml",
        _fake_sample_from_toml,
        raising=False,
    )

    # Mock canonical loader to avoid reading real experiment config with unknown keys
    shared = SharedConfig(
        experiment="speakger",
        config_path=Path("config.toml"),
        project_home=Path("."),
        dataset_dir=Path("."),
        train_out_dir=Path("."),
        sample_out_dir=Path("."),
    )
    mocker.patch(
        "ml_playground.config_loader.load_full_experiment_config",
        return_value=ExperimentConfig(
            prepare=PreparerConfig(),
            train=TrainerConfig(
                model=ModelConfig(),
                data=DataConfig(),
                optim=OptimConfig(),
                schedule=LRSchedule(),
                runtime=RuntimeConfig(out_dir=Path(".")),
            ),
            sample=SamplerConfig(
                runtime=RuntimeConfig(out_dir=Path(".")), sample=SampleConfig()
            ),
            shared=shared,
        ),
    )

    # Act: run CLI with experiment auto-resolved config (no SystemExit on success)
    cli.main(["sample", "speakger"])

    # Assert: unified path called once; legacy path not invoked
    run_sample.assert_called_once()
    assert called["count"] == 0
