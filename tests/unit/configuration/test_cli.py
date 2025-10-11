from __future__ import annotations

import pytest
from pathlib import Path
from typing import Any, Callable, Iterator, cast

from typer.testing import CliRunner

# Additional imports for consolidated CLI tests
import logging

import typer

import ml_playground.cli as cli
from ml_playground.cli import (
    CLIDependencies,
    app,
    override_cli_dependencies,
)
from ml_playground.configuration import cli as config_cli
from ml_playground.configuration import loading as config_loader
from ml_playground.configuration.models import (
    DataConfig,
    ExperimentConfig,
    LRSchedule,
    ModelConfig,
    OptimConfig,
    PreparerConfig,
    RuntimeConfig,
    SampleConfig,
    SamplerConfig,
    SharedConfig,
    TrainerConfig,
)

runner = CliRunner()


@pytest.fixture()
def dataset_dir(tmp_path: Path) -> Iterator[Path]:
    data_dir = tmp_path / "dataset"
    data_dir.mkdir(parents=True)
    (data_dir / "meta.pkl").write_text("meta")
    yield data_dir


@pytest.fixture()
def out_dirs(tmp_path: Path) -> Iterator[tuple[Path, Path]]:
    train_root = tmp_path / "train"
    sample_root = tmp_path / "sample"
    train_root.mkdir(parents=True)
    sample_root.mkdir(parents=True)
    yield train_root, sample_root


@pytest.fixture()
def shared_factory(
    dataset_dir: Path, out_dirs: tuple[Path, Path]
) -> Callable[[str], SharedConfig]:
    train_root, sample_root = out_dirs

    def _factory(experiment: str) -> SharedConfig:
        config_path = dataset_dir / f"{experiment}.toml"
        config_path.write_text("{}")
        sample_experiment_dir = sample_root / experiment
        sample_experiment_dir.mkdir(parents=True, exist_ok=True)
        (sample_experiment_dir / "meta.pkl").write_text("meta")
        return SharedConfig(
            experiment=experiment,
            config_path=config_path,
            project_home=dataset_dir.parent,
            dataset_dir=dataset_dir,
            train_out_dir=train_root,
            sample_out_dir=sample_root,
        )

    return _factory


def _make_full_experiment(
    shared: SharedConfig,
    *,
    sample: SamplerConfig | None = None,
    prepare: PreparerConfig | None = None,
) -> ExperimentConfig:
    return ExperimentConfig(
        prepare=prepare or PreparerConfig(),
        train=TrainerConfig(
            model=ModelConfig(),
            data=DataConfig(),
            optim=OptimConfig(),
            schedule=LRSchedule(),
            runtime=RuntimeConfig(out_dir=shared.train_out_dir),
        ),
        sample=sample
        or SamplerConfig(
            runtime=RuntimeConfig(out_dir=shared.sample_out_dir),
            sample=SampleConfig(),
        ),
        shared=shared,
    )


def _make_deps(
    *,
    load_experiment: Callable[[str, Path | None], ExperimentConfig],
    ensure_train: Callable[[ExperimentConfig], Any] | None = None,
    ensure_sample: Callable[[ExperimentConfig], Any] | None = None,
    run_prepare: Callable[[str, PreparerConfig, Path, SharedConfig], None]
    | None = None,
    run_train: Callable[[str, TrainerConfig, Path, SharedConfig], None] | None = None,
    run_sample: Callable[[str, SamplerConfig, Path, SharedConfig], None] | None = None,
) -> CLIDependencies:
    return CLIDependencies(
        load_experiment=load_experiment,
        ensure_train_prerequisites=ensure_train or _noop,
        ensure_sample_prerequisites=ensure_sample or _noop,
        run_prepare=run_prepare or _noop,
        run_train=run_train or _noop,
        run_sample=run_sample or _noop,
    )


def _noop(*_args: Any, **_kwargs: Any) -> None:
    return None


def test_main_prepare_shakespeare_success(
    shared_factory: Callable[[str], SharedConfig],
) -> None:
    """Test prepare command with shakespeare dataset succeeds."""

    shared = shared_factory("shakespeare")
    exp = _make_full_experiment(shared)
    calls: dict[str, int] = {"prepare": 0}

    def _run_prepare(
        experiment: str,
        prepare_cfg: PreparerConfig,
        config_path: Path,
        shared: SharedConfig,
    ) -> None:
        calls["prepare"] += 1
        assert experiment == "shakespeare"
        assert prepare_cfg is exp.prepare
        assert shared is exp.shared
        assert config_path == exp.shared.config_path

    deps = _make_deps(
        load_experiment=lambda experiment, exp_config: exp,
        run_prepare=_run_prepare,
    )

    with override_cli_dependencies(deps):
        result = runner.invoke(app, ["prepare", "shakespeare"])

    assert result.exit_code == 0
    assert calls["prepare"] == 1


def test_main_sample_missing_meta_fails(
    shared_factory: Callable[[str], SharedConfig],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Sample should fail fast when neither train meta nor runtime meta exist."""

    shared = shared_factory("shakespeare")
    exp = _make_full_experiment(shared)

    def _ensure_sample(_exp: ExperimentConfig) -> tuple[Path, Path]:
        raise ValueError("Missing required meta file for sampling")

    deps = _make_deps(
        load_experiment=lambda experiment, exp_config: exp,
        ensure_sample=_ensure_sample,
    )

    with override_cli_dependencies(deps):
        result = runner.invoke(app, ["sample", "shakespeare"])

    assert result.exit_code == 1
    assert "Missing required meta file for sampling" in caplog.messages[-1]


def test_main_prepare_bundestag_char_success(
    shared_factory: Callable[[str], SharedConfig],
) -> None:
    """Test prepare command with bundestag_char dataset succeeds."""

    shared = shared_factory("bundestag_char")
    exp = _make_full_experiment(shared)
    calls: dict[str, int] = {"prepare": 0}

    def _run_prepare(
        experiment: str,
        prepare_cfg: PreparerConfig,
        config_path: Path,
        shared: SharedConfig,
    ) -> None:
        calls["prepare"] += 1
        assert experiment == "bundestag_char"
        assert prepare_cfg is exp.prepare
        assert shared is exp.shared

    deps = CLIDependencies(
        load_experiment=lambda experiment, exp_config: exp,
        ensure_train_prerequisites=_noop,
        ensure_sample_prerequisites=_noop,
        run_prepare=_run_prepare,
        run_train=_noop,
        run_sample=_noop,
    )

    with override_cli_dependencies(deps):
        result = runner.invoke(app, ["prepare", "bundestag_char"])

    assert result.exit_code == 0
    assert calls["prepare"] == 1


def test_main_prepare_unknown_dataset_fails(caplog: pytest.LogCaptureFixture) -> None:
    """Unknown experiment should surface as a CLI error exit."""

    def _load_experiment(_experiment: str, _path: Path | None) -> ExperimentConfig:
        raise FileNotFoundError("Config not found")

    deps = _make_deps(load_experiment=_load_experiment)

    with override_cli_dependencies(deps):
        result = runner.invoke(app, ["prepare", "unknown"])

    assert result.exit_code == 1
    assert "Config not found" in caplog.messages[-1]


def test_main_train_success(shared_factory: Callable[[str], SharedConfig]) -> None:
    """Train command loads config and invokes trainer once prerequisites pass."""

    shared = shared_factory("shakespeare")
    exp = _make_full_experiment(shared)
    calls: dict[str, int] = {"train": 0}

    def _run_train(
        experiment: str,
        train_cfg: TrainerConfig,
        config_path: Path,
        shared: SharedConfig,
    ) -> None:
        calls["train"] += 1
        assert experiment == "shakespeare"
        assert train_cfg is exp.train
        assert shared is exp.shared

    deps = _make_deps(
        load_experiment=lambda experiment, exp_config: exp,
        ensure_train=lambda e: shared.dataset_dir / "meta.pkl",
        run_train=_run_train,
    )

    with override_cli_dependencies(deps):
        result = runner.invoke(app, ["train", "shakespeare"])

    assert result.exit_code == 0
    assert calls["train"] == 1


def test_main_train_no_train_block_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Train command surfaces loader errors as CLI exits."""

    def _load_experiment(_experiment: str, _path: Path | None) -> ExperimentConfig:
        raise ValueError("Missing train config")

    deps = CLIDependencies(
        load_experiment=_load_experiment,
        ensure_train_prerequisites=_noop,
        ensure_sample_prerequisites=_noop,
        run_prepare=_noop,
        run_train=_noop,
        run_sample=_noop,
    )

    with override_cli_dependencies(deps):
        result = runner.invoke(app, ["train", "shakespeare"])

    assert result.exit_code == 1
    assert "Missing train config" in caplog.messages[-1]


def test_main_sample_success(shared_factory: Callable[[str], SharedConfig]) -> None:
    """Sample command loads config and invokes sampler once prerequisites pass."""

    shared = shared_factory("shakespeare")
    sample_cfg = SamplerConfig(
        runtime=RuntimeConfig(out_dir=shared.sample_out_dir),
        sample=SampleConfig(start="x"),
    )
    exp = _make_full_experiment(shared, sample=sample_cfg)
    calls: dict[str, int] = {"sample": 0}

    def _run_sample(
        experiment: str,
        sample_cfg: SamplerConfig,
        config_path: Path,
        shared: SharedConfig,
    ) -> None:
        calls["sample"] += 1
        assert experiment == "shakespeare"
        assert sample_cfg.sample.start == "x"

    train_meta = shared.dataset_dir / "meta.pkl"
    runtime_meta = shared.sample_out_dir / shared.experiment / "meta.pkl"
    deps = _make_deps(
        load_experiment=lambda experiment, exp_config: exp,
        ensure_sample=lambda e: (train_meta, runtime_meta),
        run_sample=_run_sample,
    )

    with override_cli_dependencies(deps):
        result = runner.invoke(app, ["sample", "shakespeare"])

    assert result.exit_code == 0
    assert calls["sample"] == 1


def test_main_sample_no_sample_block_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Sample command raises when configuration validation fails."""

    def _load_experiment(_experiment: str, _path: Path | None) -> ExperimentConfig:
        raise ValueError("Missing sample config")

    deps = CLIDependencies(
        load_experiment=_load_experiment,
        ensure_train_prerequisites=_noop,
        ensure_sample_prerequisites=_noop,
        run_prepare=_noop,
        run_train=_noop,
        run_sample=_noop,
    )

    with override_cli_dependencies(deps):
        result = runner.invoke(app, ["sample", "shakespeare"])

    assert result.exit_code == 1
    assert "Missing sample config" in caplog.messages[-1]


def test_strict_mode_has_no_override_functions() -> None:
    # Strict mode: configuration is TOML-only, no override helpers are exposed
    assert not hasattr(config_loader, "apply_train_overrides")
    assert not hasattr(config_loader, "apply_sample_overrides")


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


def test_global_setup_enables_tf32_when_cuda_available() -> None:
    import torch

    # Simulate CUDA availability via DI; ensure flags become True after setup
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass
    cli._global_device_setup("cuda", "bfloat16", seed=1, cuda_is_available=lambda: True)
    # Flags should be enabled
    assert getattr(torch.backends.cuda.matmul, "allow_tf32", True) is True
    assert getattr(torch.backends.cudnn, "allow_tf32", True) is True


def test_global_setup_no_crash_without_cuda() -> None:
    # Should not raise even if CUDA unavailable
    cli._global_device_setup("cuda", "float16", seed=42)


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


def test_get_cfg_path_explicit_and_default(tmp_path: Path):
    explicit = tmp_path / "x.toml"
    # Explicit path returned as-is
    p = config_cli.cfg_path_for("some_exp", explicit)
    assert p == explicit
    # Default path under experiments root
    d = config_cli.cfg_path_for("bundestag_char", None)
    expected_suffix = Path(
        "src/ml_playground/experiments/bundestag_char/config.toml"
    ).as_posix()
    assert d.as_posix().endswith(expected_suffix)


def test_load_config_error_branches(tmp_path: Path):
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
    test_defaults_path = tmp_path / "default_config.toml"
    test_defaults_path.write_text("this is not valid toml")

    with pytest.raises(Exception) as ei3:
        config_loader.load_train_config(cfg, default_config_path=test_defaults_path)
    assert "default_config.toml" in str(ei3.value).lower()

    # 3) Experiment config invalid -> exit mentioning cfg path
    bad_cfg = exp_dir / "bad.toml"
    bad_cfg.write_text("this is not valid toml")
    with pytest.raises(Exception) as ei4:
        config_loader.load_train_config(bad_cfg)
    assert "bad.toml" in str(ei4.value).lower()


@pytest.mark.parametrize(
    "args, expect",
    [
        (["--help"], "usage"),
        (["prepare", "--help"], "prepare"),
        (["train", "--help"], "train"),
        (["sample", "--help"], "sample"),
        (["analyze", "--help"], "analyze"),
    ],
)
def test_cli_help_and_version(args, expect):
    """CLI help text should surface command summaries quickly via Typer runner."""

    result = runner.invoke(app, args)
    assert result.exit_code == 0
    out = (result.stdout or "").lower()
    assert expect in out


def test_cli_global_exp_config_missing_exits(tmp_path, caplog):
    # Point to a definitely missing path
    missing = tmp_path / "nope.toml"

    deps = _make_deps(load_experiment=lambda _exp, _path: None)

    with (
        override_cli_dependencies(deps),
        caplog.at_level(logging.ERROR, logger="ml_playground.cli"),
    ):
        result = runner.invoke(
            app, ["--exp-config", str(missing), "prepare", "shakespeare"]
        )

    # Typer exits with code 2 for callback-triggered validation failures
    assert result.exit_code == 2
    assert any("config file not found" in msg.lower() for msg in caplog.messages)


def test_sample_routes_to_injected_sampler(
    shared_factory: Callable[[str], SharedConfig],
) -> None:
    """CLI should dispatch 'sample speakger' to the injected sampler run path."""

    shared = shared_factory("speakger")
    exp = _make_full_experiment(shared)
    calls: dict[str, int] = {"run_sample": 0}

    def _run_sample(
        experiment: str,
        sample_cfg: SamplerConfig,
        config_path: Path,
        shared_cfg: SharedConfig,
    ) -> None:
        calls["run_sample"] += 1
        assert experiment == "speakger"
        assert shared_cfg is shared

    deps = _make_deps(
        load_experiment=lambda experiment, exp_config: exp,
        ensure_sample=lambda e: (
            shared.dataset_dir / "meta.pkl",
            shared.sample_out_dir / shared.experiment / "meta.pkl",
        ),
        run_sample=_run_sample,
    )

    with override_cli_dependencies(deps):
        cli.main(["sample", "speakger"])

    assert calls["run_sample"] == 1


def test_complete_experiments_delegates_to_loader():
    """_complete_experiments should delegate to config_loading.list_experiments_with_config."""
    from types import SimpleNamespace

    calls: list[str] = []

    def fake_list_experiments(incomplete: str) -> list[str]:
        calls.append(incomplete)
        return ["exp1", "exp2"]

    fake_config_loading = SimpleNamespace(
        list_experiments_with_config=fake_list_experiments
    )
    original = cli.config_loading

    cli.config_loading = fake_config_loading  # type: ignore[misc]
    try:
        result = cli._complete_experiments(SimpleNamespace(), "test")  # type: ignore[arg-type]
        assert calls == ["test"]
        assert result == ["exp1", "exp2"]
    finally:
        cli.config_loading = original  # type: ignore[misc]


def test_log_dir_handles_permission_error(tmp_path: Path):
    """_log_dir should handle PermissionError when listing directory contents."""

    class ListLogger:
        def __init__(self) -> None:
            self.infos: list[str] = []

        def info(self, message: str) -> None:
            self.infos.append(str(message))

    logger = ListLogger()
    existing = tmp_path / "restricted"
    existing.mkdir()

    class RestrictedPath(type(existing)):  # type: ignore[misc]
        def iterdir(self):  # type: ignore[override]
            raise PermissionError("Access denied")

    restricted_path = RestrictedPath(existing)
    cli._log_dir("tag", "restricted", restricted_path, logger)  # type: ignore[arg-type]
    assert any("exists" in msg for msg in logger.infos)
    assert not any("Contents" in msg for msg in logger.infos)


def test_log_dir_handles_os_error(tmp_path: Path):
    """_log_dir should handle OSError when listing directory contents."""

    class ListLogger:
        def __init__(self) -> None:
            self.infos: list[str] = []

        def info(self, message: str) -> None:
            self.infos.append(str(message))

    logger = ListLogger()
    existing = tmp_path / "broken"
    existing.mkdir()

    class BrokenPath(type(existing)):  # type: ignore[misc]
        def iterdir(self):  # type: ignore[override]
            raise OSError("Disk error")

    broken_path = BrokenPath(existing)
    cli._log_dir("tag", "broken", broken_path, logger)  # type: ignore[arg-type]
    assert any("exists" in msg for msg in logger.infos)


def test_log_command_status_handles_exceptions(tmp_path: Path):
    """_log_command_status should handle OSError/ValueError/TypeError gracefully."""

    class ListLogger:
        def __init__(self) -> None:
            self.messages: list[str] = []

        def info(self, message: str) -> None:
            self.messages.append(str(message))

    logger = ListLogger()
    shared = SharedConfig(
        experiment="test",
        config_path=tmp_path / "config.toml",
        project_home=tmp_path,
        dataset_dir=tmp_path / "dataset",
        train_out_dir=tmp_path / "train",
        sample_out_dir=tmp_path / "sample",
    )

    original_log_dir = cli._log_dir

    for exc_type in [OSError, ValueError, TypeError]:

        def mock_log_dir(*args: Any, **kwargs: Any) -> None:
            raise exc_type("Simulated error")

        cli._log_dir = mock_log_dir  # type: ignore[assignment]
        try:
            cli._log_command_status("tag", shared, tmp_path, logger)
        finally:
            cli._log_dir = original_log_dir  # type: ignore[assignment]


def test_run_analyze_bundestag_char_logs_message(caplog: pytest.LogCaptureFixture):
    """_run_analyze for bundestag_char should log a not-implemented message."""
    with caplog.at_level(logging.INFO, logger="ml_playground.cli"):
        cli._run_analyze("bundestag_char", "localhost", 8050, True)

    assert any("not implemented" in msg.lower() for msg in caplog.messages)


def test_analyze_command_invokes_run_analyze():
    """analyze CLI command should invoke _run_analyze."""
    calls: list[tuple[str, str, int, bool]] = []

    def fake_run_analyze(
        experiment: str, host: str, port: int, open_browser: bool
    ) -> None:
        calls.append((experiment, host, port, open_browser))

    original = cli._run_analyze
    cli._run_analyze = fake_run_analyze  # type: ignore[assignment]
    try:
        result = runner.invoke(cli.app, ["analyze", "bundestag_char"])
        assert result.exit_code == 0
        assert calls == [("bundestag_char", "127.0.0.1", 8050, True)]
    finally:
        cli._run_analyze = original  # type: ignore[assignment]


def test_run_train_cmd_uses_default_deps(shared_factory: Callable[[str], SharedConfig]):
    """_run_train_cmd should use get_cli_dependencies when deps=None."""
    shared = shared_factory("demo")
    exp = _make_full_experiment(shared)

    deps_called: dict[str, int] = {"get_cli_dependencies": 0}

    def fake_get_deps() -> CLIDependencies:
        deps_called["get_cli_dependencies"] += 1
        return _make_deps(load_experiment=lambda name, path: exp)

    original = cli.get_cli_dependencies
    cli.get_cli_dependencies = fake_get_deps  # type: ignore[assignment]
    try:
        cli._run_train_cmd("demo", None, deps=None)
        assert deps_called["get_cli_dependencies"] == 1
    finally:
        cli.get_cli_dependencies = original  # type: ignore[assignment]


def test_run_sample_cmd_uses_default_deps(
    shared_factory: Callable[[str], SharedConfig],
):
    """_run_sample_cmd should use get_cli_dependencies when deps=None."""
    shared = shared_factory("demo")
    exp = _make_full_experiment(shared)

    deps_called: dict[str, int] = {"get_cli_dependencies": 0}

    def fake_get_deps() -> CLIDependencies:
        deps_called["get_cli_dependencies"] += 1
        return _make_deps(load_experiment=lambda name, path: exp)

    original = cli.get_cli_dependencies
    cli.get_cli_dependencies = fake_get_deps  # type: ignore[assignment]
    try:
        cli._run_sample_cmd("demo", None, deps=None)
        assert deps_called["get_cli_dependencies"] == 1
    finally:
        cli.get_cli_dependencies = original  # type: ignore[assignment]
