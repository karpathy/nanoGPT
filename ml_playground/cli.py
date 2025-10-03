from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Callable, Optional

import torch
import typer
from typer.main import get_command

from ml_playground.configuration.models import (
    ExperimentConfig,
    PreparerConfig,
    SamplerConfig,
    SharedConfig,
    TrainerConfig,
)
from ml_playground.configuration import loading as config_loading
from ml_playground.configuration import cli as config_cli
from ml_playground.data_pipeline.preparer import create_pipeline
from ml_playground.sampling.runner import Sampler
from ml_playground.training.loop.runner import Trainer as CoreTrainer
from ml_playground.experiments import registry

# (Removed unused type aliases)


__all__ = ["main"]


@dataclass(frozen=True)
class CLIDependencies:
    load_experiment: Callable[[str, Path | None], ExperimentConfig]
    ensure_train_prerequisites: Callable[[ExperimentConfig], Any]
    ensure_sample_prerequisites: Callable[[ExperimentConfig], Any]
    run_prepare: Callable[[str, PreparerConfig, Path, SharedConfig], None]
    run_train: Callable[[str, TrainerConfig, Path, SharedConfig], None]
    run_sample: Callable[[str, SamplerConfig, Path, SharedConfig], None]


def _run_prepare_impl(
    experiment: str,
    prepare_cfg: PreparerConfig,
    config_path: Path,
    shared: Any,
) -> None:
    """Run the full prepare flow for an experiment."""
    prepare_cfg.logger.info(f"Running pipeline for experiment: {experiment}")
    pipeline = create_pipeline(prepare_cfg, shared)
    pipeline.run()
    prepare_cfg.logger.info(f"Pipeline for {experiment} finished.")


def _run_train_impl(
    experiment: str,
    train_cfg: TrainerConfig,
    config_path: Path,
    shared: Any,
) -> None:
    """Run the full training flow for an experiment."""
    if not train_cfg.runtime:
        train_cfg.logger.error("Runtime configuration is missing for training.")
        raise typer.Exit(1)

    _global_device_setup(
        train_cfg.runtime.device,
        train_cfg.runtime.dtype,
        train_cfg.runtime.seed,
    )

    train_cfg.logger.info(f"Running trainer for experiment: {experiment}")
    _log_command_status("pre-train", shared, shared.train_out_dir, train_cfg.logger)

    trainer = CoreTrainer(train_cfg, shared)
    trainer.run()

    train_cfg.logger.info(f"Trainer for {experiment} finished.")
    _log_command_status("post-train", shared, shared.train_out_dir, train_cfg.logger)


def _run_sample_impl(
    experiment: str,
    sample_cfg: SamplerConfig,
    config_path: Path,
    shared: Any,
) -> None:
    """Run the full sampling flow for an experiment."""
    if not sample_cfg.runtime:
        sample_cfg.logger.error("Runtime configuration is missing for sampling.")
        raise typer.Exit(1)

    _global_device_setup(
        sample_cfg.runtime.device,
        sample_cfg.runtime.dtype,
        sample_cfg.runtime.seed,
    )

    sample_cfg.logger.info(f"Running sampler for experiment: {experiment}")
    _log_command_status("pre-sample", shared, shared.sample_out_dir, sample_cfg.logger)
    sampler = Sampler(sample_cfg, shared)
    sampler.run()
    sample_cfg.logger.info(f"Sampler for {experiment} finished.")
    _log_command_status("post-sample", shared, shared.sample_out_dir, sample_cfg.logger)


def default_cli_dependencies() -> CLIDependencies:
    return CLIDependencies(
        load_experiment=config_cli.load_experiment,
        ensure_train_prerequisites=config_cli.ensure_train_prerequisites,
        ensure_sample_prerequisites=config_cli.ensure_sample_prerequisites,
        run_prepare=_run_prepare_impl,
        run_train=_run_train_impl,
        run_sample=_run_sample_impl,
    )


_CLI_DEPENDENCIES: CLIDependencies = default_cli_dependencies()


def get_cli_dependencies() -> CLIDependencies:
    return _CLI_DEPENDENCIES


@contextmanager
def override_cli_dependencies(deps: CLIDependencies):
    global _CLI_DEPENDENCIES
    previous = _CLI_DEPENDENCIES
    _CLI_DEPENDENCIES = deps
    try:
        yield
    finally:
        _CLI_DEPENDENCIES = previous


# --- Global device setup ---------------------------------------------------
def _global_device_setup(
    device: str,
    dtype: str,
    seed: int,
    *,
    cuda_is_available: Optional[Callable[[], bool]] = None,
) -> None:
    """Set global seeds and enable TF32 as needed.

    Centralizes side-effectful setup so other modules don't repeat it.
    """
    try:
        torch.manual_seed(seed)
        _cuda_available = (
            cuda_is_available()
            if cuda_is_available is not None
            else torch.cuda.is_available()
        )
        if _cuda_available:
            torch.cuda.manual_seed(seed)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    except (RuntimeError, AttributeError, OSError):
        # Never fail CLI due to environment-specific torch issues
        pass


# --- Typer helpers ---------------------------------------------------------
def _complete_experiments(ctx: typer.Context, incomplete: str) -> list[str]:
    """Auto-complete experiment names based on directories with a config.toml."""
    # Delegate to loader to keep FS access centralized for configuration
    return config_loading.list_experiments_with_config(incomplete)


# --- CLI plumbing ----------------------------------------------------------


def _extract_exp_config(ctx: typer.Context) -> Path | None:
    """Extract the --exp-config path from the Typer context."""
    obj = getattr(ctx, "obj", None)
    if not isinstance(obj, dict):
        return None
    return obj.get("exp_config")


def run_or_exit(
    func: Callable[[], None],
    *,
    keyboard_interrupt_msg: str | None = None,
    exception_exit_code: int = 1,
) -> None:
    """Run a function and exit gracefully on exceptions.

    - KeyboardInterrupt: print optional message and return (no exit), per tests.
    - Other exceptions: echo message and exit with provided code.
    """
    try:
        func()
    except FileNotFoundError as e:
        logger = logging.getLogger(__name__)
        logger.error(f"{e}")
        raise typer.Exit(exception_exit_code)
    except (ValueError, TypeError) as e:
        logger = logging.getLogger(__name__)
        logger.error(f"{e}")
        raise typer.Exit(exception_exit_code)
    except KeyboardInterrupt:
        if keyboard_interrupt_msg:
            logger = logging.getLogger(__name__)
            logger.info(keyboard_interrupt_msg)
        # Do not exit on KeyboardInterrupt in this helper
        return
    except (RuntimeError, OSError, ImportError, SystemError, ConnectionError) as e:
        # Generic mapping for unexpected exceptions: echo and exit with provided code
        logger = logging.getLogger(__name__)
        logger.error(f"{e}")
        raise typer.Exit(exception_exit_code)


def _log_dir(tag: str, dir_name: str, dir_path: Path | None, logger) -> None:
    """Log information about a directory path."""
    if dir_path is None:
        logger.info(f"[{tag}] {dir_name}: <not set>")
    elif isinstance(dir_path, Path):
        if dir_path.exists():
            try:
                contents = sorted([p.name for p in dir_path.iterdir()])
                logger.info(f"[{tag}] {dir_name} (exists): {dir_path}")
                logger.info(f"[{tag}]   Contents: {contents}")
            except (OSError, PermissionError):
                # If listing fails, still indicate existence
                logger.info(f"[{tag}] {dir_name} (exists): {dir_path}")
        else:
            logger.info(f"[{tag}] {dir_name} (missing): {dir_path}")


# --- Command runners -------------------------------------------------------


def _log_command_status(
    tag: str, shared: "SharedConfig", out_dir: Path, logger
) -> None:
    """Log known file-based artifacts for the given config."""
    try:
        _log_dir(tag, "out_dir", out_dir, logger)
        _log_dir(tag, "dataset_dir", shared.dataset_dir, logger)
    except (OSError, ValueError, TypeError):
        # Never fail due to logging
        pass


def _run_prepare(
    experiment: str,
    prepare_cfg: PreparerConfig,
    config_path: Path,
    shared: Any,
) -> None:
    """Run the full prepare flow for an experiment."""
    prepare_cfg.logger.info(f"Running pipeline for experiment: {experiment}")
    pipeline = create_pipeline(prepare_cfg, shared)
    pipeline.run()
    prepare_cfg.logger.info(f"Pipeline for {experiment} finished.")


def _run_train(
    experiment: str,
    train_cfg: TrainerConfig,
    config_path: Path,
    shared: Any,
) -> None:
    """Run the full training flow for an experiment."""
    if not train_cfg.runtime:
        train_cfg.logger.error("Runtime configuration is missing for training.")
        raise typer.Exit(1)

    _global_device_setup(
        train_cfg.runtime.device,
        train_cfg.runtime.dtype,
        train_cfg.runtime.seed,
    )

    train_cfg.logger.info(f"Running trainer for experiment: {experiment}")
    _log_command_status("pre-train", shared, shared.train_out_dir, train_cfg.logger)

    trainer = CoreTrainer(train_cfg, shared)
    trainer.run()

    train_cfg.logger.info(f"Trainer for {experiment} finished.")
    _log_command_status("post-train", shared, shared.train_out_dir, train_cfg.logger)


def _run_sample(
    experiment: str,
    sample_cfg: SamplerConfig,
    config_path: Path,
    shared: Any,
) -> None:
    """Run the full sampling flow for an experiment."""
    if not sample_cfg.runtime:
        sample_cfg.logger.error("Runtime configuration is missing for sampling.")
        raise typer.Exit(1)

    # Global setup is now handled inside the Sampler class
    _global_device_setup(
        sample_cfg.runtime.device,
        sample_cfg.runtime.dtype,
        sample_cfg.runtime.seed,
    )

    sample_cfg.logger.info(f"Running sampler for experiment: {experiment}")
    _log_command_status("pre-sample", shared, shared.sample_out_dir, sample_cfg.logger)
    sampler = Sampler(sample_cfg, shared)
    sampler.run()
    sample_cfg.logger.info(f"Sampler for {experiment} finished.")
    _log_command_status("post-sample", shared, shared.sample_out_dir, sample_cfg.logger)


def _run_analyze(experiment: str, host: str, port: int, open_browser: bool) -> None:
    """Run analysis for an experiment.

    Only 'bundestag_char' is currently supported.
    """
    # Raise for any experiment other than 'bundestag_char'
    if experiment != "bundestag_char":
        raise RuntimeError("analyze currently supports only 'bundestag_char'")
    # Placeholder for actual analysis logic for bundestag_char
    logger = logging.getLogger(__name__)
    logger.info(
        f"Analysis for '{experiment}' not implemented. Host={host}, Port={port}, Open={open_browser}"
    )


# --- CLI definition --------------------------------------------------------


EXPERIMENT_HELP = "Experiment name (directory in ml_playground/experiments)"


ExperimentArg = Annotated[
    str,
    typer.Argument(
        help=EXPERIMENT_HELP,
        autocompletion=_complete_experiments,
    ),
]


# Typer-based CLI
app = typer.Typer(
    no_args_is_help=True,
    help=(
        "ML Playground CLI: prepare data, train models, sample outputs, and export models.\n"
        "This CLI loads and validates TOML configs and injects the resulting configuration\n"
        "objects into experiment code. Experiments must not read TOML directly."
    ),
)


@app.callback()
def global_options(
    ctx: typer.Context,
    exp_config: Annotated[
        Path | None,
        typer.Option(
            "--exp-config",
            help=(
                "Path to an experiment-specific config TOML. When provided, it replaces "
                "the experiment's config.toml. default_config.toml is still loaded first."
            ),
        ),
    ] = None,
) -> None:
    """Global options applied to all subcommands."""
    # Validate --exp-config immediately if provided
    if exp_config is not None and not exp_config.exists():
        logger = logging.getLogger(__name__)
        logger.error(f"Config file not found: {exp_config}")
        raise typer.Exit(2)

    try:
        # Ensure INFO-level logs (including status) are visible by default
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            logging.basicConfig(level=logging.INFO, format="%(message)s")
        ctx.ensure_object(dict)
    except (AttributeError, TypeError):
        # Fallback: if ensure_object fails, safely ignore and avoid crashing
        return
    ctx.obj = {"exp_config": exp_config}


@app.command()
def prepare(
    ctx: typer.Context,
    experiment: ExperimentArg,
) -> None:
    """Prepare data for an experiment."""
    exp_config_path = _extract_exp_config(ctx)
    deps = get_cli_dependencies()

    def _do_prepare() -> None:
        exp = deps.load_experiment(experiment, exp_config_path)
        deps.run_prepare(experiment, exp.prepare, exp.shared.config_path, exp.shared)

    run_or_exit(
        _do_prepare,
        keyboard_interrupt_msg="\nData preparation cancelled.",
    )


@app.command()
def train(
    ctx: typer.Context,
    experiment: ExperimentArg,
) -> None:
    """Train a model for an experiment."""
    exp_config_path = _extract_exp_config(ctx)
    deps = get_cli_dependencies()

    run_or_exit(
        lambda: _run_train_cmd(experiment, exp_config_path, deps),
        keyboard_interrupt_msg="\nTraining cancelled.",
    )


@app.command()
def sample(
    ctx: typer.Context,
    experiment: ExperimentArg,
) -> None:
    """Sample from a trained model."""
    exp_config_path = _extract_exp_config(ctx)
    deps = get_cli_dependencies()

    run_or_exit(
        lambda: _run_sample_cmd(experiment, exp_config_path, deps),
        keyboard_interrupt_msg="\nSampling cancelled.",
    )


@app.command()
def analyze(
    ctx: typer.Context,
    experiment: ExperimentArg,
    host: str = typer.Option(
        "127.0.0.1", help="Host for the analysis server (not implemented)"
    ),
    port: int = typer.Option(
        8050, help="Port for the analysis server (not implemented)"
    ),
    open_browser: bool = typer.Option(
        True, help="Whether to open the browser automatically (not implemented)"
    ),
) -> None:
    """Run analysis for an experiment (not implemented)."""
    _run_analyze(experiment, host, port, open_browser)


def main(argv: list[str] | None = None) -> int | None:
    """Programmatic entry point used by tests; does not sys.exit.

    Passes standalone_mode=False so Click returns instead of exiting.
    """
    # Load experiment preparers explicitly at startup
    registry.load_preparers()

    cmd = get_command(app)
    return cmd.main(args=argv or [], standalone_mode=False)


# ---------------------------------------------------------------------------
# Simplified command implementations
# ---------------------------------------------------------------------------


def _run_train_cmd(
    experiment: str,
    exp_config_path: Path | None,
    deps: CLIDependencies | None = None,
) -> None:
    """Run train command: load full ExperimentConfig once and pass section."""
    if deps is None:
        deps = get_cli_dependencies()
    exp = deps.load_experiment(experiment, exp_config_path)
    deps.ensure_train_prerequisites(exp)
    deps.run_train(experiment, exp.train, exp.shared.config_path, exp.shared)


def _run_sample_cmd(
    experiment: str,
    exp_config_path: Path | None,
    deps: CLIDependencies | None = None,
) -> None:
    """Run sample command: load full ExperimentConfig once and pass section."""
    if deps is None:
        deps = get_cli_dependencies()
    exp = deps.load_experiment(experiment, exp_config_path)
    deps.ensure_sample_prerequisites(exp)
    deps.run_sample(experiment, exp.sample, exp.shared.config_path, exp.shared)


if __name__ == "__main__":
    # When executed as a script, run with default behavior (may exit)
    get_command(app)()
