from __future__ import annotations

import typer
from typer.main import get_command

import logging
from pathlib import Path
from typing import Annotated, Any, Callable
import torch

from ml_playground.config import (
    TrainerConfig,
    SamplerConfig,
    PreparerConfig,
)
import ml_playground.config_loader as config_loader
import ml_playground.prepare as prepare_mod
import ml_playground.sampler as sampler_mod
import ml_playground.trainer as trainer_mod

# (Removed unused type aliases)


# --- Global device setup ---------------------------------------------------
def _global_device_setup(device: str, dtype: str, seed: int) -> None:
    """Set global seeds and enable TF32 as needed.

    Centralizes side-effectful setup so other modules don't repeat it.
    """
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            # Enable TF32 for better perf on Ampere+ when using CUDA
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    except Exception:
        # Never fail CLI due to environment-specific torch issues
        pass


# --- Typer helpers ---------------------------------------------------------
def _complete_experiments(ctx: typer.Context, incomplete: str) -> list[str]:
    """Auto-complete experiment names based on directories with a config.toml."""
    # Delegate to loader to keep FS access centralized for configuration
    return config_loader.list_experiments_with_config(incomplete)


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
        print(f"[error] {e}")
        raise typer.Exit(exception_exit_code)
    except (ValueError, TypeError) as e:
        print(f"[error] {e}")
        raise typer.Exit(exception_exit_code)
    except KeyboardInterrupt:
        if keyboard_interrupt_msg:
            print(keyboard_interrupt_msg)
        # Do not exit on KeyboardInterrupt in this helper
        return
    except Exception as e:
        # Generic mapping for unexpected exceptions: echo and exit with provided code
        print(f"[error] {e}")
        raise typer.Exit(exception_exit_code)


def _cfg_path_for(experiment: str, exp_config: Path | None) -> Path:
    """Resolve the path to the experiment config TOML strictly.

    - If exp_config is provided, use it.
    - Else, use experiments/<experiment>/config.toml under this package's experiments dir.
    """
    return config_loader.get_cfg_path(experiment, exp_config)


def _read_toml(p: Path) -> dict[str, Any]:
    # Canonical loader: delegate to config_loader.read_toml_dict
    return config_loader.read_toml_dict(p)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        bv = out.get(k)
        if isinstance(bv, dict) and isinstance(v, dict):
            out[k] = _deep_merge(bv, v)
        else:
            out[k] = v
    return out

    # All config TOML loading and merging lives in config_loader.


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
            except Exception:
                # If listing fails, still indicate existence
                logger.info(f"[{tag}] {dir_name} (exists): {dir_path}")
        else:
            logger.info(f"[{tag}] {dir_name} (missing): {dir_path}")


# --- Command runners -------------------------------------------------------


def _log_command_status(tag: str, cfg: Any) -> None:
    """Log known file-based artifacts for the given config.

    The function inspects common path fields of the configuration and prints
    their existence and contents. It is best-effort and will never raise.
    """
    logger = logging.getLogger(__name__)
    try:
        # Handle top-level out_dir or nested runtime.out_dir
        out_dir = getattr(cfg, "out_dir", None)
        if out_dir is None and hasattr(cfg, "runtime"):
            out_dir = getattr(cfg.runtime, "out_dir", None)
        _log_dir(tag, "out_dir", out_dir, logger)

        # Handle top-level dataset_dir or nested data.dataset_dir
        ds_dir = getattr(cfg, "dataset_dir", None)
        if ds_dir is None and hasattr(cfg, "data"):
            ds_dir = getattr(cfg.data, "dataset_dir", None)
        _log_dir(tag, "dataset_dir", ds_dir, logger)
    except Exception:
        # Never fail due to logging
        pass


def _run_prepare(
    experiment: str,
    prepare_cfg: PreparerConfig,
    config_path: Path,
    shared: Any,
) -> None:
    """Run the full prepare flow for an experiment."""
    print(f"---\nRunning preparer for experiment: {experiment}")
    preparer = prepare_mod.make_preparer(prepare_cfg)
    preparer(shared)
    print(f"Preparer for {experiment} finished.\n---")


def _run_train(
    experiment: str,
    train_cfg: TrainerConfig,
    config_path: Path,
    shared: Any,
) -> None:
    """Run the full training flow for an experiment."""
    # Global setup
    if not train_cfg.runtime:
        print("[error] Runtime configuration is missing for training.")
        raise typer.Exit(1)

    _global_device_setup(
        train_cfg.runtime.device,
        train_cfg.runtime.dtype,
        train_cfg.runtime.seed,
    )

    print(f"---\nRunning trainer for experiment: {experiment}")
    _log_command_status("pre-train", train_cfg.runtime)
    trainer_mod.train(train_cfg, shared)
    print(f"Trainer for {experiment} finished.")
    _log_command_status("post-train", train_cfg.runtime)
    print("---")


def _run_sample(
    experiment: str,
    sample_cfg: SamplerConfig,
    config_path: Path,
    shared: Any,
) -> None:
    """Run the full sampling flow for an experiment."""
    if not sample_cfg.runtime:
        print("[error] Runtime configuration is missing for sampling.")
        raise typer.Exit(1)

    # Global setup
    _global_device_setup(
        sample_cfg.runtime.device,
        sample_cfg.runtime.dtype,
        sample_cfg.runtime.seed,
    )

    print(f"---\nRunning sampler for experiment: {experiment}")
    _log_command_status("pre-sample", sample_cfg.runtime)
    sampler_mod.sample(sample_cfg, shared)
    print(f"Sampler for {experiment} finished.")
    _log_command_status("post-sample", sample_cfg.runtime)
    print("---")


def _run_analyze(experiment: str, host: str, port: int, open_browser: bool) -> None:
    """Run analysis for an experiment.

    Only 'bundestag_char' is currently supported.
    """
    # Raise for any experiment other than 'bundestag_char'
    if experiment != "bundestag_char":
        raise RuntimeError("analyze currently supports only 'bundestag_char'")
    # Placeholder for actual analysis logic for bundestag_char
    print(
        f"[analyze] Analysis for '{experiment}' not implemented. Host={host}, Port={port}, Open={open_browser}"
    )


def _run_loop(
    experiment: str,
    config_path: Path,
    prepare_cfg: PreparerConfig,
    train_cfg: TrainerConfig,
    sample_cfg: SamplerConfig,
    shared: Any,
) -> None:
    """Run the full prepare->train->sample loop for an experiment."""
    # Determine if prepare can be skipped by checking data artifacts via SharedConfig
    skip_prepare = False
    try:
        ds_dir = shared.dataset_dir
        req_paths = [ds_dir / "train.bin", ds_dir / "val.bin", ds_dir / "meta.pkl"]
        skip_prepare = all(p.exists() for p in req_paths)
    except Exception:
        skip_prepare = False
    if not skip_prepare:
        _run_prepare(experiment, prepare_cfg, config_path, shared)
    _run_train(experiment, train_cfg, config_path, shared)
    _run_sample(experiment, sample_cfg, config_path, shared)


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
        print(f"Config file not found: {exp_config}")
        raise typer.Exit(2)

    try:
        # Ensure INFO-level logs (including status) are visible by default
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            logging.basicConfig(level=logging.INFO, format="%(message)s")
        ctx.ensure_object(dict)
    except Exception:
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
    run_or_exit(
        lambda: _run_prepare_cmd(experiment, exp_config_path),
        keyboard_interrupt_msg="\nData preparation cancelled.",
    )


@app.command()
def train(
    ctx: typer.Context,
    experiment: ExperimentArg,
) -> None:
    """Train a model for an experiment."""
    exp_config_path = _extract_exp_config(ctx)
    run_or_exit(
        lambda: _run_train_cmd(experiment, exp_config_path),
        keyboard_interrupt_msg="\nTraining cancelled.",
    )


@app.command()
def sample(
    ctx: typer.Context,
    experiment: ExperimentArg,
) -> None:
    """Sample from a trained model."""
    exp_config_path = _extract_exp_config(ctx)
    run_or_exit(
        lambda: _run_sample_cmd(experiment, exp_config_path),
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


@app.command()
def loop(
    ctx: typer.Context,
    experiment: ExperimentArg,
) -> None:
    """Run the full prepare, train, and sample loop for an experiment."""
    exp_config_path = _extract_exp_config(ctx)
    run_or_exit(
        lambda: _run_loop_cmd(experiment, exp_config_path),
        keyboard_interrupt_msg="\nLoop cancelled.",
    )


# (convert command removed as part of refactor; exporting handled by experiment-specific tooling)


# --- Main entrypoint ---------------------------------------------------------


def main(argv: list[str] | None = None) -> int | None:
    """Programmatic entry point used by tests; does not sys.exit.

    Passes standalone_mode=False so Click returns instead of exiting.
    """
    cmd = get_command(app)
    return cmd.main(args=argv or [], standalone_mode=False)


if __name__ == "__main__":
    # When executed as a script, run with default behavior (may exit)
    get_command(app)()

# ---------------------------------------------------------------------------
# Simplified command implementations
# ---------------------------------------------------------------------------


def _run_prepare_cmd(experiment: str, exp_config_path: Path | None) -> None:
    """Run prepare command: load full ExperimentConfig once and pass section."""
    cfg_path = _cfg_path_for(experiment, exp_config_path)
    project_home = Path(__file__).resolve().parent.parent
    exp = config_loader.load_full_experiment_config(cfg_path, project_home, experiment)
    _run_prepare(experiment, exp.prepare, cfg_path, exp.shared)


def _run_train_cmd(experiment: str, exp_config_path: Path | None) -> None:
    """Run train command: load full ExperimentConfig once and pass section."""
    cfg_path = _cfg_path_for(experiment, exp_config_path)
    project_home = Path(__file__).resolve().parent.parent
    exp = config_loader.load_full_experiment_config(cfg_path, project_home, experiment)
    # E1.2/E5: Validate meta existence for train using SharedConfig only
    train_meta = exp.shared.dataset_dir / "meta.pkl"
    if not config_loader.fs_path_exists(train_meta):
        raise ValueError(
            f"Missing required meta file for training: {train_meta}.\n"
            "Run 'prepare' first or ensure your preparer writes meta.pkl."
        )
    _run_train(experiment, exp.train, cfg_path, exp.shared)


def _run_sample_cmd(experiment: str, exp_config_path: Path | None) -> None:
    """Run sample command: load full ExperimentConfig once and pass section."""
    cfg_path = _cfg_path_for(experiment, exp_config_path)
    project_home = Path(__file__).resolve().parent.parent
    exp = config_loader.load_full_experiment_config(cfg_path, project_home, experiment)
    # E1.2/E5: Validate meta existence for sample using SharedConfig only
    train_meta = exp.shared.dataset_dir / "meta.pkl"
    runtime_meta = exp.shared.sample_out_dir / experiment / "meta.pkl"
    if not (
        config_loader.fs_path_exists(train_meta)
        or config_loader.fs_path_exists(runtime_meta)
    ):
        raise ValueError(
            "Missing required meta file for sampling. Checked: "
            f"train.meta={train_meta}, runtime.meta={runtime_meta}.\n"
            "Run 'prepare' and 'train' first or place meta.pkl in one of the expected locations."
        )
    _run_sample(experiment, exp.sample, cfg_path, exp.shared)


def _run_loop_cmd(experiment: str, exp_config_path: Path | None) -> None:
    """Run loop command: load full ExperimentConfig once and pass sections."""
    cfg_path = _cfg_path_for(experiment, exp_config_path)
    project_home = Path(__file__).resolve().parent.parent
    exp = config_loader.load_full_experiment_config(cfg_path, project_home, experiment)
    _run_loop(experiment, cfg_path, exp.prepare, exp.train, exp.sample, exp.shared)
