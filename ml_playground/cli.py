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
    root = Path(__file__).resolve().parent / "experiments"
    if not root.exists():
        return []

    try:
        return sorted(
            [
                p.name
                for p in root.iterdir()
                if p.is_dir()
                and (p / "config.toml").exists()
                and p.name.startswith(incomplete)
            ]
        )
    except OSError:
        # Autocomplete is a non-critical UX nicety; return empty on FS errors
        return []


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


def _resolve_and_load_configs(exp_name: str, exp_cfg: Path | None):
    """Resolve config path and return raw TOML dicts.

    Exists to be monkeypatched in tests.
    """
    cfg_path = _cfg_path_for(exp_name, exp_cfg)
    raw = _read_toml(cfg_path)
    return cfg_path, raw, {}


def _cfg_path_for(experiment: str, exp_config: Path | None) -> Path:
    """Resolve the path to the experiment config TOML strictly.

    - If exp_config is provided, use it.
    - Else, use experiments/<experiment>/config.toml under this package's experiments dir.
    """
    if exp_config is not None:
        return exp_config
    exp_dir = Path(__file__).resolve().parent / "experiments" / experiment
    return exp_dir / "config.toml"


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


def _load_merged_raw(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load default_config.toml (if exists) and experiment config, return (defaults, merged)."""
    # default_config.toml expected at the parent of experiments/
    defaults_path = config_path.parents[2] / "default_config.toml"
    defaults_err: Exception | None = None
    if defaults_path.exists():
        try:
            defaults = _read_toml(defaults_path)
        except Exception as e:
            # Defer raising; experiment config error should take precedence if present
            defaults = {}
            defaults_err = ValueError(f"default_config.toml: {e}")
    else:
        defaults = {}

    # Read experiment config; if it fails, prefer this error mentioning the cfg path
    try:
        raw = _read_toml(config_path)
    except Exception as e:
        raise ValueError(f"{config_path.name}: {e}")

    # If experiment config succeeded but defaults had an error, raise it now
    if defaults_err is not None:
        raise defaults_err
    merged = _deep_merge(defaults, raw)
    return defaults, merged


def _resolve_path_if_relative(base: Path, path_str: str) -> str:
    """Resolve a path string relative to base if it's not absolute."""
    if path_str.startswith("/"):
        return path_str
    return str((base / path_str).resolve())


def _resolve_relative_paths(merged: dict[str, Any], cfg_path: Path) -> dict[str, Any]:
    """Resolve known relative paths relative to the config file directory."""
    base = cfg_path.parent
    out = dict(merged)
    # prepare.dataset_dir, prepare.raw_dir
    prep = out.get("prepare")
    if isinstance(prep, dict):
        for key in ("dataset_dir", "raw_dir"):
            if (
                key in prep
                and isinstance(prep[key], str)
                and not prep[key].startswith("/")
            ):
                prep[key] = _resolve_path_if_relative(base, prep[key])
    # train.data.dataset_dir and train.runtime.out_dir
    train = out.get("train")
    if isinstance(train, dict):
        data = train.get("data")
        if (
            isinstance(data, dict)
            and "dataset_dir" in data
            and isinstance(data["dataset_dir"], str)
            and not data["dataset_dir"].startswith("/")
        ):
            data["dataset_dir"] = _resolve_path_if_relative(base, data["dataset_dir"])
        rt = train.get("runtime")
        if (
            isinstance(rt, dict)
            and "out_dir" in rt
            and isinstance(rt["out_dir"], str)
            and not rt["out_dir"].startswith("/")
        ):
            rt["out_dir"] = _resolve_path_if_relative(base, rt["out_dir"])
    # sample.runtime.out_dir (if provided directly)
    sample = out.get("sample")
    if isinstance(sample, dict):
        rt = sample.get("runtime")
        if (
            isinstance(rt, dict)
            and "out_dir" in rt
            and isinstance(rt["out_dir"], str)
            and not rt["out_dir"].startswith("/")
        ):
            rt["out_dir"] = _resolve_path_if_relative(base, rt["out_dir"])
    return out


def _clean_runtime_config(config_dict: dict, allowed_fields: set) -> dict:
    """Clean runtime configuration by removing unknown fields."""
    rt = config_dict.get("runtime")
    if isinstance(rt, dict):
        config_dict["runtime"] = {k: v for k, v in rt.items() if k in allowed_fields}
    return config_dict


def _load_effective_config(
    experiment: str, exp_config: Path | None, config_type: str
) -> tuple[Path, Any]:
    """Generic function to load and validate configuration strictly from TOML."""
    config_path = _cfg_path_for(experiment, exp_config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    _, merged = _load_merged_raw(config_path)
    merged = _resolve_relative_paths(merged, config_path)

    if config_type == "prepare":
        config_dict = dict(merged.get("prepare", {}))
        config_dict["logger"] = logging.getLogger(__name__)
        cfg = PreparerConfig.model_validate(config_dict)
    elif config_type == "train":
        config_dict = dict(merged.get("train", {}))
        # Ensure required nested sections exist; allow defaults when omitted
        if "model" not in config_dict:
            config_dict["model"] = {}
        if "data" not in config_dict:
            config_dict["data"] = {}
        if "optim" not in config_dict:
            config_dict["optim"] = {}
        if "schedule" not in config_dict:
            config_dict["schedule"] = {}
        if "runtime" not in config_dict:
            config_dict["runtime"] = {}
        # Coerce paths to Path objects for strict schema
        from pathlib import Path as _Path

        if isinstance(config_dict.get("data", {}).get("dataset_dir"), str):
            config_dict["data"]["dataset_dir"] = _Path(
                config_dict["data"]["dataset_dir"]
            )  # type: ignore[index]
        if isinstance(config_dict.get("runtime", {}).get("out_dir"), str):
            config_dict["runtime"]["out_dir"] = _Path(config_dict["runtime"]["out_dir"])  # type: ignore[index]
        config_dict["logger"] = logging.getLogger(__name__)
        cfg = TrainerConfig.model_validate(config_dict)
    elif config_type == "sample":
        config_dict = dict(merged.get("sample", {}))
        if not isinstance(config_dict, dict):
            raise ValueError("Missing [sample] section in config")
        if "sample" not in config_dict or not isinstance(
            config_dict.get("sample"), dict
        ):
            config_dict = dict(config_dict)
            config_dict["sample"] = {}
        # Coerce paths to Path objects for strict schema
        from pathlib import Path as _Path

        if isinstance(config_dict.get("runtime", {}).get("out_dir"), str):
            config_dict.setdefault("runtime", {})
            config_dict["runtime"]["out_dir"] = _Path(config_dict["runtime"]["out_dir"])  # type: ignore[index]
        config_dict["logger"] = logging.getLogger(__name__)
        cfg = SamplerConfig.model_validate(config_dict)
    else:
        raise ValueError(f"Unknown config type: {config_type}")

    return config_path, cfg


def def_load_effective_prepare(
    experiment: str, exp_config: Path | None
) -> tuple[Path, PreparerConfig]:
    """Load and validate preparer configuration strictly from TOML."""
    return _load_effective_config(experiment, exp_config, "prepare")


def def_load_effective_train(
    experiment: str, exp_config: Path | None
) -> tuple[Path, TrainerConfig]:
    """Load and validate training configuration strictly from TOML."""
    return _load_effective_config(experiment, exp_config, "train")


def def_load_effective_sample(
    experiment: str, exp_config: Path | None
) -> tuple[Path, SamplerConfig]:
    """Load and validate sampling configuration strictly from TOML.

    Supports schema-level runtime_ref to inherit train.runtime and resolves paths relative to the config.
    """
    return _load_effective_config(experiment, exp_config, "sample")


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
        if out_dir is None:
            logger.info(f"[{tag}] out_dir: <not set>")
        elif isinstance(out_dir, Path):
            if out_dir.exists():
                try:
                    contents = sorted([p.name for p in out_dir.iterdir()])
                    logger.info(f"[{tag}] out_dir (exists): {out_dir}")
                    logger.info(f"[{tag}]   Contents: {contents}")
                except Exception:
                    # If listing fails, still indicate existence
                    logger.info(f"[{tag}] out_dir (exists): {out_dir}")
            else:
                logger.info(f"[{tag}] out_dir (missing): {out_dir}")

        # Handle top-level dataset_dir or nested data.dataset_dir
        ds_dir = getattr(cfg, "dataset_dir", None)
        if ds_dir is None and hasattr(cfg, "data"):
            ds_dir = getattr(cfg.data, "dataset_dir", None)
        if ds_dir is None:
            logger.info(f"[{tag}] dataset_dir: <not set>")
        elif isinstance(ds_dir, Path):
            if ds_dir.exists():
                try:
                    contents = sorted([p.name for p in ds_dir.iterdir()])
                    logger.info(f"[{tag}] dataset_dir (exists): {ds_dir}")
                    logger.info(f"[{tag}]   Contents: {contents}")
                except Exception:
                    logger.info(f"[{tag}] dataset_dir (exists): {ds_dir}")
            else:
                logger.info(f"[{tag}] dataset_dir (missing): {ds_dir}")
    except Exception:
        # Never fail due to logging
        pass


def _run_prepare(
    experiment: str,
    prepare_cfg: PreparerConfig,
    config_path: Path,
) -> None:
    """Run the full prepare flow for an experiment."""
    print(f"---\nRunning preparer for experiment: {experiment}")
    preparer = prepare_mod.make_preparer(prepare_cfg)
    preparer()
    print(f"Preparer for {experiment} finished.\n---")


def _run_train(
    experiment: str,
    train_cfg: TrainerConfig,
    config_path: Path,
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
    trainer_mod.train(train_cfg)
    print(f"Trainer for {experiment} finished.")
    _log_command_status("post-train", train_cfg.runtime)
    print("---")


def _run_sample(
    experiment: str,
    sample_cfg: SamplerConfig,
    config_path: Path,
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
    sampler_mod.sample(sample_cfg)
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
) -> None:
    """Run the full prepare->train->sample loop for an experiment."""
    # If dataset artifacts already exist, skip prepare to allow env overrides-driven loops
    # Determine if prepare can be skipped by checking data artifacts from DataConfig
    skip_prepare = False
    try:
        data_cfg = train_cfg.data
        req_paths = [data_cfg.train_path, data_cfg.val_path]
        # meta is optional
        if data_cfg.meta_path is not None:
            req_paths.append(data_cfg.meta_path)
        skip_prepare = all(p.exists() for p in req_paths)
    except Exception:
        skip_prepare = False
    if not skip_prepare:
        _run_prepare(experiment, prepare_cfg, config_path)
    _run_train(experiment, train_cfg, config_path)
    _run_sample(experiment, sample_cfg, config_path)


# --- CLI definition --------------------------------------------------------


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


@app.command(name="prepare")
def prepare_command(
    ctx: typer.Context,
    experiment: Annotated[
        str,
        typer.Argument(
            help="Experiment name (directory in ml_playground/experiments)",
            autocompletion=_complete_experiments,
        ),
    ],
) -> None:
    """Prepare data for an experiment."""
    exp_config_path = _extract_exp_config(ctx)
    run_or_exit(
        lambda: _run_prepare_cmd(experiment, exp_config_path),
        keyboard_interrupt_msg="\nPreparation cancelled.",
    )


@app.command(name="train")
def train_command(
    ctx: typer.Context,
    experiment: Annotated[
        str,
        typer.Argument(
            help="Experiment name (directory in ml_playground/experiments)",
            autocompletion=_complete_experiments,
        ),
    ],
) -> None:
    """Train a model for an experiment."""
    exp_config_path = _extract_exp_config(ctx)
    run_or_exit(
        lambda: _run_train_cmd(experiment, exp_config_path),
        keyboard_interrupt_msg="\nTraining cancelled.",
    )


@app.command(name="sample")
def sample_command(
    ctx: typer.Context,
    experiment: Annotated[
        str,
        typer.Argument(
            help="Experiment name (directory in ml_playground/experiments)",
            autocompletion=_complete_experiments,
        ),
    ],
) -> None:
    """Sample from a trained model."""
    exp_config_path = _extract_exp_config(ctx)
    run_or_exit(
        lambda: _run_sample_cmd(experiment, exp_config_path),
        keyboard_interrupt_msg="\nSampling cancelled.",
    )


@app.command(name="analyze")
def analyze_command(
    ctx: typer.Context,
    experiment: Annotated[
        str,
        typer.Argument(
            help="Experiment name (directory in ml_playground/experiments)",
            autocompletion=_complete_experiments,
        ),
    ],
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
    experiment: Annotated[
        str,
        typer.Argument(
            help="Experiment name (directory in ml_playground/experiments)",
            autocompletion=_complete_experiments,
        ),
    ],
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
    exp = config_loader.load_full_experiment_config(cfg_path)
    _run_prepare(experiment, exp.prepare, cfg_path)


def _run_train_cmd(experiment: str, exp_config_path: Path | None) -> None:
    """Run train command: load full ExperimentConfig once and pass section."""
    cfg_path = _cfg_path_for(experiment, exp_config_path)
    exp = config_loader.load_full_experiment_config(cfg_path)
    _run_train(experiment, exp.train, cfg_path)


def _run_sample_cmd(experiment: str, exp_config_path: Path | None) -> None:
    """Run sample command: load full ExperimentConfig once and pass section."""
    cfg_path = _cfg_path_for(experiment, exp_config_path)
    exp = config_loader.load_full_experiment_config(cfg_path)
    _run_sample(experiment, exp.sample, cfg_path)


def _run_loop_cmd(experiment: str, exp_config_path: Path | None) -> None:
    """Run loop command: load full ExperimentConfig once and pass sections."""
    cfg_path = _cfg_path_for(experiment, exp_config_path)
    exp = config_loader.load_full_experiment_config(cfg_path)
    _run_loop(experiment, cfg_path, exp.prepare, exp.train, exp.sample)
