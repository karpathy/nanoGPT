from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import tomllib

from ml_playground.config import (
    ExperimentConfig,
    PreparerConfig,
    SamplerConfig,
    TrainerConfig,
)

# Module-level logger
logger = logging.getLogger(__name__)


def get_cfg_path(experiment: str, exp_config: Path | None) -> Path:
    """Return the path to an experiment's config.toml."""
    if exp_config:
        return exp_config
    return Path(__file__).resolve().parent / "experiments" / experiment / "config.toml"


def list_experiments_with_config(prefix: str = "") -> list[str]:
    """List experiment directory names under experiments/ that contain a config.toml.

    This is used by CLI autocompletion, centralizing the filesystem touchpoint for
    configuration discovery.
    """
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
                and p.name.startswith(prefix)
            ]
        )
    except OSError:
        return []


def read_toml_dict(path: Path) -> dict[str, Any]:
    """Reads a TOML file and returns a dictionary.

    Canonical loader: Path-only, reads UTF-8 text, uses tomllib.loads.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        # Include the filename in the error message for clearer diagnostics
        # and to satisfy tests that assert the offending filename is present.
        raise Exception(f"{path.name}: {e}")


# Centralized filesystem query helpers (single FS boundary policy)
def fs_path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except Exception:
        return False


def fs_is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except Exception:
        return False


def fs_is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except Exception:
        return False


def _default_config_path_from_root(project_root: Path) -> Path:
    """Compute the canonical default_config.toml from the project root."""
    return (
        project_root / "ml_playground" / "experiments" / "default_config.toml"
    ).resolve()


def deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merges two dictionaries. `override` wins."""
    out: dict[str, Any] = dict(base)
    for k, v in override.items():
        existing = out.get(k)
        if isinstance(existing, dict) and isinstance(v, dict):
            out[k] = deep_merge_dicts(existing, v)
        else:
            out[k] = v
    return out


def _load_and_merge_configs(
    config_path: Path, project_home: Path, experiment_name: str
) -> dict[str, Any]:
    """Load and merge configurations from default, experiment, and .ldres files."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw_exp = read_toml_dict(config_path)

    defaults_path = _default_config_path_from_root(project_home)
    defaults_raw = read_toml_dict(defaults_path) if defaults_path.exists() else {}

    ldres_config = (
        project_home
        / ".ldres"
        / "etc"
        / "ml_playground"
        / "experiments"
        / experiment_name
        / "config.toml"
    )
    ldres_raw = read_toml_dict(ldres_config) if ldres_config.exists() else {}

    # Merge order: defaults -> experiment config -> .ldres config
    merged = deep_merge_dicts(deepcopy(defaults_raw), deepcopy(raw_exp))
    return deep_merge_dicts(merged, deepcopy(ldres_raw))


def load_full_experiment_config(
    config_path: Path, project_home: Path, experiment_name: str
) -> ExperimentConfig:
    """Canonical loader for a full experiment configuration."""
    effective_config = _load_and_merge_configs(
        config_path, project_home, experiment_name
    )

    # Populated by the model validator
    effective_config.setdefault("shared", {})
    effective_config["shared"]["config_path"] = config_path
    effective_config["shared"]["project_home"] = project_home
    effective_config["shared"]["experiment"] = experiment_name

    return ExperimentConfig.model_validate(effective_config)


def load_train_config(config_path: Path) -> TrainerConfig:
    """Load config from a file path."""
    raw_exp = read_toml_dict(config_path)
    project_root = Path(__file__).resolve().parent.parent
    defaults_path = _default_config_path_from_root(project_root)
    defaults_raw = read_toml_dict(defaults_path) if defaults_path.exists() else {}

    raw_merged = deep_merge_dicts(deepcopy(defaults_raw), deepcopy(raw_exp))

    # The train config expects a 'train' section.
    train_data = raw_merged.get("train", {})

    # Pass config_path in context for path resolution
    context = {"config_path": config_path}
    cfg = TrainerConfig.model_validate(train_data, context=context)

    info = {"raw": raw_merged, "context": {"config_path": str(config_path)}}
    cfg.extras["provenance"] = info
    return cfg


def load_sample_config(config_path: Path) -> SamplerConfig:
    """Load config from a file path."""
    raw_exp = read_toml_dict(config_path)
    project_root = Path(__file__).resolve().parent.parent
    defaults_path = _default_config_path_from_root(project_root)
    defaults_raw = read_toml_dict(defaults_path) if defaults_path.exists() else {}

    raw_merged = deep_merge_dicts(deepcopy(defaults_raw), deepcopy(raw_exp))

    if "sample" not in raw_exp:
        raise ValueError("Config must contain a [sample] section")

    sample_data = raw_merged.get("sample", {})

    context = {"config_path": config_path}
    cfg = SamplerConfig.model_validate(sample_data, context=context)

    info = {"raw": raw_merged, "context": {"config_path": str(config_path)}}
    cfg.extras["provenance"] = info
    return cfg


def load_prepare_config(path: Path) -> PreparerConfig:
    """Public wrapper to load and validate preparer config."""
    raw_exp = read_toml_dict(path)
    project_root = Path(__file__).resolve().parent.parent
    defaults_path = _default_config_path_from_root(project_root)
    defaults_raw = read_toml_dict(defaults_path) if defaults_path.exists() else {}

    raw_merged = deep_merge_dicts(deepcopy(defaults_raw), deepcopy(raw_exp))

    if "prepare" not in raw_merged:
        raise ValueError("Config must contain a [prepare] section")

    prepare_data = raw_merged.get("prepare", {})

    context = {"config_path": path}
    cfg = PreparerConfig.model_validate(prepare_data, context=context)

    info = {"raw": raw_merged, "context": {"config_path": str(path)}}
    cfg.extras["provenance"] = info
    return cfg


# Strict mode: no override functions; configuration is TOML-only.
