from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, TypedDict, cast

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
    """Compute the canonical configuration path for an experiment."""

    if exp_config:
        return exp_config
    return Path(__file__).resolve().parent / "experiments" / experiment / "config.toml"


def get_default_config_path(project_root: Path | None = None) -> Path:
    """Return the repository-wide default experiment configuration path."""

    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    return _default_config_path_from_root(project_root)


def list_experiments_with_config(prefix: str = "") -> list[str]:
    """Enumerate experiment directories that provide a configuration file.

    Args:
        prefix: Optional name prefix used by Typer auto-completion.

    Returns:
        Sorted list of experiment names that contain a `config.toml` file.
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


TomlMapping = Dict[str, Any]


class ExperimentPayload(TypedDict, total=False):
    shared: TomlMapping
    prepare: TomlMapping
    train: TomlMapping
    sample: TomlMapping


def _ensure_mapping(value: Any, context: str) -> TomlMapping:
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected mapping for {context}")
    return dict(value)


def read_toml_dict(path: Path) -> TomlMapping:
    """Read a TOML document from disk into a dictionary.

    Args:
        path: Absolute path to the TOML file.

    Returns:
        Parsed TOML content as a dictionary.

    Raises:
        FileNotFoundError: If the requested path does not exist.
        Exception: If TOML parsing fails, augmented with the filename.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        # Include the filename in the error message for clearer diagnostics
        # and to satisfy tests that assert the offending filename is present.
        raise Exception(f"{path.name}: {e}")
    if not isinstance(data, dict):
        raise TypeError(f"TOML root in {path} must be a mapping")
    return cast(TomlMapping, data)


def _default_config_path_from_root(project_root: Path) -> Path:
    """Compute the path to the repository-wide default configuration file."""
    return project_root / "ml_playground" / "experiments" / "default_config.toml"


def deep_merge_dicts(base: TomlMapping, override: TomlMapping) -> TomlMapping:
    """Recursively merge two dictionaries preferring values from ``override``."""
    out: TomlMapping = dict(base)
    for k, v in override.items():
        existing = out.get(k)
        if isinstance(existing, dict) and isinstance(v, dict):
            out[k] = deep_merge_dicts(existing, v)
        else:
            out[k] = deepcopy(v)
    return out


def _load_and_merge_configs(
    config_path: Path, project_home: Path, experiment_name: str
) -> ExperimentPayload:
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

    # Merge order with strict semantics:
    # - Only merge defaults into sections that are explicitly present in the experiment config
    # - Within a present section, only merge keys that are also present in the experiment section
    #   (do not introduce missing subsections from defaults).
    def _merge_present(
        dflt: Mapping[str, Any], exp: Mapping[str, Any], ctx: str
    ) -> TomlMapping:
        defaults_map = _ensure_mapping(dflt, f"defaults for {ctx}")
        exp_map = _ensure_mapping(exp, ctx)
        out: TomlMapping = {}
        for key, exp_val in exp_map.items():
            default_val = defaults_map.get(key)
            if isinstance(default_val, Mapping) and isinstance(exp_val, Mapping):
                out[key] = _merge_present(
                    default_val,
                    exp_val,
                    f"{ctx}.{key}" if ctx else key,
                )
            else:
                out[key] = deepcopy(exp_val)
        return out

    merged: TomlMapping = {}
    for k, v in raw_exp.items():
        dv = defaults_raw.get(k)
        if isinstance(v, Mapping) and isinstance(dv, Mapping):
            merged[k] = _merge_present(dv, v, f"experiment[{k}]")
        else:
            merged[k] = deepcopy(v)

    # Apply .ldres overrides last. This may add keys, but typical test scenarios do not rely on .ldres.
    merged_payload = deep_merge_dicts(merged, ldres_raw)
    return cast(ExperimentPayload, merged_payload)


def load_full_experiment_config(
    config_path: Path, project_home: Path, experiment_name: str
) -> ExperimentConfig:
    """Load and validate the fully merged experiment configuration."""

    effective_config = _load_and_merge_configs(
        config_path, project_home, experiment_name
    )

    shared = _ensure_mapping(effective_config.setdefault("shared", {}), "[shared]")
    shared["config_path"] = config_path
    shared["project_home"] = project_home
    shared["experiment"] = experiment_name
    effective_config["shared"] = shared

    return ExperimentConfig.model_validate(effective_config)


def load_train_config(config_path: Path) -> TrainerConfig:
    """Load the `[train]` section from a configuration file."""
    raw_exp = read_toml_dict(config_path)
    project_root = Path(__file__).resolve().parent.parent
    defaults_path = _default_config_path_from_root(project_root)
    defaults_raw = read_toml_dict(defaults_path) if defaults_path.exists() else {}

    raw_merged = deep_merge_dicts(deepcopy(defaults_raw), deepcopy(raw_exp))

    # The train config expects a 'train' section.
    train_data = _ensure_mapping(
        raw_merged.get("train", {}),
        "[train] section",
    )

    # Pass config_path in context for path resolution
    context = {"config_path": config_path}
    cfg = TrainerConfig.model_validate(train_data, context=context)

    info = {"raw": raw_merged, "context": {"config_path": str(config_path)}}
    cfg.extras["provenance"] = info
    return cfg


__all__ = [
    "get_cfg_path",
    "get_default_config_path",
    "load_full_experiment_config",
    "load_train_config",
    "load_sample_config",
    "load_prepare_config",
]


def load_sample_config(config_path: Path) -> SamplerConfig:
    """Load the `[sample]` section from a configuration file."""
    raw_exp = read_toml_dict(config_path)
    project_root = Path(__file__).resolve().parent.parent
    defaults_path = _default_config_path_from_root(project_root)
    defaults_raw = read_toml_dict(defaults_path) if defaults_path.exists() else {}

    raw_merged = deep_merge_dicts(deepcopy(defaults_raw), deepcopy(raw_exp))

    if "sample" not in raw_exp:
        raise ValueError("Config must contain a [sample] section")

    sample_data = _ensure_mapping(
        raw_merged.get("sample", {}),
        "[sample] section",
    )

    context = {"config_path": config_path}
    cfg = SamplerConfig.model_validate(sample_data, context=context)

    info = {"raw": raw_merged, "context": {"config_path": str(config_path)}}
    cfg.extras["provenance"] = info
    return cfg


def load_prepare_config(path: Path) -> PreparerConfig:
    """Load the `[prepare]` section from a configuration file."""
    raw_exp = read_toml_dict(path)
    project_root = Path(__file__).resolve().parent.parent
    defaults_path = _default_config_path_from_root(project_root)
    defaults_raw = read_toml_dict(defaults_path) if defaults_path.exists() else {}

    raw_merged = deep_merge_dicts(deepcopy(defaults_raw), deepcopy(raw_exp))

    if "prepare" not in raw_merged:
        raise ValueError("Config must contain a [prepare] section")

    prepare_data = _ensure_mapping(
        raw_merged.get("prepare", {}),
        "[prepare] section",
    )

    context = {"config_path": path}
    cfg = PreparerConfig.model_validate(prepare_data, context=context)

    info = {"raw": raw_merged, "context": {"config_path": str(path)}}
    cfg.extras["provenance"] = info
    return cfg


# Strict mode: no override functions; configuration is TOML-only.
