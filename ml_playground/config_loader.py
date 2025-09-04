from __future__ import annotations

import logging
import tomllib
from pathlib import Path
from typing import Any
from copy import deepcopy

from pydantic import BaseModel

from ml_playground.config import (
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


def read_toml_dict(path: Path) -> dict[str, Any]:
    """Reads a TOML file and returns a dictionary."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("rb") as f:
        return tomllib.load(f)


def get_default_config_path(config_path: Path) -> Path:
    """Return the default_config.toml path adjacent to experiments/.

    Given an experiment config like .../experiments/<exp>/config.toml, the
    defaults live at .../default_config.toml.
    """
    return (config_path.parent.parent / "default_config.toml").resolve()


def deep_merge_dicts(base: dict, override: dict) -> dict:
    """Recursively merges two dictionaries. `override` wins."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(out.get(k), dict) and isinstance(v, dict):
            out[k] = deep_merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


def load_train_config(config_path: Path) -> TrainerConfig:
    """Load config from a file path."""
    raw_exp = read_toml_dict(config_path)
    defaults_path = get_default_config_path(config_path)
    defaults_raw = read_toml_dict(defaults_path) if defaults_path.exists() else {}
    raw_merged = deep_merge_dicts(deepcopy(defaults_raw), deepcopy(raw_exp))
    cfg = TrainerConfig(**raw_merged.get("train", {}))
    info = {"raw": raw_merged, "context": {"config_path": str(config_path)}}
    cfg.extras["provenance"] = info
    return cfg


def load_sample_config(config_path: Path) -> SamplerConfig:
    """Load config from a file path."""
    raw_exp = read_toml_dict(config_path)
    defaults_path = get_default_config_path(config_path)
    defaults_raw = read_toml_dict(defaults_path) if defaults_path.exists() else {}
    raw_merged = deep_merge_dicts(deepcopy(defaults_raw), deepcopy(raw_exp))
    # Provide strict error when [sample] section is absent
    if "sample" not in raw_merged:
        raise ValueError("Config must contain a [sample] section")
    # Apply runtime_ref on raw before validation
    raw_effective = _apply_runtime_ref_to_raw(raw_merged)
    cfg = SamplerConfig.model_validate(raw_effective.get("sample", {}))
    info = {"raw": raw_merged, "context": {"config_path": str(config_path)}}
    cfg.extras["provenance"] = info
    return cfg


def load_sample_config_from_raw(
    raw_exp: dict, defaults_raw: dict | None = None
) -> SamplerConfig:
    """Loads and validates the sampling config from raw dictionaries."""
    defaults_raw = defaults_raw or {}
    raw_merged = deep_merge_dicts(deepcopy(defaults_raw), deepcopy(raw_exp))
    # Apply runtime_ref on raw before validation
    raw_effective = _apply_runtime_ref_to_raw(raw_merged)
    cfg = SamplerConfig.model_validate(raw_effective.get("sample", {}))
    info = {"raw": raw_merged, "context": {}}
    cfg.extras["provenance"] = info
    return cfg


def _apply_runtime_ref_to_raw(raw_exp: dict) -> dict:
    """Apply supported runtime_ref merges on raw dict and return a new dict.

    Strict: only supports 'train.runtime'. Does not perform any path rewriting.
    """
    out = deepcopy(raw_exp)
    sample_cfg_raw = deepcopy(out.get("sample", {}))
    if sample_cfg_raw.get("runtime_ref") == "train.runtime":
        if "train" in out and "runtime" in out["train"]:
            train_runtime = deepcopy(out["train"]["runtime"])
            sample_rt = deepcopy(sample_cfg_raw.get("runtime", {}))
            sample_cfg_raw["runtime"] = deep_merge_dicts(train_runtime, sample_rt)
    out["sample"] = sample_cfg_raw
    return out


def load_train_config_from_raw(raw_exp: dict, defaults_raw: dict) -> TrainerConfig:
    """Loads and validates the training config from raw dictionaries."""
    raw = deep_merge_dicts(defaults_raw, raw_exp)
    if "train" not in raw:
        raise ValueError("Config must contain a [train] section")

    # Use Pydantic for validation
    cfg = TrainerConfig.model_validate(raw["train"])
    return cfg


def load_prepare_config(path: Path) -> PreparerConfig:
    """Public wrapper to load and validate preparer config."""
    raw_exp = read_toml_dict(path)
    if "prepare" not in raw_exp:
        raise ValueError("Config must contain a [prepare] section")

    # Use Pydantic for validation
    cfg = PreparerConfig.model_validate(raw_exp["prepare"])
    return cfg


# Strict mode: no override functions; configuration is TOML-only.
