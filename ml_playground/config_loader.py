from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import tomllib
from pydantic_core import ValidationError

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


def _coerce_known_paths_to_path(eff: dict[str, Any]) -> dict[str, Any]:
    """Convert known path fields to Path objects to satisfy strict models."""
    out = dict(eff)
    # prepare paths
    prep = out.get("prepare")
    if isinstance(prep, dict):
        for k in ("raw_dir",):
            if k in prep and isinstance(prep[k], str):
                prep[k] = Path(prep[k])
    # train paths
    train = out.get("train")
    if isinstance(train, dict):
        data = train.get("data")
        if (
            isinstance(data, dict)
            and "dataset_dir" in data
            and isinstance(data["dataset_dir"], str)
        ):
            data["dataset_dir"] = Path(data["dataset_dir"])
        rt = train.get("runtime")
        if isinstance(rt, dict) and "out_dir" in rt and isinstance(rt["out_dir"], str):
            rt["out_dir"] = Path(rt["out_dir"])
    # sample paths
    sample = out.get("sample")
    if isinstance(sample, dict):
        rt = sample.get("runtime")
        if isinstance(rt, dict) and "out_dir" in rt and isinstance(rt["out_dir"], str):
            rt["out_dir"] = Path(rt["out_dir"])
    # shared paths
    shared = out.get("shared")
    if isinstance(shared, dict):
        for key in (
            "config_path",
            "project_home",
            "dataset_dir",
            "train_out_dir",
            "sample_out_dir",
        ):
            if key in shared and isinstance(shared[key], str):
                shared[key] = Path(shared[key])
    return out


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
        for key in ("raw_dir",):
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
        # Resolve train.runtime.out_dir for partial loader
        runtime = train.get("runtime")
        if (
            isinstance(runtime, dict)
            and "out_dir" in runtime
            and isinstance(runtime["out_dir"], str)
            and not runtime["out_dir"].startswith("/")
        ):
            runtime["out_dir"] = _resolve_path_if_relative(base, runtime["out_dir"])
    # sample.runtime.out_dir: do not resolve; preserve as provided
    sample = out.get("sample")
    if isinstance(sample, dict):
        pass
    return out


def load_full_experiment_config(
    config_path: Path, project_home: Path, experiment_name: str
) -> ExperimentConfig:
    """Canonical loader for a full experiment configuration.

    - Reads default_config.toml (if present), experiment config, and special .ldres config (if present).
    - Merges defaults -> experiment -> .ldres config (.ldres config overrides all).
    - Resolves known relative paths relative to the config file dir.
    - Validates the entire config into an ExperimentConfig (prepare, train, sample mandatory).
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw_exp = read_toml_dict(config_path)

    # Strict validation: require mandatory sections in experiment config
    required_sections = ["prepare", "train", "sample"]
    for section in required_sections:
        if section not in raw_exp:
            raise ValidationError.from_exception_data(
                "ExperimentConfig",
                [{"type": "missing", "loc": (section,), "input": raw_exp}],
            )

    # Validate train subsections
    if "train" in raw_exp and isinstance(raw_exp["train"], dict):
        train_required = ["model", "data", "optim", "schedule", "runtime"]
        for subsection in train_required:
            if subsection not in raw_exp["train"]:
                raise ValidationError.from_exception_data(
                    "TrainerConfig",
                    [
                        {
                            "type": "missing",
                            "loc": ("train", subsection),
                            "input": raw_exp["train"],
                        }
                    ],
                )

    # Validate sample subsections
    if "sample" in raw_exp and isinstance(raw_exp["sample"], dict):
        sample_required = ["runtime", "sample"]
        for subsection in sample_required:
            if subsection not in raw_exp["sample"]:
                raise ValidationError.from_exception_data(
                    "SamplerConfig",
                    [
                        {
                            "type": "missing",
                            "loc": ("sample", subsection),
                            "input": raw_exp["sample"],
                        }
                    ],
                )

    defaults_path = _default_config_path_from_root(project_home)
    defaults_raw = {}
    if defaults_path.exists():
        try:
            defaults_raw = read_toml_dict(defaults_path)
        except Exception as e:
            raise Exception(f"default_config.toml: {e}")

    # --- Check for special .ldres experiment config ---
    ldres_config = (
        project_home
        / ".ldres"
        / "etc"
        / "ml_playground"
        / "experiments"
        / experiment_name
        / "config.toml"
    )
    ldres_raw = {}
    if ldres_config.exists():
        try:
            ldres_raw = read_toml_dict(ldres_config)
        except Exception as e:
            raise Exception(f".ldres experiment config: {e}")
    # --- END ---

    # Merge order: defaults -> experiment config -> .ldres config
    merged = deep_merge_dicts(deepcopy(defaults_raw), deepcopy(raw_exp))
    merged = deep_merge_dicts(merged, deepcopy(ldres_raw))
    # Resolve relative strings and coerce known paths to Path before strict validation
    merged = _resolve_relative_paths(merged, config_path)
    effective = _coerce_known_paths_to_path(merged)

    # Clean up fields that have been moved to SharedConfig
    if "prepare" in effective and isinstance(effective["prepare"], dict):
        effective["prepare"].pop("dataset_dir", None)

    return ExperimentConfig(**effective)


def load_train_config(config_path: Path) -> TrainerConfig:
    """Load config from a file path."""
    raw_exp = read_toml_dict(config_path)
    # Derive project root from package location for partial loader
    project_root = Path(__file__).resolve().parent.parent
    defaults_path = _default_config_path_from_root(project_root)
    if defaults_path.exists():
        try:
            defaults_raw = read_toml_dict(defaults_path)
        except Exception as e:
            raise Exception(f"default_config.toml: {e}")
    else:
        defaults_raw = {}
    raw_merged = deep_merge_dicts(deepcopy(defaults_raw), deepcopy(raw_exp))
    # Resolve relative paths and coerce known paths to Path objects
    merged = _resolve_relative_paths(raw_merged, config_path)
    effective = _coerce_known_paths_to_path(merged)
    td = dict(effective.get("train", {}))
    # Populate required nested sections with defaults when omitted
    td.setdefault("model", {})
    td.setdefault("data", {})
    td.setdefault("optim", {})
    td.setdefault("schedule", {})
    td.setdefault("runtime", {})
    td["logger"] = logger
    cfg = TrainerConfig.model_validate(td)
    info = {"raw": raw_merged, "context": {"config_path": str(config_path)}}
    cfg.extras["provenance"] = info
    return cfg


def load_sample_config(config_path: Path) -> SamplerConfig:
    """Load config from a file path."""
    raw_exp = read_toml_dict(config_path)
    project_root = Path(__file__).resolve().parent.parent
    defaults_path = _default_config_path_from_root(project_root)
    if defaults_path.exists():
        try:
            defaults_raw = read_toml_dict(defaults_path)
        except Exception as e:
            raise Exception(f"default_config.toml: {e}")
    else:
        defaults_raw = {}
    # Strict: require [sample] section in the raw file itself (not provided by defaults)
    if "sample" not in raw_exp:
        raise ValueError("Config must contain a [sample] section")
    raw_merged = deep_merge_dicts(deepcopy(defaults_raw), deepcopy(raw_exp))
    # No runtime_ref mechanics; strict SamplerConfig requires runtime
    sd = dict(raw_merged.get("sample", {}))
    # Resolve relative out_dir against config directory for partial loader
    base = config_path.parent
    rt = sd.get("runtime", {})
    if isinstance(rt, dict):
        od = rt.get("out_dir")
        if isinstance(od, str) and not od.startswith("/"):
            rt["out_dir"] = (base / od).resolve()
        elif isinstance(od, Path) and not od.is_absolute():
            rt["out_dir"] = (base / od).resolve()
    # Ensure nested 'sample' subsection exists for SamplerConfig
    if "sample" not in sd or not isinstance(sd.get("sample"), dict):
        sd["sample"] = {}
    sd["logger"] = logger
    cfg = SamplerConfig.model_validate(sd)
    info = {"raw": raw_merged, "context": {"config_path": str(config_path)}}
    cfg.extras["provenance"] = info
    return cfg


def load_sample_config_from_raw(
    raw_exp: dict, defaults_raw: dict | None = None
) -> SamplerConfig:
    """Loads and validates the sampling config from raw dictionaries."""
    defaults_raw = defaults_raw or {}
    raw_merged = deep_merge_dicts(deepcopy(defaults_raw), deepcopy(raw_exp))
    eff = _coerce_known_paths_to_path(raw_merged)
    cfg = SamplerConfig.model_validate(eff.get("sample", {}))
    info = {"raw": raw_merged, "context": {}}
    cfg.extras["provenance"] = info
    return cfg


# runtime_ref support removed: configurations must specify explicit runtime under [sample]


def load_train_config_from_raw(raw_exp: dict, defaults_raw: dict) -> TrainerConfig:
    """Loads and validates the training config from raw dictionaries."""
    raw = deep_merge_dicts(defaults_raw, raw_exp)
    if "train" not in raw:
        raise ValueError("Config must contain a [train] section")

    # Use Pydantic for validation
    cfg = TrainerConfig.model_validate(raw["train"])
    return cfg


def load_prepare_config(path: Path) -> PreparerConfig:
    """Public wrapper to load and validate preparer config.

    Only parses the [prepare] section, merging defaults -> experiment and resolving paths.
    Does not require [train] or [sample] sections.
    """
    raw_exp = read_toml_dict(path)
    # Derive project root from package location for partial loader
    project_root = Path(__file__).resolve().parent.parent
    defaults_path = _default_config_path_from_root(project_root)
    defaults_raw = read_toml_dict(defaults_path) if defaults_path.exists() else {}
    merged = deep_merge_dicts(deepcopy(defaults_raw), deepcopy(raw_exp))
    if "prepare" not in merged or not isinstance(merged.get("prepare"), dict):
        raise ValueError("Config must contain a [prepare] section")
    prep_dict = dict(merged.get("prepare", {}))
    prep_dict["logger"] = logger
    cfg = PreparerConfig.model_validate(prep_dict)
    info = {"raw": merged, "context": {"config_path": str(path)}}
    cfg.extras["provenance"] = info
    return cfg


# Strict mode: no override functions; configuration is TOML-only.
