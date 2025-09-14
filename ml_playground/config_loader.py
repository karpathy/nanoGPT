from __future__ import annotations

import logging
import tomllib
from pathlib import Path
from typing import Any
from copy import deepcopy


from ml_playground.config import (
    PreparerConfig,
    SamplerConfig,
    TrainerConfig,
    ExperimentConfig,
    SharedConfig,
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


def get_default_config_path(config_path: Path) -> Path:
    """Return the default_config.toml path adjacent to experiments/.

    Given an experiment config like .../experiments/<exp>/config.toml, the
    defaults live at .../default_config.toml.
    """
    # config_path: .../experiments/<exp>/config.toml
    # defaults live under the experiments directory: .../experiments/default_config.toml
    return (config_path.parent.parent / "default_config.toml").resolve()


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
        for k in ("dataset_dir", "raw_dir"):
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
        for key in ("dataset_dir", "raw_dir"):
            if (
                key in prep
                and isinstance(prep[key], str)
                and not prep[key].startswith("/")
            ):
                prep[key] = _resolve_path_if_relative(base, prep[key])
    # train.data.dataset_dir (keep runtime.out_dir as provided to preserve relative semantics)
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
    defaults_path = get_default_config_path(config_path)
    defaults_raw = {}
    if defaults_path.exists():
        try:
            defaults_raw = read_toml_dict(defaults_path)
        except Exception as e:
            raise Exception(f"default_config.toml: {e}")
    else:
        # Also support defaults placed as a sibling to the experiments/ directory
        alt_defaults = config_path.parent.parent.parent / "default_config.toml"
        if alt_defaults.exists():
            try:
                defaults_raw = read_toml_dict(alt_defaults)
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
    merged = _resolve_relative_paths(merged, config_path)
    effective = _coerce_known_paths_to_path(merged)
    # Enforce unknown top-level keys: if extras beyond allowed are present, trigger validation error
    allowed_top = {"prepare", "train", "sample"}
    extras = set(effective.keys()) - allowed_top
    if extras:
        # Will raise a ValidationError due to extra keys and missing required 'shared'
        ExperimentConfig.model_validate(effective)
    # Build section configs explicitly and attach SharedConfig
    prep_dict = dict(effective.get("prepare", {}))
    train_dict = dict(effective.get("train", {}))
    sample_dict = dict(effective.get("sample", {}))
    # Validate sections
    prepare = PreparerConfig.model_validate(prep_dict)
    train = TrainerConfig.model_validate(train_dict)
    sample = SamplerConfig.model_validate(sample_dict)
    # Construct SharedConfig from validated sections
    shared = SharedConfig(
        experiment=experiment_name,
        config_path=config_path,
        project_home=project_home,
        dataset_dir=train.data.dataset_dir,
        train_out_dir=train.runtime.out_dir,
        sample_out_dir=sample.runtime.out_dir,
    )
    return ExperimentConfig(prepare=prepare, train=train, sample=sample, shared=shared)


def load_train_config(config_path: Path) -> TrainerConfig:
    """Load config from a file path."""
    raw_exp = read_toml_dict(config_path)
    defaults_path = get_default_config_path(config_path)
    if defaults_path.exists():
        try:
            defaults_raw = read_toml_dict(defaults_path)
        except Exception as e:
            raise Exception(f"default_config.toml: {e}")
    else:
        # Also support defaults placed as a sibling to the experiments/ directory
        alt_defaults = config_path.parent.parent.parent / "default_config.toml"
        if alt_defaults.exists():
            try:
                defaults_raw = read_toml_dict(alt_defaults)
            except Exception as e:
                raise Exception(f"default_config.toml: {e}")
        else:
            defaults_raw = {}
    raw_merged = deep_merge_dicts(deepcopy(defaults_raw), deepcopy(raw_exp))
    eff = _coerce_known_paths_to_path(raw_merged)
    td = dict(eff.get("train", {}))
    # Resolve relative paths against config directory for partial loader
    base = config_path.parent
    try:
        data = td.get("data", {})
        if isinstance(data, dict):
            ds = data.get("dataset_dir")
            if isinstance(ds, Path) and not ds.is_absolute():
                data["dataset_dir"] = (base / ds).resolve()
        rt = td.get("runtime", {})
        if isinstance(rt, dict):
            od = rt.get("out_dir")
            if isinstance(od, Path) and not od.is_absolute():
                rt["out_dir"] = (base / od).resolve()
    except Exception:
        pass
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
    defaults_path = get_default_config_path(config_path)
    if defaults_path.exists():
        try:
            defaults_raw = read_toml_dict(defaults_path)
        except Exception as e:
            raise Exception(f"default_config.toml: {e}")
    else:
        alt_defaults = config_path.parent.parent.parent / "default_config.toml"
        if alt_defaults.exists():
            try:
                defaults_raw = read_toml_dict(alt_defaults)
            except Exception as e:
                raise Exception(f"default_config.toml: {e}")
        else:
            defaults_raw = {}
    raw_merged = deep_merge_dicts(deepcopy(defaults_raw), deepcopy(raw_exp))
    # Provide strict error when [sample] section is absent
    if "sample" not in raw_merged:
        raise ValueError("Config must contain a [sample] section")
    # No runtime_ref mechanics; strict SamplerConfig requires runtime
    eff = _coerce_known_paths_to_path(raw_merged)
    sd = dict(eff.get("sample", {}))
    # Resolve relative paths against config directory for partial loader
    base = config_path.parent
    try:
        rt = sd.get("runtime", {})
        if isinstance(rt, dict):
            od = rt.get("out_dir")
            if isinstance(od, Path) and not od.is_absolute():
                rt["out_dir"] = (base / od).resolve()
    except Exception:
        pass
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
    defaults_path = get_default_config_path(path)
    defaults_raw = read_toml_dict(defaults_path) if defaults_path.exists() else {}
    merged = deep_merge_dicts(deepcopy(defaults_raw), deepcopy(raw_exp))
    merged = _resolve_relative_paths(merged, path)
    if "prepare" not in merged or not isinstance(merged.get("prepare"), dict):
        raise ValueError("Config must contain a [prepare] section")
    eff = _coerce_known_paths_to_path(merged)
    prep_dict = dict(eff.get("prepare", {}))
    prep_dict["logger"] = logger
    cfg = PreparerConfig.model_validate(prep_dict)
    info = {"raw": merged, "context": {"config_path": str(path)}}
    cfg.extras["provenance"] = info
    return cfg


# Strict mode: no override functions; configuration is TOML-only.
