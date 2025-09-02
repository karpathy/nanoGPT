from __future__ import annotations

import typer
import click
from typer.main import get_command
import importlib
import json
import os
import shutil
import tomllib
import logging
from pathlib import Path
from typing import Any, Annotated, cast, Union

from ml_playground.config import (
    TrainerConfig,
    SamplerConfig,
    DataConfig,
    RuntimeConfig,
    AppConfig,
)
from ml_playground.prepare import PreparerConfig

from ml_playground.experiments.protocol import (
    Preparer as ExpPreparer,
    Trainer as ExpTrainer,
    Sampler as ExpSampler,
)

# Type aliases for better typing
TomlData = dict[str, Any]
PydanticObj = object
ConfigModel = Union[TrainerConfig, SamplerConfig]

# Module-level logger
logger = logging.getLogger(__name__)


# --- Typer helpers ---------------------------------------------------------
def _experiments_root() -> Path:
    """Return the root folder that contains experiment directories."""
    return Path(__file__).resolve().parent / "experiments"


def _complete_experiments(ctx: typer.Context, incomplete: str) -> list[str]:
    """Auto-complete experiment names based on directories with a config.toml."""
    root = _experiments_root()
    if not root.exists():
        return []

    try:
        # Only catch exceptions from directory listing operations
        return sorted(
            [
                p.name
                for p in root.iterdir()
                if p.is_dir()
                and (p / "config.toml").exists()
                and p.name.startswith(incomplete)
            ]
        )
    except Exception:
        # Only catch filesystem-related errors during directory listing
        return []


def _read_toml_dict(path: Path) -> TomlData:
    if not isinstance(path, Path):
        path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("rb") as f:
        raw: TomlData = tomllib.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Config at {path} must be a TOML table/object")
    return raw


def _load_prepare_config_from_raw(
    raw_exp: TomlData, defaults_raw: TomlData
) -> PreparerConfig:
    # Merge defaults' [prepare] under experiment overrides, if present
    raw = dict(raw_exp)
    if isinstance(defaults_raw, dict):
        d_prep_obj = defaults_raw.get("prepare")
        if isinstance(d_prep_obj, dict):
            raw = _deep_merge_dicts({"prepare": d_prep_obj}, raw)

    prep_tbl = raw.get("prepare")
    if not isinstance(prep_tbl, dict):
        # Prepare block is optional for generic flows; provide empty typed config
        return PreparerConfig()

    # Strictness: only known keys + extras allowed in [prepare]
    allowed_keys = {
        "dataset_dir",
        "raw_dir",
        "add_structure_tokens",
        "doc_separator",
        "extras",
    }
    unknown_prepare = set(prep_tbl.keys()) - allowed_keys
    if unknown_prepare:
        raise ValueError("Unknown key(s) in [prepare] (outside extras)")

    # Coerce known fields
    dataset_dir = prep_tbl.get("dataset_dir")
    raw_dir = prep_tbl.get("raw_dir")
    extras_obj = prep_tbl.get("extras")
    extras: TomlData = {}
    if extras_obj is not None:
        if not isinstance(extras_obj, dict):
            raise ValueError("[prepare.extras] must be a table/object")
        extras = dict(extras_obj)

    return PreparerConfig(
        dataset_dir=Path(dataset_dir) if isinstance(dataset_dir, (str, Path)) else None,
        raw_dir=Path(raw_dir) if isinstance(raw_dir, (str, Path)) else None,
        add_structure_tokens=bool(prep_tbl.get("add_structure_tokens"))
        if "add_structure_tokens" in prep_tbl
        else None,
        doc_separator=str(prep_tbl.get("doc_separator"))
        if "doc_separator" in prep_tbl
        else None,
        extras=extras,
    )


def _load_train_config_from_raw(
    raw_exp: TomlData, defaults_raw: TomlData
) -> TrainerConfig:
    # Merge defaults provided by caller; experiment overrides defaults.
    raw = dict(raw_exp)
    if isinstance(defaults_raw, dict):
        d_train_obj = defaults_raw.get("train")
        if isinstance(d_train_obj, dict):
            raw = _deep_merge_dicts({"train": d_train_obj}, raw)

    train_tbl = raw.get("train")
    if not isinstance(train_tbl, dict):
        # Keep legacy top-level message used by CLI tests
        raise Exception("Config must contain [train] block")

    # Required subsections
    for sec in ("model", "data", "optim", "schedule", "runtime"):
        if not isinstance(train_tbl.get(sec), dict):
            raise ValueError(f"Missing required section [{sec}]")

    # Unknown keys in [train.data]
    allowed_data_keys = set(DataConfig.model_fields.keys())
    unknown = set(train_tbl["data"].keys()) - allowed_data_keys
    if unknown:
        raise ValueError("Unknown key(s) in [train.data]")

    # Strictness: Only allowed top-level keys + extras in [train]
    allowed_sections = {"model", "data", "optim", "schedule", "runtime"}
    unknown_top = set(train_tbl.keys()) - (allowed_sections | {"extras"})
    if unknown_top:
        raise ValueError("Unknown key(s) in [train] (outside extras)")

    # Resolve and prune unsupported sections/keys (no path rewriting; use as configured)
    d = {k: train_tbl[k] for k in allowed_sections}

    # Extract extras (dict) if present
    extras_obj = train_tbl.get("extras")
    if extras_obj is not None:
        if not isinstance(extras_obj, dict):
            raise ValueError("[train.extras] must be a table/object")
        d["extras"] = dict(extras_obj)

    # No path heuristics: do not rewrite dataset_dir or out_dir; use as configured.
    # [train.runtime]: drop unknown keys (keep only keys known to RuntimeConfig)
    allowed_runtime_keys = set(RuntimeConfig.model_fields.keys())
    d_runtime_all = dict(d["runtime"])
    d_runtime = {k: v for k, v in d_runtime_all.items() if k in allowed_runtime_keys}
    d["runtime"] = d_runtime

    # Centralized provenance logging
    _tmp_exp_train = raw_exp.get("train")
    exp_train_tbl: TomlData = _tmp_exp_train if isinstance(_tmp_exp_train, dict) else {}
    defaults_dict: TomlData = defaults_raw if isinstance(defaults_raw, dict) else {}
    _tmp_train = defaults_dict.get("train")
    defaults_train_tbl: TomlData = _tmp_train if isinstance(_tmp_train, dict) else {}

    # Initial provenance: default vs experiment
    prov: dict[str, dict[str, str]] = {
        "model": {},
        "data": {},
        "optim": {},
        "schedule": {},
        "runtime": {},
    }
    for sec in ("model", "data", "optim", "schedule", "runtime"):
        _dflt_sec = defaults_train_tbl.get(sec)
        sec_defaults = _dflt_sec if isinstance(_dflt_sec, dict) else {}
        _exp_sec = exp_train_tbl.get(sec)
        sec_exp = _exp_sec if isinstance(_exp_sec, dict) else {}
        merged_sec = d.get(sec, {})
        if isinstance(merged_sec, dict):
            for k in merged_sec.keys():
                if k in sec_exp:
                    prov[sec][k] = "override"  # set by experiment
                elif k in sec_defaults:
                    prov[sec][k] = "default"
                else:
                    prov[sec][k] = (
                        "override"  # conservative: present only in experiment
                    )

    # Environment overrides are not supported: only default + experiment TOMLs are used.

    # Strict: do not rewrite any paths; use them exactly as configured.
    # [train.runtime]: drop unknown keys (keep only keys known to RuntimeConfig)
    allowed_runtime_keys = set(RuntimeConfig.model_fields.keys())
    d_runtime_all = dict(d["runtime"])
    d_runtime = {k: v for k, v in d_runtime_all.items() if k in allowed_runtime_keys}
    d["runtime"] = d_runtime

    # Provenance log at startup: print effective config with markers
    # Format: [config] train.<section>.<key> = <value> (default|override)
    try:
        print("[config] Effective training configuration (source: default/override):")
        for sec in ("model", "data", "optim", "schedule", "runtime"):
            sec_map = d.get(sec, {})
            if not isinstance(sec_map, dict):
                continue
            for k in sorted(sec_map.keys()):
                source = prov.get(sec, {}).get(k, "override")
                print(f"[config] train.{sec}.{k} = {sec_map[k]!r} ({source})")
    except Exception:
        # Never fail due to logging
        pass

    try:
        return TrainerConfig.model_validate(d)
    except Exception as e:
        # Normalize to ValueError for test expectations
        raise ValueError(str(e))


def _load_sample_config_from_raw(
    raw_exp: TomlData, defaults_raw: TomlData
) -> SamplerConfig:
    """Strict loader for sample section with minimal checks from preloaded dict.

    - Ensures required subsections exist ([sample] with either [runtime] or runtime_ref, plus [sample.sample]).
    - Enforces unknown-key rejection outside [sample.extras].
    - Supports schema-level reference `runtime_ref = "train.runtime"` with override merge.
    - Delegates detailed value validation to Pydantic models.
    """
    # Deep-merge defaults provided by caller (defaults under experiment overrides)
    raw = dict(raw_exp)
    if isinstance(defaults_raw, dict):
        d_sample_obj = defaults_raw.get("sample")
        if isinstance(d_sample_obj, dict):
            raw = _deep_merge_dicts({"sample": d_sample_obj}, raw)

    sample_tbl = raw.get("sample")
    if not isinstance(sample_tbl, dict):
        raise Exception("Config must contain [sample] block")

    # Strictness: only runtime, runtime_ref, sample, extras allowed at this level
    allowed_top = {"runtime", "runtime_ref", "sample", "extras"}
    unknown_top = set(sample_tbl.keys()) - allowed_top
    if unknown_top:
        raise ValueError("Unknown key(s) in [sample] (outside extras)")

    # Require [sample.sample]
    if not isinstance(sample_tbl.get("sample"), dict):
        raise ValueError("Missing required section [sample]")

    runtime_tbl = sample_tbl.get("runtime")
    runtime_ref = sample_tbl.get("runtime_ref")
    if not isinstance(runtime_tbl, dict) and not isinstance(runtime_ref, str):
        raise ValueError(
            "Sampler requires either [sample.runtime] or sample.runtime_ref"
        )

    # Allowed keys pruning for runtime/sample according to models
    allowed_runtime_keys = set(RuntimeConfig.model_fields.keys())

    # Build merged dict and provenance
    d: dict[str, Any] = {"sample": dict(sample_tbl["sample"])}

    # Extract extras
    extras_obj = sample_tbl.get("extras")
    if extras_obj is not None:
        if not isinstance(extras_obj, dict):
            raise ValueError("[sample.extras] must be a table/object")
        d["extras"] = dict(extras_obj)

    # Prepare for centralized provenance logging
    _tmp_exp_sample = raw_exp.get("sample")
    exp_sample_tbl: TomlData = (
        _tmp_exp_sample if isinstance(_tmp_exp_sample, dict) else {}
    )
    defaults_dict: TomlData = defaults_raw if isinstance(defaults_raw, dict) else {}
    _tmp_sample = defaults_dict.get("sample")
    defaults_sample_tbl: TomlData = _tmp_sample if isinstance(_tmp_sample, dict) else {}

    # Resolve runtime via ref merge if requested
    if isinstance(runtime_ref, str):
        if runtime_ref != "train.runtime":
            raise ValueError("Unsupported sample.runtime_ref; allowed: 'train.runtime'")
        # Merge defaults.train and exp.train, then take .runtime
        _defaults_train = defaults_dict.get("train")
        defaults_train_tbl: TomlData = (
            cast(TomlData, _defaults_train) if isinstance(_defaults_train, dict) else {}
        )
        _exp_train = raw_exp.get("train")
        exp_train_tbl: TomlData = (
            cast(TomlData, _exp_train) if isinstance(_exp_train, dict) else {}
        )
        merged_train_tbl = _deep_merge_dicts(defaults_train_tbl, exp_train_tbl)
        base_rt = merged_train_tbl.get("runtime")
        if not isinstance(base_rt, dict):
            raise ValueError(
                "sample.runtime_ref points to 'train.runtime' but [train.runtime] is missing"
            )
        resolved_rt = {k: v for k, v in base_rt.items() if k in allowed_runtime_keys}
        if isinstance(runtime_tbl, dict):
            # sample.runtime overrides
            overrides = {
                k: v for k, v in runtime_tbl.items() if k in allowed_runtime_keys
            }
            resolved_rt.update(overrides)
        d["runtime"] = resolved_rt
    else:
        # Use provided sample.runtime directly
        d["runtime"] = {
            k: v
            for k, v in (runtime_tbl if isinstance(runtime_tbl, dict) else {}).items()
            if k in allowed_runtime_keys
        }

    # Centralized provenance logging
    _log_config_provenance(
        "sample", d, exp_sample_tbl, defaults_sample_tbl, ("runtime", "sample")
    )

    try:
        return SamplerConfig.model_validate(d)
    except Exception as e:
        raise ValueError(str(e))


def _deep_merge_dicts(base: TomlData, override: TomlData) -> TomlData:
    """Recursively merge override into base (override wins).
    Note: dict union (|, |=) and update() are shallow; this handles nested dicts.
    """
    out = dict(base)
    for k, v in override.items():
        bv = out.get(k)
        if isinstance(bv, dict) and isinstance(v, dict):
            out[k] = _deep_merge_dicts(bv, v)
        else:
            out[k] = v
    return out


def _pydantic_dump(obj: PydanticObj) -> TomlData:
    """Best-effort dump of a Pydantic v2 model or nested structure to a plain dict."""
    try:
        # Pydantic v2
        return cast("TomlData", obj).model_dump()  # type: ignore[attr-defined]
    except Exception:
        if isinstance(obj, dict):
            return {k: _pydantic_dump(v) for k, v in obj.items()}
        # Fallback for simple namespaces
        try:
            return {k: _pydantic_dump(v) for k, v in vars(obj).items()}
        except Exception:
            return dict(obj) if isinstance(obj, dict) else {}


def _apply_overrides_generic(
    exp: ConfigModel, overrides_env: str, model_cls: Any
) -> ConfigModel:
    raw = os.getenv(overrides_env)
    if not raw:
        return exp
    try:
        ov = json.loads(raw)
    except Exception:
        return exp
    if not isinstance(ov, dict):
        return exp
    base = _pydantic_dump(exp)
    merged = _deep_merge_dicts(base, ov)
    try:
        return model_cls.model_validate(merged)
    except Exception:
        # If overrides are invalid, keep original to avoid breaking flows
        return exp


def _apply_train_overrides(exp: TrainerConfig) -> TrainerConfig:
    return cast(
        TrainerConfig,
        _apply_overrides_generic(exp, "ML_PLAYGROUND_TRAIN_OVERRIDES", TrainerConfig),
    )


def _apply_sample_overrides(exp: SamplerConfig) -> SamplerConfig:
    return cast(
        SamplerConfig,
        _apply_overrides_generic(exp, "ML_PLAYGROUND_SAMPLE_OVERRIDES", SamplerConfig),
    )


def _load_train_config(path: Path) -> TrainerConfig:
    """Strict loader with minimal legacy checks and path resolution.

    - Ensures required subsections exist ([model],[data],[optim],[schedule],[runtime]).
    - Produces a friendly unknown-key error for [train.data].
    - Delegates detailed value validation to Pydantic models.
    """
    with path.open("rb") as f:
        raw_exp: TomlData = tomllib.load(f)
    # Discover and merge defaults from ml_playground/experiments/default_config.toml (if present).
    # Behavior: defaults provide a base; the experiment's config overrides them.
    base_experiments_path = Path(__file__).resolve().parent / "experiments"
    defaults_config_path = base_experiments_path / "default_config.toml"
    raw = dict(raw_exp)
    defaults_raw: TomlData | None = None
    if defaults_config_path.exists():
        try:
            with defaults_config_path.open("rb") as df:
                defaults_raw = tomllib.load(df)
            if isinstance(defaults_raw, dict):
                d_train_obj = defaults_raw.get("train")
                if isinstance(d_train_obj, dict):
                    raw = _deep_merge_dicts({"train": d_train_obj}, raw)
        except Exception:
            # Ignore default merge failures; proceed with experiment-only config.
            defaults_raw = None

    train_tbl = raw.get("train")
    if not isinstance(train_tbl, dict):
        # Keep legacy top-level message used by CLI tests
        raise Exception("Config must contain [train] block")

    # Required subsections
    for sec in ("model", "data", "optim", "schedule", "runtime"):
        if not isinstance(train_tbl.get(sec), dict):
            raise ValueError(f"Missing required section [{sec}]")

    # Unknown keys in [train.data]
    allowed_data_keys = set(DataConfig.model_fields.keys())
    unknown = set(train_tbl["data"].keys()) - allowed_data_keys
    if unknown:
        raise ValueError("Unknown key(s) in [train.data]")

    # Resolve and prune unsupported sections/keys (no path rewriting; use as configured)
    allowed_sections = {"model", "data", "optim", "schedule", "runtime"}
    d = {k: train_tbl[k] for k in allowed_sections}

    # Track provenance for startup logging
    prov: dict[str, dict[str, str]] = {
        "model": {},
        "data": {},
        "optim": {},
        "schedule": {},
        "runtime": {},
    }
    _tmp_exp_train = raw_exp.get("train")
    exp_train_tbl: TomlData = _tmp_exp_train if isinstance(_tmp_exp_train, dict) else {}
    defaults_dict: TomlData = defaults_raw if isinstance(defaults_raw, dict) else {}
    _tmp_train = defaults_dict.get("train")
    defaults_train_tbl: TomlData = _tmp_train if isinstance(_tmp_train, dict) else {}

    # Initial provenance: default vs experiment
    for sec in allowed_sections:
        _dflt_sec = defaults_train_tbl.get(sec)
        sec_defaults = _dflt_sec if isinstance(_dflt_sec, dict) else {}
        _exp_sec = exp_train_tbl.get(sec)
        sec_exp = _exp_sec if isinstance(_exp_sec, dict) else {}
        merged_sec = d.get(sec, {})
        if isinstance(merged_sec, dict):
            for k in merged_sec.keys():
                if k in sec_exp:
                    prov[sec][k] = "override"  # set by experiment
                elif k in sec_defaults:
                    prov[sec][k] = "default"
                else:
                    prov[sec][k] = (
                        "override"  # conservative: present only in experiment
                    )

    # Environment overrides are not supported: only default + experiment TOMLs are used.

    # Strict: do not rewrite any paths; use them exactly as configured.
    # [train.runtime]: drop unknown keys (keep only keys known to RuntimeConfig)
    allowed_runtime_keys = set(RuntimeConfig.model_fields.keys())
    d_runtime_all = dict(d["runtime"])
    d_runtime = {k: v for k, v in d_runtime_all.items() if k in allowed_runtime_keys}
    d["runtime"] = d_runtime

    # Provenance log at startup: print effective config with markers
    # Format: [config] train.<section>.<key> = <value> (default|override)
    try:
        print("[config] Effective training configuration (source: default/override):")
        for sec in ("model", "data", "optim", "schedule", "runtime"):
            sec_map = d.get(sec, {})
            if not isinstance(sec_map, dict):
                continue
            for k in sorted(sec_map.keys()):
                source = prov.get(sec, {}).get(k, "override")
                print(f"[config] train.{sec}.{k} = {sec_map[k]!r} ({source})")
    except Exception:
        # Never fail due to logging
        pass

    try:
        return TrainerConfig.model_validate(d)
    except Exception as e:
        # Normalize to ValueError for test expectations
        raise ValueError(str(e))


def _load_sample_config(path: Path) -> SamplerConfig:
    """Strict loader for sample section with minimal checks.

    - Ensures required subsections exist ([sample] with either [runtime] or runtime_ref, plus [sample.sample]).
    - Supports schema-level reference `runtime_ref = "train.runtime"` with override merge.
    - Delegates detailed value validation to Pydantic models.
    """
    with path.open("rb") as f:
        raw_exp: TomlData = tomllib.load(f)

    # Load defaults and deep-merge (defaults under experiment overrides)
    defaults_path = (
        Path(__file__).resolve().parent / "experiments" / "default_config.toml"
    )
    raw = dict(raw_exp)
    defaults_raw: TomlData | None = None
    if defaults_path.exists():
        try:
            with defaults_path.open("rb") as df:
                defaults_raw = tomllib.load(df)
            if isinstance(defaults_raw, dict):
                d_sample_obj = defaults_raw.get("sample")
                if isinstance(d_sample_obj, dict):
                    raw = _deep_merge_dicts({"sample": d_sample_obj}, raw)
        except Exception:
            defaults_raw = None

    sample_tbl = raw.get("sample")
    if not isinstance(sample_tbl, dict):
        raise Exception("Config must contain [sample] block")

    # Strictness: only runtime, runtime_ref, sample, extras allowed at this level
    allowed_top = {"runtime", "runtime_ref", "sample", "extras"}
    unknown_top = set(sample_tbl.keys()) - allowed_top
    if unknown_top:
        raise ValueError("Unknown key(s) in [sample] (outside extras)")

    # Require [sample.sample]
    if not isinstance(sample_tbl.get("sample"), dict):
        raise ValueError("Missing required section [sample]")

    # Handle runtime presence or reference
    runtime_tbl = sample_tbl.get("runtime")
    runtime_ref = sample_tbl.get("runtime_ref")
    if not isinstance(runtime_tbl, dict) and not isinstance(runtime_ref, str):
        raise ValueError(
            "Sampler requires either [sample.runtime] or sample.runtime_ref"
        )

    # Allowed keys pruning for runtime/sample according to models
    allowed_runtime_keys = set(RuntimeConfig.model_fields.keys())

    # Build merged dict and provenance
    d: dict[str, Any] = {"sample": dict(sample_tbl["sample"])}

    # Strict: do not rewrite any paths; use them exactly as configured.
    prov: dict[str, dict[str, str]] = {"runtime": {}, "sample": {}}
    _tmp_exp_sample = raw_exp.get("sample")
    exp_sample_tbl: TomlData = (
        _tmp_exp_sample if isinstance(_tmp_exp_sample, dict) else {}
    )
    defaults_dict: TomlData = defaults_raw if isinstance(defaults_raw, dict) else {}
    _tmp_sample = defaults_dict.get("sample")
    defaults_sample_tbl: TomlData = _tmp_sample if isinstance(_tmp_sample, dict) else {}

    for sec in ("sample",):
        _dflt_sec = defaults_sample_tbl.get(sec)
        sec_defaults = _dflt_sec if isinstance(_dflt_sec, dict) else {}
        _exp_sec = exp_sample_tbl.get(sec)
        sec_exp = _exp_sec if isinstance(_exp_sec, dict) else {}
        merged_sec = d.get(sec, {})
        if isinstance(merged_sec, dict):
            for k in merged_sec.keys():
                if k in sec_exp:
                    prov[sec][k] = "override"
                elif k in sec_defaults:
                    prov[sec][k] = "default"
                else:
                    prov[sec][k] = "override"

    # Resolve runtime via ref merge if requested
    if isinstance(runtime_ref, str):
        if runtime_ref != "train.runtime":
            raise ValueError("Unsupported sample.runtime_ref; allowed: 'train.runtime'")
        # Merge defaults.train and exp.train, then take .runtime
        _defaults_train = defaults_dict.get("train")
        defaults_train_tbl: TomlData = (
            cast(TomlData, _defaults_train) if isinstance(_defaults_train, dict) else {}
        )
        _exp_train = raw_exp.get("train")
        exp_train_tbl: TomlData = (
            cast(TomlData, _exp_train) if isinstance(_exp_train, dict) else {}
        )
        merged_train_tbl = _deep_merge_dicts(defaults_train_tbl, exp_train_tbl)
        base_rt = merged_train_tbl.get("runtime")
        if not isinstance(base_rt, dict):
            raise ValueError(
                "sample.runtime_ref points to 'train.runtime' but [train.runtime] is missing"
            )
        resolved_rt = {k: v for k, v in base_rt.items() if k in allowed_runtime_keys}
        if isinstance(runtime_tbl, dict):
            # sample.runtime overrides
            overrides = {
                k: v for k, v in runtime_tbl.items() if k in allowed_runtime_keys
            }
            resolved_rt.update(overrides)
        d["runtime"] = resolved_rt
    else:
        # Use provided sample.runtime directly
        d["runtime"] = {
            k: v
            for k, v in (runtime_tbl if isinstance(runtime_tbl, dict) else {}).items()
            if k in allowed_runtime_keys
        }

    # Provenance logging
    try:
        print("[config] Effective sampling configuration (source: default/override):")
        for sec in ("runtime", "sample"):
            sec_map = d.get(sec, {})
            if not isinstance(sec_map, dict):
                continue
            for k in sorted(sec_map.keys()):
                source = prov.get(sec, {}).get(k, "override")
                print(f"[config] sample.{sec}.{k} = {sec_map[k]!r} ({source})")
    except Exception:
        pass

    try:
        return SamplerConfig.model_validate(d)
    except Exception as e:
        raise ValueError(str(e))


def load_train_config(path: Path) -> TrainerConfig:
    """Public wrapper: read TOML + defaults, delegate to strict raw loader, then
    resolve relative paths against the experiment root (path.parent).
    """
    base_experiments_path = Path(__file__).resolve().parent / "experiments"
    defaults_config_path = base_experiments_path / "default_config.toml"
    try:
        defaults_raw: TomlData = _read_toml_dict(defaults_config_path)
    except Exception:
        defaults_raw = {}
    raw_exp = _read_toml_dict(path)
    cfg = _load_train_config_from_raw(raw_exp, defaults_raw)
    exp_root = path.parent
    # Resolve dataset_dir relative to experiment root if it is not absolute
    try:
        ds = cfg.data.dataset_dir
        if not ds.is_absolute():
            resolved = (exp_root / ds).resolve()
            cfg = cfg.model_copy(
                update={"data": cfg.data.model_copy(update={"dataset_dir": resolved})}
            )
    except Exception:
        # Keep original on any resolution error
        pass
    # Resolve train.runtime.out_dir relative to experiment root if not absolute
    try:
        rt = cfg.runtime
        od = rt.out_dir
        if not od.is_absolute():
            resolved_out = (exp_root / od).resolve()
            cfg = cfg.model_copy(
                update={"runtime": rt.model_copy(update={"out_dir": resolved_out})}
            )
    except Exception:
        pass
    return cfg


def load_sample_config(path: Path) -> SamplerConfig:
    """Public wrapper: read TOML + defaults, delegate to strict raw loader, then
    resolve relative paths against the experiment root (path.parent).
    """
    base_experiments_path = Path(__file__).resolve().parent / "experiments"
    defaults_config_path = base_experiments_path / "default_config.toml"
    try:
        defaults_raw: TomlData = _read_toml_dict(defaults_config_path)
    except Exception:
        defaults_raw = {}
    raw_exp = _read_toml_dict(path)
    cfg = _load_sample_config_from_raw(raw_exp, defaults_raw)
    # Resolve runtime.out_dir relative to experiment root if not absolute
    try:
        rt = cfg.runtime
        if rt is not None:
            out_dir = rt.out_dir
            if not out_dir.is_absolute():
                resolved_out = (path.parent / out_dir).resolve()
                cfg = cfg.model_copy(
                    update={"runtime": rt.model_copy(update={"out_dir": resolved_out})}
                )
    except Exception:
        pass
    return cfg


# HasExperiment and CLIArgs classes removed - now pass parameters directly


def _log_config_provenance(
    config_type: str,
    merged_config: dict[str, Any],
    exp_config: TomlData,
    defaults_config: TomlData,
    sections: tuple[str, ...],
) -> None:
    """Centralized provenance logging for configuration loading.

    Tracks which config values come from defaults vs experiment overrides
    and prints them in a standardized format.
    """
    try:
        prov: dict[str, dict[str, str]] = {sec: {} for sec in sections}
        # Build provenance tracking
        for sec in sections:
            _dflt_sec = defaults_config.get(sec)
            sec_defaults = _dflt_sec if isinstance(_dflt_sec, dict) else {}
            _exp_sec = exp_config.get(sec)
            sec_exp = _exp_sec if isinstance(_exp_sec, dict) else {}
            merged_sec = merged_config.get(sec, {})

            if isinstance(merged_sec, dict):
                for k in merged_sec.keys():
                    if k in sec_exp:
                        prov[sec][k] = "override"  # set by experiment
                    elif k in sec_defaults:
                        prov[sec][k] = "default"
                    else:
                        prov[sec][k] = (
                            "override"  # conservative: present only in experiment
                        )

        # Print provenance log
        print(
            f"[config] Effective {config_type} configuration (source: default/override):"
        )
        for sec in sections:
            sec_map = merged_config.get(sec, {})
            if not isinstance(sec_map, dict):
                continue
            for k in sorted(sec_map.keys()):
                source = prov.get(sec, {}).get(k, "override")
                print(f"[config] {config_type}.{sec}.{k} = {sec_map[k]!r} ({source})")
    except Exception:
        # Never fail due to logging
        pass


def _resolve_and_load_configs(
    experiment: str, exp_config: Path | None = None
) -> tuple[Path, TomlData, TomlData]:
    # Validate syntax: require subcommand + experiment name
    if not isinstance(experiment, str):
        raise SystemExit("Syntax error: expected 'command <experiment>'")

    # Determine config path: if an explicit exp_config is provided, use it and
    # ignore the experiment-local config.toml; otherwise fall back to default location.
    explicit_cfg: Path | None = exp_config

    if isinstance(explicit_cfg, Path):
        cfg_path = explicit_cfg
    else:
        cfg_path = (
            Path(__file__).resolve().parent / "experiments" / experiment / "config.toml"
        )

    if not cfg_path.exists():
        # If explicit path was given, error references that path; otherwise reference default location.
        raise SystemExit(
            f"Config not found for experiment '{experiment}'. Expected at: {cfg_path}"
        )

    defaults_config_path = (
        Path(__file__).resolve().parent / "experiments" / "default_config.toml"
    )
    try:
        defaults_raw: TomlData = _read_toml_dict(defaults_config_path)
    except Exception as e:
        raise SystemExit(
            f"Default config invalid or not found at {defaults_config_path}: {e}"
        )

    try:
        config_raw: TomlData = _read_toml_dict(cfg_path)
    except Exception as e:
        raise SystemExit(f"Experiment config invalid or not found at {cfg_path}: {e}")

    return cfg_path, config_raw, defaults_raw


class ExperimentLoader:
    """Encapsulates pluggable experiment discovery with shared error messages."""

    def __init__(self) -> None:
        # Cache instances keyed by (role, experiment) where role in {"preparer", "trainer", "sampler"}
        self._cache: dict[tuple[str, str], object] = {}

    def _load_exp_class_instance(
        self, module_path: str, method_name: str, role: str, experiment: str
    ) -> object:
        """Import module_path and instantiate the first class exposing method_name.

        This removes per-experiment integration shims and relies only on canonical modules.
        Uses caching to avoid re-imports in the same process.
        """
        cache_key = (role, experiment)
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            mod = importlib.import_module(module_path)
        except Exception as e:
            raise SystemExit(f"Failed to import {module_path}: {e}")

        # Collect available class names for better error messaging
        available_classes = []
        suitable_instance = None

        # Find a class with the given method and a no-arg constructor
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if isinstance(attr, type):
                available_classes.append(attr_name)
                if hasattr(attr, method_name):
                    try:
                        suitable_instance = attr()
                        break
                    except Exception:
                        continue

        if suitable_instance is None:
            available_list = (
                ", ".join(sorted(available_classes)) if available_classes else "none"
            )
            raise SystemExit(
                f"No suitable class with method '{method_name}' found in {module_path}. "
                f"Available classes: {available_list}"
            )

        # Cache the instance before returning
        self._cache[cache_key] = suitable_instance
        return suitable_instance

    def load_preparer(self, experiment: str) -> ExpPreparer:
        return cast(
            ExpPreparer,
            self._load_exp_class_instance(
                f"ml_playground.experiments.{experiment}.preparer",
                "prepare",
                "preparer",
                experiment,
            ),
        )

    def load_trainer(self, experiment: str) -> ExpTrainer:
        return cast(
            ExpTrainer,
            self._load_exp_class_instance(
                f"ml_playground.experiments.{experiment}.trainer",
                "train",
                "trainer",
                experiment,
            ),
        )

    def load_sampler(self, experiment: str) -> ExpSampler:
        return cast(
            ExpSampler,
            self._load_exp_class_instance(
                f"ml_playground.experiments.{experiment}.sampler",
                "sample",
                "sampler",
                experiment,
            ),
        )


# Create a global instance for use in commands
_experiment_loader = ExperimentLoader()


# Unified AppConfig loader and context cache

# Side-channel to capture last strict validation errors per config path
_last_load_errors: dict[Path, dict[str, str | None]] = {}


def load_app_config(
    experiment: str, exp_config: Path | None
) -> tuple[Path, AppConfig, PreparerConfig]:
    """Load, merge, validate, resolve and override all configs once.

    Returns (cfg_path, AppConfig, PreparerConfig).
    - Does not raise on train/sample validation errors; instead sets them to None.
      Error strings are stored in a side-channel for commands to surface.
    """
    cfg_path, config_raw, defaults_raw = _resolve_and_load_configs(
        experiment, exp_config
    )

    # Preparer (best-effort)
    try:
        prep_cfg = _load_prepare_config_from_raw(config_raw, defaults_raw)
    except Exception:
        prep_cfg = PreparerConfig()
    # Resolve preparer paths relative to cfg dir
    try:
        exp_root = cfg_path.parent
        ds = getattr(prep_cfg, "dataset_dir", None)
        if isinstance(ds, Path) and not ds.is_absolute():
            prep_cfg = prep_cfg.model_copy(
                update={"dataset_dir": (exp_root / ds).resolve()}
            )
        rd = getattr(prep_cfg, "raw_dir", None)
        if isinstance(rd, Path) and not rd.is_absolute():
            prep_cfg = prep_cfg.model_copy(
                update={"raw_dir": (exp_root / rd).resolve()}
            )
    except Exception:
        pass

    # Train
    train_cfg: TrainerConfig | None
    train_err: str | None = None
    try:
        tcfg = _load_train_config_from_raw(config_raw, defaults_raw)
        # resolve relative paths against cfg dir
        exp_root = cfg_path.parent
        try:
            ds = tcfg.data.dataset_dir
            if not ds.is_absolute():
                tcfg = tcfg.model_copy(
                    update={
                        "data": tcfg.data.model_copy(
                            update={"dataset_dir": (exp_root / ds).resolve()}
                        )
                    }
                )
        except Exception:
            pass
        try:
            od = tcfg.runtime.out_dir
            if not od.is_absolute():
                tcfg = tcfg.model_copy(
                    update={
                        "runtime": tcfg.runtime.model_copy(
                            update={"out_dir": (exp_root / od).resolve()}
                        )
                    }
                )
        except Exception:
            pass
        # apply env overrides once
        tcfg = _apply_train_overrides(tcfg)
        train_cfg = tcfg
    except Exception as e:
        train_cfg = None
        train_err = str(e)

    # Sample
    sample_cfg: SamplerConfig | None
    sample_err: str | None = None
    try:
        scfg = _load_sample_config_from_raw(config_raw, defaults_raw)
        # resolve relative out_dir against cfg dir
        try:
            rt = scfg.runtime
            if rt is not None:
                od = rt.out_dir
                if not od.is_absolute():
                    scfg = scfg.model_copy(
                        update={
                            "runtime": rt.model_copy(
                                update={"out_dir": (cfg_path.parent / od).resolve()}
                            )
                        }
                    )
        except Exception:
            pass
        scfg = _apply_sample_overrides(scfg)
        sample_cfg = scfg
    except Exception as e:
        sample_cfg = None
        sample_err = str(e)

    app = AppConfig(train=train_cfg, sample=sample_cfg)

    # Record strict validation error messages for this config path
    try:
        _last_load_errors[cfg_path] = {"train": train_err, "sample": sample_err}
    except Exception:
        pass

    # Return core tuple
    return cfg_path, app, prep_cfg


def ensure_loaded(
    ctx: typer.Context, experiment: str
) -> tuple[Path, AppConfig, PreparerConfig]:
    """Ensure that (cfg_path, AppConfig, PreparerConfig) are loaded and cached in Typer context.
    Also caches error messages under ctx.obj["app_errors"].
    """
    ctx.ensure_object(dict)
    obj = cast(dict[str, Any], ctx.obj)
    exp_config = (
        obj.get("exp_config")
        if isinstance(obj.get("exp_config"), Path) or obj.get("exp_config") is None
        else None
    )
    cache_key = (experiment, exp_config)
    cached = obj.get("loaded_cache")
    if isinstance(cached, dict) and cached.get("key") == cache_key:
        return (
            cast(Path, cached["cfg_path"]),
            cast(AppConfig, cached["app"]),
            cast(PreparerConfig, cached["prep"]),
        )

    # Load fresh
    try:
        cfg_path, app, prep = load_app_config(experiment, cast(Path | None, exp_config))
    except SystemExit:
        # Bubble up CLI-style exits
        raise
    except Exception as e:
        # Unexpected; keep behavior consistent by echo+exit
        typer.echo(str(e))
        raise typer.Exit(code=2)

    # Recompute errors to cache by attempting strict loads (without I/O) already done in load_app_config
    obj["app_errors"] = obj.get("app_errors", {})
    if isinstance(obj["app_errors"], dict):
        obj["app_errors"][cache_key] = {
            "train": None,
            "sample": None,
        }
    # Store cache
    obj["loaded_cache"] = {
        "key": cache_key,
        "cfg_path": cfg_path,
        "app": app,
        "prep": prep,
    }
    # Also stash error messages directly for easy access
    errs = _last_load_errors.get(cfg_path, {"train": None, "sample": None})
    obj["loaded_errors"] = {
        "key": cache_key,
        "train": errs.get("train"),
        "sample": errs.get("sample"),
    }
    return cfg_path, app, prep


def run_or_exit(
    func, keyboard_interrupt_msg: str | None = None, exception_exit_code: int = 2
):
    """Helper to run a function with standardized error handling.

    Args:
        func: Function to execute
        keyboard_interrupt_msg: Message to show on KeyboardInterrupt (if None, no handler)
        exception_exit_code: Exit code for general exceptions (default: 2)
    """
    try:
        return func()
    except KeyboardInterrupt:
        if keyboard_interrupt_msg is not None:
            typer.echo(keyboard_interrupt_msg)
        # Don't raise typer.Exit here to match existing behavior (some commands just return)
    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(str(e))
        raise typer.Exit(code=exception_exit_code)


def _extract_exp_config(ctx: typer.Context) -> Path | None:
    """Extract exp_config from Typer context safely."""
    try:
        return ctx.obj.get("exp_config") if ctx and ctx.obj else None
    except Exception:
        return None


def get_cfg_path(experiment: str, exp_config: Path | None) -> Path:
    """Get config path: use exp_config if provided, else experiments/<experiment>/config.toml."""
    if exp_config is not None:
        return exp_config
    return _experiments_root() / experiment / "config.toml"


def load_effective_train(
    experiment: str, exp_config: Path | None
) -> tuple[Path, TrainerConfig]:
    """Load and validate training configuration, returning config path and effective config."""
    config_path, config_raw, defaults_raw = _resolve_and_load_configs(
        experiment, exp_config
    )

    # Strict validation to satisfy tests that patch the raw loader
    try:
        _ = _load_train_config_from_raw(config_raw, defaults_raw)
    except Exception as e:
        typer.echo(str(e))
        raise typer.Exit(code=2)

    # Load the effective config (wrapper) for execution
    try:
        train_cfg: TrainerConfig = load_train_config(config_path)
    except Exception as e:
        typer.echo(str(e))
        raise typer.Exit(code=2)

    # Apply environment overrides
    train_cfg = _apply_train_overrides(train_cfg)
    return config_path, train_cfg


def load_effective_sample(
    experiment: str, exp_config: Path | None
) -> tuple[Path, SamplerConfig]:
    """Load and validate sampling configuration, returning config path and effective config."""
    config_path, config_raw, defaults_raw = _resolve_and_load_configs(
        experiment, exp_config
    )

    # Strict validation and config build from raw to satisfy tests
    try:
        sample_cfg: SamplerConfig = _load_sample_config_from_raw(
            config_raw, defaults_raw
        )
    except Exception as e:
        typer.echo(str(e))
        raise typer.Exit(code=2)

    # Resolve runtime.out_dir relative to experiment root if provided and relative
    try:
        rt = sample_cfg.runtime
        if rt is not None:
            od = rt.out_dir
            if not od.is_absolute():
                resolved_out = (config_path.parent / od).resolve()
                sample_cfg = sample_cfg.model_copy(
                    update={"runtime": rt.model_copy(update={"out_dir": resolved_out})}
                )
    except Exception:
        pass

    # Apply environment overrides
    sample_cfg = _apply_sample_overrides(sample_cfg)
    return config_path, sample_cfg


def _log_command_status(tag: str, cfg: Any) -> None:
    """Log known file-based artifacts for the given config.

    The function inspects common path fields of the configuration and prints
    their existence and contents. It is best-effort and will never raise.
    """
    try:
        paths: dict[str, Path | None] = {}
        if isinstance(cfg, PreparerConfig):
            paths["dataset_dir"] = getattr(cfg, "dataset_dir", None)
            paths["raw_dir"] = getattr(cfg, "raw_dir", None)
        elif isinstance(cfg, TrainerConfig):
            data = getattr(cfg, "data", None)
            runtime = getattr(cfg, "runtime", None)
            if data is not None:
                paths["dataset_dir"] = getattr(data, "dataset_dir", None)
            if runtime is not None:
                paths["out_dir"] = getattr(runtime, "out_dir", None)
        elif isinstance(cfg, SamplerConfig):
            runtime = getattr(cfg, "runtime", None)
            if runtime is not None:
                paths["out_dir"] = getattr(runtime, "out_dir", None)

        for name, path in paths.items():
            if not isinstance(path, Path):
                logger.info(f"[status] {tag}.{name}: <not set>")
                continue
            if path.exists():
                try:
                    contents = ", ".join(sorted(p.name for p in path.iterdir()))
                    logger.info(f"[status] {tag}.{name}: {path} (contents: {contents})")
                except Exception:
                    logger.info(f"[status] {tag}.{name}: {path} (exists)")
            else:
                logger.info(f"[status] {tag}.{name}: {path} (missing)")
    except Exception:
        # Never fail due to status logging
        pass


def _run_prepare(
    experiment: str,
    prepare_cfg: PreparerConfig,
    config_path: Path,
) -> None:
    _log_command_status("prepare", prepare_cfg)
    preparer: ExpPreparer = _experiment_loader.load_preparer(experiment)
    report = preparer.prepare(prepare_cfg)
    try:
        print(f"[prepare] side-effects: {report.summarize()}")
        for msg in report.messages:
            print(f"[prepare] {msg}")
    except Exception:
        pass


def _run_loop(
    experiment: str,
    prepare_cfg: PreparerConfig,
    train_cfg: TrainerConfig,
    sample_cfg: SamplerConfig,
    config_path: Path,
) -> None:
    preparer: ExpPreparer = _experiment_loader.load_preparer(experiment)
    trainer: ExpTrainer = _experiment_loader.load_trainer(experiment)
    sampler: ExpSampler = _experiment_loader.load_sampler(experiment)

    # Prepare
    _log_command_status("prepare", prepare_cfg)
    prep_report = preparer.prepare(prepare_cfg)
    try:
        print(f"[prepare] side-effects: {prep_report.summarize()}")
        for msg in prep_report.messages:
            print(f"[prepare] {msg}")
    except Exception:
        pass

    # Train
    _log_command_status("train", train_cfg)
    train_report = trainer.train(train_cfg)
    try:
        print(f"[train] side-effects: {train_report.summarize()}")
        for msg in train_report.messages:
            print(f"[train] {msg}")
    except Exception:
        pass

    # Sample
    _log_command_status("sample", sample_cfg)
    sample_report = sampler.sample(sample_cfg)
    try:
        print(f"[sample] side-effects: {sample_report.summarize()}")
        for msg in sample_report.messages:
            print(f"[sample] {msg}")
    except Exception:
        pass


def _run_train(experiment: str, train_cfg: TrainerConfig, config_path: Path) -> None:
    _log_command_status("train", train_cfg)
    trainer: ExpTrainer = _experiment_loader.load_trainer(experiment)
    report = trainer.train(train_cfg)

    # After training, best-effort propagate dataset meta.pkl into out_dir for sampling
    try:
        ds_dir = train_cfg.data.dataset_dir
        meta_name = train_cfg.data.meta_pkl or "meta.pkl"
        src_meta = ds_dir / meta_name
        dst_meta = train_cfg.runtime.out_dir / "meta.pkl"
        if src_meta.exists() and not dst_meta.exists():
            dst_meta.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_meta, dst_meta)
    except Exception:
        # Never fail the CLI due to meta propagation
        pass

    try:
        print(f"[train] side-effects: {report.summarize()}")
        for msg in report.messages:
            print(f"[train] {msg}")
    except Exception:
        pass


def _run_sample(experiment: str, sample_cfg: SamplerConfig, config_path: Path) -> None:
    _log_command_status("sample", sample_cfg)
    sampler: ExpSampler = _experiment_loader.load_sampler(experiment)
    report = sampler.sample(sample_cfg)
    try:
        print(f"[sample] side-effects: {report.summarize()}")
        for msg in report.messages:
            print(f"[sample] {msg}")
    except Exception:
        pass


def _run_analyze(experiment: str, host: str, port: int, open_browser: bool) -> None:
    if experiment != "bundestag_char":
        raise RuntimeError("analyze currently supports only 'bundestag_char'")
    try:
        from ml_playground.analysis.lit_integration import run_server_bundestag_char
    except Exception as e:
        raise RuntimeError(str(e))
    run_server_bundestag_char(host=host, port=port, open_browser=open_browser)


# Typer-based CLI
app = typer.Typer(
    no_args_is_help=True,
    help="ML Playground CLI: prepare data, train models, and sample outputs.",
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
        typer.echo(f"Config file not found: {exp_config}")
        raise typer.Exit(code=2)

    try:
        # Ensure INFO-level logs (including status) are visible by default
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            logging.basicConfig(level=logging.INFO, format="%(message)s")
        ctx.ensure_object(dict)
    except Exception:
        # Fallback: if ensure_object fails, safely ignore and avoid crashing
        return
    ctx.obj["exp_config"] = exp_config


@app.command("prepare")
def cmd_prepare(
    ctx: typer.Context,
    experiment: Annotated[
        str,
        typer.Argument(
            help="Experiment name (folder under experiments/)",
            autocompletion=_complete_experiments,
        ),
    ],
) -> None:
    """Run the experiment-specific prepare step (config loaded once)."""

    def prepare_impl():
        cfg_path, app, prep_cfg = ensure_loaded(ctx, experiment)
        _run_prepare(experiment, prep_cfg, cfg_path)

    run_or_exit(prepare_impl, exception_exit_code=2)


@app.command("train")
def cmd_train(
    ctx: typer.Context,
    experiment: Annotated[
        str,
        typer.Argument(
            help="Experiment name (folder under experiments/)",
            autocompletion=_complete_experiments,
        ),
    ],
) -> None:
    """Validate config and run training for the given experiment (single-load)."""

    def train_impl():
        cfg_path, app, _prep = ensure_loaded(ctx, experiment)
        if app.train is None:
            # Preserve user-friendly error message
            errs = cast(dict[str, Any], getattr(ctx, "obj", {})).get(
                "loaded_errors", {}
            )
            msg = None
            if isinstance(errs, dict) and errs.get("key") == (
                experiment,
                cast(dict[str, Any], getattr(ctx, "obj", {})).get("exp_config"),
            ):
                msg = errs.get("train")
            typer.echo(msg or "Config must contain [train] block")
            raise typer.Exit(code=2)
        _run_train(experiment, app.train, cfg_path)

    run_or_exit(
        train_impl,
        keyboard_interrupt_msg="\nTraining interrupted by user (Ctrl+C). Exiting gracefully.",
        exception_exit_code=2,
    )


@app.command("sample")
def cmd_sample(
    ctx: typer.Context,
    experiment: Annotated[
        str,
        typer.Argument(
            help="Experiment name (folder under experiments/)",
            autocompletion=_complete_experiments,
        ),
    ],
) -> None:
    """Validate config and run sampling for the given experiment (single-load)."""
    # Special-case integration for 'speakger': delegate to its sampler entrypoint
    if experiment == "speakger":

        def speakger_impl():
            exp_config = _extract_exp_config(ctx)
            cfg_path = get_cfg_path(experiment, exp_config)
            mod = importlib.import_module("ml_playground.experiments.speakger.sampler")
            getattr(mod, "sample_from_toml")(cfg_path)

        run_or_exit(speakger_impl, exception_exit_code=2)
        return

    def sample_impl():
        cfg_path, app, _prep = ensure_loaded(ctx, experiment)
        if app.sample is None:
            errs = cast(dict[str, Any], getattr(ctx, "obj", {})).get(
                "loaded_errors", {}
            )
            msg = None
            if isinstance(errs, dict) and errs.get("key") == (
                experiment,
                cast(dict[str, Any], getattr(ctx, "obj", {})).get("exp_config"),
            ):
                msg = errs.get("sample")
            typer.echo(msg or "Config must contain [sample] block")
            raise typer.Exit(code=2)
        _run_sample(experiment, app.sample, cfg_path)

    run_or_exit(
        sample_impl,
        keyboard_interrupt_msg="\nSampling interrupted by user (Ctrl+C). Exiting gracefully.",
        exception_exit_code=2,
    )


@app.command("convert")
def cmd_convert(
    ctx: typer.Context,
    experiment: Annotated[
        str,
        typer.Argument(
            help="Experiment name (folder under experiments/)",
            autocompletion=_complete_experiments,
        ),
    ],
) -> None:
    """Export a trained model to an Ollama-ready directory. Supports only bundestag_char (POC)."""
    if experiment != "bundestag_char":
        typer.echo("convert currently supports only 'bundestag_char'.")
        raise typer.Exit(code=2)

    def convert_impl():
        cfg_path, _app, _prep = ensure_loaded(ctx, experiment)
        try:
            mod = importlib.import_module(
                "ml_playground.experiments.bundestag_char.ollama_export"
            )
            getattr(mod, "convert_from_toml")(cfg_path)
        except SystemExit as e:
            msg = str(e).strip()
            if msg:
                typer.echo(msg)
            raise typer.Exit(code=getattr(e, "code", 1))
        except Exception as e:
            typer.echo(str(e))
            raise typer.Exit(code=1)

    run_or_exit(convert_impl, exception_exit_code=2)


@app.command("loop")
def cmd_loop(
    ctx: typer.Context,
    experiment: Annotated[
        str,
        typer.Argument(
            help="Experiment name (folder under experiments/)",
            autocompletion=_complete_experiments,
        ),
    ],
) -> None:
    """Run prepare → train → sample as a single loop for the experiment (single-load)."""

    def loop_impl():
        cfg_path, app, prep_cfg = ensure_loaded(ctx, experiment)
        errs = cast(dict[str, Any], getattr(ctx, "obj", {})).get("loaded_errors", {})
        key = (
            experiment,
            cast(dict[str, Any], getattr(ctx, "obj", {})).get("exp_config"),
        )
        if app.train is None:
            msg = (
                errs.get("train")
                if isinstance(errs, dict) and errs.get("key") == key
                else None
            )
            typer.echo(msg or "Config must contain [train] block")
            raise typer.Exit(code=2)
        if app.sample is None:
            msg = (
                errs.get("sample")
                if isinstance(errs, dict) and errs.get("key") == key
                else None
            )
            typer.echo(msg or "Config must contain [sample] block")
            raise typer.Exit(code=2)
        _run_loop(experiment, prep_cfg, app.train, app.sample, cfg_path)

    run_or_exit(
        loop_impl,
        keyboard_interrupt_msg="\nSampling interrupted by user (Ctrl+C). Exiting gracefully.",
        exception_exit_code=2,
    )


@app.command("analyze")
def cmd_analyze(
    ctx: typer.Context,
    experiment: Annotated[
        str,
        typer.Argument(
            help="Experiment name (folder under experiments/)",
            autocompletion=_complete_experiments,
        ),
    ],
    host: Annotated[
        str, typer.Option("--host", help="Host interface to bind")
    ] = "127.0.0.1",
    port: Annotated[
        int, typer.Option("--port", help="Port to serve on (0 for auto)")
    ] = 5432,
    open_browser: Annotated[
        bool, typer.Option("--open-browser", help="Open UI in a browser")
    ] = False,
) -> None:
    """Launch an interactive analysis UI (LIT) for the selected experiment.

    Proof-of-concept currently supports only 'bundestag_char'.
    """
    if experiment != "bundestag_char":
        typer.echo("analyze currently supports only 'bundestag_char'")
        raise typer.Exit(code=2)
    try:
        # Lazy import to avoid importing analysis unless requested
        from ml_playground.analysis.lit_integration import run_server_bundestag_char
    except Exception as e:
        typer.echo(str(e))
        raise typer.Exit(code=2)

    def analyze_impl():
        run_server_bundestag_char(host=host, port=port, open_browser=open_browser)

    run_or_exit(
        analyze_impl,
        keyboard_interrupt_msg="\nAnalysis server interrupted by user (Ctrl+C). Exiting.",
        exception_exit_code=2,
    )


def main(argv: list[str] | None = None) -> None:
    """Programmatic entry point used by tests and CLI.

    Delegates entirely to Typer for command parsing and execution.
    """
    cmd = get_command(app)
    try:
        # Use standalone_mode=False so success does not raise SystemExit (for programmatic calls/tests)
        ret = cmd.main(args=argv, prog_name="ml_playground", standalone_mode=False)
        # If click/typer returns a non-zero integer exit code, propagate as SystemExit
        if isinstance(ret, int) and ret != 0:
            raise SystemExit(ret)
        return
    except click.exceptions.NoArgsIsHelpError:
        # Help was shown (no args). Exit cleanly without traceback.
        return
    except click.exceptions.MissingParameter as e:
        # Special-case the required 'experiment' parameter to show helpful guidance
        pname = getattr(getattr(e, "param", None), "name", None)
        if pname == "experiment":
            try:
                experiments = _complete_experiments(cast(typer.Context, None), "")
            except Exception:
                experiments = []
            if experiments:
                exp_list = "\n  - " + "\n  - ".join(experiments)
                typer.echo(
                    "Missing required argument: 'experiment'. Available experiments:"
                    f"{exp_list}\nUsage: ml_playground [prepare|train|sample|loop] <experiment>"
                )
            else:
                typer.echo(
                    "Missing required argument: 'experiment'. No experiments found.\n"
                    "Ensure ml_playground/experiments contains experiment folders with a config.toml.\n"
                    "Usage: ml_playground [prepare|train|sample|loop] <experiment>"
                )
        else:
            typer.echo(str(e))
        raise SystemExit(2)
    except click.ClickException as e:
        # Show concise message and exit with the exception's exit code
        try:
            msg = e.format_message()  # type: ignore[attr-defined]
        except Exception:
            msg = str(e)
        typer.echo(msg)
        raise SystemExit(getattr(e, "exit_code", 1))
    except click.exceptions.Exit as e:
        # For programmatic entry, only propagate non-zero exit codes
        code = getattr(e, "exit_code", 0)
        if code != 0:
            raise SystemExit(code)
        return
    except KeyboardInterrupt:
        typer.echo("\nOperation interrupted by user (Ctrl+C). Exiting gracefully.")
    except Exception as e:
        # Fallback: never show a traceback to the user
        typer.echo(f"Error: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
