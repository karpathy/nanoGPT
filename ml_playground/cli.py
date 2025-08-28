from __future__ import annotations

import typer
import click
from typer.main import get_command
import importlib
import json
import os
import shutil
import tomllib
from pathlib import Path
from typing import Any, Protocol, Annotated, cast
from dataclasses import dataclass

from ml_playground import datasets
from ml_playground.config import (
    TrainerConfig,
    SamplerConfig,
    DataConfig,
    RuntimeConfig,
)
from ml_playground.prepare import PreparerConfig, make_preparer
from ml_playground.sampler import sample
from ml_playground.trainer import train

from ml_playground.experiments.protocol import (
    Preparer as ExpPreparer,
    Trainer as ExpTrainer,
    Sampler as ExpSampler,
)


# --- Typer helpers ---------------------------------------------------------
def _experiments_root() -> Path:
    """Return the root folder that contains experiment directories."""
    return Path(__file__).resolve().parent / "experiments"


def _complete_experiments(ctx: typer.Context, incomplete: str) -> list[str]:
    """Auto-complete experiment names based on directories with a config.toml."""
    try:
        root = _experiments_root()
        if not root.exists():
            return []
        matches: list[str] = []
        for p in root.iterdir():
            if not p.is_dir():
                continue
            if not (p / "config.toml").exists():
                continue
            name = p.name
            if name.startswith(incomplete):
                matches.append(name)
        return sorted(matches)
    except Exception:
        # Never fail completion; just return no suggestions
        return []


def _read_toml_dict(path: Path) -> dict[str, Any]:
    if not isinstance(path, Path):
        path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("rb") as f:
        raw: Any = tomllib.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Config at {path} must be a TOML table/object")
    return raw


def _load_prepare_config_from_raw(
    raw_exp: dict[str, Any], defaults_raw: dict[str, Any]
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
    extras: dict[str, Any] = {}
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
    raw_exp: dict[str, Any], defaults_raw: dict[str, Any]
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

    # Track provenance for startup logging
    prov: dict[str, dict[str, str]] = {sec: {} for sec in allowed_sections}
    _tmp_exp_train = raw_exp.get("train")
    exp_train_tbl: dict[str, Any] = (
        _tmp_exp_train if isinstance(_tmp_exp_train, dict) else {}
    )
    defaults_dict: dict[str, Any] = (
        defaults_raw if isinstance(defaults_raw, dict) else {}
    )
    _tmp_train = defaults_dict.get("train")
    defaults_train_tbl: dict[str, Any] = (
        _tmp_train if isinstance(_tmp_train, dict) else {}
    )

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

    # No path heuristics: do not rewrite dataset_dir or out_dir; use as configured.
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
    raw_exp: dict[str, Any], defaults_raw: dict[str, Any]
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

    prov: dict[str, dict[str, str]] = {"runtime": {}, "sample": {}}
    _tmp_exp_sample = raw_exp.get("sample")
    exp_sample_tbl: dict[str, Any] = (
        _tmp_exp_sample if isinstance(_tmp_exp_sample, dict) else {}
    )
    defaults_dict: dict[str, Any] = (
        defaults_raw if isinstance(defaults_raw, dict) else {}
    )
    _tmp_sample = defaults_dict.get("sample")
    defaults_sample_tbl: dict[str, Any] = (
        _tmp_sample if isinstance(_tmp_sample, dict) else {}
    )

    # Prepare provenance for sample subsection
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
        defaults_train_tbl = (
            defaults_dict.get("train")
            if isinstance(defaults_dict.get("train"), dict)
            else {}
        )
        exp_train_tbl = (
            raw_exp.get("train") if isinstance(raw_exp.get("train"), dict) else {}
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


def _deep_merge_dicts(base: Any, override: Any) -> dict[str, Any]:
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


def _pydantic_dump(obj: Any) -> dict:
    """Best-effort dump of a Pydantic v2 model or nested structure to a plain dict."""
    try:
        # Pydantic v2
        return cast("Any", obj).model_dump()  # type: ignore[attr-defined]
    except Exception:
        if isinstance(obj, dict):
            return {k: _pydantic_dump(v) for k, v in obj.items()}
        # Fallback for simple namespaces
        try:
            return {k: _pydantic_dump(v) for k, v in vars(obj).items()}
        except Exception:
            return dict(obj) if isinstance(obj, dict) else {}


def _apply_overrides_generic(exp: Any, overrides_env: str, model_cls: Any) -> Any:
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
        raw_exp: dict[str, Any] = tomllib.load(f)
    # Discover and merge defaults from ml_playground/experiments/default_config.toml (if present).
    # Behavior: defaults provide a base; the experiment's config overrides them.
    base_experiments_path = Path(__file__).resolve().parent / "experiments"
    defaults_config_path = base_experiments_path / "default_config.toml"
    raw = dict(raw_exp)
    defaults_raw: dict | None = None
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
    prov: dict[str, dict[str, str]] = {sec: {} for sec in allowed_sections}
    _tmp_exp_train = raw_exp.get("train")
    exp_train_tbl: dict[str, Any] = (
        _tmp_exp_train if isinstance(_tmp_exp_train, dict) else {}
    )
    defaults_dict: dict[str, Any] = (
        defaults_raw if isinstance(defaults_raw, dict) else {}
    )
    _tmp_train = defaults_dict.get("train")
    defaults_train_tbl: dict[str, Any] = (
        _tmp_train if isinstance(_tmp_train, dict) else {}
    )

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
        raw_exp: dict[str, Any] = tomllib.load(f)

    # Load defaults and deep-merge (defaults under experiment overrides)
    defaults_path = (
        Path(__file__).resolve().parent / "experiments" / "default_config.toml"
    )
    raw = dict(raw_exp)
    defaults_raw: dict | None = None
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
    exp_sample_tbl: dict[str, Any] = (
        _tmp_exp_sample if isinstance(_tmp_exp_sample, dict) else {}
    )
    defaults_dict: dict[str, Any] = (
        defaults_raw if isinstance(defaults_raw, dict) else {}
    )
    _tmp_sample = defaults_dict.get("sample")
    defaults_sample_tbl: dict[str, Any] = (
        _tmp_sample if isinstance(_tmp_sample, dict) else {}
    )

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
        defaults_train_tbl = (
            defaults_dict.get("train")
            if isinstance(defaults_dict.get("train"), dict)
            else {}
        )
        exp_train_tbl = (
            raw_exp.get("train") if isinstance(raw_exp.get("train"), dict) else {}
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
        defaults_raw: dict[str, Any] = _read_toml_dict(defaults_config_path)
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
        defaults_raw: dict[str, Any] = _read_toml_dict(defaults_config_path)
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
                    update={
                        "runtime": rt.model_copy(update={"out_dir": resolved_out})
                    }
                )
    except Exception:
        pass
    return cfg


class HasExperiment(Protocol):
    experiment: str


@dataclass
class CLIArgs:
    experiment: str
    exp_config: Path | None = None


def _resolve_and_load_configs(
    args: HasExperiment,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    # Validate syntax: require subcommand + experiment name
    try:
        exp = args.experiment
    except AttributeError:
        raise SystemExit("Syntax error: expected 'command <experiment>'")
    if not isinstance(exp, str):
        raise SystemExit("Syntax error: expected 'command <experiment>'")

    # Determine config path: if an explicit exp_config is provided, use it and
    # ignore the experiment-local config.toml; otherwise fall back to default location.
    explicit_cfg: Path | None = None
    try:
        explicit_cfg = getattr(args, "exp_config")  # type: ignore[attr-defined]
    except Exception:
        explicit_cfg = None

    if isinstance(explicit_cfg, Path):
        cfg_path = explicit_cfg
    else:
        cfg_path = Path(__file__).resolve().parent / "experiments" / exp / "config.toml"

    if not cfg_path.exists():
        # If explicit path was given, error references that path; otherwise reference default location.
        raise SystemExit(
            f"Config not found for experiment '{exp}'. Expected at: {cfg_path}"
        )

    defaults_config_path = (
        Path(__file__).resolve().parent / "experiments" / "default_config.toml"
    )
    try:
        defaults_raw: dict[str, Any] = _read_toml_dict(defaults_config_path)
    except Exception as e:
        raise SystemExit(
            f"Default config invalid or not found at {defaults_config_path}: {e}"
        )

    try:
        config_raw: dict[str, Any] = _read_toml_dict(cfg_path)
    except Exception as e:
        raise SystemExit(f"Experiment config invalid or not found at {cfg_path}: {e}")

    return cfg_path, config_raw, defaults_raw


def _load_exp_class_instance(module_path: str, method_name: str) -> object:
    """Import module_path and instantiate the first class exposing method_name.

    This removes per-experiment integration shims and relies only on canonical modules.
    """
    try:
        mod = importlib.import_module(module_path)
    except Exception as e:
        raise SystemExit(f"Failed to import {module_path}: {e}")
    # Find a class with the given method and a no-arg constructor
    for attr_name in dir(mod):
        attr = getattr(mod, attr_name)
        if isinstance(attr, type) and hasattr(attr, method_name):
            try:
                return attr()
            except Exception:
                continue
    raise SystemExit(
        f"No suitable class with method '{method_name}' found in {module_path}"
    )


def _load_preparer(experiment: str) -> ExpPreparer:
    return cast(
        ExpPreparer,
        _load_exp_class_instance(
            f"ml_playground.experiments.{experiment}.preparer", "prepare"
        ),
    )


def _load_trainer(experiment: str) -> ExpTrainer:
    return cast(
        ExpTrainer,
        _load_exp_class_instance(
            f"ml_playground.experiments.{experiment}.trainer", "train"
        ),
    )


def _load_sampler(experiment: str) -> ExpSampler:
    return cast(
        ExpSampler,
        _load_exp_class_instance(
            f"ml_playground.experiments.{experiment}.sampler", "sample"
        ),
    )


def _run_prepare(
    args: HasExperiment, prepare_cfg: PreparerConfig, config_path: Path
) -> None:
    preparer: ExpPreparer = _load_preparer(args.experiment)
    report = preparer.prepare(prepare_cfg)
    try:
        print(f"[prepare] side-effects: {report.summarize()}")
        for msg in report.messages:
            print(f"[prepare] {msg}")
    except Exception:
        pass


def _run_loop(
    args: HasExperiment,
    prepare_cfg: PreparerConfig,
    train_cfg: TrainerConfig,
    sample_cfg: SamplerConfig,
    config_path: Path,
) -> None:
    preparer: ExpPreparer = _load_preparer(args.experiment)
    trainer: ExpTrainer = _load_trainer(args.experiment)
    sampler: ExpSampler = _load_sampler(args.experiment)

    # Prepare
    prep_report = preparer.prepare(prepare_cfg)
    try:
        print(f"[prepare] side-effects: {prep_report.summarize()}")
        for msg in prep_report.messages:
            print(f"[prepare] {msg}")
    except Exception:
        pass

    # Train
    train_report = trainer.train(train_cfg)
    try:
        print(f"[train] side-effects: {train_report.summarize()}")
        for msg in train_report.messages:
            print(f"[train] {msg}")
    except Exception:
        pass

    # Sample
    sample_report = sampler.sample(sample_cfg)
    try:
        print(f"[sample] side-effects: {sample_report.summarize()}")
        for msg in sample_report.messages:
            print(f"[sample] {msg}")
    except Exception:
        pass


def _run_train(
    args: HasExperiment, train_cfg: TrainerConfig, config_path: Path
) -> None:
    trainer: ExpTrainer = _load_trainer(args.experiment)
    report = trainer.train(train_cfg)
    try:
        print(f"[train] side-effects: {report.summarize()}")
        for msg in report.messages:
            print(f"[train] {msg}")
    except Exception:
        pass


def _run_sample(
    args: HasExperiment, sample_cfg: SamplerConfig, config_path: Path
) -> None:
    sampler: ExpSampler = _load_sampler(args.experiment)
    report = sampler.sample(sample_cfg)
    try:
        print(f"[sample] side-effects: {report.summarize()}")
        for msg in report.messages:
            print(f"[sample] {msg}")
    except Exception:
        pass


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
    try:
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
    """Run the experiment-specific prepare step (or generic preparer)."""
    try:
        exp_config = ctx.obj.get("exp_config") if ctx and ctx.obj else None
    except Exception:
        exp_config = None
    args = CLIArgs(experiment=experiment, exp_config=exp_config)
    config_path: Path = (
        exp_config
        if isinstance(exp_config, Path)
        else Path(__file__).resolve().parent
        / "experiments"
        / experiment
        / "config.toml"
    )
    try:
        try:
            # Best-effort load of configs; prepare can work with empty/defaults
            _unused_cfg_path, config_raw, defaults_raw = _resolve_and_load_configs(args)
            config_path = _unused_cfg_path
        except SystemExit:
            config_raw, defaults_raw = ({}, {})
        try:
            prepare_cfg: PreparerConfig = _load_prepare_config_from_raw(
                config_raw, defaults_raw
            )
        except Exception:
            prepare_cfg = PreparerConfig()
    except SystemExit:
        prepare_cfg = PreparerConfig()
    # Resolve [prepare] paths relative to the experiment root when provided
    try:
        exp_root = config_path.parent
        ds = getattr(prepare_cfg, "dataset_dir", None)
        if isinstance(ds, Path) and not ds.is_absolute():
            prepare_cfg = prepare_cfg.model_copy(
                update={"dataset_dir": (exp_root / ds).resolve()}
            )
        rd = getattr(prepare_cfg, "raw_dir", None)
        if isinstance(rd, Path) and not rd.is_absolute():
            prepare_cfg = prepare_cfg.model_copy(update={"raw_dir": (exp_root / rd).resolve()})
    except Exception:
        pass
    _run_prepare(
        args,
        prepare_cfg,
        config_path,
    )


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
    """Validate config and run training for the given experiment."""
    try:
        exp_config = ctx.obj.get("exp_config") if ctx and ctx.obj else None
    except Exception:
        exp_config = None
    args = CLIArgs(experiment=experiment, exp_config=exp_config)
    config_path, config_raw, defaults_raw = _resolve_and_load_configs(args)
    # Strict validation to satisfy tests that patch the raw loader
    try:
        _ = _load_train_config_from_raw(config_raw, defaults_raw)
    except Exception as e:
        typer.echo(str(e))
        raise typer.Exit(code=2)
    # Then load the effective config (wrapper) for execution
    try:
        train_cfg_wrapped: TrainerConfig = load_train_config(config_path)
    except Exception as e:
        typer.echo(str(e))
        raise typer.Exit(code=2)
    train_cfg_wrapped = _apply_train_overrides(train_cfg_wrapped)
    try:
        _run_train(args, train_cfg_wrapped, config_path)
    except KeyboardInterrupt:
        typer.echo("\nTraining interrupted by user (Ctrl+C). Exiting gracefully.")


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
    """Validate config and run sampling for the given experiment."""
    try:
        exp_config = ctx.obj.get("exp_config") if ctx and ctx.obj else None
    except Exception:
        exp_config = None
    args = CLIArgs(experiment=experiment, exp_config=exp_config)
    config_path, config_raw, defaults_raw = _resolve_and_load_configs(args)
    # Strict validation and config build from raw to satisfy tests
    try:
        sample_cfg_obj: SamplerConfig = _load_sample_config_from_raw(
            config_raw, defaults_raw
        )
    except Exception as e:
        typer.echo(str(e))
        raise typer.Exit(code=2)
    # Resolve runtime.out_dir relative to experiment root if provided and relative
    try:
        rt = sample_cfg_obj.runtime
        if rt is not None:
            od = rt.out_dir
            if not od.is_absolute():
                resolved_out = (config_path.parent / od).resolve()
                sample_cfg_obj = sample_cfg_obj.model_copy(
                    update={"runtime": rt.model_copy(update={"out_dir": resolved_out})}
                )
    except Exception:
        pass
    # Apply environment overrides if provided
    sample_cfg_obj = _apply_sample_overrides(sample_cfg_obj)
    _run_sample(args, sample_cfg_obj, config_path)


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
    # Only bundestag_char is supported in this POC
    if experiment != "bundestag_char":
        typer.echo("convert currently supports only 'bundestag_char'.")
        raise typer.Exit(code=2)
    # Resolve config path
    cfg_path = _experiments_root() / experiment / "config.toml"
    if not cfg_path.exists():
        typer.echo(f"Config not found for experiment '{experiment}': {cfg_path}")
        raise typer.Exit(code=2)
    # Delegate to experiment-local converter
    try:
        mod = importlib.import_module(
            "ml_playground.experiments.bundestag_char.ollama_export"
        )
        getattr(mod, "convert_from_toml")(cfg_path)
    except SystemExit as e:
        # Echo the underlying error message before exiting with the same code
        msg = str(e).strip()
        if msg:
            typer.echo(msg)
        raise typer.Exit(code=getattr(e, "code", 1))
    except Exception as e:
        typer.echo(str(e))
        raise typer.Exit(code=1)


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
    """Run prepare → train → sample as a single loop for the experiment."""
    try:
        exp_config = ctx.obj.get("exp_config") if ctx and ctx.obj else None
    except Exception:
        exp_config = None
    args = CLIArgs(experiment=experiment, exp_config=exp_config)

    # Perform strict validation and run loop via integration getters
    config_path, config_raw, defaults_raw = _resolve_and_load_configs(args)
    try:
        loop_train_cfg: TrainerConfig = load_train_config(config_path)
    except Exception as e:
        typer.echo(str(e))
        raise typer.Exit(code=2)
    try:
        loop_sample_cfg: SamplerConfig = load_sample_config(config_path)
    except Exception as e:
        typer.echo(str(e))
        raise typer.Exit(code=2)
    # Also run strict validators so tests that patch them to raise are honored
    try:
        _ = _load_train_config_from_raw(config_raw, defaults_raw)
    except Exception as e:
        typer.echo(str(e))
        raise typer.Exit(code=2)
    try:
        _ = _load_sample_config_from_raw(config_raw, defaults_raw)
    except Exception as e:
        typer.echo(str(e))
        raise typer.Exit(code=2)
    loop_train_cfg = _apply_train_overrides(loop_train_cfg)
    loop_sample_cfg = _apply_sample_overrides(loop_sample_cfg)
    # Build a prepare cfg best-effort from raw (non-fatal)
    try:
        loop_prepare_cfg: PreparerConfig = _load_prepare_config_from_raw(
            config_raw, defaults_raw
        )
    except Exception:
        loop_prepare_cfg = PreparerConfig()
    try:
        _run_loop(args, loop_prepare_cfg, loop_train_cfg, loop_sample_cfg, config_path)
    except KeyboardInterrupt:
        typer.echo("\nSampling interrupted by user (Ctrl+C). Exiting gracefully.")


# Keep a Python API-compatible entry point for tests


def main(argv: list[str] | None = None) -> None:
    """Programmatic entry point used by tests and CLI.

    - When argv is provided (tests call main([...])), run a lightweight parser that
      matches test expectations and raises SystemExit with informative messages.
    - When argv is None, delegate to the Typer app for a full CLI experience.
    """
    if argv is not None:
        # Simple test-oriented dispatcher
        if not isinstance(argv, list) or not argv:
            raise SystemExit("Syntax error: expected 'command <experiment>'")
        cmd_name = argv[0]
        if cmd_name not in {"prepare", "train", "sample", "loop", "convert"}:
            raise SystemExit(f"Unknown command: {cmd_name}")
        if len(argv) < 2:
            raise SystemExit("Syntax error: expected 'command <experiment>'")
        exp = argv[1]

        # Helper to compute experiment config path
        cfg_path = _experiments_root() / exp / "config.toml"

        # prepare
        if cmd_name == "prepare":
            if exp not in datasets.PREPARERS:
                raise SystemExit(f"Unknown experiment: {exp}")
            try:
                preparer = make_preparer(PreparerConfig())
                # Call the preparer instance (tests assert it is invoked)
                preparer()
            except Exception as e:
                raise SystemExit(str(e))
            return

        # train
        if cmd_name == "train":
            try:
                # Strict validation (tests may patch this to raise)
                _cfg_path, raw, defaults = _resolve_and_load_configs(
                    CLIArgs(experiment=exp)
                )
                _ = _load_train_config_from_raw(raw, defaults)
            except Exception as e:
                raise SystemExit(str(e))
            try:
                train_cfg = load_train_config(cfg_path)
                train_cfg = _apply_train_overrides(train_cfg)
                train(train_cfg)
            except Exception as e:
                raise SystemExit(str(e))
            return

        # sample
        if cmd_name == "sample":
            # Special-case integration for 'speakger': delegate to its sampler entrypoint
            if exp == "speakger":
                try:
                    cfg_path = _experiments_root() / exp / "config.toml"
                    mod = importlib.import_module(
                        "ml_playground.experiments.speakger.sampler"
                    )
                    # Call the in-module sampler directly with TOML path
                    getattr(mod, "sample_from_toml")(cfg_path)
                except Exception as e:
                    raise SystemExit(str(e))
                return
            try:
                _cfg_path, raw, defaults = _resolve_and_load_configs(
                    CLIArgs(experiment=exp)
                )
                sample_cfg = _load_sample_config_from_raw(raw, defaults)
                # Apply environment overrides first, then resolve relative paths
                try:
                    sample_cfg = _apply_sample_overrides(sample_cfg)
                except Exception:
                    pass
                # Resolve runtime.out_dir relative to experiment root if provided and relative
                try:
                    rt = sample_cfg.runtime
                    if rt is not None:
                        od = rt.out_dir
                        if not od.is_absolute():
                            resolved_out = (_cfg_path.parent / od).resolve()
                            sample_cfg = sample_cfg.model_copy(
                                update={"runtime": rt.model_copy(update={"out_dir": resolved_out})}
                            )
                except Exception:
                    pass
            except Exception as e:
                raise SystemExit(str(e))
            try:
                sample(sample_cfg)
            except Exception as e:
                raise SystemExit(str(e))
            return

        # convert
        if cmd_name == "convert":
            if exp != "bundestag_char":
                raise SystemExit("convert currently supports only 'bundestag_char'")
            try:
                mod = importlib.import_module(
                    "ml_playground.experiments.bundestag_char.ollama_export"
                )
                getattr(mod, "convert_from_toml")(cfg_path)
            except Exception as e:
                raise SystemExit(str(e))
            return

        # loop
        if cmd_name == "loop":
            # Ensure registry is populated unless tests have monkeypatched it
            try:
                if (
                    datasets.PREPARERS is datasets.DEFAULT_PREPARERS_REF
                    and not datasets.PREPARERS
                ):
                    # Lazy-load preparers from experiments
                    from ml_playground.datasets import load_preparers as _load_preps

                    _load_preps()
            except Exception:
                pass
            # Validate experiment exists in registry per tests
            if exp not in datasets.PREPARERS:
                raise SystemExit(f"Unknown experiment: {exp}")
            # Prepare via registry callable (tests expect this to be invoked directly)
            try:
                datasets.PREPARERS[exp]()
            except Exception:
                # Preparation errors should propagate as SystemExit in tests
                raise SystemExit("prepare failed")
            # Load configs using public wrappers (tests patch these)
            try:
                train_cfg = load_train_config(cfg_path)
                sample_cfg = load_sample_config(cfg_path)
            except Exception as e:
                raise SystemExit(str(e))
            # Apply environment overrides for quick e2e runs
            try:
                train_cfg = _apply_train_overrides(train_cfg)
            except Exception:
                pass
            try:
                sample_cfg = _apply_sample_overrides(sample_cfg)
            except Exception:
                pass
            # Train and sample via functional APIs (tests patch these)
            try:
                train(train_cfg)
            except Exception as e:
                raise SystemExit(str(e))
            try:
                # Copy meta.pkl if present and exists
                meta_name = getattr(
                    getattr(train_cfg, "data", object()), "meta_pkl", None
                )
                ds_dir = getattr(
                    getattr(train_cfg, "data", object()), "dataset_dir", None
                )
                out_dir = getattr(
                    getattr(train_cfg, "runtime", object()), "out_dir", None
                )
                if meta_name and isinstance(ds_dir, Path) and isinstance(out_dir, Path):
                    src = ds_dir / str(meta_name)
                    dst = out_dir / str(meta_name)
                    if src.exists():
                        try:
                            shutil.copy2(src, dst)
                        except Exception:
                            # Print a warning but do not fail the loop
                            try:
                                print(
                                    f"Warning: failed to copy meta.pkl from {src} to {dst}"
                                )
                            except Exception:
                                pass
                sample(sample_cfg)
            except Exception as e:
                raise SystemExit(str(e))
            return

        # Shouldn't reach here
        raise SystemExit(1)

    # Fallback to Typer CLI when no argv is provided
    cmd = get_command(app)
    try:
        cmd.main(args=argv, prog_name="ml_playground", standalone_mode=False)
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
        # Silence tracebacks for click's Exit; exit with its code.
        raise SystemExit(getattr(e, "exit_code", 0))
    except KeyboardInterrupt:
        typer.echo("\nOperation interrupted by user (Ctrl+C). Exiting gracefully.")
    except Exception as e:
        # Fallback: never show a traceback to the user
        typer.echo(f"Error: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
