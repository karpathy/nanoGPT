from __future__ import annotations

import typer
import click
from typer.main import get_command
import importlib
import json
import os
import pickle
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

# --- Typer helpers ---------------------------------------------------------
def _experiments_root() -> Path:
    """Return the root folder that contains experiment directories."""
    return Path(__file__).resolve().parent / "experiments"

def _complete_experiments(
    ctx: typer.Context, param: typer.CallbackParam, incomplete: str
) -> list[str]:
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

    - Ensures required subsections exist ([runtime], [sample]).
    - Enforces unknown-key rejection outside [sample.extras].
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

    # Strictness: only runtime, sample, extras allowed at this level
    allowed_top = {"runtime", "sample", "extras"}
    unknown_top = set(sample_tbl.keys()) - allowed_top
    if unknown_top:
        raise ValueError("Unknown key(s) in [sample] (outside extras)")

    for sec in ("runtime", "sample"):
        if not isinstance(sample_tbl.get(sec), dict):
            raise ValueError(f"Missing required section [{sec}]")

    # Allowed keys pruning for runtime/sample according to models
    allowed_runtime_keys = set(RuntimeConfig.model_fields.keys())

    # Build merged dict and provenance
    d = {"runtime": dict(sample_tbl["runtime"]), "sample": dict(sample_tbl["sample"])}

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

    for sec in ("runtime", "sample"):
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

    # No path heuristics; prune unknown runtime keys only
    d["runtime"] = {k: v for k, v in d["runtime"].items() if k in allowed_runtime_keys}

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


def _deep_merge_dicts(base: dict, override: dict) -> dict:
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

    - Ensures required subsections exist ([runtime], [sample]).
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

    for sec in ("runtime", "sample"):
        if not isinstance(sample_tbl.get(sec), dict):
            raise ValueError(f"Missing required section [{sec}]")

    # Allowed keys pruning for runtime/sample according to models
    allowed_runtime_keys = set(RuntimeConfig.model_fields.keys())
    # SampleConfig uses its own schema; we keep all keys and let pydantic forbid extras on model_validate

    # Build merged dict and provenance
    d = {"runtime": dict(sample_tbl["runtime"]), "sample": dict(sample_tbl["sample"])}

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

    for sec in ("runtime", "sample"):
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

    # Environment overrides are not supported: only default + experiment TOMLs are used.

    # No path heuristics; prune unknown runtime keys only
    d["runtime"] = {k: v for k, v in d["runtime"].items() if k in allowed_runtime_keys}

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
    # Resolve dataset_dir relative to experiment root if it is not absolute
    try:
        ds = cfg.data.dataset_dir
        if not ds.is_absolute():
            resolved = (path.parent / ds).resolve()
            cfg = cfg.model_copy(
                update={"data": cfg.data.model_copy(update={"dataset_dir": resolved})}
            )
    except Exception:
        # Keep original on any resolution error
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
        out_dir = cfg.runtime.out_dir
        if not out_dir.is_absolute():
            resolved_out = (path.parent / out_dir).resolve()
            cfg = cfg.model_copy(
                update={
                    "runtime": cfg.runtime.model_copy(update={"out_dir": resolved_out})
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
    cfg_path = Path(__file__).resolve().parent / "experiments" / exp / "config.toml"
    if not cfg_path.exists():
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


class ExperimentIntegration(Protocol):
    def prepare(self, cfg: PreparerConfig) -> None: ...

    def train(self, cfg: TrainerConfig) -> None: ...

    def sample(self, cfg: SamplerConfig) -> None: ...

    def loop(
        self, prepare: PreparerConfig, train: TrainerConfig, sample: SamplerConfig
    ) -> None: ...


def _try_load_experiment_integration(experiment: str) -> ExperimentIntegration | None:
    """
    Dynamic import of an experiment integration module:
    ml_playground.experiments.<experiment>.integration

    If present, returns a module object assumed to implement ExperimentIntegration.
    If the module is not found, returns None to allow generic fallback.
    """
    try:
        mod = importlib.import_module(
            f"ml_playground.experiments.{experiment}.integration"
        )
    except ModuleNotFoundError:
        return None
    except Exception as e:
        # Fail fast on other import errors
        raise SystemExit(f"Failed to import integration for '{experiment}': {e}")
    return cast(ExperimentIntegration, mod)


def _run_prepare(
    args: HasExperiment, prepare_cfg: PreparerConfig, config_path: Path
) -> None:
    # Try experiment-specific integration first
    integration = _try_load_experiment_integration(args.experiment)
    if integration:
        integration.prepare(prepare_cfg)
        return

    # Validate experiment name using registry, then construct and run the default preparer instance
    if datasets.PREPARERS is datasets.DEFAULT_PREPARERS_REF:
        datasets.load_preparers()
    registry = datasets.PREPARERS
    if args.experiment not in registry:
        raise SystemExit(f"Unknown experiment: {args.experiment}")

    preparer = make_preparer(prepare_cfg)
    preparer()


def _run_loop(
    args: HasExperiment,
    prepare_cfg: PreparerConfig,
    train_cfg: TrainerConfig,
    sample_cfg: SamplerConfig,
    config_path: Path,
) -> None:
    # Try experiment-specific integration loop first
    integration = _try_load_experiment_integration(args.experiment)
    if integration:
        integration.loop(prepare_cfg, train_cfg, sample_cfg)
        return

    # Deterministic plugin load and dispatch with DI support
    if datasets.PREPARERS is datasets.DEFAULT_PREPARERS_REF:
        datasets.load_preparers()
    registry = datasets.PREPARERS
    prepare_fn = registry.get(args.experiment)
    if prepare_fn is None:
        raise SystemExit(f"Unknown experiment: {args.experiment}")
    # 1) prepare
    prepare_fn()
    # 2) train with typed config
    train(train_cfg)
    # Ensure out_dir/meta.pkl matches dataset meta for sampling (best-effort)
    data_cfg = train_cfg.data
    if data_cfg.meta_pkl is not None:
        src_meta = data_cfg.dataset_dir / data_cfg.meta_pkl
        if src_meta.exists():
            dst_meta = train_cfg.runtime.out_dir / "meta.pkl"
            train_cfg.runtime.out_dir.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(src_meta, dst_meta)
            except Exception as e:
                try:
                    print(
                        f"[warn] Failed to copy meta.pkl from {src_meta} to {dst_meta}: {e}"
                    )
                except Exception:
                    pass
            else:
                # Validate only if the file is present and readable
                try:
                    if dst_meta.exists():
                        with dst_meta.open("rb") as f:
                            meta = pickle.load(f)
                        if not isinstance(meta, dict) or "meta_version" not in meta:
                            print(
                                f"[warn] Invalid meta.pkl copied from {src_meta}: missing required 'meta_version'. "
                                "Re-run prepare to regenerate dataset artifacts."
                            )
                except Exception:
                    # Non-fatal; continue to sampling
                    pass
    # 3) sample with typed config
    try:
        sample(sample_cfg)
    except KeyboardInterrupt:
        print("\nSampling interrupted by user (Ctrl+C). Exiting gracefully.")
    return


def _run_train(
    args: HasExperiment, train_cfg: TrainerConfig, config_path: Path
) -> None:
    # Try experiment-specific integration trainer
    integration = _try_load_experiment_integration(args.experiment)
    if integration:
        # Allow integration to handle its own preparation if needed
        integration.train(train_cfg)
        return

    train(train_cfg)


def _run_sample(
    args: HasExperiment, sample_cfg: SamplerConfig, config_path: Path
) -> None:
    # Try experiment-specific integration sampler
    integration = _try_load_experiment_integration(args.experiment)
    if integration:
        integration.sample(sample_cfg)
        return

    # In CLI sample mode, avoid filesystem coupling to dataset artifacts; delegate to sampler.
    sample(sample_cfg)


 # Typer-based CLI
app = typer.Typer(no_args_is_help=True, help="ML Playground CLI: prepare data, train models, and sample outputs.")


@app.command("prepare")
def cmd_prepare(
    experiment: Annotated[
        str,
        typer.Argument(
            help="Experiment name (folder under experiments/)",
            autocompletion=_complete_experiments,
        ),
    ]
) -> None:
    """Run the experiment-specific prepare step (or generic preparer)."""
    args = CLIArgs(experiment=experiment)
    try:
        try:
            _unused_cfg, config_raw, defaults_raw = _resolve_and_load_configs(args)
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
    _run_prepare(
        args,
        prepare_cfg,
        Path(
            f"{Path(__file__).resolve().parent / 'experiments' / experiment / 'config.toml'}"
        ),
    )


@app.command("train")
def cmd_train(
    experiment: Annotated[
        str,
        typer.Argument(
            help="Experiment name (folder under experiments/)",
            autocompletion=_complete_experiments,
        ),
    ]
) -> None:
    """Validate config and run training for the given experiment."""
    args = CLIArgs(experiment=experiment)
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
    experiment: Annotated[
        str,
        typer.Argument(
            help="Experiment name (folder under experiments/)",
            autocompletion=_complete_experiments,
        ),
    ]
) -> None:
    """Validate config and run sampling for the given experiment."""
    args = CLIArgs(experiment=experiment)
    config_path, config_raw, defaults_raw = _resolve_and_load_configs(args)
    # Strict validation and config build from raw to satisfy tests
    try:
        sample_cfg_obj: SamplerConfig = _load_sample_config_from_raw(
            config_raw, defaults_raw
        )
    except Exception as e:
        typer.echo(str(e))
        raise typer.Exit(code=2)
    # Apply environment overrides if provided
    sample_cfg_obj = _apply_sample_overrides(sample_cfg_obj)
    _run_sample(args, sample_cfg_obj, config_path)


@app.command("loop")
def cmd_loop(
    experiment: Annotated[
        str,
        typer.Argument(
            help="Experiment name (folder under experiments/)",
            autocompletion=_complete_experiments,
        ),
    ]
) -> None:
    """Run prepare → train → sample as a single loop for the experiment."""
    args = CLIArgs(experiment=experiment)
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
    cmd = get_command(app)
    try:
        cmd.main(args=argv, prog_name="ml_playground", standalone_mode=False)
    except click.exceptions.NoArgsIsHelpError:
        # Help was shown (no args). Exit cleanly without traceback.
        return


if __name__ == "__main__":
    main()
