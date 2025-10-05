from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, TypedDict, cast

import tomllib

from ml_playground.configuration.models import (
    ExperimentConfig,
    PreparerConfig,
    SamplerConfig,
    TrainerConfig,
)
from ml_playground.configuration.merge_utils import merge_mappings

logger = logging.getLogger(__name__)

TomlMapping = Dict[str, Any]


class ExperimentPayload(TypedDict, total=False):
    shared: TomlMapping
    prepare: TomlMapping
    train: TomlMapping
    sample: TomlMapping


def get_cfg_path(experiment: str, exp_config: Path | None) -> Path:
    if exp_config:
        return exp_config
    return (
        Path(__file__).resolve().parent.parent
        / "experiments"
        / experiment
        / "config.toml"
    )


def get_default_config_path(project_root: Path | None = None) -> Path:
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    return _default_config_path_from_root(project_root)


def list_experiments_with_config(prefix: str = "") -> list[str]:
    root = Path(__file__).resolve().parent.parent / "experiments"
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


def _ensure_mapping(value: Any, context: str) -> TomlMapping:
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected mapping for {context}")
    return dict(value)


def read_toml_dict(
    path: Path,
    *,
    toml_loader: Callable[[str], Mapping[str, Any]] | None = None,
) -> TomlMapping:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        loader = toml_loader if toml_loader is not None else tomllib.loads
        data = loader(text)
    except tomllib.TOMLDecodeError as exc:
        raise Exception(f"{path.name}: {exc}")
    if not isinstance(data, dict):
        raise TypeError(f"TOML root in {path} must be a mapping")
    return cast(TomlMapping, data)


def _default_config_path_from_root(project_root: Path) -> Path:
    return project_root / "ml_playground" / "experiments" / "default_config.toml"


def _load_and_merge_configs(
    config_path: Path, project_home: Path, experiment_name: str
) -> ExperimentPayload:
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

    merged = merge_mappings(defaults_raw, raw_exp, override_only=True)

    merged_payload = merge_mappings(merged, ldres_raw)
    return cast(ExperimentPayload, merged_payload)


def load_full_experiment_config(
    config_path: Path, project_home: Path, experiment_name: str
) -> ExperimentConfig:
    effective_config = _load_and_merge_configs(
        config_path, project_home, experiment_name
    )

    shared = _ensure_mapping(effective_config.setdefault("shared", {}), "[shared]")
    shared["config_path"] = config_path
    shared["project_home"] = project_home
    shared["experiment"] = experiment_name
    effective_config["shared"] = shared

    return ExperimentConfig.model_validate(
        effective_config,
        context={"config_path": config_path},
    )


def load_train_config(
    config_path: Path, *, default_config_path: Path | None = None
) -> TrainerConfig:
    raw_exp = read_toml_dict(config_path)
    project_root = Path(__file__).resolve().parent.parent.parent
    defaults_path = (
        default_config_path
        if default_config_path is not None
        else _default_config_path_from_root(project_root)
    )
    defaults_raw = read_toml_dict(defaults_path) if defaults_path.exists() else {}

    raw_merged = merge_mappings(defaults_raw, raw_exp)

    train_data = _ensure_mapping(raw_merged.get("train", {}), "[train] section")

    context = {"config_path": config_path}
    cfg = TrainerConfig.model_validate(train_data, context=context)

    info = {"raw": raw_merged, "context": {"config_path": str(config_path)}}
    cfg.extras["provenance"] = info
    return cfg


def load_sample_config(
    config_path: Path, *, default_config_path: Path | None = None
) -> SamplerConfig:
    raw_exp = read_toml_dict(config_path)
    project_root = Path(__file__).resolve().parent.parent.parent
    defaults_path = _default_config_path_from_root(project_root)
    defaults_raw = read_toml_dict(defaults_path) if defaults_path.exists() else {}

    raw_merged = merge_mappings(defaults_raw, raw_exp)

    if "sample" not in raw_exp:
        raise ValueError("Config must contain a [sample] section")

    sample_data = _ensure_mapping(raw_merged.get("sample", {}), "[sample] section")

    context = {"config_path": config_path}
    cfg = SamplerConfig.model_validate(sample_data, context=context)

    info = {"raw": raw_merged, "context": {"config_path": str(config_path)}}
    cfg.extras["provenance"] = info
    return cfg


def load_prepare_config(
    config_path: Path, *, default_config_path: Path | None = None
) -> PreparerConfig:
    raw_exp = read_toml_dict(config_path)
    project_root = Path(__file__).resolve().parent.parent.parent
    defaults_path = _default_config_path_from_root(project_root)
    defaults_raw = read_toml_dict(defaults_path) if defaults_path.exists() else {}

    raw_merged = merge_mappings(defaults_raw, raw_exp)

    if "prepare" not in raw_merged:
        raise ValueError("Config must contain a [prepare] section")

    prepare_data = _ensure_mapping(raw_merged.get("prepare", {}), "[prepare] section")

    context = {"config_path": config_path}
    cfg = PreparerConfig.model_validate(prepare_data, context=context)

    info = {"raw": raw_merged, "context": {"config_path": str(config_path)}}
    cfg.extras["provenance"] = info
    return cfg


def load_experiment_toml(path: Path) -> ExperimentConfig:
    project_home = Path(__file__).resolve().parent.parent
    experiment_name = path.parent.name
    return load_full_experiment_config(path, project_home, experiment_name)


__all__ = [
    "ExperimentPayload",
    "TomlMapping",
    "get_cfg_path",
    "get_default_config_path",
    "list_experiments_with_config",
    "read_toml_dict",
    "merge_mappings",
    "load_full_experiment_config",
    "load_train_config",
    "load_sample_config",
    "load_prepare_config",
    "load_experiment_toml",
]
