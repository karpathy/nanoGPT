from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from ml_playground.configuration.loading import (
    get_cfg_path,
    load_full_experiment_config,
)
from ml_playground.configuration.models import ExperimentConfig

_PROJECT_HOME = Path(__file__).resolve().parent.parent.parent


def cfg_path_for(experiment: str, exp_config: Optional[Path]) -> Path:
    """Return the canonical path to an experiment configuration file."""
    return get_cfg_path(experiment, exp_config)


def load_experiment(experiment: str, exp_config: Optional[Path]) -> ExperimentConfig:
    """Load the fully merged configuration for a CLI invocation."""
    cfg_path = cfg_path_for(experiment, exp_config)
    return load_full_experiment_config(cfg_path, _PROJECT_HOME, experiment)


def ensure_train_prerequisites(exp: ExperimentConfig) -> Path:
    """Ensure required training artifacts exist before running ``train``."""
    train_meta = exp.shared.dataset_dir / "meta.pkl"
    if not train_meta.exists():
        raise ValueError(
            "Missing required meta file for training: "
            f"{train_meta}.\nRun 'prepare' first or ensure your preparer writes meta.pkl."
        )
    return train_meta


def ensure_sample_prerequisites(exp: ExperimentConfig) -> Tuple[Path, Path]:
    """Ensure sampling has access to metadata produced during training."""
    train_meta = exp.shared.dataset_dir / "meta.pkl"
    runtime_meta = exp.shared.sample_out_dir / exp.shared.experiment / "meta.pkl"
    if not (train_meta.exists() or runtime_meta.exists()):
        raise ValueError(
            "Missing required meta file for sampling. Checked: "
            f"train.meta={train_meta}, runtime.meta={runtime_meta}.\n"
            "Run 'prepare' and 'train' first or place meta.pkl in one of the expected locations."
        )
    return train_meta, runtime_meta


__all__ = [
    "cfg_path_for",
    "load_experiment",
    "ensure_train_prerequisites",
    "ensure_sample_prerequisites",
]
