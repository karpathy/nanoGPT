from __future__ import annotations
from pathlib import Path
# Re-export Pydantic v2 models as canonical config types
# The legacy dataclass-based models and bespoke load_toml() were removed.
# Public imports should use these names as before:
#   from ml_playground.config import TrainExperiment, SampleExperiment, ...
# For direct model access prefer: from ml_playground.config_models import AppConfig

from .config_models import (
    DeviceKind,
    DTypeKind,
    OptimConfig,
    LRSchedule,
    ModelConfig,
    DataConfig,
    RuntimeConfig,
    SampleConfig,
    TrainExperiment,
    SampleExperiment,
    AppConfig,
)
from .cli_config import load_config as _load_config


def load_toml(path: Path) -> AppConfig:
    """Compatibility wrapper for legacy tests. Delegates to CLI loader.

    Note: Only the CLI performs TOML I/O; this function simply forwards to
    ml_playground.cli_config.load_config.
    """
    return _load_config(path)


__all__ = [
    "DeviceKind",
    "DTypeKind",
    "OptimConfig",
    "LRSchedule",
    "ModelConfig",
    "DataConfig",
    "RuntimeConfig",
    "SampleConfig",
    "TrainExperiment",
    "SampleExperiment",
    "AppConfig",
    "load_toml",
]
