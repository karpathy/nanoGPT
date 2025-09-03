from __future__ import annotations

from pathlib import Path

import pytest

from ml_playground import config_loader
from ml_playground.config import (
    TrainerConfig,
    ModelConfig,
    DataConfig,
    OptimConfig,
    LRSchedule,
    RuntimeConfig,
)


def test_strict_mode_has_no_override_functions() -> None:
    # Strict mode: configuration is TOML-only, no override helpers are exposed
    assert not hasattr(config_loader, "apply_train_overrides")
    assert not hasattr(config_loader, "apply_sample_overrides")
