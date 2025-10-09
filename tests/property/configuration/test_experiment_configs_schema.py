from __future__ import annotations

from pathlib import Path

import pytest

from ml_playground.configuration.loading import load_full_experiment_config
from ml_playground.configuration.models import ExperimentConfig

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS_DIR = PROJECT_ROOT / "ml_playground" / "experiments"


def _discover_experiment_configs() -> list[Path]:
    return sorted(EXPERIMENTS_DIR.glob("*/config.toml"))


ALL_CONFIG_PATHS = _discover_experiment_configs()


@pytest.mark.parametrize(
    "config_path",
    ALL_CONFIG_PATHS,
    ids=[p.parent.name for p in ALL_CONFIG_PATHS],
)
def test_experiment_config_validates(config_path: Path) -> None:
    experiment_name = config_path.parent.name
    cfg = load_full_experiment_config(
        config_path=config_path,
        project_home=PROJECT_ROOT,
        experiment_name=experiment_name,
    )
    assert isinstance(cfg, ExperimentConfig)
