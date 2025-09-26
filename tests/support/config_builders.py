"""Shared test configuration builders for consistent test setup."""

from pathlib import Path

from ml_playground.config import (
    DataConfig,
    LRSchedule,
    ModelConfig,
    OptimConfig,
    PreparerConfig,
    RuntimeConfig,
    SampleConfig,
    SamplerConfig,
    SharedConfig,
    TrainerConfig,
)


def create_basic_configs(
    tmp_path: Path,
) -> tuple[PreparerConfig, TrainerConfig, SamplerConfig, SharedConfig]:
    """Create basic test configurations for CLI and pipeline testing.

    Args:
        tmp_path: Temporary directory path for test outputs.

    Returns:
        Tuple of (preparer_config, trainer_config, sampler_config, shared_config)
    """
    prep_cfg = PreparerConfig()
    tcfg = TrainerConfig(
        model=ModelConfig(),
        data=DataConfig(),
        optim=OptimConfig(),
        schedule=LRSchedule(),
        runtime=RuntimeConfig(out_dir=tmp_path),
    )
    scfg = SamplerConfig(runtime=RuntimeConfig(out_dir=tmp_path), sample=SampleConfig())
    shared = SharedConfig(
        experiment="exp",
        config_path=tmp_path / "cfg.toml",
        project_home=tmp_path,
        dataset_dir=tmp_path,
        train_out_dir=tmp_path,
        sample_out_dir=tmp_path,
    )
    return prep_cfg, tcfg, scfg, shared


def create_experiment_shared_config(
    tmp_path: Path, experiment: str = "test_exp"
) -> SharedConfig:
    """Create a SharedConfig for experiment testing.

    Args:
        tmp_path: Temporary directory path.
        experiment: Experiment name.

    Returns:
        SharedConfig instance.
    """
    return SharedConfig(
        experiment=experiment,
        config_path=tmp_path / "config.toml",
        project_home=tmp_path,
        dataset_dir=tmp_path / "datasets",
        train_out_dir=tmp_path / "train",
        sample_out_dir=tmp_path / "sample",
    )
