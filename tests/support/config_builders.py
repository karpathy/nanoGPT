"""Shared test configuration builders for consistent test setup."""

from pathlib import Path

from ml_playground.configuration.models import (
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
    """Create strict-ready configs for prepare/train/sample flows."""

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_text = raw_dir / "input.txt"
    raw_text.write_text("dummy", encoding="utf-8")

    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    train_out = tmp_path / "train"
    train_out.mkdir(parents=True, exist_ok=True)
    sample_out = tmp_path / "sample"
    sample_out.mkdir(parents=True, exist_ok=True)

    prep_cfg = PreparerConfig(raw_dir=raw_dir, raw_text_path=raw_text)

    tcfg = TrainerConfig(
        model=ModelConfig(),
        data=DataConfig(),
        optim=OptimConfig(),
        schedule=LRSchedule(),
        runtime=RuntimeConfig(out_dir=train_out),
        hf_model=TrainerConfig.HFModelConfig(
            model_name="hf/model",
            gradient_checkpointing=False,
            block_size=128,
        ),
        peft=TrainerConfig.PeftConfig(enabled=False),
    )
    scfg = SamplerConfig(
        runtime=RuntimeConfig(out_dir=sample_out),
        sample=SampleConfig(),
    )
    shared = SharedConfig(
        experiment="exp",
        config_path=tmp_path / "cfg.toml",
        project_home=tmp_path,
        dataset_dir=dataset_dir,
        train_out_dir=train_out,
        sample_out_dir=sample_out,
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
