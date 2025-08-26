from __future__ import annotations

from pathlib import Path
from ml_playground.config import TrainerConfig, SamplerConfig
from ml_playground.prepare import PreparerConfig


def _config_path() -> Path:
    # Resolve the experiment-local config.toml
    return Path(__file__).resolve().parent / "config.toml"


def prepare(cfg: PreparerConfig) -> None:  # noqa: ARG001 - cfg currently unused
    # Delegate to the experiment's Gemma/PEFT integration
    from . import gemma_finetuning_mps as gm

    try:
        gm.prepare_from_toml(_config_path())
    except Exception:
        # Non-fatal: training path may surface a clearer error later
        pass


def train(cfg: TrainerConfig) -> None:  # noqa: ARG001 - cfg not used by integration
    from . import gemma_finetuning_mps as gm

    # Ensure dataset is prepared before training (idempotent)
    try:
        gm.prepare_from_toml(_config_path())
    except Exception:
        # Let the trainer surface clearer errors if needed
        pass
    gm.train_from_toml(_config_path())


def sample(cfg: SamplerConfig) -> None:  # noqa: ARG001 - cfg not used by integration
    from . import gemma_finetuning_mps as gm

    gm.sample_from_toml(_config_path())


def loop(
    prepare: PreparerConfig,  # noqa: A002 - match CLI protocol
    train: TrainerConfig,  # noqa: A002 - match CLI protocol
    sample: SamplerConfig,  # noqa: A002 - match CLI protocol
) -> None:
    from . import gemma_finetuning_mps as gm

    try:
        gm.prepare_from_toml(_config_path())
    except Exception:
        pass
    gm.train_from_toml(_config_path())
    gm.sample_from_toml(_config_path())
