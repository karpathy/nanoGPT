from __future__ import annotations
from pathlib import Path
from typing import Any, Literal, Optional

import tomllib
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Strict, single-source configuration module.
# All config models and the TOML loader live here.

DeviceKind = Literal["cpu", "mps", "cuda"]
DTypeKind = Literal["float32", "bfloat16", "float16"]


class _FrozenStrictModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)


class OptimConfig(_FrozenStrictModel):
    learning_rate: float = 6e-4
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    @field_validator("learning_rate", "weight_decay", "beta1", "beta2", "grad_clip")
    @classmethod
    def _non_negative(cls, v: float) -> float:
        if v != v:  # NaN
            raise ValueError("must not be NaN")
        if v < 0:
            raise ValueError("must be >= 0")
        return float(v)


class LRSchedule(_FrozenStrictModel):
    decay_lr: bool = True
    warmup_iters: int = 2_000
    lr_decay_iters: int = 600_000
    min_lr: float = 6e-5

    @field_validator("warmup_iters", "lr_decay_iters")
    @classmethod
    def _non_negative_int(cls, v: int) -> int:
        if v < 0:
            raise ValueError("must be >= 0")
        return int(v)

    @model_validator(mode="after")
    def _check_warmup_le_decay(self) -> "LRSchedule":
        if self.warmup_iters > self.lr_decay_iters:
            raise ValueError("warmup_iters must be <= lr_decay_iters")
        if self.min_lr < 0:
            raise ValueError("min_lr must be >= 0")
        return self


class ModelConfig(_FrozenStrictModel):
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 1024
    dropout: float = 0.0
    bias: bool = False
    vocab_size: Optional[int] = None

    @field_validator("n_layer", "n_head", "n_embd", "block_size")
    @classmethod
    def _positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be > 0")
        return int(v)

    @field_validator("dropout")
    @classmethod
    def _dropout_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("dropout must be in [0, 1]")
        return float(v)

    @field_validator("vocab_size")
    @classmethod
    def _vocab_positive(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return None
        if v <= 0:
            raise ValueError("vocab_size must be > 0 if provided")
        return int(v)


class DataConfig(_FrozenStrictModel):
    dataset_dir: Path
    train_bin: str = "train.bin"
    val_bin: str = "val.bin"
    meta_pkl: Optional[str] = "meta.pkl"
    batch_size: int = 12
    block_size: int = 1024
    grad_accum_steps: int = 40

    @field_validator("dataset_dir", mode="before")
    @classmethod
    def _coerce_path(cls, v: Path | str) -> Path:
        return Path(v)

    @field_validator("batch_size", "block_size", "grad_accum_steps")
    @classmethod
    def _positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be > 0")
        return int(v)


class RuntimeConfig(_FrozenStrictModel):
    out_dir: Path
    max_iters: int = 600_000
    eval_interval: int = 2_000
    eval_iters: int = 200
    log_interval: int = 1
    eval_only: bool = False
    always_save_checkpoint: bool = True
    seed: int = 1337
    device: DeviceKind = "cpu"
    dtype: DTypeKind = "float32"
    compile: bool = False

    # Checkpoint policy
    ckpt_last_filename: str = "ckpt_last.pt"
    ckpt_best_filename: str = "ckpt_best.pt"
    ckpt_top_k: int = 0
    ckpt_metric: Literal["val_loss", "perplexity"] = "val_loss"
    ckpt_greater_is_better: bool = False
    ckpt_atomic: bool = True
    ckpt_write_metadata: bool = True
    ckpt_time_interval_minutes: int = 0
    # Smoothed improvement + early stopping
    best_smoothing_alpha: float = 0.0
    early_stop_patience: int = 0
    # Exponential moving average of weights
    ema_decay: float = 0.0

    @field_validator("out_dir", mode="before")
    @classmethod
    def _coerce_path(cls, v: Path | str) -> Path:
        return Path(v)

    @field_validator(
        "max_iters",
        "eval_interval",
        "eval_iters",
        "log_interval",
        "seed",
        "ckpt_top_k",
        "ckpt_time_interval_minutes",
        "early_stop_patience",
    )
    @classmethod
    def _non_negative_int(cls, v: int) -> int:
        if v < 0:
            raise ValueError("must be >= 0")
        return int(v)

    @field_validator("best_smoothing_alpha", "ema_decay")
    @classmethod
    def _in_unit_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("must be in [0, 1]")
        return float(v)


class SampleConfig(_FrozenStrictModel):
    start: str = "\n"
    num_samples: int = 3
    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 200

    @field_validator("num_samples", "max_new_tokens")
    @classmethod
    def _positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be > 0")
        return int(v)

    @field_validator("top_k")
    @classmethod
    def _non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("must be >= 0")
        return int(v)

    @field_validator("temperature")
    @classmethod
    def _positive_float(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("must be > 0")
        return float(v)


class TrainExperiment(_FrozenStrictModel):
    model: ModelConfig
    data: DataConfig
    optim: OptimConfig
    schedule: LRSchedule
    runtime: RuntimeConfig


class SampleExperiment(_FrozenStrictModel):
    runtime: RuntimeConfig
    sample: SampleConfig


class AppConfig(_FrozenStrictModel):
    train: Optional[TrainExperiment] = Field(default=None)
    sample: Optional[SampleExperiment] = Field(default=None)


# Strict TOML loader (no pruning/workarounds). Invalid/incomplete sections raise.


def load_toml(path: Path) -> AppConfig:
    with path.open("rb") as f:
        raw: Any = tomllib.load(f)
    if not isinstance(raw, dict):
        raw = {}
    return AppConfig.model_validate(raw)


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
