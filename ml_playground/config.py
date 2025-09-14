from __future__ import annotations

from pathlib import Path
import logging
from typing import Literal, Optional, Any, Annotated, cast
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
    PositiveInt,
    NonNegativeInt,
    StrictFloat,
    AfterValidator,
    ValidationInfo,
)

# Read policy constants to avoid hardcoded strings in code/tests
READ_POLICY_LATEST: Literal["latest"] = "latest"
READ_POLICY_BEST: Literal["best"] = "best"
DEFAULT_READ_POLICY: Literal["latest"] = READ_POLICY_LATEST


def _deep_merge_dicts(base: Any, override: Any) -> dict[str, Any]:
    """Recursively merge override into base (override wins).
    Only merges nested dicts; other types are replaced.
    """
    out: dict[str, Any] = dict(base) if isinstance(base, dict) else {}
    if not isinstance(override, dict):
        return out
    for k, v in cast(dict[str, Any], override).items():
        bv = out.get(k)
        if isinstance(bv, dict) and isinstance(v, dict):
            out[k] = _deep_merge_dicts(bv, v)
        else:
            out[k] = v
    return out


# Strict, single-source configuration module.


# Central strict resolver for all config path fields
def _resolve_path_strict(v: Path) -> Path:
    try:
        return v.resolve()
    except OSError:
        raise ValueError(f"Invalid path: {v}")


# Section/key constants to avoid scattered magic strings
SECTION_PREPARE = "prepare"
SECTION_TRAIN = "train"
SECTION_SAMPLE = "sample"
KEY_EXTRAS = "extras"

DeviceKind = Literal["cpu", "mps", "cuda"]
DTypeKind = Literal["float32", "bfloat16", "float16"]


class _FrozenStrictModel(BaseModel):
    """
    Base model that is immutable (frozen) and forbids extra fields. Used to enforce strict schema in all configs.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_default=True,
        strict=True,  # zero coercion across all models
        arbitrary_types_allowed=True,  # allow logging.Logger
    )

    # Common logger available on all configs; allow custom test doubles
    logger: Any = Field(default_factory=lambda: logging.getLogger(__name__))


# Reusable strict type aliases
def _no_nan(v: float) -> float:
    if v != v:  # NaN check
        raise ValueError("must not be NaN")
    return v


NonNaNNonNegativeStrictFloat = Annotated[
    StrictFloat, AfterValidator(_no_nan), Field(ge=0)
]
UnitIntervalStrictFloat = Annotated[StrictFloat, Field(ge=0, le=1)]  # [0, 1]
PosUnitIntervalStrictFloat = Annotated[StrictFloat, Field(gt=0, le=1)]  # (0, 1]
PositiveStrictFloat = Annotated[StrictFloat, Field(gt=0)]

# New validated int aliases
NonNegativeStrictInt = Annotated[int, Field(ge=0)]
PositiveStrictInt = Annotated[int, Field(gt=0)]
AtLeastOneInt = Annotated[int, Field(ge=1)]
SeedInt = Annotated[int, Field(ge=0)]
MinutesNonNegative = Annotated[int, Field(ge=0)]
EpochCount = AtLeastOneInt


class PreparerConfig(_FrozenStrictModel):
    """Strict config for data preparation (owner-local)."""

    raw_dir: Path = Path("./raw")
    add_structure_tokens: bool = False
    doc_separator: str = ""
    extras: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _resolve_paths(cls, data: Any, info: ValidationInfo) -> Any:
        if not isinstance(data, dict) or not info.context:
            return data

        config_path = info.context.get("config_path")
        if not config_path or not isinstance(config_path, Path):
            return data

        base_dir = config_path.parent

        def resolve_if_relative(path_val: Any) -> Any:
            if isinstance(path_val, str) and not path_val.startswith("/"):
                return (base_dir / path_val).resolve()
            if isinstance(path_val, Path) and not path_val.is_absolute():
                return (base_dir / path_val).resolve()
            return path_val

        if "raw_dir" in data:
            data["raw_dir"] = resolve_if_relative(data["raw_dir"])

        return data


class RuntimeConfig(_FrozenStrictModel):
    """
    Configuration for runtime/environment, checkpointing, logging, early stopping,
    device+seed settings, and advanced training options.
    """

    out_dir: Path
    max_iters: NonNegativeStrictInt = 600_000
    eval_interval: AtLeastOneInt = 2_000
    eval_iters: AtLeastOneInt = 200
    log_interval: AtLeastOneInt = 1
    eval_only: bool = False
    seed: SeedInt = 1337
    device: DeviceKind = "cpu"
    dtype: DTypeKind = "float32"
    compile: bool = False
    # TensorBoard logging toggle (default: enabled)
    tensorboard_enabled: bool = True
    # Persist last/best every eval step
    always_save_checkpoint: bool = False

    # Optional epoch semantics (experiment-scoped)
    iters_per_epoch: Optional[EpochCount] = None
    max_epochs: Optional[EpochCount] = None

    # Checkpoint policy (rotated-only)
    ckpt_metric: Literal["val_loss", "perplexity"] = "val_loss"
    ckpt_greater_is_better: bool = False
    ckpt_atomic: bool = True
    ckpt_write_metadata: bool = True
    ckpt_last_filename: str = "ckpt_last.pt"
    ckpt_best_filename: str = "ckpt_best.pt"
    ckpt_top_k: NonNegativeStrictInt = 0
    ckpt_time_interval_minutes: MinutesNonNegative = 0

    class Checkpointing(_FrozenStrictModel):
        class Keep(_FrozenStrictModel):
            last: NonNegativeInt = 1
            best: NonNegativeInt = 1

        keep: Keep = Keep()
        # Which rotated checkpoint to read when loading: latest or best
        read_policy: Literal["latest", "best"] = DEFAULT_READ_POLICY

    checkpointing: Checkpointing = Checkpointing()

    # Smoothed improvement + early stopping
    best_smoothing_alpha: UnitIntervalStrictFloat = 1.0
    early_stop_patience: NonNegativeStrictInt = 0
    # Exponential moving average of weights
    ema_decay: UnitIntervalStrictFloat = 0.0

    @field_validator("out_dir", mode="after")
    @classmethod
    def _resolve_out_dir(cls, v: Path) -> Path:
        # Preserve as provided (relative or absolute). Resolution is handled by loaders when desired.
        return v


class TrainerConfig(_FrozenStrictModel):
    """
    Top-level configuration for a training session, encapsulating model, data, optimizer,
    scheduler, runtime, and extensible extras.
    """

    @model_validator(mode="before")
    @classmethod
    def _resolve_paths(cls, data: Any, info: ValidationInfo) -> Any:
        if not isinstance(data, dict) or not info.context:
            return data

        config_path = info.context.get("config_path")
        if not config_path or not isinstance(config_path, Path):
            return data

        base_dir = config_path.parent

        def resolve_if_relative(path_val: Any) -> Any:
            if isinstance(path_val, str) and not path_val.startswith("/"):
                return (base_dir / path_val).resolve()
            if isinstance(path_val, Path) and not path_val.is_absolute():
                return (base_dir / path_val).resolve()
            return path_val

        if "runtime" in data and isinstance(data["runtime"], dict):
            if "out_dir" in data["runtime"]:
                data["runtime"]["out_dir"] = resolve_if_relative(
                    data["runtime"]["out_dir"]
                )

        return data

    model: ModelConfig
    data: DataConfig
    optim: OptimConfig
    schedule: LRSchedule
    runtime: RuntimeConfig
    extras: dict[str, Any] = Field(default_factory=dict)
    checkpointing: RuntimeConfig.Checkpointing = RuntimeConfig.Checkpointing()

    @model_validator(mode="after")
    def _cross_field_checks(self) -> "TrainerConfig":
        # Ensure data.block_size <= model.block_size
        try:
            if self.data.block_size > self.model.block_size:
                raise ValueError(
                    "train.data.block_size must be <= train.model.block_size"
                )
            # If using LR decay, ensure min_lr <= learning_rate
            if (
                self.schedule.decay_lr
                and self.schedule.min_lr == self.optim.learning_rate
            ):
                raise ValueError(
                    "train.schedule.min_lr must be <= train.optim.learning_rate when decay_lr=true"
                )
            # If decay is disabled, warmup must be zero to avoid inconsistent intent
            if (not self.schedule.decay_lr) or (self.schedule.warmup_iters != 0):
                raise ValueError(
                    "train.schedule.warmup_iters must be 0 when decay_lr=false"
                )
        except Exception:
            # If any attribute missing due to prior validation, let pydantic report it
            pass
        return self


class SamplerConfig(_FrozenStrictModel):
    """
    Top-level configuration for model sampling/generation runs, including runtime and sampling parameters.
    Supports a schema-level reference to reuse runtime from the training section.
    """

    @model_validator(mode="before")
    @classmethod
    def _resolve_paths(cls, data: Any, info: ValidationInfo) -> Any:
        if not isinstance(data, dict) or not info.context:
            return data

        config_path = info.context.get("config_path")
        if not config_path or not isinstance(config_path, Path):
            return data

        base_dir = config_path.parent

        def resolve_if_relative(path_val: Any) -> Any:
            if isinstance(path_val, str) and not path_val.startswith("/"):
                return (base_dir / path_val).resolve()
            if isinstance(path_val, Path) and not path_val.is_absolute():
                return (base_dir / path_val).resolve()
            return path_val

        if "runtime" in data and isinstance(data["runtime"], dict):
            if "out_dir" in data["runtime"]:
                data["runtime"]["out_dir"] = resolve_if_relative(
                    data["runtime"]["out_dir"]
                )

        return data

    # Runtime is required; references are not supported
    runtime: RuntimeConfig

    sample: SampleConfig
    extras: dict[str, Any] = Field(default_factory=dict)


class OptimConfig(_FrozenStrictModel):
    """
    Configuration for optimizer hyperparameters (learning rate, betas, weight decay, grad clipping, etc).
    Includes validation to enforce non-negative values.
    """

    learning_rate: NonNaNNonNegativeStrictFloat = 6e-4
    weight_decay: NonNaNNonNegativeStrictFloat = 1e-1
    beta1: NonNaNNonNegativeStrictFloat = 0.9
    beta2: NonNaNNonNegativeStrictFloat = 0.95
    grad_clip: NonNaNNonNegativeStrictFloat = 1.0


class LRSchedule(_FrozenStrictModel):
    """
    Learning rate schedule parameters (warmup, decay, min lr, etc) with built-in validation checks.
    """

    decay_lr: bool = True
    warmup_iters: NonNegativeStrictInt = 2000
    lr_decay_iters: NonNegativeStrictInt = 600_000
    min_lr: NonNaNNonNegativeStrictFloat = 6e-5

    @model_validator(mode="after")
    def _check_warmup_le_decay(self) -> "LRSchedule":
        # Type aliases enforce non-negativity; only cross-field relation remains
        if self.warmup_iters > self.lr_decay_iters:
            raise ValueError("warmup_iters must be <= lr_decay_iters")
        return self


class ModelConfig(_FrozenStrictModel):
    """
    Configuration for model architecture parameters (layers, hidden size, dropout, vocab size, etc).
    Provides validation to ensure non-negativity and valid ranges.
    """

    n_layer: PositiveStrictInt = 12
    n_head: PositiveStrictInt = 12
    n_embd: PositiveStrictInt = 767
    block_size: AtLeastOneInt = 1024
    dropout: UnitIntervalStrictFloat = 0.0
    bias: bool = True
    vocab_size: Optional[PositiveInt] = None


class DataConfig(_FrozenStrictModel):
    """
    Configuration for training data input, including paths to data, batch/block sizes, sampling policy,
    and dataset-specific hyperparameters.
    """

    train_bin: str = "train.bin"
    val_bin: str = "val.bin"
    meta_pkl: str = "meta.pkl"
    batch_size: AtLeastOneInt = 12
    block_size: AtLeastOneInt = 1024
    grad_accum_steps: AtLeastOneInt = 40
    # Tokenizer selection for bundestag_char-like datasets
    tokenizer: Literal["char", "word", "tiktoken"] = "char"
    # n-gram tokenization size for character datasets (1 = pure char-level)
    ngram_size: PositiveInt = 1
    # Sampling policy: random (default) or sequential (deterministic coverage)
    sampler: Literal["random", "sequential"] = "random"

    @model_validator(mode="after")
    def _check_tokenizer_compat(self) -> "DataConfig":
        # tiktoken does not use ngram grouping; enforce neutral ngram_size
        if not self.tokenizer == "tiktoken" and self.ngram_size != 1:
            raise ValueError(
                "train.data.ngram_size must be 1 when tokenizer='tiktoken'"
            )
        return self

    # Path construction methods
    def train_path(self, dataset_dir: Path) -> Path:
        return dataset_dir / self.train_bin

    def val_path(self, dataset_dir: Path) -> Path:
        return dataset_dir / self.val_bin

    def meta_path(self, dataset_dir: Path) -> Path:
        return dataset_dir / self.meta_pkl


class SampleConfig(_FrozenStrictModel):
    """
    Configuration for sampling and text generation parameters, including temperature, nucleus/top-k sampling,
    and output size controls.
    """

    start: str = "\n"
    num_samples: AtLeastOneInt = 3
    max_new_tokens: AtLeastOneInt = 200
    temperature: PositiveStrictFloat = 0.8
    top_k: NonNegativeStrictInt = 200
    top_p: Optional[PosUnitIntervalStrictFloat] = None


class ExperimentConfig(_FrozenStrictModel):
    """Full experiment configuration parsed once and validated strictly."""

    prepare: PreparerConfig
    train: TrainerConfig
    sample: SamplerConfig
    shared: SharedConfig

    @model_validator(mode="before")
    @classmethod
    def _resolve_paths(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        shared_data = data.get("shared", {})
        config_path = None
        if isinstance(shared_data, dict):
            config_path = shared_data.get("config_path")
        elif hasattr(shared_data, "config_path"):
            config_path = shared_data.config_path
        if not config_path or not isinstance(config_path, Path):
            return data

        base_dir = config_path.parent

        def resolve_if_relative(path_val: Any) -> Any:
            if isinstance(path_val, str) and not path_val.startswith("/"):
                return (base_dir / path_val).resolve()
            if isinstance(path_val, Path) and not path_val.is_absolute():
                return (base_dir / path_val).resolve()
            return path_val

        # Defensively handle data that may already be a model instance
        train_data = data.get("train")
        if isinstance(train_data, dict):
            runtime_data = train_data.get("runtime")
            if isinstance(runtime_data, dict) and "out_dir" in runtime_data:
                runtime_data["out_dir"] = resolve_if_relative(runtime_data["out_dir"])

        sample_data = data.get("sample")
        if isinstance(sample_data, dict):
            runtime_data = sample_data.get("runtime")
            if isinstance(runtime_data, dict) and "out_dir" in runtime_data:
                runtime_data["out_dir"] = resolve_if_relative(runtime_data["out_dir"])

        prepare_data = data.get("prepare")
        if isinstance(prepare_data, dict) and "raw_dir" in prepare_data:
            prepare_data["raw_dir"] = resolve_if_relative(prepare_data["raw_dir"])

        # Populate shared paths
        if isinstance(shared_data, dict):
            train_runtime_data = data.get("train", {}).get("runtime", {})
            if isinstance(train_runtime_data, dict):
                train_out_dir = train_runtime_data.get("out_dir")
                if train_out_dir:
                    shared_data["train_out_dir"] = resolve_if_relative(train_out_dir)

            sample_runtime_data = data.get("sample", {}).get("runtime", {})
            if isinstance(sample_runtime_data, dict):
                sample_out_dir = sample_runtime_data.get("out_dir")
                if sample_out_dir:
                    shared_data["sample_out_dir"] = resolve_if_relative(sample_out_dir)

            prepare_data = data.get("prepare")
            if isinstance(prepare_data, dict):
                dataset_dir = prepare_data.pop("dataset_dir", None)
                if dataset_dir:
                    shared_data["dataset_dir"] = resolve_if_relative(dataset_dir)

        return data


def load_experiment_toml(path: Path) -> ExperimentConfig:
    """Canonical wrapper that delegates to config_loader.

    Ensures the filesystem boundary for configuration remains in config_loader.
    Accepts only a Path (no coercion) and returns a fully validated ExperimentConfig.
    """
    # Local import to avoid circular dependency at module import time
    from ml_playground import config_loader as _loader

    project_home = Path(__file__).resolve().parent.parent
    experiment_name = path.parent.name
    return _loader.load_full_experiment_config(path, project_home, experiment_name)


class SharedConfig(_FrozenStrictModel):
    """
    Common per-experiment context derived at runtime (not from TOML).

    Carries frequently accessed paths and identifiers for logging and wiring.
    """

    experiment: str
    config_path: Path
    project_home: Path
    dataset_dir: Path
    train_out_dir: Path
    sample_out_dir: Path

    @field_validator(
        "config_path",
        "project_home",
        "dataset_dir",
        "train_out_dir",
        "sample_out_dir",
        mode="after",
    )
    @classmethod
    def _as_is(cls, v: Path) -> Path:
        # Preserve given paths without resolving to keep semantics consistent with loader
        return v


# Backward-compatible aliases for newer API names used by some modules
TrainExperiment = TrainerConfig
SampleExperiment = SamplerConfig


__all__ = [
    "DeviceKind",
    "DTypeKind",
    "OptimConfig",
    "LRSchedule",
    "ModelConfig",
    "DataConfig",
    "RuntimeConfig",
    "SampleConfig",
    "TrainerConfig",
    "SamplerConfig",
    "PreparerConfig",
    "ExperimentConfig",
    "SharedConfig",
    "load_experiment_toml",
    "SECTION_PREPARE",
    "SECTION_TRAIN",
    "SECTION_SAMPLE",
    "KEY_EXTRAS",
]
