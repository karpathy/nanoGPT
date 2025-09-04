from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Any
import tomllib
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Read policy constants to avoid hardcoded strings in code/tests
READ_POLICY_LATEST: Literal["latest"] = "latest"
READ_POLICY_BEST: Literal["best"] = "best"
DEFAULT_READ_POLICY: Literal["latest"] = READ_POLICY_LATEST


def _deep_merge_dicts(base: Any, override: Any) -> dict[str, Any]:
    """Recursively merge override into base (override wins).
    Only merges nested dicts; other types are replaced.
    """
    out = dict(base) if isinstance(base, dict) else {}
    if not isinstance(override, dict):
        return out
    for k, v in override.items():
        bv = out.get(k)
        if isinstance(bv, dict) and isinstance(v, dict):
            out[k] = _deep_merge_dicts(bv, v)
        else:
            out[k] = v
    return out


# Strict, single-source configuration module.

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

    model_config = ConfigDict(frozen=False, extra="forbid", validate_default=True)


class PreparerConfig(_FrozenStrictModel):
    """Strict config for data preparation (owner-local)."""

    dataset_dir: Optional[Path] = None
    raw_dir: Optional[Path] = None
    add_structure_tokens: Optional[bool] = None
    doc_separator: Optional[str] = None
    extras: dict[str, Any] = Field(default_factory=dict)
    logger: Any | None = Field(default=None)

    @field_validator("dataset_dir", "raw_dir", mode="before")
    @classmethod
    def _coerce_path(cls, v: Path | str | None) -> Optional[Path]:
        if v is None:
            return None
        return Path(v)

    @field_validator("dataset_dir", "raw_dir", mode="after")
    @classmethod
    def _resolve_path(cls, v: Optional[Path]) -> Optional[Path]:
        if v is None:
            return None
        try:
            return v.resolve()
        except Exception:
            # Best-effort normalization; leave as-is if resolve fails
            return v


class RuntimeConfig(_FrozenStrictModel):
    """
    Configuration for runtime/environment, checkpointing, logging, early stopping,
    device+seed settings, and advanced training options.
    """

    out_dir: Path
    max_iters: int = 600_000
    eval_interval: int = 2_000
    eval_iters: int = 200
    log_interval: int = 1
    eval_only: bool = False
    seed: int = 1337
    device: DeviceKind = "cpu"
    dtype: DTypeKind = "float32"
    compile: bool = False
    # TensorBoard logging toggle (default: enabled)
    tensorboard_enabled: bool = True

    # Optional epoch semantics (experiment-scoped)
    iters_per_epoch: Optional[int] = None
    max_epochs: Optional[int] = None

    # Checkpoint policy (rotated-only)
    ckpt_metric: Literal["val_loss", "perplexity"] = "val_loss"
    ckpt_greater_is_better: bool = False
    ckpt_atomic: bool = True
    ckpt_write_metadata: bool = True
    ckpt_time_interval_minutes: int = 0

    class Checkpointing(_FrozenStrictModel):
        class Keep(_FrozenStrictModel):
            last: int = 1
            best: int = 1

            @field_validator("last", "best")
            @classmethod
            def _validate_positive(cls, v: int) -> int:
                if v < 0:
                    raise ValueError("must be >= 0")
                return int(v)

        keep: Keep = Keep()
        # Which rotated checkpoint to read when loading: latest or best
        read_policy: Literal["latest", "best"] = DEFAULT_READ_POLICY

    checkpointing: Checkpointing = Checkpointing()

    # Smoothed improvement + early stopping
    best_smoothing_alpha: float = 0.0
    early_stop_patience: int = 0
    # Exponential moving average of weights
    ema_decay: float = 0.0

    @field_validator("out_dir", mode="before")
    @classmethod
    def _coerce_out_dir(cls, v: Path | str) -> Path:
        return Path(v)

    @field_validator("out_dir", mode="after")
    @classmethod
    def _resolve_out_dir(cls, v: Path) -> Path:
        # Preserve as provided (relative or absolute). Resolution is handled by loaders when desired.
        return v

    # No stable filename paths; reading is selected via checkpointing.read_policy


class TrainerConfig(_FrozenStrictModel):
    """
    Top-level configuration for a training session, encapsulating model, data, optimizer,
    scheduler, runtime, and extensible extras.
    """

    model: ModelConfig
    data: DataConfig
    optim: OptimConfig
    schedule: LRSchedule
    runtime: RuntimeConfig
    extras: dict[str, Any] = Field(default_factory=dict)
    logger: Any | None = Field(default=None)
    checkpointing: RuntimeConfig.Checkpointing = RuntimeConfig.Checkpointing()


class SamplerConfig(_FrozenStrictModel):
    """
    Top-level configuration for model sampling/generation runs, including runtime and sampling parameters.
    Supports a schema-level reference to reuse runtime from the training section.
    """

    # Either provide runtime directly or a reference to an existing section
    runtime: Optional[RuntimeConfig] = None
    runtime_ref: Optional[Literal["train.runtime"]] = None

    sample: SampleConfig
    extras: dict[str, Any] = Field(default_factory=dict)
    logger: Any | None = Field(default=None)

    @model_validator(mode="after")
    def _check_runtime_or_ref(self) -> "SamplerConfig":
        if self.runtime is None and self.runtime_ref is None:
            raise ValueError(
                "SamplerConfig requires either 'runtime' or 'runtime_ref'."
            )
        return self


class OptimConfig(_FrozenStrictModel):
    """
    Configuration for optimizer hyperparameters (learning rate, betas, weight decay, grad clipping, etc).
    Includes validation to enforce non-negative values.
    """

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
    """
    Learning rate schedule parameters (warmup, decay, min lr, etc) with built-in validation checks.
    """

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
    """
    Configuration for model architecture parameters (layers, hidden size, dropout, vocab size, etc).
    Provides validation to ensure non-negativity and valid ranges.
    """

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
    """
    Configuration for training data input, including paths to data, batch/block sizes, sampling policy,
    and dataset-specific hyperparameters.
    """

    dataset_dir: Path
    train_bin: str = "train.bin"
    val_bin: str = "val.bin"
    meta_pkl: Optional[str] = "meta.pkl"
    batch_size: int = 12
    block_size: int = 1024
    grad_accum_steps: int = 40
    # Tokenizer selection for bundestag_char-like datasets
    tokenizer: Literal["char", "word", "tiktoken"] = "char"
    # n-gram tokenization size for character datasets (1 = pure char-level)
    ngram_size: int = 1
    # Sampling policy: random (default) or sequential (deterministic coverage)
    sampler: Literal["random", "sequential"] = "random"

    @field_validator("dataset_dir", mode="before")
    @classmethod
    def _coerce_path(cls, v: Path | str) -> Path:
        return Path(v)

    @field_validator("dataset_dir", mode="after")
    @classmethod
    def _resolve_dataset_dir(cls, v: Path) -> Path:
        try:
            return v.resolve()
        except Exception:
            return v

    @field_validator("batch_size", "block_size", "grad_accum_steps", "ngram_size")
    @classmethod
    def _positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be > 0")
        return int(v)

    # Computed, read-only paths
    @property
    def train_path(self) -> Path:
        return self.dataset_dir / self.train_bin

    @property
    def val_path(self) -> Path:
        return self.dataset_dir / self.val_bin

    @property
    def meta_path(self) -> Optional[Path]:
        if self.meta_pkl is None:
            return None
        return self.dataset_dir / self.meta_pkl


class SampleConfig(_FrozenStrictModel):
    """
    Configuration for sampling and text generation parameters, including temperature, nucleus/top-k sampling,
    and output size controls.
    """

    start: str = "\n"
    num_samples: int = 3
    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 200
    top_p: Optional[float] = None

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

    @field_validator("top_p")
    @classmethod
    def _unit_range_exclusive_zero(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return None
        # Accept (0, 1] -> strictly greater than 0, less than or equal to 1.0
        if not (0.0 < v <= 1.0):
            raise ValueError("top_p must be in (0, 1]")
        return float(v)


class AppConfig(_FrozenStrictModel):
    """
    Top-level config capturing the overall application configuration, possibly including both training
    and sampling configuration blocks.
    """

    train: Optional[TrainerConfig] = Field(default=None)
    sample: Optional[SamplerConfig] = Field(default=None)


def load_toml(path: Path) -> "AppConfig":
    """Load a TOML file into a strongly-typed AppConfig.

    This helper is used by experiment preparers to optionally read
    experiment-scoped configuration values. It performs strict typing
    via Pydantic models and raises on invalid structure.

    NOTE: AppConfig is strict (forbids extras). To allow experiment-specific
    top-level sections (e.g., [export]), we filter the loaded TOML to only
    include keys that AppConfig knows about ("train" and "sample").
    """
    if not isinstance(path, Path):
        path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")
    with path.open("rb") as f:
        raw = tomllib.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Config at {path} must be a TOML table/object")
    filtered: dict[str, Any] = {}
    if SECTION_TRAIN in raw:
        filtered[SECTION_TRAIN] = raw[SECTION_TRAIN]
    if SECTION_SAMPLE in raw:
        filtered[SECTION_SAMPLE] = raw[SECTION_SAMPLE]
    return AppConfig.model_validate(filtered)


class ExperimentConfig(_FrozenStrictModel):
    """Full experiment configuration parsed once and validated strictly."""

    prepare: PreparerConfig
    train: TrainerConfig
    sample: SamplerConfig

    @model_validator(mode="after")
    def _resolve_references(self) -> "ExperimentConfig":
        # Resolve [sample.runtime] via schema-level reference if provided
        s = self.sample
        if s.runtime_ref == "train.runtime":
            # Merge train.runtime into sample.runtime overrides (if any)
            base = self.train.runtime.model_dump()
            if s.runtime is not None:
                overrides = s.runtime.model_dump()
                base.update(overrides)
            resolved_rt = RuntimeConfig.model_validate(base)
            # rebuild sample without the ref
            self.sample = SamplerConfig(
                runtime=resolved_rt,
                sample=s.sample,
                extras=s.extras,
                logger=s.logger,
            )
        # You could add more reference resolutions here in future
        return self


def load_experiment_toml(path: Path | str) -> ExperimentConfig:
    """Parse the entire TOML once, returning a fully validated ExperimentConfig."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("rb") as f:
        raw = tomllib.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Config at {p} must be a TOML table/object")
    # No filtering: rely on ExperimentConfig's strict schema to forbid unknown sections
    return ExperimentConfig.model_validate(raw)


# Backward-compatible aliases for newer API names used by some modules
TrainExperiment = TrainerConfig
SampleExperiment = SamplerConfig


def validate_config_field(
    value: Any,
    field_name: str,
    expected_type: type,
    required: bool = True,
    min_value: Any = None,
    max_value: Any = None,
) -> None:
    """
    Validate a configuration field with type and range checks.

    This function checks if the provided value is of the expected type and within the specified range.
    It raises a ValueError if the validation fails.

    Args:
        value: The value to validate
        field_name: Name of the field (for error messages)
        expected_type: Expected type of the value
        required: Whether the field is required (cannot be None)
        min_value: Minimum allowed value (for numeric types)
        max_value: Maximum allowed value (for numeric types)

    Raises:
        ValueError: If validation fails
    """
    # Check if required
    if required and value is None:
        raise ValueError(f"Required configuration field '{field_name}' is missing")

    # Skip further validation if None and not required
    if value is None:
        return

    # Type check
    if not isinstance(value, expected_type):
        raise ValueError(
            f"Configuration field '{field_name}' must be of type {expected_type.__name__}, "
            f"got {type(value).__name__}"
        )

    # Range checks for numeric types
    if isinstance(value, (int, float)):
        if min_value is not None and value < min_value:
            raise ValueError(
                f"Configuration field '{field_name}' must be >= {min_value}, got {value}"
            )
        if max_value is not None and value > max_value:
            raise ValueError(
                f"Configuration field '{field_name}' must be <= {max_value}, got {value}"
            )


def validate_path_exists(
    path: Path, field_name: str, must_be_file: bool = False, must_be_dir: bool = False
) -> None:
    """
    Validate that a path exists and optionally check if it's a file or directory.

    This function checks if the provided path exists and optionally checks if it's a file or directory.
    It raises a ValueError if the validation fails.

    Args:
        path: Path to validate
        field_name: Name of the field (for error messages)
        must_be_file: If True, path must exist and be a file
        must_be_dir: If True, path must exist and be a directory

    Raises:
        ValueError: If validation fails
    """
    if not path.exists():
        raise ValueError(f"Path specified in '{field_name}' does not exist: {path}")

    if must_be_file and not path.is_file():
        raise ValueError(f"Path specified in '{field_name}' must be a file: {path}")

    if must_be_dir and not path.is_dir():
        raise ValueError(
            f"Path specified in '{field_name}' must be a directory: {path}"
        )


# Expose utility functions
validate_config_field = validate_config_field
validate_path_exists = validate_path_exists

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
    "AppConfig",
    "ExperimentConfig",
    "load_toml",
    "load_experiment_toml",
    "SECTION_PREPARE",
    "SECTION_TRAIN",
    "SECTION_SAMPLE",
    "KEY_EXTRAS",
    "validate_config_field",
    "validate_path_exists",
]
