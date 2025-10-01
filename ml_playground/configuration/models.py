from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Any, Literal, Optional
import typing as _t

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    AfterValidator,
    NonNegativeInt,
    PositiveInt,
    ValidationInfo,
    computed_field,
    field_validator,
    model_validator,
)

from ml_playground.core.logging_protocol import LoggerLike

READ_POLICY_LATEST: Literal["latest"] = "latest"
READ_POLICY_BEST: Literal["best"] = "best"
DEFAULT_READ_POLICY: Literal["best"] = READ_POLICY_BEST


def merge_configs(base_config: Any, override_config: Any) -> dict[str, Any]:
    """Merge two configuration dictionaries with nested support."""

    if not isinstance(base_config, dict) or not isinstance(override_config, dict):
        return override_config if isinstance(override_config, dict) else base_config

    merged = dict(base_config)
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_path_strict(v: Path) -> Path:
    try:
        return v.resolve()
    except OSError as exc:  # pragma: no cover - resolution failure path
        raise ValueError(f"Invalid path: {v}") from exc


def _resolve_if_relative(value: Any, base_dir: Path) -> Any:
    if isinstance(value, str) and not value.startswith("/"):
        return (base_dir / value).resolve()
    if isinstance(value, Path) and not value.is_absolute():
        return (base_dir / value).resolve()
    return value


def _resolve_fields_relative(
    data: dict[str, Any], keys: list[str], base_dir: Path
) -> None:
    for key in keys:
        if key in data:
            data[key] = _resolve_if_relative(data[key], base_dir)


SECTION_PREPARE = "prepare"
SECTION_TRAIN = "train"
SECTION_SAMPLE = "sample"
KEY_EXTRAS = "extras"

DeviceKind = Literal["cpu", "mps", "cuda"]
DTypeKind = Literal["float32", "bfloat16", "float16"]


class _ConfigCrossFieldValidator:
    """Centralized cross-field validation helpers for configuration models."""

    @staticmethod
    def runtime(runtime: "RuntimeConfig") -> None:
        if runtime.log_interval > runtime.eval_interval:
            raise ValueError(
                "train.runtime.log_interval must be <= train.runtime.eval_interval"
            )

    @staticmethod
    def trainer(trainer: "TrainerConfig") -> None:
        if trainer.data.block_size > trainer.model.block_size:
            raise ValueError("train.data.block_size must be <= train.model.block_size")

        if (
            trainer.schedule.decay_lr
            and trainer.schedule.min_lr > trainer.optim.learning_rate
        ):
            raise ValueError(
                "train.schedule.min_lr must be <= train.optim.learning_rate when decay_lr=true"
            )

        if not trainer.schedule.decay_lr and trainer.schedule.warmup_iters != 0:
            raise ValueError(
                "train.schedule.warmup_iters must be 0 when decay_lr=false"
            )

    @staticmethod
    def lr_schedule(schedule: "LRSchedule") -> None:
        if schedule.warmup_iters > schedule.lr_decay_iters:
            raise ValueError("warmup_iters must be <= lr_decay_iters")

    @staticmethod
    def data(data: "DataConfig") -> None:
        if data.tokenizer != "tiktoken" and data.ngram_size != 1:
            raise ValueError(
                "train.data.ngram_size must be 1 when tokenizer='tiktoken'"
            )


class _FrozenStrictModel(BaseModel):
    """Base model that is immutable, strict, and forbids extra fields."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_default=True,
        strict=True,
        arbitrary_types_allowed=True,
    )

    logger: LoggerLike = Field(default_factory=lambda: logging.getLogger(__name__))


def _no_nan(v: float) -> float:
    if v != v:  # NaN check
        raise ValueError("must not be NaN")
    return v


NonNaNNonNegativeStrictFloat = Annotated[float, AfterValidator(_no_nan), Field(ge=0)]
UnitIntervalStrictFloat = Annotated[float, Field(ge=0, le=1)]
PosUnitIntervalStrictFloat = Annotated[float, Field(gt=0, le=1)]
PositiveStrictFloat = Annotated[float, Field(gt=0)]

NonNegativeStrictInt = Annotated[int, Field(ge=0)]
PositiveStrictInt = Annotated[int, Field(gt=0)]
AtLeastOneInt = Annotated[int, Field(ge=1)]
SeedInt = Annotated[int, Field(ge=0)]
MinutesNonNegative = Annotated[int, Field(ge=0)]
EpochCount = AtLeastOneInt


class PreparerConfig(_FrozenStrictModel):
    raw_dir: Path = Path("./raw")
    raw_text_path: Path | None = None
    tokenizer_type: Literal["char", "word", "tiktoken"] = "char"
    add_structure_tokens: bool = False
    doc_separator: str = ""
    extras: dict[str, Any] = Field(default_factory=dict)
    # Optional DI hooks (keep generic to avoid import cycles)
    # Function to read text from a path (e.g., Path -> str)
    read_text_fn: Optional[_t.Callable[..., Any]] = None
    # Factory to create a tokenizer from a kind
    tokenizer_factory: Optional[_t.Callable[..., Any]] = None

    @model_validator(mode="before")
    @classmethod
    def _resolve_paths(cls, data: Any, info: ValidationInfo) -> Any:
        if not isinstance(data, dict) or not info.context:
            return data
        config_path = info.context.get("config_path")
        if not config_path or not isinstance(config_path, Path):
            return data
        base_dir = config_path.parent
        _resolve_fields_relative(data, ["raw_dir", "raw_text_path"], base_dir)
        return data


class RuntimeConfig(_FrozenStrictModel):
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
    tensorboard_enabled: bool = True
    tensorboard_update_mode: Literal["eval", "log"] = "eval"
    always_save_checkpoint: bool = False
    iters_per_epoch: Optional[EpochCount] = None
    max_epochs: Optional[EpochCount] = None
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
        read_policy: Literal["latest", "best"] = DEFAULT_READ_POLICY

    checkpointing: Checkpointing = Checkpointing()
    best_smoothing_alpha: UnitIntervalStrictFloat = 1.0
    early_stop_patience: NonNegativeStrictInt = 0
    ema_decay: UnitIntervalStrictFloat = 0.0

    @field_validator("out_dir", mode="after")
    @classmethod
    def _resolve_out_dir(cls, v: Path) -> Path:
        return v

    @model_validator(mode="after")
    def _check_logging_intervals(self) -> "RuntimeConfig":
        _ConfigCrossFieldValidator.runtime(self)
        return self

    @computed_field(return_type=int)
    def total_eval_steps(self) -> int:
        if self.eval_interval <= 0:
            return 0
        return int(self.max_iters // self.eval_interval)


class TrainerConfig(_FrozenStrictModel):
    @model_validator(mode="before")
    @classmethod
    def _resolve_paths(cls, data: Any, info: ValidationInfo) -> Any:
        if not isinstance(data, dict) or not info.context:
            return data
        config_path = info.context.get("config_path")
        if not config_path or not isinstance(config_path, Path):
            return data
        base_dir = config_path.parent
        if "runtime" in data and isinstance(data["runtime"], dict):
            _resolve_fields_relative(data["runtime"], ["out_dir"], base_dir)
        return data

    class HFModelConfig(_FrozenStrictModel):
        model_name: str
        gradient_checkpointing: bool = False
        block_size: AtLeastOneInt = 1024

    class PeftConfig(_FrozenStrictModel):
        enabled: bool = False
        r: PositiveStrictInt = 8
        lora_alpha: PositiveStrictFloat = 16.0
        lora_dropout: UnitIntervalStrictFloat = 0.0
        bias: Literal["none", "all", "lora_only"] = "none"
        target_modules: tuple[str, ...] = ()
        extend_mlp_targets: bool = False

        @model_validator(mode="before")
        @classmethod
        def _coerce_target_modules(cls, data: Any) -> Any:
            if isinstance(data, dict) and "target_modules" in data:
                modules = data["target_modules"]
                if isinstance(modules, list):
                    data["target_modules"] = tuple(str(m) for m in modules)
            return data

    model: "ModelConfig"
    data: "DataConfig"
    optim: "OptimConfig"
    schedule: "LRSchedule"
    runtime: RuntimeConfig
    extras: dict[str, Any] = Field(default_factory=dict)
    hf_model: HFModelConfig | None = None
    peft: PeftConfig | None = None
    checkpointing: RuntimeConfig.Checkpointing = RuntimeConfig.Checkpointing()
    # Optional DI callables (kept generic to avoid import cycles)
    # Hooks around a training step
    before_step_hook: Optional[_t.Callable[..., Any]] = None
    after_step_hook: Optional[_t.Callable[..., Any]] = None
    # Checkpoint save/load indirections
    checkpoint_save_fn: Optional[_t.Callable[..., Any]] = None
    checkpoint_load_fn: Optional[_t.Callable[..., Any]] = None

    @model_validator(mode="after")
    def _cross_field_checks(self) -> "TrainerConfig":
        _ConfigCrossFieldValidator.trainer(self)
        return self


class SamplerConfig(_FrozenStrictModel):
    @model_validator(mode="before")
    @classmethod
    def _resolve_paths(cls, data: Any, info: ValidationInfo) -> Any:
        if not isinstance(data, dict) or not info.context:
            return data
        config_path = info.context.get("config_path")
        if not config_path or not isinstance(config_path, Path):
            return data
        base_dir = config_path.parent
        if "runtime" in data and isinstance(data["runtime"], dict):
            _resolve_fields_relative(data["runtime"], ["out_dir"], base_dir)
        return data

    runtime: RuntimeConfig
    sample: "SampleConfig"
    extras: dict[str, Any] = Field(default_factory=dict)
    # Optional DI callables for sampling
    checkpoint_load_fn: Optional[_t.Callable[..., Any]] = None
    model_factory: Optional[_t.Callable[..., Any]] = None


class OptimConfig(_FrozenStrictModel):
    learning_rate: NonNaNNonNegativeStrictFloat = 6e-4
    weight_decay: NonNaNNonNegativeStrictFloat = 1e-1
    beta1: NonNaNNonNegativeStrictFloat = 0.9
    beta2: NonNaNNonNegativeStrictFloat = 0.95
    grad_clip: NonNaNNonNegativeStrictFloat = 1.0


class LRSchedule(_FrozenStrictModel):
    decay_lr: bool = True
    warmup_iters: NonNegativeStrictInt = 2000
    lr_decay_iters: NonNegativeStrictInt = 600_000
    min_lr: NonNaNNonNegativeStrictFloat = 6e-5

    @model_validator(mode="after")
    def _check_warmup_le_decay(self) -> "LRSchedule":
        _ConfigCrossFieldValidator.lr_schedule(self)
        return self


class ModelConfig(_FrozenStrictModel):
    n_layer: PositiveStrictInt = 12
    n_head: PositiveStrictInt = 12
    n_embd: PositiveStrictInt = 767
    block_size: AtLeastOneInt = 1024
    dropout: UnitIntervalStrictFloat = 0.0
    bias: bool = True
    vocab_size: Optional[PositiveInt] = None


class DataConfig(_FrozenStrictModel):
    train_bin: str = "train.bin"
    val_bin: str = "val.bin"
    meta_pkl: str = "meta.pkl"
    batch_size: AtLeastOneInt = 12
    block_size: AtLeastOneInt = 1024
    grad_accum_steps: AtLeastOneInt = 40
    tokenizer: Literal["char", "word", "tiktoken"] = "char"
    ngram_size: PositiveInt = 1
    sampler: Literal["random", "sequential"] = "random"

    @model_validator(mode="after")
    def _check_tokenizer_compat(self) -> "DataConfig":
        _ConfigCrossFieldValidator.data(self)
        return self

    def train_path(self, dataset_dir: Path) -> Path:
        return dataset_dir / self.train_bin

    def val_path(self, dataset_dir: Path) -> Path:
        return dataset_dir / self.val_bin

    def meta_path(self, dataset_dir: Path) -> Path:
        return dataset_dir / self.meta_pkl


class SampleConfig(_FrozenStrictModel):
    start: str = "\n"
    num_samples: AtLeastOneInt = 3
    max_new_tokens: AtLeastOneInt = 200
    temperature: PositiveStrictFloat = 0.8
    top_k: NonNegativeStrictInt = 200
    top_p: Optional[PosUnitIntervalStrictFloat] = None


class ExperimentConfig(_FrozenStrictModel):
    prepare: PreparerConfig
    train: TrainerConfig
    sample: SamplerConfig
    shared: "SharedConfig"

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

        if isinstance(data.get("train"), dict):
            runtime_data = data["train"].get("runtime")
            if isinstance(runtime_data, dict) and "out_dir" in runtime_data:
                od = runtime_data["out_dir"]
                if isinstance(od, str):
                    runtime_data["out_dir"] = Path(od)

        if isinstance(data.get("sample"), dict):
            runtime_data = data["sample"].get("runtime")
            if isinstance(runtime_data, dict) and "out_dir" in runtime_data:
                od = runtime_data["out_dir"]
                if isinstance(od, str):
                    runtime_data["out_dir"] = Path(od)

        if isinstance(data.get("prepare"), dict):
            prepare_data = data["prepare"]
            if "raw_dir" in prepare_data:
                prepare_data["raw_dir"] = _resolve_if_relative(
                    prepare_data["raw_dir"], base_dir
                )

        if isinstance(shared_data, dict):
            for key in (
                "dataset_dir",
                "train_out_dir",
                "sample_out_dir",
                "config_path",
                "project_home",
            ):
                v = shared_data.get(key)
                if isinstance(v, str):
                    shared_data[key] = Path(v)

            train_runtime_data = data.get("train", {}).get("runtime", {})
            if isinstance(train_runtime_data, dict):
                train_out_dir = train_runtime_data.get("out_dir")
                if train_out_dir:
                    shared_data["train_out_dir"] = (
                        Path(train_out_dir)
                        if isinstance(train_out_dir, str)
                        else train_out_dir
                    )

            sample_runtime_data = data.get("sample", {}).get("runtime", {})
            if isinstance(sample_runtime_data, dict):
                sample_out_dir = sample_runtime_data.get("out_dir")
                if sample_out_dir:
                    shared_data["sample_out_dir"] = (
                        Path(sample_out_dir)
                        if isinstance(sample_out_dir, str)
                        else sample_out_dir
                    )

            prepare_data = data.get("prepare")
            if isinstance(prepare_data, dict):
                dataset_dir = prepare_data.pop("dataset_dir", None)
                if dataset_dir:
                    shared_data["dataset_dir"] = (
                        Path(dataset_dir)
                        if isinstance(dataset_dir, str)
                        else dataset_dir
                    )

        return data


class SharedConfig(_FrozenStrictModel):
    experiment: str
    config_path: Path
    project_home: Path
    dataset_dir: Path
    train_out_dir: Path
    sample_out_dir: Path

    @model_validator(mode="before")
    @classmethod
    def _resolve_shared_paths(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        cfg_path = data.get("config_path")
        try:
            base_dir = Path(cfg_path).parent if cfg_path is not None else None
        except (TypeError, ValueError, OSError):
            base_dir = None
        if base_dir is None:
            return data

        def norm(v: Any) -> Any:
            if isinstance(v, (str, Path)):
                p = Path(v)
                if not p.is_absolute():
                    return (base_dir / p).resolve()
                return p
            return v

        for key in ("project_home", "dataset_dir", "train_out_dir", "sample_out_dir"):
            if key in data:
                data[key] = norm(data[key])
        return data

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
        return v


__all__ = [
    "merge_configs",
    "READ_POLICY_LATEST",
    "READ_POLICY_BEST",
    "DEFAULT_READ_POLICY",
    "SECTION_PREPARE",
    "SECTION_TRAIN",
    "SECTION_SAMPLE",
    "KEY_EXTRAS",
    "DeviceKind",
    "DTypeKind",
    "PreparerConfig",
    "RuntimeConfig",
    "TrainerConfig",
    "SamplerConfig",
    "OptimConfig",
    "LRSchedule",
    "ModelConfig",
    "DataConfig",
    "SampleConfig",
    "ExperimentConfig",
    "SharedConfig",
]
