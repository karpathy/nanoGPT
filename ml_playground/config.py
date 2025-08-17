from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional
import tomllib

DeviceKind = Literal["cpu", "mps", "cuda"]
DTypeKind = Literal["float32", "bfloat16", "float16"]


@dataclass(frozen=True)
class OptimConfig:
    learning_rate: float = 6e-4
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0


@dataclass(frozen=True)
class LRSchedule:
    decay_lr: bool = True
    warmup_iters: int = 2_000
    lr_decay_iters: int = 600_000
    min_lr: float = 6e-5


@dataclass(frozen=True)
class ModelConfig:
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 1024
    dropout: float = 0.0
    bias: bool = False
    vocab_size: Optional[int] = None


@dataclass(frozen=True)
class DataConfig:
    dataset_dir: Path
    train_bin: str = "train.bin"
    val_bin: str = "val.bin"
    meta_pkl: Optional[str] = "meta.pkl"
    batch_size: int = 12
    block_size: int = 1024
    grad_accum_steps: int = 40


@dataclass(frozen=True)
class RuntimeConfig:
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

    # Checkpoint policy (improved defaults)
    ckpt_last_filename: str = "ckpt_last.pt"
    ckpt_best_filename: str = "ckpt_best.pt"
    ckpt_top_k: int = 0  # keep N extra best checkpoints, 0 disables
    ckpt_metric: Literal["val_loss", "perplexity"] = "val_loss"
    ckpt_greater_is_better: bool = False  # loss: False; perplexity: False
    ckpt_atomic: bool = True
    ckpt_write_metadata: bool = True
    ckpt_time_interval_minutes: int = 0  # 0 disables time-based saves
    # Smoothed improvement + early stopping
    best_smoothing_alpha: float = 0.0  # 0 disables EMA smoothing
    early_stop_patience: int = 0  # 0 disables patience-based early stopping
    # Exponential moving average of weights (for evaluation/best saves)
    ema_decay: float = 0.0  # 0 disables EMA


@dataclass(frozen=True)
class SampleConfig:
    start: str = "\n"
    num_samples: int = 3
    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 200


@dataclass(frozen=True)
class TrainExperiment:
    model: ModelConfig
    data: DataConfig
    optim: OptimConfig
    schedule: LRSchedule
    runtime: RuntimeConfig


@dataclass(frozen=True)
class SampleExperiment:
    runtime: RuntimeConfig
    sample: SampleConfig


@dataclass(frozen=True)
class AppConfig:
    train: Optional[TrainExperiment] = None
    sample: Optional[SampleExperiment] = None


def load_toml(path: Path) -> AppConfig:
    with path.open("rb") as f:
        data = tomllib.load(f)

    def mm_model(d: dict | None) -> ModelConfig | None:
        return ModelConfig(**d) if d is not None else None

    def mm_data(d: dict | None) -> DataConfig | None:
        if d is None:
            return None
        d = dict(d)
        d["dataset_dir"] = Path(d["dataset_dir"])  # coerce
        return DataConfig(**d)

    def mm_runtime(d: dict | None) -> RuntimeConfig | None:
        if d is None:
            return None
        d = dict(d)
        d["out_dir"] = Path(d["out_dir"])  # coerce
        return RuntimeConfig(**d)

    def mm_optim(d: dict | None) -> OptimConfig | None:
        return OptimConfig(**d) if d is not None else None

    def mm_sched(d: dict | None) -> LRSchedule | None:
        return LRSchedule(**d) if d is not None else None

    def mm_sample(d: dict | None) -> SampleConfig | None:
        return SampleConfig(**d) if d is not None else None

    train = None
    if (t := data.get("train")) is not None:
        model_cfg = mm_model(t.get("model"))
        data_cfg = mm_data(t.get("data"))
        optim_cfg = mm_optim(t.get("optim"))
        schedule_cfg = mm_sched(t.get("schedule"))
        runtime_cfg = mm_runtime(t.get("runtime"))

        if all(
            x is not None
            for x in [model_cfg, data_cfg, optim_cfg, schedule_cfg, runtime_cfg]
        ):
            train = TrainExperiment(
                model=model_cfg,  # type: ignore[arg-type]
                data=data_cfg,  # type: ignore[arg-type]
                optim=optim_cfg,  # type: ignore[arg-type]
                schedule=schedule_cfg,  # type: ignore[arg-type]
                runtime=runtime_cfg,  # type: ignore[arg-type]
            )
    sample = None
    if (s := data.get("sample")) is not None:
        runtime_cfg = mm_runtime(s.get("runtime"))
        sample_cfg = mm_sample(s.get("sample"))

        if runtime_cfg is not None and sample_cfg is not None:
            sample = SampleExperiment(
                runtime=runtime_cfg,
                sample=sample_cfg,
            )
    return AppConfig(train=train, sample=sample)
