from __future__ import annotations

import json
import math
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Literal, cast

import torch
from torch import nn
from torch.utils.data import Dataset

from typing import TYPE_CHECKING
import importlib.util

# TensorBoard is required when this module is used for training. Fail fast if missing.
if importlib.util.find_spec("torch.utils.tensorboard") is None:  # pragma: no cover
    raise SystemExit(
        "tensorboard is required for this experiment. Install via UV: `uv add tensorboard`."
    )


# PyTorch profiler → TensorBoard (lightweight, scheduled)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from peft import LoraConfig, PeftModel  # noqa: F401


def _require_peft():
    try:
        from peft import LoraConfig, PeftModel, get_peft_model  # type: ignore
    except ImportError as e:  # pragma: no cover - only raised if peft missing
        raise SystemExit(
            "peft is required for this integration. Install via UV: `uv add peft`."
        ) from e
    return LoraConfig, PeftModel, get_peft_model


# -----------------------------
# Config parsing (TOML-light)
# -----------------------------


@dataclass(frozen=True)
class PrepareCfg:
    raw_dir: Path
    dataset_dir: Path
    add_structure_tokens: bool = True
    doc_separator: str = "<DOC_SEP>"


@dataclass(frozen=True)
class HFModelCfg:
    model_name: str = "Qwen/Qwen2.5-1.5B"
    gradient_checkpointing: bool = True
    block_size: int = 512


@dataclass(frozen=True)
class PeftCfg:
    enabled: bool = True
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    bias: Literal["none", "all", "lora_only"] = "none"
    target_modules: Tuple[str, ...] = ("q_proj", "v_proj", "o_proj")
    extend_mlp_targets: bool = False


@dataclass(frozen=True)
class DataCfg:
    dataset_dir: Path
    batch_size: int = 8
    grad_accum_steps: int = 8
    block_size: int = 512
    shuffle: bool = True


@dataclass(frozen=True)
class OptimCfg:
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0


@dataclass(frozen=True)
class SchedCfg:
    decay_lr: bool = True
    warmup_iters: int = 500
    lr_decay_iters: int = 200_000
    min_lr: float = 2e-5


@dataclass(frozen=True)
class RuntimeCfg:
    out_dir: Path
    max_iters: int = 200_000
    eval_interval: int = 300
    eval_iters: int = 100
    log_interval: int = 100
    eval_only: bool = False
    always_save_checkpoint: bool = True
    seed: int = 1337
    device: str = "mps"
    dtype: str = "float16"
    compile: bool = False
    ckpt_time_interval_minutes: int = 10
    ckpt_atomic: bool = False
    best_smoothing_alpha: float = 0.0
    early_stop_patience: int = 5
    ema_decay: float = 0.0
    save_merged_on_best: bool = False
    keep_last_n: int = 0


@dataclass(frozen=True)
class SampleRuntime:
    out_dir: Path
    device: str = "mps"
    dtype: str = "float16"
    compile: bool = False
    seed: int = 1337


@dataclass(frozen=True)
class SampleCfg:
    start: str = "\n"
    num_samples: int = 1
    max_new_tokens: int = 1024
    temperature: float = 0.8
    top_k: int = 200
    top_p: float = 0.9


@dataclass(frozen=True)
class TrainTOML:
    hf_model: HFModelCfg
    peft: PeftCfg
    data: DataCfg
    optim: OptimCfg
    schedule: SchedCfg
    runtime: RuntimeCfg


@dataclass(frozen=True)
class SampleTOML:
    runtime: SampleRuntime
    sample: SampleCfg


@dataclass(frozen=True)
class AppTOML:
    prepare: PrepareCfg
    train: TrainTOML
    sample: SampleTOML


# -----------------------------
# Helpers
# -----------------------------


def _read_toml(config_path: Path) -> dict:
    import tomllib

    with config_path.open("rb") as f:
        return tomllib.load(f)


def _coerce_path(d: dict, key: str) -> Path:
    return Path(d[key])


def _parse_bias(val: object) -> Literal["none", "all", "lora_only"]:
    s = str(val)
    if s not in ("none", "all", "lora_only"):
        s = "none"
    return cast(Literal["none", "all", "lora_only"], s)


def _parse_app(config_path: Path) -> AppTOML:
    d = _read_toml(config_path)
    p = d.get("prepare", {})
    t = d.get("train", {})
    s = d.get("sample", {})

    prepare = PrepareCfg(
        raw_dir=_coerce_path(p, "raw_dir"),
        dataset_dir=_coerce_path(p, "dataset_dir"),
        add_structure_tokens=bool(p.get("add_structure_tokens", True)),
        doc_separator=str(p.get("doc_separator", "<DOC_SEP>")),
    )
    hf_model = HFModelCfg(**t.get("hf_model", {}))
    peft_cfg = PeftCfg(
        enabled=bool(t.get("peft", {}).get("enabled", True)),
        r=int(t.get("peft", {}).get("r", 8)),
        lora_alpha=int(t.get("peft", {}).get("lora_alpha", 16)),
        lora_dropout=float(t.get("peft", {}).get("lora_dropout", 0.05)),
        bias=_parse_bias(t.get("peft", {}).get("bias", "none")),
        target_modules=tuple(
            t.get("peft", {}).get("target_modules", ["q_proj", "v_proj", "o_proj"])
        ),
        extend_mlp_targets=bool(t.get("peft", {}).get("extend_mlp_targets", False)),
    )
    data_cfg = DataCfg(
        dataset_dir=_coerce_path(t.get("data", {}), "dataset_dir"),
        batch_size=int(t.get("data", {}).get("batch_size", 8)),
        grad_accum_steps=int(t.get("data", {}).get("grad_accum_steps", 8)),
        block_size=int(t.get("data", {}).get("block_size", hf_model.block_size)),
        shuffle=bool(t.get("data", {}).get("shuffle", True)),
    )
    optim_cfg = OptimCfg(**t.get("optim", {}))
    sched_cfg = SchedCfg(**t.get("schedule", {}))
    runtime_cfg = RuntimeCfg(
        out_dir=_coerce_path(t.get("runtime", {}), "out_dir"),
        max_iters=int(t.get("runtime", {}).get("max_iters", 200_000)),
        eval_interval=int(t.get("runtime", {}).get("eval_interval", 300)),
        eval_iters=int(t.get("runtime", {}).get("eval_iters", 100)),
        log_interval=int(t.get("runtime", {}).get("log_interval", 100)),
        eval_only=bool(t.get("runtime", {}).get("eval_only", False)),
        always_save_checkpoint=bool(
            t.get("runtime", {}).get("always_save_checkpoint", True)
        ),
        seed=int(t.get("runtime", {}).get("seed", 1337)),
        device=str(t.get("runtime", {}).get("device", "mps")),
        dtype=str(t.get("runtime", {}).get("dtype", "float16")),
        compile=bool(t.get("runtime", {}).get("compile", False)),
        ckpt_time_interval_minutes=int(
            t.get("runtime", {}).get("ckpt_time_interval_minutes", 10)
        ),
        ckpt_atomic=bool(t.get("runtime", {}).get("ckpt_atomic", False)),
        best_smoothing_alpha=float(
            t.get("runtime", {}).get("best_smoothing_alpha", 0.0)
        ),
        early_stop_patience=int(t.get("runtime", {}).get("early_stop_patience", 5)),
        ema_decay=float(t.get("runtime", {}).get("ema_decay", 0.0)),
        save_merged_on_best=bool(
            t.get("runtime", {}).get("save_merged_on_best", False)
        ),
        keep_last_n=int(t.get("runtime", {}).get("keep_last_n", 0)),
    )

    sample_runtime = SampleRuntime(
        out_dir=_coerce_path(s.get("runtime", {}), "out_dir"),
        device=str(s.get("runtime", {}).get("device", "mps")),
        dtype=str(s.get("runtime", {}).get("dtype", "float16")),
        compile=bool(s.get("runtime", {}).get("compile", False)),
        seed=int(s.get("runtime", {}).get("seed", 1337)),
    )
    sample_cfg = SampleCfg(**s.get("sample", {}))

    return AppTOML(
        prepare=prepare,
        train=TrainTOML(
            hf_model=hf_model,
            peft=peft_cfg,
            data=data_cfg,
            optim=optim_cfg,
            schedule=sched_cfg,
            runtime=runtime_cfg,
        ),
        sample=SampleTOML(runtime=sample_runtime, sample=sample_cfg),
    )


# -----------------------------
# Prepare step
# -----------------------------

_SPECIAL_TOKENS = [
    # deactivated for now, since we have no code yet, that inserts these tokens
    # "<SPEECH_START>",
    # "<SPEECH_END>",
    # "<SPEAKER>",
    # "</SPEAKER>",
    # "<APPLAUSE>",
    # "<LAUGHTER>",
    # "<INTERRUPTION>",
    "<DOC_SEP>",
]


def _read_all_texts(raw_dir: Path) -> List[str]:
    # Deterministic order: sorted by path
    files = [p for p in sorted(raw_dir.glob("**/*.txt")) if p.is_file()]
    # Fallback to input.txt directly in raw_dir
    if not files and (raw_dir / "input.txt").exists():
        files = [raw_dir / "input.txt"]
    texts: List[str] = []
    for p in files:
        texts.append(p.read_text(encoding="utf-8"))
    return texts


def _decorate_texts(texts: List[str], add_tokens: bool) -> List[str]:
    if not add_tokens:
        return texts
    decorated: List[str] = []
    for t in texts:
        s = f"<SPEECH_START><SPEAKER>Unknown</SPEAKER>\n{t}\n<SPEECH_END>"
        decorated.append(s)
    return decorated


def prepare_from_toml(config_path: Path) -> None:
    raise SystemExit("prepare_from_toml has been removed. Use injected config via CLI.")


# -----------------------------
# Data packing
# -----------------------------


class PackedJSONL(Dataset):
    def __init__(
        self,
        jsonl_path: Path,
        tokenizer,
        block_size: int,
        sep_token: Optional[str] = None,
    ):
        self.block_size = block_size
        self.tokenizer = tokenizer
        texts: List[str] = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                texts.append(obj.get("text", ""))
        # pack with separator
        sep = sep_token if sep_token is not None else (tokenizer.eos_token or "")
        joined = (sep or "").join(texts)
        ids: List[int] = tokenizer.encode(joined, add_special_tokens=False)
        self.ids = torch.tensor(ids, dtype=torch.long)
        # number of full blocks
        self.num_blocks = max(0, (len(self.ids) - 1) // block_size)

    def __len__(self) -> int:
        return self.num_blocks

    def __getitem__(self, idx: int):
        start = idx * self.block_size
        x = self.ids[start : start + self.block_size]
        y = self.ids[start + 1 : start + 1 + self.block_size]
        return x, y


# -----------------------------
# Training utilities
# -----------------------------


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _autocast_ctx(device_str: str, dtype: torch.dtype):
    dev: Literal["cpu", "cuda", "mps"] = (
        "mps" if device_str == "mps" else ("cuda" if device_str == "cuda" else "cpu")
    )
    # Use top-level torch.autocast for better type stub compatibility and MPS support
    return torch.autocast(device_type=dev, dtype=dtype)  # type: ignore[attr-defined]


def _get_lr(
    it: int, warmup: int, decay_iters: int, min_lr: float, base_lr: float
) -> float:
    if it < warmup:
        return base_lr * it / max(1, warmup)
    if it > decay_iters:
        return min_lr
    decay_ratio = (it - warmup) / max(1, (decay_iters - warmup))
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (base_lr - min_lr)


def _save_adapters(model: nn.Module, save_dir: Path, atomic: bool = False) -> None:
    tmp = save_dir.with_suffix(".tmp") if atomic else save_dir
    os.makedirs(tmp, exist_ok=True)
    # For PeftModel.save_pretrained: save into directory
    save_fn = getattr(model, "save_pretrained", None)
    if callable(save_fn):
        # If PEFT is present and embeddings were resized, avoid warning by being explicit.
        try:
            save_fn(str(tmp), save_embedding_layers=True)  # type: ignore[call-arg]
        except TypeError:
            # Fallback for save_pretrained signatures that don't accept this kwarg
            save_fn(str(tmp))
    if atomic:
        # Ensure destination directory exists before moving files from tmp
        os.makedirs(save_dir, exist_ok=True)
        if save_dir.exists():
            for p in save_dir.iterdir():
                if p.is_file():
                    p.unlink()
        # rename tmp -> save_dir
        for src in tmp.iterdir():
            dst = save_dir / src.name
            if dst.exists():
                dst.unlink()
            src.replace(dst)
        tmp.rmdir()


def _prune_old_iters(adapters_root: Path, keep_last_n: int, current_dir: Path) -> None:
    """
    Keep only the latest N iter_* directories under adapters_root.
    Does not touch 'best' or 'last'. No-op if keep_last_n <= 0.
    """
    try:
        if keep_last_n <= 0:
            return
        if not adapters_root.exists():
            return
        candidates = []
        for p in adapters_root.iterdir():
            if p.is_dir() and p.name.startswith("iter_"):
                try:
                    num = int(p.name.split("_", 1)[1])
                except Exception:
                    continue
                candidates.append((num, p))
        if not candidates:
            return
        candidates.sort(key=lambda x: x[0])  # ascending by iter
        # Keep only the last N
        to_delete = candidates[:-keep_last_n] if keep_last_n > 0 else candidates
        for _, p in to_delete:
            try:
                if p.resolve() == current_dir.resolve():
                    continue
                shutil.rmtree(p)
            except Exception:
                # best effort pruning
                pass
    except Exception:
        # fully best-effort
        pass


# -----------------------------
# Train step (HF + PEFT on MPS)
# -----------------------------


def train_from_toml(config_path: Path) -> None:
    raise SystemExit("train_from_toml has been removed. Use injected config via CLI.")


# -----------------------------
# Sampling
# -----------------------------


def sample_from_toml(config_path: Path) -> None:
    raise SystemExit("sample_from_toml has been removed. Use injected config via CLI.")


# -----------------------------
# Composite loop
# -----------------------------


def loop(config_path: Path) -> None:
    raise SystemExit(
        "loop has been removed. Use explicit CLI steps with injected config."
    )
