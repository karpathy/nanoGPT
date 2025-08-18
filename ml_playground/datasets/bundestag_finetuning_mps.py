from __future__ import annotations

import json
import math
import os
import random
import time
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Literal, cast

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from peft import LoraConfig, PeftModel  # noqa: F401


def _require_peft():
    try:
        from peft import LoraConfig, PeftModel, get_peft_model  # type: ignore
    except Exception as e:  # pragma: no cover - only raised if peft missing
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
    cfg = _parse_app(config_path)
    p = cfg.prepare
    p.dataset_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.train.hf_model.model_name)
    special_tokens: List[str] = []
    if p.add_structure_tokens:
        # Ensure DOC_SEP present (may be redundant if user sets add_structure_tokens True)
        special_tokens = list(_SPECIAL_TOKENS)
        if cfg.prepare.doc_separator not in special_tokens:
            special_tokens.append(cfg.prepare.doc_separator)
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    tok_dir = p.dataset_dir / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(tok_dir)

    # Texts
    texts = _read_all_texts(p.raw_dir)
    texts = _decorate_texts(texts, p.add_structure_tokens)

    # Split 90/10
    n = len(texts)
    n_train = max(1, int(0.9 * n))
    train_texts = texts[:n_train]
    val_texts = texts[n_train:]
    if not val_texts and texts:
        val_texts = texts[-1:]

    # Save jsonl
    def _write_jsonl(path: Path, items: List[str]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for s in items:
                f.write(json.dumps({"text": s}, ensure_ascii=False) + "\n")

    train_path = p.dataset_dir / "train.jsonl"
    val_path = p.dataset_dir / "val.jsonl"
    _write_jsonl(train_path, train_texts)
    _write_jsonl(val_path, val_texts)

    # meta.json
    meta = {
        "base_model": cfg.train.hf_model.model_name,
        "special_tokens": special_tokens,
        "format": "jsonl",
        "block_size": cfg.train.hf_model.block_size,
        "notes": "uses tokenizer from tokenizer/",
    }
    (p.dataset_dir / "meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"[prepare] wrote {train_path}, {val_path}, tokenizer/, meta.json")


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
    app = _parse_app(config_path)
    t = app.train
    rt = t.runtime

    _set_seed(rt.seed)
    device = torch.device(rt.device)
    if rt.device != "mps":
        print(
            f"[train] Warning: requested device {rt.device}, integration optimized for MPS."
        )

    out_dir = rt.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "adapters").mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    # Tokenizer: load from dataset and copy to out_dir
    tok_dir = app.prepare.dataset_dir / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    # Avoid tokenizer max-length warning when building long packed streams
    try:
        tokenizer.model_max_length = int(1e12)  # effectively "no limit" for our packing
    except Exception:
        pass
    out_tok_dir = out_dir / "tokenizer"
    if not out_tok_dir.exists():
        out_tok_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(out_tok_dir)

    # Model
    torch_dtype = torch.float16 if rt.dtype == "float16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        t.hf_model.model_name, torch_dtype=torch_dtype
    )
    # Ensure compatibility with gradient checkpointing
    if t.hf_model.gradient_checkpointing:
        try:
            if hasattr(model, "config") and hasattr(model.config, "use_cache"):
                model.config.use_cache = False
        except Exception:
            pass
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()  # type: ignore[attr-defined]

    # Resize embeddings if tokenizer was extended with special tokens
    if hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tokenizer))  # type: ignore[attr-defined]

    # LoRA
    peft_targets = list(t.peft.target_modules)
    if t.peft.extend_mlp_targets:
        for m in ["up_proj", "down_proj", "gate_proj"]:
            if m not in peft_targets:
                peft_targets.append(m)
    if t.peft.enabled:
        LoraConfig, _, get_peft_model = _require_peft()
        lora_cfg = LoraConfig(
            r=t.peft.r,
            lora_alpha=t.peft.lora_alpha,
            lora_dropout=t.peft.lora_dropout,
            bias=cast(Literal["none", "all", "lora_only"], t.peft.bias),
            task_type="CAUSAL_LM",
            target_modules=peft_targets,
        )
        model = get_peft_model(model, lora_cfg)

    cast(nn.Module, model).to(device)  # type: ignore[arg-type]

    # Data
    block_size = t.data.block_size
    doc_sep = app.prepare.doc_separator
    train_ds = PackedJSONL(
        app.prepare.dataset_dir / "train.jsonl", tokenizer, block_size, doc_sep
    )
    val_ds = PackedJSONL(
        app.prepare.dataset_dir / "val.jsonl", tokenizer, block_size, doc_sep
    )

    # Preflight dataset checks to avoid RandomSampler(num_samples=0) and zero-batch infinite loops
    train_blocks = len(train_ds)
    if train_blocks == 0:
        raise SystemExit(
            f"[train] No training samples available. The tokenized corpus is shorter than block_size+1 (block_size={block_size}). "
            f"Consider reducing train.data.block_size in the TOML (e.g., 128 or 256) or adding more raw text to {app.prepare.dataset_dir}."
        )
    drop_last_train = True
    if train_blocks < t.data.batch_size:
        print(
            f"[train] Note: dataset provides {train_blocks} training blocks but batch_size={t.data.batch_size}. "
            f"Using drop_last=False to allow small final batches."
        )
        drop_last_train = False

    def _collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
        xs, ys = zip(*batch)
        x = torch.stack(xs)
        y = torch.stack(ys)
        return x.to(device), y.to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=t.data.batch_size,
        shuffle=t.data.shuffle,
        drop_last=drop_last_train,
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=t.data.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=_collate,
    )

    # Optimizer and scheduler
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=t.optim.learning_rate,
        betas=(t.optim.beta1, t.optim.beta2),
        weight_decay=t.optim.weight_decay,
    )

    it = 0
    best_val = float("inf")
    tokens_seen = 0
    last_ckpt_time = time.time()

    def evaluate() -> float:
        model.eval()
        losses: List[float] = []
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                if i >= rt.eval_iters:
                    break
                with _autocast_ctx(rt.device, torch.float16):
                    out = model(input_ids=x, labels=y)
                    loss = out.loss
                losses.append(float(loss.detach().cpu()))
        model.train()
        return float(sum(losses) / max(1, len(losses)))

    # Resume if possible
    state_path = out_dir / "ckpt_last.json"
    adapters_last = out_dir / "adapters" / "last"
    if state_path.exists() and adapters_last.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            it = int(state.get("iter", 0))
            best_val = float(state.get("best_val", best_val))
            tokens_seen = int(state.get("tokens_seen", 0))
            # load adapters
            _, PeftModel, _ = _require_peft()
            base = AutoModelForCausalLM.from_pretrained(
                t.hf_model.model_name, torch_dtype=torch_dtype
            )
            # Ensure compatibility with gradient checkpointing on resume
            if t.hf_model.gradient_checkpointing:
                try:
                    if hasattr(base, "config") and hasattr(base.config, "use_cache"):
                        base.config.use_cache = False
                except Exception:
                    pass
            base.resize_token_embeddings(len(tokenizer))  # type: ignore[attr-defined]
            model = PeftModel.from_pretrained(base, str(adapters_last))
            cast(nn.Module, model).to(device)  # type: ignore[arg-type]
            if t.hf_model.gradient_checkpointing and hasattr(
                model, "gradient_checkpointing_enable"
            ):
                model.gradient_checkpointing_enable()  # type: ignore[attr-defined]
            print(f"[train] Resumed from iter {it}, best_val={best_val}")
        except Exception as e:  # pragma: no cover - best effort resume
            print(f"[train] Warning: resume failed: {e}")

    model.train()

    # Training loop
    accum = t.data.grad_accum_steps
    total_steps = t.runtime.max_iters
    log_every = max(1, rt.log_interval)

    while it < total_steps:
        t0 = time.time()
        for step, (x, y) in enumerate(train_loader):
            # Learning rate schedule
            lr = (
                _get_lr(
                    it,
                    t.schedule.warmup_iters,
                    t.schedule.lr_decay_iters,
                    t.schedule.min_lr,
                    t.optim.learning_rate,
                )
                if t.schedule.decay_lr
                else t.optim.learning_rate
            )
            for pg in optim.param_groups:
                pg["lr"] = lr

            with _autocast_ctx(rt.device, torch.float16):
                out = model(input_ids=x, labels=y)
                loss = out.loss / accum
            loss.backward()

            if t.optim.grad_clip and t.optim.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), t.optim.grad_clip)

            if (step + 1) % accum == 0:
                optim.step()
                optim.zero_grad(set_to_none=True)
                it += 1

                if it % log_every == 0:
                    dt = time.time() - t0
                    tokens_in_step = t.data.batch_size * accum * (t.data.block_size)
                    tokens_sec = tokens_in_step / max(1e-6, dt)
                    print(
                        f"iter {it} | loss {float(loss.detach().cpu()) * accum:.4f} | lr {lr:.3e} | tok/s {tokens_sec:.0f}"
                    )
                    t0 = time.time()

                # Eval + checkpoint
                if (it % rt.eval_interval == 0) or (rt.eval_only):
                    val = evaluate()
                    print(f"[eval] iter {it} | val_loss {val:.4f}")
                    # Save last adapters
                    adapters_iter_dir = out_dir / "adapters" / f"iter_{it}"
                    _save_adapters(model, adapters_iter_dir, atomic=rt.ckpt_atomic)
                    # Update last pointer
                    if (out_dir / "adapters" / "last").exists():
                        # replace symlink or dir
                        try:
                            if (out_dir / "adapters" / "last").is_symlink():
                                (out_dir / "adapters" / "last").unlink()
                        except Exception:
                            pass
                    # point "last" to latest directory
                    try:
                        os.symlink(
                            adapters_iter_dir.name, out_dir / "adapters" / "last"
                        )
                    except Exception:
                        # Fallback: copy files
                        _save_adapters(
                            model, out_dir / "adapters" / "last", atomic=rt.ckpt_atomic
                        )

                    # Retention: keep only the latest N iter_* checkpoints if configured
                    if rt.keep_last_n and rt.keep_last_n > 0:
                        _prune_old_iters(out_dir / "adapters", rt.keep_last_n, adapters_iter_dir)

                    # Track best
                    improved = val < best_val
                    if improved:
                        best_val = val
                        _save_adapters(
                            model, out_dir / "adapters" / "best", atomic=rt.ckpt_atomic
                        )

                    # Save small state json
                    state = {
                        "iter": it,
                        "best_val": best_val,
                        "tokens_seen": tokens_seen,
                        "val_loss": val,
                    }
                    (out_dir / "ckpt_last.json").write_text(
                        json.dumps(state, indent=2), encoding="utf-8"
                    )

                    if rt.eval_only:
                        return

                # Time-based checkpointing
                if (
                    rt.ckpt_time_interval_minutes > 0
                    and (time.time() - last_ckpt_time)
                    >= rt.ckpt_time_interval_minutes * 60
                ):
                    adapters_time_dir = (
                        out_dir / "adapters" / f"time_{int(time.time())}"
                    )
                    _save_adapters(model, adapters_time_dir, atomic=rt.ckpt_atomic)
                    last_ckpt_time = time.time()

            tokens_seen += x.numel()

        # If we finish the loader and still have it < total_steps, continue looping over data
        if it >= total_steps:
            break

    print("[train] finished training")


# -----------------------------
# Sampling
# -----------------------------


def sample_from_toml(config_path: Path) -> None:
    app = _parse_app(config_path)
    s = app.sample

    device = torch.device(s.runtime.device)
    torch_dtype = torch.float16 if s.runtime.dtype == "float16" else torch.float32

    # Tokenizer: prefer out_dir/tokenizer
    out_tok_dir = s.runtime.out_dir / "tokenizer"
    tok_dir = (
        out_tok_dir if out_tok_dir.exists() else (app.prepare.dataset_dir / "tokenizer")
    )
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)

    # Model + adapters
    base = AutoModelForCausalLM.from_pretrained(
        app.train.hf_model.model_name, torch_dtype=torch_dtype
    )
    # Try to load adapters first; if size mismatch on embeddings/lm_head occurs, reload a fresh base, align vocab, and retry.
    adapters_dir = s.runtime.out_dir / "adapters" / "best"
    if not adapters_dir.exists():
        adapters_dir = s.runtime.out_dir / "adapters" / "last"
    if adapters_dir.exists():
        _, PeftModel, _ = _require_peft()
        try:
            model = PeftModel.from_pretrained(base, str(adapters_dir))
        except RuntimeError as e:
            msg = str(e)
            if "size mismatch" in msg and ("embed_tokens" in msg or "lm_head" in msg):
                # Reload a clean base to avoid duplicate peft_config, then align vocab and retry
                fresh_base = AutoModelForCausalLM.from_pretrained(
                    app.train.hf_model.model_name, torch_dtype=torch_dtype
                )
                try:
                    fresh_base.resize_token_embeddings(len(tokenizer))  # type: ignore[attr-defined]
                except Exception:
                    pass
                model = PeftModel.from_pretrained(fresh_base, str(adapters_dir))
            else:
                raise
    else:
        print("[sample] Warning: adapters not found; using base model.")
        model = base

    # Ensure final embedding size matches tokenizer (handles both larger/smaller cases safely).
    try:
        current_vocab = cast(nn.Module, model).get_input_embeddings().weight.shape[0]  # type: ignore[attr-defined]
    except Exception:
        current_vocab = base.get_input_embeddings().weight.shape[0]  # type: ignore[attr-defined]
    target_vocab = len(tokenizer)
    if target_vocab != current_vocab:
        cast(nn.Module, model).resize_token_embeddings(target_vocab)  # type: ignore[attr-defined]

    cast(nn.Module, model).to(device)  # type: ignore[arg-type]
    model.eval()

    # Seed
    _set_seed(s.runtime.seed)

    prompt = s.sample.start
    if prompt.startswith("FILE:"):
        prompt = Path(prompt[5:]).read_text(encoding="utf-8")

    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        with _autocast_ctx(s.runtime.device, torch.float16):
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=s.sample.max_new_tokens,
                do_sample=True,
                temperature=s.sample.temperature,
                top_k=s.sample.top_k,
                top_p=s.sample.top_p,
                pad_token_id=tokenizer.eos_token_id,
            )

    text = tokenizer.decode(out[0], skip_special_tokens=False)
    samples_dir = s.runtime.out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    out_path = samples_dir / f"sample-{ts}.txt"
    out_path.write_text(text, encoding="utf-8")
    print(f"[sample] wrote {out_path}")


# -----------------------------
# Composite loop
# -----------------------------


def loop(config_path: Path) -> None:
    prepare_from_toml(config_path)
    train_from_toml(config_path)
    sample_from_toml(config_path)
