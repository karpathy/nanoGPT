from __future__ import annotations
import math
import time
import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Tuple, List
import pickle
import torch
from dataclasses import dataclass
from .model import GPTConfig, GPT
from .config import TrainExperiment
from .device import setup
from .data import SimpleBatches


@dataclass
class _CkptInfo:
    path: Path
    metric: float
    iter_num: int
    created_at: float


def _atomic_save(obj, path: Path, atomic: bool) -> None:
    if not atomic:
        torch.save(obj, path)
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_sidecar(path: Path, *, metric: float, iter_num: int, filename: str) -> None:
    meta = {
        "metric": metric,
        "iter_num": iter_num,
        "filename": filename,
        "created_at": time.time(),
        "sha256": _sha256_of_file(path.with_name(filename)),
    }
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


class _EMA:
    def __init__(self, model: GPT, decay: float, device: str):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone().to(device)
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model: GPT) -> None:
        d = self.decay
        for k, v in model.state_dict().items():
            if k in self.shadow and v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v, alpha=1.0 - d)

    @torch.no_grad()
    def apply_to(self, model: GPT) -> None:
        sd = model.state_dict()
        for k, v in self.shadow.items():
            sd[k].copy_(v)


@torch.no_grad()
def _estimate_loss(
    model: GPT, batches: SimpleBatches, eval_iters: int, ctx, device: str
) -> Dict[str, float]:
    model.eval()
    losses: Dict[str, float] = {}
    for split in ("train", "val"):
        acc = 0.0
        for _ in range(eval_iters):
            X, Y = batches.get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            acc += float(loss.item())
        losses[split] = acc / float(eval_iters)
    model.train()
    return losses


def _get_lr(
    it: int, *, warmup: int, decay_iters: int, min_lr: float, base_lr: float
) -> float:
    if it < warmup:
        return base_lr * (it + 1) / (warmup + 1)
    if it > decay_iters:
        return min_lr
    decay_ratio = (it - warmup) / (decay_iters - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (base_lr - min_lr)


def _load_meta_vocab_size(meta_path: Path) -> int | None:
    if meta_path.exists():
        with meta_path.open("rb") as f:
            meta = pickle.load(f)
        vs = meta.get("vocab_size")
        return int(vs) if isinstance(vs, int) else None
    return None


def train(exp: TrainExperiment) -> Tuple[int, float]:
    rt = exp.runtime
    model_cfg = exp.model

    rt.out_dir.mkdir(parents=True, exist_ok=True)

    device_type, ptdtype, ctx = setup(rt.device, rt.dtype, rt.seed)

    # Data
    batches = SimpleBatches(exp.data, device=device_type)

    # Determine vocab size (from config or dataset meta)
    vocab_size = model_cfg.vocab_size
    if vocab_size is None and exp.data.meta_pkl is not None:
        vocab_size = (
            _load_meta_vocab_size(exp.data.dataset_dir / exp.data.meta_pkl) or 50304
        )
    if vocab_size is None:
        vocab_size = 50304

    # Build model args from config
    model_args = {
        "n_layer": model_cfg.n_layer,
        "n_head": model_cfg.n_head,
        "n_embd": model_cfg.n_embd,
        "block_size": model_cfg.block_size,
        "bias": model_cfg.bias,
        "vocab_size": vocab_size,
        "dropout": model_cfg.dropout,
    }

    # Auto-resume if checkpoint exists (use ckpt_last)
    ckpt_path = rt.out_dir / rt.ckpt_last_filename
    checkpoint = None
    if ckpt_path.exists():
        print(f"[train] Resuming from checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device_type)
        # Override model_args with those from checkpoint for strictness
        ckpt_model_args = checkpoint.get("model_args", {})
        for k in [
            "n_layer",
            "n_head",
            "n_embd",
            "block_size",
            "bias",
            "vocab_size",
            "dropout",
        ]:
            if k in ckpt_model_args:
                model_args[k] = ckpt_model_args[k]

    # Instantiate model and (if resuming) load weights
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf).to(device_type)

    if checkpoint is not None:
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k in list(state_dict.keys()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    else:
        # Fresh run; optionally crop block size down if requested smaller than default
        if model_cfg.block_size < model.config.block_size:
            model.crop_block_size(model_cfg.block_size)

    # Optimizer and scaler
    scaler = torch.amp.GradScaler(
        enabled=(rt.dtype == "float16" and device_type == "cuda")
    )
    optim = model.configure_optimizers(
        weight_decay=exp.optim.weight_decay,
        learning_rate=exp.optim.learning_rate,
        betas=(exp.optim.beta1, exp.optim.beta2),
        device_type=device_type,
    )
    if checkpoint is not None and "optimizer" in checkpoint:
        optim.load_state_dict(checkpoint["optimizer"])  # resume optimizer state
    if checkpoint is not None and "scaler" in checkpoint:
        try:
            scaler.load_state_dict(checkpoint["scaler"])  # resume scaler state
        except Exception:
            pass

    # Restore RNG states if available
    if checkpoint is not None and "rng" in checkpoint:
        try:
            torch.set_rng_state(checkpoint["rng"].get("torch"))
        except Exception:
            pass
        try:
            if torch.cuda.is_available() and checkpoint["rng"].get("cuda") is not None:
                torch.cuda.set_rng_state_all(checkpoint["rng"].get("cuda"))
        except Exception:
            pass

    # free memory ASAP for state dicts loaded
    state_dict = None  # type: ignore

    raw_model = model
    if rt.compile:
        model = torch.compile(model)

    # Initialize EMA/retention/smoothing state
    ema = _EMA(raw_model, rt.ema_decay, device_type) if rt.ema_decay > 0.0 else None
    best_ckpts: List[_CkptInfo] = []
    last_time_ckpt = time.time()

    if rt.ckpt_greater_is_better:
        best_metric = float("-inf")
    else:
        best_metric = float("inf")
    smoothed_metric: float | None = None
    no_improve_evals = 0

    # Initialize training counters (override if resuming)
    iter_num = 0
    best_val = float("inf")
    if checkpoint is not None:
        iter_num = int(checkpoint.get("iter_num", 0))
        best_val = float(checkpoint.get("best_val_loss", float("inf")))
        print(f"[train] Resumed at iter {iter_num}, best_val_loss {best_val}")
        # Do not clear checkpoint yet; it may be needed for best_metric seed; but we proceed
        checkpoint = None

    X, Y = batches.get_batch("train")
    t0 = time.time()

    tokens_per_iter = (
        exp.data.grad_accum_steps * exp.data.batch_size * exp.data.block_size
    )
    print(f"tokens per iteration: {tokens_per_iter:,}")

    while iter_num <= rt.max_iters:
        lr = (
            _get_lr(
                iter_num,
                warmup=exp.schedule.warmup_iters,
                decay_iters=exp.schedule.lr_decay_iters,
                min_lr=exp.schedule.min_lr,
                base_lr=exp.optim.learning_rate,
            )
            if exp.schedule.decay_lr
            else exp.optim.learning_rate
        )
        for g in optim.param_groups:
            g["lr"] = lr

        if iter_num % rt.eval_interval == 0:
            losses = _estimate_loss(model, batches, rt.eval_iters, ctx, device_type)
            val_loss = float(losses["val"])  # canonical source
            print(f"step {iter_num}: train {losses['train']:.4f}, val {val_loss:.4f}")
            metric = val_loss if rt.ckpt_metric == "val_loss" else math.exp(val_loss)

            # Optional smoothing (EMA of metric values)
            if rt.best_smoothing_alpha > 0.0:
                if smoothed_metric is None:
                    smoothed_metric = metric
                else:
                    a = rt.best_smoothing_alpha
                    smoothed_metric = a * metric + (1.0 - a) * smoothed_metric
                decision_metric = smoothed_metric
            else:
                decision_metric = metric

            is_improved = (
                (decision_metric > best_metric)
                if rt.ckpt_greater_is_better
                else (decision_metric < best_metric)
            )

            # Construct base checkpoint payload
            ckpt_base = {
                "model": raw_model.state_dict(),
                "optimizer": optim.state_dict(),
                "scaler": scaler.state_dict(),
                "rng": {
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None,
                },
                "model_args": {
                    "n_layer": model_cfg.n_layer,
                    "n_head": model_cfg.n_head,
                    "n_embd": model_cfg.n_embd,
                    "block_size": model_cfg.block_size,
                    "bias": model_cfg.bias,
                    "vocab_size": vocab_size,
                    "dropout": model_cfg.dropout,
                },
                "iter_num": iter_num,
                "best_val_loss": best_val,
            }

            # Save robust "last" checkpoint
            last_path = rt.out_dir / rt.ckpt_last_filename
            _atomic_save(ckpt_base, last_path, rt.ckpt_atomic)
            if rt.ckpt_write_metadata:
                _write_sidecar(
                    last_path.with_suffix(".json"),
                    metric=metric,
                    iter_num=iter_num,
                    filename=last_path.name,
                )

            # Optional time-based safety checkpoint (updates timer after save)
            if (
                rt.ckpt_time_interval_minutes > 0
                and (time.time() - last_time_ckpt) >= rt.ckpt_time_interval_minutes * 60
            ):
                last_time_ckpt = time.time()

            # Save/update the "best" checkpoint when improved or per policy
            if is_improved or rt.always_save_checkpoint:
                best_metric = decision_metric
                best_val = min(best_val, val_loss)
                if iter_num > 0:
                    payload = dict(ckpt_base)
                    if ema is not None:
                        payload["model"] = ema.shadow
                    best_path = rt.out_dir / rt.ckpt_best_filename
                    _atomic_save(payload, best_path, rt.ckpt_atomic)
                    if rt.ckpt_write_metadata:
                        _write_sidecar(
                            best_path.with_suffix(".json"),
                            metric=metric,
                            iter_num=iter_num,
                            filename=best_path.name,
                        )

                    # Optional top-k archive of best checkpoints
                    if rt.ckpt_top_k > 0:
                        stamp = int(time.time())
                        numbered = rt.out_dir / f"ckpt_best-{stamp}.pt"
                        _atomic_save(payload, numbered, rt.ckpt_atomic)
                        best_ckpts.append(
                            _CkptInfo(numbered, metric, iter_num, time.time())
                        )
                        # prune worst beyond k
                        best_ckpts.sort(
                            key=lambda x: x.metric, reverse=rt.ckpt_greater_is_better
                        )
                        while len(best_ckpts) > rt.ckpt_top_k:
                            old = best_ckpts.pop()
                            try:
                                old.path.unlink(missing_ok=True)
                                side = old.path.with_suffix(".json")
                                if side.exists():
                                    side.unlink()
                            except Exception as e:
                                print(
                                    f"[ckpt] warning: failed to delete {old.path}: {e}"
                                )
                no_improve_evals = 0
            else:
                no_improve_evals += 1
                if 0 < rt.early_stop_patience <= no_improve_evals:
                    print(
                        f"[train] early stopping after {no_improve_evals} evals without improvement"
                    )
                    break

        if iter_num == 0 and rt.eval_only:
            break

        # Gradient accumulation
        for _ in range(exp.data.grad_accum_steps):
            with ctx:
                _, loss = model(X, Y)
                loss = loss / exp.data.grad_accum_steps
            X, Y = batches.get_batch("train")
            scaler.scale(loss).backward()

        if exp.optim.grad_clip > 0:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), exp.optim.grad_clip)

        scaler.step(optim)
        scaler.update()
        optim.zero_grad(set_to_none=True)

        # Update EMA after optimizer step
        if ema is not None:
            ema.update(raw_model)

        if iter_num % rt.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            total_loss = float(loss.item()) * exp.data.grad_accum_steps
            print(
                f"iter {iter_num}: loss {total_loss:.4f}, step_time {dt * 1000:.1f}ms"
            )

        iter_num += 1

    return iter_num, best_val
