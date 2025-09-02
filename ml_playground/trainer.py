from __future__ import annotations
import math
import time
import os
import json
import hashlib
from pathlib import Path
import shutil
from typing import Any, Dict, Tuple, Protocol, Optional, List, cast
import pickle
import torch
from dataclasses import dataclass
from ml_playground.model import GPTConfig, GPT
from ml_playground.config import TrainerConfig
from ml_playground.device import setup
from ml_playground.data import SimpleBatches
from ml_playground.error_handling import (
    DataError,
    ModelError,
    CheckpointError,
    setup_logging,
)
import logging

"""
Centralized training utilities for ml_playground experiments.

This module provides standardized utilities for model training including:
- Checkpoint management with atomic operations
- Exponential Moving Average (EMA) support
- TensorBoard logging integration
- Progress reporting and metrics tracking
- Error handling with centralized exception types

All experiments should use these utilities to ensure consistency and proper error handling.
"""

# Add TensorBoard (best-effort)
try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:  # pragma: no cover - allow training without tensorboard installed

    class SummaryWriter:  # type: ignore
        def __init__(self, *args, **kwargs) -> None:
            pass

        def add_scalar(self, *args, **kwargs) -> None:
            pass

        def add_histogram(self, *args, **kwargs) -> None:
            pass

        def close(self) -> None:
            pass


class Trainer(Protocol):
    def __call__(self, cfg: TrainerConfig) -> Tuple[int, float]: ...


@dataclass(frozen=True)
class Checkpoint:
    """A strongly-typed checkpoint object."""

    model: Dict[str, Any]
    optimizer: Dict[str, Any]
    model_args: Dict[str, Any]
    iter_num: int
    best_val_loss: float
    config: Dict[str, Any]
    ema: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint to dictionary for serialization."""
        result = {
            "model": self.model,
            "optimizer": self.optimizer,
            "model_args": self.model_args,
            "iter_num": self.iter_num,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        if self.ema is not None:
            result["ema"] = self.ema
        return result

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to attributes."""
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        """Allow 'in' operator to check for attribute existence."""
        return hasattr(self, key)


@dataclass
class _CkptInfo:
    """Internal checkpoint metadata for management."""

    path: Path
    metric: float
    iter_num: int
    created_at: float


class CheckpointManager:
    """A utility class for managing checkpoints with advanced features."""

    def __init__(
        self, out_dir: Path, atomic: bool = True, keep_last: int = 1, keep_best: int = 1
    ):
        self.out_dir = out_dir
        self.atomic = atomic
        if keep_last < 0 or keep_best < 0:
            raise CheckpointError(
                f"Invalid checkpoint keep policy: keep_last={keep_last}, keep_best={keep_best} (must be >= 0)"
            )
        self.keep_last = keep_last
        self.keep_best = keep_best
        self.last_checkpoints: List[_CkptInfo] = []
        self.best_checkpoints: List[_CkptInfo] = []
        # Discover any existing checkpoints so behavior persists across restarts
        self._discover_existing()

    def _discover_existing(self) -> None:
        """Scan filesystem for existing rotated checkpoints and rebuild state."""
        try:
            # last: ckpt_last_XXXXXXXX.pt
            for p in sorted(self.out_dir.glob("ckpt_last_*.pt")):
                try:
                    # iter from filename suffix
                    stem = p.stem  # e.g., ckpt_last_00000010
                    iter_str = stem.split("_")[-1]
                    it = int(iter_str)
                except Exception:
                    it = 0
                created = p.stat().st_mtime
                self.last_checkpoints.append(_CkptInfo(p, float("inf"), it, created))
            # best: ckpt_best_XXXXXXXX_*.pt (metric may be encoded)
            for p in sorted(self.out_dir.glob("ckpt_best_*.pt")):
                stem = p.stem  # e.g., ckpt_best_00000010_1.234567
                parts = stem.split("_")
                it = 0
                metric = float("inf")
                if len(parts) >= 3:
                    try:
                        it = int(parts[2])
                    except Exception:
                        it = 0
                    # try parse metric from suffix if present
                    try:
                        metric = float(parts[3]) if len(parts) >= 4 else float("inf")
                    except Exception:
                        metric = float("inf")
                created = p.stat().st_mtime
                self.best_checkpoints.append(_CkptInfo(p, metric, it, created))
        except Exception:
            # Best-effort; if anything fails, keep lists possibly partial
            pass

    def _update_stable_pointer(self, rotated_path: Path, stable_filename: str) -> None:
        """Point the stable filename to the rotated file via symlink if possible, else copy."""
        stable_path = self.out_dir / stable_filename
        try:
            if stable_path.exists() or stable_path.is_symlink():
                try:
                    stable_path.unlink()
                except Exception:
                    pass
            # Create relative symlink if supported
            try:
                stable_path.symlink_to(rotated_path.name)
                return
            except Exception:
                # Fallback: copy the file
                shutil.copy2(rotated_path, stable_path)
        except Exception:
            # Do not fail training on pointer update issues
            pass

    def save_checkpoint(
        self,
        checkpoint: Checkpoint,
        base_filename: str,
        metric: float,
        iter_num: int,
        logger: Optional[logging.Logger] = None,
        is_best: bool = False,
    ) -> Path:
        """Save a checkpoint with metadata and manage last and best checkpoints.

        Returns the rotated checkpoint path that was written.
        """
        # Determine rotated filename based on kind
        if is_best:
            rotated_name = f"ckpt_best_{iter_num:08d}_{metric:.6f}.pt"
        else:
            rotated_name = f"ckpt_last_{iter_num:08d}.pt"
        path = self.out_dir / rotated_name

        # Save the checkpoint
        _atomic_save(checkpoint.to_dict(), path, self.atomic)

        ckpt_info = _CkptInfo(path, metric, iter_num, time.time())

        # Manage last checkpoints
        if self.keep_last > 0 and not is_best:
            # Remove any existing checkpoint with the same path
            self.last_checkpoints = [
                ckpt for ckpt in self.last_checkpoints if ckpt.path != path
            ]
            self.last_checkpoints.append(ckpt_info)
            # Keep only the specified number of last checkpoints
            if len(self.last_checkpoints) > self.keep_last:
                # Remove oldest checkpoints
                to_remove = self.last_checkpoints[
                    : len(self.last_checkpoints) - self.keep_last
                ]
                self.last_checkpoints = self.last_checkpoints[
                    len(self.last_checkpoints) - self.keep_last :
                ]

                # Delete the files
                for ckpt in to_remove:
                    try:
                        ckpt.path.unlink()
                        # Also remove sidecar file if it exists
                        sidecar = ckpt.path.with_suffix(ckpt.path.suffix + ".json")
                        if sidecar.exists():
                            sidecar.unlink()
                        if logger:
                            logger.info(f"Removed old last checkpoint: {ckpt.path}")
                    except Exception as e:
                        if logger:
                            logger.warning(
                                f"Failed to remove old last checkpoint {ckpt.path}: {e}"
                            )

        # Manage best checkpoints
        if is_best and self.keep_best > 0:
            # Remove any existing checkpoint with the same path
            self.best_checkpoints = [
                ckpt for ckpt in self.best_checkpoints if ckpt.path != path
            ]
            self.best_checkpoints.append(ckpt_info)
            # Sort by metric (assuming lower is better by default)
            self.best_checkpoints.sort(key=lambda x: x.metric, reverse=False)

            # Keep only the specified number of best checkpoints
            if len(self.best_checkpoints) > self.keep_best:
                # Remove worst checkpoints
                to_remove = self.best_checkpoints[self.keep_best :]
                self.best_checkpoints = self.best_checkpoints[: self.keep_best]

                # Delete the files
                for ckpt in to_remove:
                    try:
                        ckpt.path.unlink()
                        # Also remove sidecar file if it exists
                        sidecar = ckpt.path.with_suffix(ckpt.path.suffix + ".json")
                        if sidecar.exists():
                            sidecar.unlink()
                        if logger:
                            logger.info(f"Removed old best checkpoint: {ckpt.path}")
                    except Exception as e:
                        if logger:
                            logger.warning(
                                f"Failed to remove old best checkpoint {ckpt.path}: {e}"
                            )

        # Update stable pointer (symlink or copy) to an actually kept checkpoint under the base filename
        try:
            target_for_pointer: Path = path
            if is_best:
                if self.best_checkpoints:
                    # best_checkpoints is sorted ascending by metric; choose the best
                    target_for_pointer = self.best_checkpoints[0].path
            else:
                if self.last_checkpoints:
                    # choose the most recent by created_at
                    target_for_pointer = max(self.last_checkpoints, key=lambda x: x.created_at).path
            self._update_stable_pointer(target_for_pointer, base_filename)
        except Exception:
            pass

        if logger:
            logger.info(f"Saved checkpoint to {path}")
        return path

    def load_latest_checkpoint(
        self, device: str, logger: Optional[logging.Logger] = None
    ) -> Optional[Checkpoint]:
        """Load the latest checkpoint from the last checkpoints list."""
        if not self.last_checkpoints:
            # try discovering from disk
            self._discover_existing()
            if not self.last_checkpoints:
                return None

        # Get the most recent checkpoint
        latest_ckpt = max(self.last_checkpoints, key=lambda x: x.created_at)

        try:
            checkpoint_dict = torch.load(str(latest_ckpt.path), map_location=device)
            if not isinstance(checkpoint_dict, dict):
                raise ValueError("Checkpoint file does not contain a dictionary")

            # Validate required keys
            required_keys = [
                "model",
                "optimizer",
                "model_args",
                "iter_num",
                "best_val_loss",
                "config",
            ]
            for key in required_keys:
                if key not in checkpoint_dict:
                    raise ValueError(f"Checkpoint missing required key: {key}")

            # Create Checkpoint object
            checkpoint = Checkpoint(
                model=checkpoint_dict["model"],
                optimizer=checkpoint_dict["optimizer"],
                model_args=checkpoint_dict["model_args"],
                iter_num=checkpoint_dict["iter_num"],
                best_val_loss=checkpoint_dict["best_val_loss"],
                config=checkpoint_dict["config"],
                ema=checkpoint_dict.get("ema"),
            )

            if logger:
                logger.info(f"Loaded checkpoint from {latest_ckpt.path}")
            return checkpoint
        except Exception as e:
            if logger:
                logger.error(f"Error loading checkpoint from {latest_ckpt.path}: {e}")
            return None

    def load_best_checkpoint(
        self, device: str, logger: Optional[logging.Logger] = None
    ) -> Optional[Checkpoint]:
        """Load the best checkpoint from the best checkpoints list."""
        if not self.best_checkpoints:
            self._discover_existing()
            if not self.best_checkpoints:
                return None

        # Get the best checkpoint (lowest metric)
        best_ckpt = min(self.best_checkpoints, key=lambda x: x.metric)

        try:
            checkpoint_dict = torch.load(str(best_ckpt.path), map_location=device)
            if not isinstance(checkpoint_dict, dict):
                raise ValueError("Checkpoint file does not contain a dictionary")

            # Validate required keys
            required_keys = [
                "model",
                "optimizer",
                "model_args",
                "iter_num",
                "best_val_loss",
                "config",
            ]
            for key in required_keys:
                if key not in checkpoint_dict:
                    raise ValueError(f"Checkpoint missing required key: {key}")

            # Create Checkpoint object
            checkpoint = Checkpoint(
                model=checkpoint_dict["model"],
                optimizer=checkpoint_dict["optimizer"],
                model_args=checkpoint_dict["model_args"],
                iter_num=checkpoint_dict["iter_num"],
                best_val_loss=checkpoint_dict["best_val_loss"],
                config=checkpoint_dict["config"],
                ema=checkpoint_dict.get("ema"),
            )

            if logger:
                logger.info(f"Loaded best checkpoint from {best_ckpt.path}")
            return checkpoint
        except Exception as e:
            if logger:
                logger.error(
                    f"Error loading best checkpoint from {best_ckpt.path}: {e}"
                )
            return None


def _atomic_save(obj, path: Path, atomic: bool) -> None:
    if not atomic:
        torch.save(obj, path)
        return
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp_path)
    tmp_path.replace(path)


def _sha256_of_file(path: Path) -> str:
    """Compute SHA256 of file contents."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _write_sidecar(
    *,
    sidecar_path: Path,
    pt_path: Path,
    kind: str,
    iter_num: int,
    tokens_seen: int,
    lr: float,
    eval_iters: int,
    metric_name: str,
    greater_is_better: bool,
    metric_raw: float,
    smoothing_alpha: float,
    decision_metric: float,
    ema_used_for_saved_model: bool,
    eval_metric_on_ema: float | None,
    device: str,
    dtype: str,
    dataset_meta: Dict[str, int] | None = None,
    progress: Dict[str, float | int] | None = None,
) -> None:
    """Write a sidecar JSON file with training metadata."""
    try:
        sidecar = {
            "kind": kind,
            "iter_num": iter_num,
            "tokens_seen": tokens_seen,
            "lr": lr,
            "eval_iters": eval_iters,
            "metric_name": metric_name,
            "greater_is_better": greater_is_better,
            "metric_raw": metric_raw,
            "smoothing_alpha": smoothing_alpha,
            "decision_metric": decision_metric,
            "ema_used_for_saved_model": ema_used_for_saved_model,
            "eval_metric_on_ema": eval_metric_on_ema,
            "device": device,
            "dtype": dtype,
            "dataset_meta": dataset_meta or {},
            "progress": progress or {},
            "sha256": _sha256_of_file(pt_path),
        }
        with sidecar_path.open("w") as f:
            json.dump(sidecar, f, indent=2)
    except Exception as e:
        # Log but don't fail training for sidecar write errors
        print(f"Warning: Failed to write sidecar {sidecar_path}: {e}")


class _EMA:
    """Exponential Moving Average for model weights."""

    def __init__(self, model: GPT, decay: float, device: str):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone().to(device)
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }

    def update(self, model: GPT) -> None:
        for name, param in model.state_dict().items():
            if param.dtype.is_floating_point and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.detach().to(self.shadow[name].device), alpha=1.0 - self.decay
                )

    def apply_to(self, model: GPT) -> None:
        state_dict = model.state_dict()
        for name, param in state_dict.items():
            if name in self.shadow:
                state_dict[name].copy_(self.shadow[name])


def _estimate_loss(
    model: GPT, batches: SimpleBatches, eval_iters: int, ctx: Any
) -> Dict[str, float]:
    """Estimate loss on train/val splits."""
    out = {}
    # Handle DDP model
    if hasattr(model, "module"):
        raw_model = model.module
    else:
        raw_model = model
    
    # Ensure raw_model is a GPT instance
    if not isinstance(raw_model, GPT):
        raise TypeError(f"Expected GPT model, got {type(raw_model)}")
    
    raw_model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = batches.get_batch(split)
            with ctx:
                logits, loss = raw_model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    raw_model.train()
    return out


def _get_lr(
    it: int, *, warmup: int, decay_iters: int, min_lr: float, base_lr: float
) -> float:
    """Learning rate decay scheduler (cosine with warmup)."""
    if it < warmup:
        return base_lr * it / warmup
    if it > decay_iters:
        return min_lr
    decay_ratio = (it - warmup) / (decay_iters - warmup)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (base_lr - min_lr)


def _load_meta_vocab_size(meta_path: Path) -> int | None:
    """Load vocab_size from meta.pkl if it exists."""
    try:
        with meta_path.open("rb") as f:
            meta = pickle.load(f)
        if not isinstance(meta, dict):
            return None
        return meta.get("vocab_size")
    except Exception:
        return None


def validate_checkpoint(checkpoint: dict, required_keys: set) -> None:
    """Validate that a checkpoint contains all required keys."""
    missing_keys = required_keys - set(checkpoint.keys())
    if missing_keys:
        raise CheckpointError(f"Checkpoint missing required keys: {missing_keys}")


def extract_model_args_from_checkpoint(checkpoint: dict) -> dict:
    """Extract model arguments from a checkpoint, with backward compatibility."""
    if "model_args" in checkpoint:
        return checkpoint["model_args"]

    # Backward compatibility: try to extract from config
    if "config" in checkpoint and "model" in checkpoint["config"]:
        return checkpoint["config"]["model"]

    # If we can't find model args, return empty dict
    return {}


def train(exp: TrainerConfig) -> Tuple[int, float]:
    rt = exp.runtime
    model_cfg = exp.model

    # Set up logging
    logger = setup_logging("ml_playground.train")

    try:
        rt.out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise DataError(f"Failed to create output directory {rt.out_dir}: {e}") from e

    # Initialize TensorBoard (configurable, default enabled)
    tb_dir = rt.out_dir / "logs" / "tb"
    if getattr(rt, "tensorboard_enabled", True):
        writer = SummaryWriter(log_dir=str(tb_dir))
    else:

        class _NoopTB:
            def add_scalar(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                pass

            def add_histogram(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                pass

            def close(self) -> None:
                pass

        writer = _NoopTB()  # type: ignore[assignment]

    # meta.pkl fallback copying removed - sampling must find meta.pkl in the expected location

    device_type, ptdtype, ctx = setup(rt.device, rt.dtype, rt.seed)

    # Data
    try:
        batches = SimpleBatches(exp.data, device=device_type)
    except Exception as e:
        raise DataError(f"Failed to initialize data batches: {e}") from e

    try:
        print(f"sampler: {exp.data.sampler}")
    except Exception:
        pass

    # Determine vocab size (from config or dataset meta)
    vocab_size = model_cfg.vocab_size
    if vocab_size is None and exp.data.meta_pkl is not None:
        try:
            vocab_size = (
                _load_meta_vocab_size(exp.data.dataset_dir / exp.data.meta_pkl) or 50304
            )
        except Exception as e:
            logger.warning(f"Failed to load vocab_size from meta.pkl: {e}")
            vocab_size = 50304
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
    resume_from: Path | None = None
    # Always try to resume from ckpt_last_filename if it exists
    resume_from = rt.out_dir / rt.ckpt_last_filename
    if not resume_from.exists():
        logger.info(f"[train] No checkpoint found at {resume_from}; starting fresh")
        resume_from = None

    # Init these up here so we can save checkpoints in the training loop
    ckpt_manager = CheckpointManager(
        rt.out_dir,
        atomic=rt.ckpt_atomic,
        keep_last=rt.checkpointing.keep.last,
        keep_best=rt.checkpointing.keep.best,
    )
    ema: _EMA | None = None

    if resume_from is not None:
        logger.info(f"[train] Resuming from checkpoint: {resume_from}")
        try:
            checkpoint = torch.load(
                resume_from, map_location=device_type, weights_only=False
            )
        except Exception as e:
            raise CheckpointError(
                f"Failed to load checkpoint from {resume_from}: {e}"
            ) from e

        # Validate checkpoint strictly
        required = {"model", "optimizer", "model_args", "iter_num", "best_val_loss", "config"}
        validate_checkpoint(checkpoint, required)

        # Resume training state
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
        tokens_seen = checkpoint.get("tokens_seen", 0)

        # Load model
        try:
            model = GPT(GPTConfig(**checkpoint["model_args"]))
            model.load_state_dict(checkpoint["model"])
        except Exception as e:
            raise ModelError(f"Failed to load model from checkpoint: {e}") from e

        # Load optimizer
        try:
            optimizer = model.configure_optimizers(
                weight_decay=exp.optim.weight_decay,
                learning_rate=exp.optim.learning_rate,
                betas=(exp.optim.beta1, exp.optim.beta2),
                device_type=device_type,
            )
            optimizer.load_state_dict(checkpoint["optimizer"])
        except Exception as e:
            raise ModelError(f"Failed to load optimizer from checkpoint: {e}") from e

        # Load EMA if present
        if "ema" in checkpoint:
            try:
                ema = _EMA(model, rt.ema_decay, device_type)
                ema.shadow = checkpoint["ema"]
            except Exception as e:
                logger.warning(f"Failed to load EMA from checkpoint: {e}")
    else:
        logger.info("[train] Initializing a new model from scratch")
        iter_num = 0
        best_val_loss = 1e9
        tokens_seen = 0
        model = GPT(GPTConfig(**model_args))
        optimizer = model.configure_optimizers(
            weight_decay=exp.optim.weight_decay,
            learning_rate=exp.optim.learning_rate,
            betas=(exp.optim.beta1, exp.optim.beta2),
            device_type=device_type,
        )
        if rt.ema_decay > 0:
            ema = _EMA(model, rt.ema_decay, device_type)

    # Compile model if enabled
    if rt.compile:
        logger.info("[train] Compiling model...")
        try:
            model = cast(GPT, torch.compile(model))  # ensure type remains GPT for type checkers
        except Exception as e:
            logger.warning(f"Failed to compile model: {e}")

    # Wrap model and optimizer in DDP if enabled
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        raise NotImplementedError("DDP training is not yet implemented")
        # TODO: Implement DDP training

    model.to(device_type)

    # Training loop
    logger.info("[train] Starting training loop...")
    X, Y = batches.get_batch("train")
    t0 = time.time()
    local_iter_num = 0
    running_mfu = -1.0

    # Predefine optional variables for type-checker safety
    loss: torch.Tensor | None = None
    raw_model: GPT | None = None
    original_weights: dict[str, torch.Tensor] | None = None

    # Training metrics tracking
    train_loss_tracker = 0.0
    train_loss_count = 0

    # EMA metrics tracking
    ema_val_loss_tracker = 0.0
    ema_val_loss_count = 0

    # Optional evaluation caches for safe later access
    losses: Dict[str, float] | None = None
    ema_losses: Dict[str, float] | None = None

    while True:
        # Determine and set the learning rate for this iteration
        lr = _get_lr(
            iter_num,
            warmup=exp.schedule.warmup_iters,
            decay_iters=exp.schedule.lr_decay_iters,
            min_lr=exp.schedule.min_lr,
            base_lr=exp.optim.learning_rate,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Evaluate the loss on train/val sets
        if iter_num % rt.eval_interval == 0:
            try:
                # model is GPT (may be compiled); safe to pass
                losses = _estimate_loss(model, batches, rt.eval_iters, ctx)
                logger.info(
                    f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                )

                # Update TensorBoard
                writer.add_scalar("loss/train", losses["train"], iter_num)
                writer.add_scalar("loss/val", losses["val"], iter_num)
                writer.add_scalar("lr", lr, iter_num)
                writer.add_scalar("mfu", running_mfu * 100, iter_num)

                # Track training metrics
                train_loss_tracker += losses["train"]
                train_loss_count += 1

                # Decide and save best/last per policy
                val_loss = losses["val"]
                checkpoint = Checkpoint(
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    model_args=model_args,
                    iter_num=iter_num,
                    best_val_loss=(val_loss if iter_num == 0 else best_val_loss),
                    config=exp.model_dump(),
                    ema=ema.shadow if ema is not None else None,
                )

                # Sidecar payload builder
                def _sidecar(kind: str, ema_used: bool = False) -> Dict[str, Any]:
                    return {
                        "kind": kind,
                        "iter_num": iter_num,
                        "tokens_seen": tokens_seen,
                        "lr": lr,
                        "eval_iters": rt.eval_iters,
                        "metric_name": "val_loss",
                        "greater_is_better": False,
                        "metric_raw": val_loss,
                        "smoothing_alpha": rt.best_smoothing_alpha,
                        "decision_metric": (val_loss if kind == "best" else best_val_loss),
                        "ema_used_for_saved_model": ema_used,
                        "eval_metric_on_ema": (ema_losses["val"] if (ema_losses is not None and "val" in ema_losses) else None),
                        "device": device_type,
                        "dtype": str(ptdtype),
                        "dataset_meta": getattr(batches, "meta", None) or {},
                        "progress": {
                            "train_loss_avg": train_loss_tracker / train_loss_count if train_loss_count > 0 else 0.0,
                            "ema_val_loss_avg": ema_val_loss_tracker / ema_val_loss_count if ema_val_loss_count > 0 else 0.0,
                        },
                    }

                if iter_num == 0:
                    # First checkpoint: save both best and last
                    best_val_loss = val_loss
                    best_path = ckpt_manager.save_checkpoint(
                        checkpoint,
                        rt.ckpt_best_filename,
                        best_val_loss,
                        iter_num,
                        logger,
                        is_best=True,
                    )
                    _write_sidecar(
                        sidecar_path=best_path.with_suffix(best_path.suffix + ".json"),
                        pt_path=best_path,
                        **_sidecar("best", ema_used=False),
                    )
                    last_path = ckpt_manager.save_checkpoint(
                        checkpoint,
                        rt.ckpt_last_filename,
                        best_val_loss,
                        iter_num,
                        logger,
                        is_best=False,
                    )
                    _write_sidecar(
                        sidecar_path=last_path.with_suffix(last_path.suffix + ".json"),
                        pt_path=last_path,
                        **_sidecar("last", ema_used=False),
                    )
                elif val_loss < best_val_loss:
                    # Improvement: update best only; do NOT save last in same iteration
                    best_val_loss = val_loss
                    best_path = ckpt_manager.save_checkpoint(
                        checkpoint,
                        rt.ckpt_best_filename,
                        best_val_loss,
                        iter_num,
                        logger,
                        is_best=True,
                    )
                    _write_sidecar(
                        sidecar_path=best_path.with_suffix(best_path.suffix + ".json"),
                        pt_path=best_path,
                        **_sidecar("best", ema_used=False),
                    )
                else:
                    # No improvement: update last
                    last_path = ckpt_manager.save_checkpoint(
                        checkpoint,
                        rt.ckpt_last_filename,
                        best_val_loss,
                        iter_num,
                        logger,
                        is_best=False,
                    )
                    _write_sidecar(
                        sidecar_path=last_path.with_suffix(last_path.suffix + ".json"),
                        pt_path=last_path,
                        **_sidecar("last", ema_used=False),
                    )
            except Exception as e:
                logger.error(f"Error during evaluation: {e}")
                raise

        # Evaluate with EMA model if enabled
        if rt.ema_decay > 0 and iter_num % 100 == 0 and iter_num >= 100:
            try:
                if ema is not None:
                    # Handle DDP model
                    if hasattr(model, "module"):
                        raw_model = cast(GPT, model.module)
                    else:
                        raw_model = cast(GPT, model)
                    
                    # Ensure raw_model is a GPT instance
                    if not isinstance(raw_model, GPT):
                        raise TypeError(f"Expected GPT model, got {type(raw_model)}")
                    
                    # Save original weights
                    original_weights = {
                        k: v.detach().clone() for k, v in raw_model.state_dict().items()
                    }
                    # Apply EMA weights
                    ema.apply_to(raw_model)
                    # Evaluate
                    ema_losses = _estimate_loss(raw_model, batches, rt.eval_iters, ctx)
                    logger.info(
                        f"step {iter_num}: EMA val loss {ema_losses['val']:.4f}"
                    )
                    # Restore original weights
                    raw_model.load_state_dict(original_weights)

                    # Update TensorBoard
                    writer.add_scalar("loss/ema_val", ema_losses["val"], iter_num)

                    # Track EMA metrics
                    ema_val_loss_tracker += ema_losses["val"]
                    ema_val_loss_count += 1
            except Exception as e:
                logger.error(f"Error during EMA evaluation: {e}")
                # Restore original weights even if evaluation fails
                if raw_model is not None and original_weights is not None:
                    try:
                        raw_model.load_state_dict(original_weights)
                    except Exception:
                        pass

        # Termination conditions
        if iter_num > rt.max_iters:
            break

        # Forward backward update, with optional gradient accumulation to simulate larger batch size
        try:
            # Handle DDP model
            if hasattr(model, "module"):
                raw_model = cast(GPT, model.module)
            else:
                raw_model = cast(GPT, model)
            
            # Ensure raw_model is a GPT instance
            if not isinstance(raw_model, GPT):
                raise TypeError(f"Expected GPT model, got {type(raw_model)}")
            
            for micro_step in range(exp.data.grad_accum_steps):
                if ddp:
                    # Not implemented yet
                    pass
                else:
                    logits, loss_t = raw_model(X, Y)
                    loss = loss_t / exp.data.grad_accum_steps
                X, Y = batches.get_batch("train")
                optimizer.zero_grad(set_to_none=True)
                if loss is None:
                    raise RuntimeError("Loss is None before backward; this indicates an unhandled branch in training loop.")
                loss.backward()
                optimizer.step()

                # Update EMA if enabled
                if ema is not None:
                    ema.update(raw_model)

                # Update tokens seen
                tokens_seen += X.numel()
        except Exception as e:
            logger.error(f"Error during training step: {e}")
            raise

        # Log metrics periodically
        if iter_num % rt.log_interval == 0 and local_iter_num >= 1:
            dt = time.time() - t0
            t0 = time.time()
            tokens_per_sec = tokens_seen / dt if dt > 0 else 0
            tokens_seen = 0
            if loss is not None:
                logger.info(
                    f"iter {iter_num}: loss {loss.item():.4f}, time {dt*1000:.2f}ms, tokens/sec {tokens_per_sec:.2f}"
                )
            if iter_num >= 100:
                mfu = raw_model.estimate_mfu(exp.data.batch_size * exp.data.grad_accum_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                logger.info(f"MFU: {running_mfu * 100:.2f}%")

            # Update TensorBoard
            if loss is not None:
                writer.add_scalar("train/loss", loss.item(), iter_num)
                writer.add_scalar("train/tokens_per_sec", tokens_per_sec, iter_num)
                writer.add_scalar("train/step_time_ms", dt * 1000, iter_num)

        # Time-based checkpointing not implemented; explicit policies above handle persistence.
        if rt.ckpt_time_interval_minutes > 0:
            pass

        iter_num += 1
        local_iter_num += 1

    # Finalize training
    writer.close()
    logger.info("[train] Training complete.")

    # Save final checkpoint
    try:
        checkpoint = Checkpoint(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            model_args=model_args,
            iter_num=iter_num,
            best_val_loss=best_val_loss,
            config=exp.model_dump(),
            ema=ema.shadow if ema is not None else None,
        )
        ckpt_manager.save_checkpoint(
            checkpoint,
            rt.ckpt_last_filename,
            best_val_loss,
            iter_num,
            logger,
            is_best=False,
        )
    except Exception as e:
        logger.error(f"Error saving final checkpoint: {e}")
        raise

    # Return final iteration number and best validation loss
    return iter_num, best_val_loss
