from __future__ import annotations
import math
import time
import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Tuple, List, cast, Protocol, Optional
import pickle
import shutil
import torch
from dataclasses import dataclass
from ml_playground.model import GPTConfig, GPT
from ml_playground.config import TrainerConfig
from ml_playground.device import setup
from ml_playground.data import SimpleBatches
from ml_playground.error_handling import DataError, ModelError, CheckpointError, setup_logging
import logging

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
                self.shadow[name].mul_(self.decay).add_(param.detach().to(self.shadow[name].device), alpha=1.0 - self.decay)

    def apply_to(self, model: GPT) -> None:
        state_dict = model.state_dict()
        for name, param in state_dict.items():
            if name in self.shadow:
                state_dict[name].copy_(self.shadow[name])


def _estimate_loss(
    model: GPT, batches: SimpleBatches, eval_iters: int, ctx
) -> Dict[str, float]:
    """Estimate loss on train/val splits."""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = batches.get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
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


class CheckpointManager:
    """A utility class for managing checkpoints with advanced features."""
    
    def __init__(self, out_dir: Path, atomic: bool = True, top_k: int = 5):
        self.out_dir = out_dir
        self.atomic = atomic
        self.top_k = top_k
        self.checkpoints: list[_CkptInfo] = []
    
    def save_checkpoint(
        self, 
        checkpoint: dict, 
        filename: str, 
        metric: float, 
        iter_num: int,
        logger: Optional[logging.Logger] = None
    ) -> None:
        """Save a checkpoint with metadata and manage top-k checkpoints."""
        path = self.out_dir / filename
        
        # Save the checkpoint
        _atomic_save(checkpoint, path, self.atomic)
        
        # Add to checkpoint list
        ckpt_info = _CkptInfo(path, metric, iter_num, time.time())
        self.checkpoints.append(ckpt_info)
        
        # Sort by metric (assuming lower is better by default)
        self.checkpoints.sort(key=lambda x: x.metric)
        
        # Keep only top-k checkpoints
        if len(self.checkpoints) > self.top_k and self.top_k > 0:
            # Remove worst checkpoints
            to_remove = self.checkpoints[self.top_k:]
            self.checkpoints = self.checkpoints[:self.top_k]
            
            # Delete the files
            for ckpt in to_remove:
                try:
                    ckpt.path.unlink()
                    # Also remove sidecar file if it exists
                    sidecar = ckpt.path.with_suffix(ckpt.path.suffix + '.json')
                    if sidecar.exists():
                        sidecar.unlink()
                    if logger:
                        logger.info(f"Removed old checkpoint: {ckpt.path}")
                except Exception as e:
                    if logger:
                        logger.warning(f"Failed to remove old checkpoint {ckpt.path}: {e}")
        
        if logger:
            logger.info(f"Saved checkpoint to {path}")
    
    def load_latest_checkpoint(self, device: str, logger: Optional[logging.Logger] = None) -> Optional[dict]:
        """Load the latest checkpoint based on creation time."""
        candidates = [
            self.out_dir / "ckpt_best.pt",
            self.out_dir / "ckpt_last.pt",
        ]
        
        for path in candidates:
            if path.exists():
                try:
                    ckpt = torch.load(path, map_location=device)
                    if logger:
                        logger.info(f"Loaded checkpoint from {path}")
                    return ckpt
                except Exception as e:
                    if logger:
                        logger.warning(f"Failed to load checkpoint from {path}: {e}")
        
        if logger:
            logger.info("No existing checkpoints found")
        return None
    
    def get_checkpoint_info(self) -> list[_CkptInfo]:
        """Get information about all managed checkpoints."""
        return self.checkpoints.copy()


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
    ckpt_manager = CheckpointManager(rt.out_dir, atomic=rt.ckpt_atomic, top_k=rt.top_k_checkpoints)
    ema: _EMA | None = None

    if resume_from is not None:
        logger.info(f"[train] Resuming from checkpoint: {resume_from}")
        try:
            checkpoint = torch.load(resume_from, map_location=device_type)
        except Exception as e:
            raise CheckpointError(f"Failed to load checkpoint from {resume_from}: {e}") from e
            
        # Resume training state
        iter_num = checkpoint.get("iter_num", 0)
        best_val_loss = checkpoint.get("best_val_loss", 1e9)
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
            model = torch.compile(model)  # type: ignore
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
    
    # Training metrics tracking
    train_loss_tracker = 0.0
    train_loss_count = 0
    
    # EMA metrics tracking
    ema_val_loss_tracker = 0.0
    ema_val_loss_count = 0
    
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
                
                # Update best validation loss
                if losses["val"] < best_val_loss:
                    best_val_loss = losses["val"]
                    if iter_num > 0:
                        checkpoint = {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "model_args": model_args,
                            "iter_num": iter_num,
                            "best_val_loss": best_val_loss,
                            "tokens_seen": tokens_seen,
                            "config": exp.model_dump(),
                        }
                        if ema is not None:
                            checkpoint["ema"] = ema.shadow
                        ckpt_manager.save_checkpoint(checkpoint, rt.ckpt_best_filename, best_val_loss, iter_num, logger)
            except Exception as e:
                logger.error(f"Error during evaluation: {e}")
                raise

        # Evaluate with EMA model if enabled
        if rt.ema_decay > 0 and iter_num % 100 == 0 and iter_num >= 100:
            try:
                if ema is not None:
                    # Save original weights
                    original_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    # Apply EMA weights
                    ema.apply_to(model)
                    # Evaluate
                    ema_losses = _estimate_loss(model, batches, rt.eval_iters, ctx)
                    logger.info(f"step {iter_num}: EMA val loss {ema_losses['val']:.4f}")
                    # Restore original weights
                    model.load_state_dict(original_weights)
                    
                    # Update TensorBoard
                    writer.add_scalar("loss/ema_val", ema_losses["val"], iter_num)
                    
                    # Track EMA metrics
                    ema_val_loss_tracker += ema_losses["val"]
                    ema_val_loss_count += 1
            except Exception as e:
                logger.error(f"Error during EMA evaluation: {e}")
                # Restore original weights even if evaluation fails
                if 'original_weights' in locals():
                    try:
                        model.load_state_dict(original_weights)
                    except Exception:
                        pass

        # Termination conditions
        if iter_num > rt.max_iters:
            break

        # Forward backward update, with optional gradient accumulation to simulate larger batch size
        try:
            for micro_step in range(exp.data.grad_accum_steps):
                if ddp:
                    # Not implemented yet
                    pass
                else:
                    logits, loss = model(X, Y)
                    loss = loss / exp.data.grad_accum_steps
                X, Y = batches.get_batch("train")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                
                # Update EMA if enabled
                if ema is not None:
                    ema.update(model)
                
                # Update tokens seen
                tokens_seen += X.numel()
        except Exception as e:
            logger.error(f"Error during training step: {e}")
            raise

        # Timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % rt.log_interval == 0:
            # Get loss as float. note: this is a CPU-GPU sync point
            lossf = loss.item() * exp.data.grad_accum_steps
            if local_iter_num >= 5:  # let the training loop settle a bit
                mfu = model.estimate_mfu(exp.data.batch_size * exp.data.grad_accum_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            logger.info(
                f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
            )
            
            # Update TensorBoard
            writer.add_scalar("loss/step", lossf, iter_num)
            writer.add_scalar("time/step", dt * 1000, iter_num)

        # Save checkpoint periodically based on time interval
        if rt.ckpt_time_interval_minutes > 0:
            # Save checkpoint every ckpt_time_interval_minutes
            pass
        elif iter_num % 1000 == 0 and iter_num > 0:  # Default fallback: save every 1000 iterations
            try:
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "tokens_seen": tokens_seen,
                    "config": exp.model_dump(),
                }
                if ema is not None:
                    checkpoint["ema"] = ema.shadow
                ckpt_manager.save_checkpoint(checkpoint, rt.ckpt_last_filename, best_val_loss, iter_num, logger)
                
                # Write sidecar with metadata
                sidecar_path = rt.out_dir / (rt.ckpt_last_filename + ".json")
                _write_sidecar(
                    sidecar_path=sidecar_path,
                    pt_path=rt.out_dir / rt.ckpt_last_filename,
                    kind="last",
                    iter_num=iter_num,
                    tokens_seen=tokens_seen,
                    lr=lr,
                    eval_iters=rt.eval_iters,
                    metric_name="val_loss",
                    greater_is_better=False,
                    metric_raw=losses.get("val", 0.0),
                    smoothing_alpha=rt.best_smoothing_alpha,
                    decision_metric=best_val_loss,
                    ema_used_for_saved_model=False,
                    eval_metric_on_ema=ema_losses.get("val", None) if 'ema_losses' in locals() else None,
                    device=device_type,
                    dtype=str(ptdtype),
                    dataset_meta=getattr(batches, "meta", None),
                    progress={
                        "train_loss_avg": train_loss_tracker / train_loss_count if train_loss_count > 0 else 0.0,
                        "ema_val_loss_avg": ema_val_loss_tracker / ema_val_loss_count if ema_val_loss_count > 0 else 0.0,
                    } if train_loss_count > 0 or ema_val_loss_count > 0 else None,
                )
            except Exception as e:
                logger.error(f"Error saving checkpoint: {e}")
                raise

        iter_num += 1
        local_iter_num += 1

    # Finalize training
    writer.close()
    logger.info("[train] Training complete.")
    
    # Return final iteration number and best validation loss
    return iter_num, best_val_loss
