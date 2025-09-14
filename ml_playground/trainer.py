from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter

from ml_playground.checkpoint import Checkpoint, CheckpointManager
from ml_playground.config import (
    DataConfig,
    ModelConfig,
    OptimConfig,
    RuntimeConfig,
    TrainerConfig,
    LRSchedule,
    READ_POLICY_BEST,
    SharedConfig,
)
from ml_playground.data import SimpleBatches
from ml_playground.ema import EMA
from ml_playground.error_handling import CheckpointError, setup_logging
from ml_playground.estimator import estimate_loss
from ml_playground import lr_scheduler
from ml_playground.model import GPT


def _setup_data_loader(
    data_cfg: DataConfig, runtime_cfg: RuntimeConfig, dataset_dir: Path
) -> SimpleBatches:
    """Initialize data loader."""
    return SimpleBatches(data=data_cfg, device=runtime_cfg.device, dataset_dir=dataset_dir)


def get_lr(it: int, schedule: LRSchedule, optim: OptimConfig) -> float:
    """Wrapper for learning rate according to schedule and optimizer.

    - If schedule.decay_lr is False, return constant learning rate.
    - Otherwise delegate to cosine scheduler with warmup.
    """
    if not schedule.decay_lr:
        return optim.learning_rate
    return lr_scheduler.get_lr(
        it,
        warmup=schedule.warmup_iters,
        decay_iters=schedule.lr_decay_iters,
        min_lr=schedule.min_lr,
        base_lr=optim.learning_rate,
    )


def _setup_model(
    model_cfg: ModelConfig,
    runtime_cfg: RuntimeConfig,
    optim_cfg: OptimConfig,
    logger: logging.Logger,
) -> tuple[GPT, torch.optim.Optimizer]:
    """Initialize model and optimizer."""
    logger.info("Initializing model")
    model = GPT(model_cfg)
    model.to(runtime_cfg.device)

    logger.info("Initializing optimizer")
    optimizer = model.configure_optimizers(
        optim_cfg.weight_decay,
        optim_cfg.learning_rate,
        (optim_cfg.beta1, optim_cfg.beta2),
        runtime_cfg.device,
    )

    return model, optimizer


def train(cfg: TrainerConfig, shared: SharedConfig) -> tuple[int, float]:
    """Main training loop."""
    # --- Setup -------------------------------------------------------------------
    runtime_cfg = cfg.runtime
    model_cfg = cfg.model
    data_cfg = cfg.data
    optim_cfg = cfg.optim
    schedule_cfg = cfg.schedule

    # Use shared-config paths for all filesystem locations (no fallback)
    out_dir = shared.train_out_dir
    setup_logging(str(out_dir))
    logger = logging.getLogger(__name__)

    # --- Set random seeds -------------------------------------------------------
    torch.manual_seed(runtime_cfg.seed)
    torch.cuda.manual_seed(runtime_cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    # --- Device setup -----------------------------------------------------------
    device_type = "cuda" if "cuda" in runtime_cfg.device else "cpu"  # for later use
    # note: float16 data type will automatically use a GradScaler
    pt_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[runtime_cfg.dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else autocast(device_type=device_type, dtype=pt_dtype)
    )

    # --- Data loader ------------------------------------------------------------
    batches = _setup_data_loader(data_cfg, runtime_cfg, shared.dataset_dir)

    # --- Model and optimizer ----------------------------------------------------
    model, optimizer = _setup_model(model_cfg, runtime_cfg, optim_cfg, logger)

    # --- Checkpoint loading -----------------------------------------------------
    iter_num = 0
    best_val_loss = 1e9
    checkpoint: Checkpoint | None = None

    # --- Checkpoint manager setup -----------------------------------------------
    # Ensure output directory exists before any checkpointing/logging
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    ckpt_mgr = CheckpointManager(
        out_dir=out_dir,
        atomic=runtime_cfg.ckpt_atomic,
        keep_last=runtime_cfg.checkpointing.keep.last,
        keep_best=runtime_cfg.checkpointing.keep.best,
    )

    if out_dir.exists():
        try:
            if runtime_cfg.checkpointing.read_policy == READ_POLICY_BEST:
                checkpoint = ckpt_mgr.load_best_checkpoint(
                    device=runtime_cfg.device, logger=logger
                )
            else:
                checkpoint = ckpt_mgr.load_latest_checkpoint(
                    device=runtime_cfg.device, logger=logger
                )
        except CheckpointError as e:
            logger.warning(
                f"Could not load checkpoint ({runtime_cfg.checkpointing.read_policy}): {e}"
            )

    if checkpoint:
        logger.info("Resuming training from latest checkpoint")
        model.load_state_dict(checkpoint.model, strict=False)
        optimizer.load_state_dict(checkpoint.optimizer)
        iter_num = checkpoint.iter_num
        best_val_loss = checkpoint.best_val_loss

    # --- Compile model ----------------------------------------------------------
    if runtime_cfg.compile:
        logger.info("Compiling the model... (takes a ~minute)")
        model = torch.compile(model)  # type: ignore

    # --- GradScaler setup -------------------------------------------------------
    # Use CUDA GradScaler via torch.amp; enable only when on CUDA and using float16.
    scaler = GradScaler(
        enabled=(device_type == "cuda" and runtime_cfg.dtype == "float16")
    )

    # --- EMA (Exponential Moving Average) setup ------------------------------
    ema: EMA | None = None
    if runtime_cfg.ema_decay > 0.0:
        ema = EMA(model, runtime_cfg.ema_decay, runtime_cfg.device)  # type: ignore
        if checkpoint and checkpoint.ema:
            ema.shadow = checkpoint.ema

    # --- TensorBoard logging -------------------------------------------------
    writer: SummaryWriter | None = None
    if runtime_cfg.tensorboard_enabled:
        writer = SummaryWriter(log_dir=str(out_dir))

    # --- Training loop ----------------------------------------------------------
    logger.info("Starting training loop")
    X, Y = batches.get_batch("train")  # fetch the first batch
    t0 = time.time()
    local_iter_num = 0  # number of iterations in the lifetime of this process
    raw_model = model._orig_mod if runtime_cfg.compile else model  # type: ignore
    running_mfu = -1.0

    while True:
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num, schedule_cfg, optim_cfg)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % runtime_cfg.eval_interval == 0 and runtime_cfg.eval_iters > 0:
            losses = estimate_loss(model, batches, runtime_cfg.eval_iters, ctx)  # type: ignore
            logger.info(
                f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            if writer:
                writer.add_scalar("Loss/train", losses["train"], iter_num)
                writer.add_scalar("Loss/val", losses["val"], iter_num)
                writer.add_scalar("LR", lr, iter_num)

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    ckpt_mgr.save_checkpoint(
                        Checkpoint(
                            model=raw_model.state_dict(),  # type: ignore
                            optimizer=optimizer.state_dict(),
                            model_args=model_cfg.model_dump(),
                            iter_num=iter_num,
                            best_val_loss=best_val_loss,
                            config=cfg.model_dump(),
                            ema=ema.shadow if ema else None,
                        ),
                        # Base filename is unused in strict rotated mode; preserved for signature compatibility
                        base_filename="ckpt_best.pt",
                        metric=best_val_loss,
                        iter_num=iter_num,
                        logger=logger,
                        is_best=True,
                    )

        if iter_num == 0 and runtime_cfg.eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        loss = 0.0
        for micro_step in range(data_cfg.grad_accum_steps):
            with ctx:
                logits, loss = model(X, Y)
                # scale the loss to account for gradient accumulation
                loss = loss / data_cfg.grad_accum_steps
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = batches.get_batch("train")
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()  # type: ignore

        # clip the gradient
        if optim_cfg.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), optim_cfg.grad_clip)  # type: ignore

        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()

        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # ema update
        if ema:
            ema.update(model)  # type: ignore

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % runtime_cfg.log_interval == 0:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * data_cfg.grad_accum_steps  # type: ignore
            if local_iter_num >= 5:  # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(  # type: ignore
                    data_cfg.batch_size * data_cfg.grad_accum_steps, dt
                )
                running_mfu = (
                    mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                )
            # Clamp MFU into [0, 100] for display and format as percentage
            mfu_pct = max(0.0, min(float(running_mfu), 100.0))
            logger.info(
                f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {mfu_pct:.2f}%"
            )
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > runtime_cfg.max_iters:
            break

    # --- Save final checkpoint --------------------------------------------------
    ckpt_mgr.save_checkpoint(
        Checkpoint(
            model=raw_model.state_dict(),  # type: ignore
            optimizer=optimizer.state_dict(),
            model_args=model_cfg.model_dump(),
            iter_num=iter_num,
            best_val_loss=best_val_loss,
            config=cfg.model_dump(),
            ema=ema.shadow if ema else None,
        ),
        # Base filename is unused in strict rotated mode; preserved for signature compatibility
        base_filename="ckpt_last.pt",
        metric=best_val_loss,
        iter_num=iter_num,
        logger=logger,
        is_best=False,
    )

    # Propagate dataset metadata to out_dir for downstream sampling utilities
    try:
        meta_src = data_cfg.dataset_dir / data_cfg.meta_pkl  # filenames from DataConfig
        if meta_src and meta_src.exists():
            # copy to out_dir preserving filename
            meta_dst = out_dir / meta_src.name
            # best-effort copy; avoid crashing training if fails
            import shutil

            shutil.copy2(meta_src, meta_dst)
    except Exception:
        pass

    if writer:
        writer.close()

    return iter_num, best_val_loss


# Explicit public API for this module
__all__ = [
    "get_lr",
    "train",
]
