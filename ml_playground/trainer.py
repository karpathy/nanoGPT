from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from typing import cast

import torch
from torch import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter

from ml_playground.checkpoint import Checkpoint, CheckpointManager
from ml_playground.config import (
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


class Trainer:
    def __init__(self, cfg: TrainerConfig, shared: SharedConfig):
        """Initialize the trainer."""
        self.cfg = cfg
        self.shared = shared
        self.runtime_cfg = cfg.runtime
        self.model_cfg = cfg.model
        self.data_cfg = cfg.data
        self.optim_cfg = cfg.optim
        self.schedule_cfg = cfg.schedule

        self.out_dir = shared.train_out_dir
        self.logger = logging.getLogger(__name__)
        setup_logging(str(self.out_dir))

        self._setup_torch_env()

        self.batches = self._setup_data_loader()
        self.model, self.optimizer = self._setup_model()

        self.iter_num = 0
        self.best_val_loss = 1e9

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_mgr = self._setup_checkpoint_manager()

        self._load_checkpoint()
        self._setup_components()

    def _setup_torch_env(self):
        torch.manual_seed(self.runtime_cfg.seed)
        torch.cuda.manual_seed(self.runtime_cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.device_type = "cuda" if "cuda" in self.runtime_cfg.device else "cpu"
        pt_dtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[self.runtime_cfg.dtype]
        self.ctx = (
            nullcontext()
            if self.device_type == "cpu"
            else autocast(device_type=self.device_type, dtype=pt_dtype)
        )

    def _setup_data_loader(self) -> SimpleBatches:
        return SimpleBatches(
            data=self.data_cfg,
            device=self.runtime_cfg.device,
            dataset_dir=self.shared.dataset_dir,
        )

    def _setup_model(self) -> tuple[GPT, torch.optim.Optimizer]:
        self.logger.info("Initializing model and optimizer")
        model = GPT(self.model_cfg)
        model.to(self.runtime_cfg.device)
        optimizer = model.configure_optimizers(
            self.optim_cfg.weight_decay,
            self.optim_cfg.learning_rate,
            (self.optim_cfg.beta1, self.optim_cfg.beta2),
            self.runtime_cfg.device,
        )
        return model, optimizer

    def _setup_checkpoint_manager(self) -> CheckpointManager:
        return CheckpointManager(
            out_dir=self.out_dir,
            atomic=self.runtime_cfg.ckpt_atomic,
            keep_last=self.runtime_cfg.checkpointing.keep.last,
            keep_best=self.runtime_cfg.checkpointing.keep.best,
        )

    def _setup_components(self):
        if self.runtime_cfg.compile:
            self.logger.info("Compiling the model... (takes a ~minute)")
            self.model = cast(GPT, torch.compile(self.model))

        self.scaler = GradScaler(
            enabled=(self.device_type == "cuda" and self.runtime_cfg.dtype == "float16")
        )

        self.ema: EMA | None = None
        if self.runtime_cfg.ema_decay > 0.0:
            self.ema = EMA(
                self.model, self.runtime_cfg.ema_decay, self.runtime_cfg.device
            )

        self.writer: SummaryWriter | None = None
        if self.runtime_cfg.tensorboard_enabled:
            self.writer = SummaryWriter(log_dir=str(self.out_dir))

    def _load_checkpoint(self):
        checkpoint: Checkpoint | None = None
        if self.out_dir.exists():
            try:
                if self.runtime_cfg.checkpointing.read_policy == READ_POLICY_BEST:
                    checkpoint = self.ckpt_mgr.load_best_checkpoint(
                        device=self.runtime_cfg.device, logger=self.logger
                    )
                else:
                    checkpoint = self.ckpt_mgr.load_latest_checkpoint(
                        device=self.runtime_cfg.device, logger=self.logger
                    )
            except CheckpointError as e:
                self.logger.warning(
                    f"Could not load checkpoint ({self.runtime_cfg.checkpointing.read_policy}): {e}"
                )

        if checkpoint:
            self.logger.info("Resuming training from latest checkpoint")
            self.model.load_state_dict(checkpoint.model, strict=False)
            self.optimizer.load_state_dict(checkpoint.optimizer)
            self.iter_num = checkpoint.iter_num
            self.best_val_loss = checkpoint.best_val_loss
            if self.ema and checkpoint.ema:
                self.ema.shadow = checkpoint.ema

    def run(self) -> tuple[int, float]:
        """Main training loop."""
        self.logger.info("Starting training loop")
        X, Y = self.batches.get_batch("train")
        t0 = time.time()
        local_iter_num = 0
        # Access original module when compiled; cast to GPT for static typing
        raw_model = cast(GPT, getattr(self.model, "_orig_mod", self.model))
        running_mfu = -1.0

        while True:
            lr = get_lr(self.iter_num, self.schedule_cfg, self.optim_cfg)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            if (
                self.iter_num % self.runtime_cfg.eval_interval == 0
                and self.runtime_cfg.eval_iters > 0
            ):
                self._evaluate(lr, raw_model)

            if self.iter_num == 0 and self.runtime_cfg.eval_only:
                break

            loss = self._train_step(X, Y)
            X, Y = self.batches.get_batch("train")

            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if self.iter_num % self.runtime_cfg.log_interval == 0:
                running_mfu = self._log_step(
                    loss, dt, local_iter_num, raw_model, running_mfu
                )

            self.iter_num += 1
            local_iter_num += 1

            if self.iter_num > self.runtime_cfg.max_iters:
                break

        self._save_checkpoint(raw_model, is_best=False)
        self._propagate_meta()

        if self.writer:
            self.writer.close()

        return self.iter_num, self.best_val_loss

    def _log_step(
        self,
        loss: torch.Tensor,
        dt: float,
        local_iter_num: int,
        raw_model: GPT,
        running_mfu: float,
    ) -> float:
        lossf = loss.item() * self.data_cfg.grad_accum_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(
                self.data_cfg.batch_size * self.data_cfg.grad_accum_steps, dt
            )
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu

        mfu_pct = max(0.0, min(float(running_mfu), 100.0))
        self.logger.info(
            f"iter {self.iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {mfu_pct:.2f}%"
        )
        return running_mfu

    def _evaluate(self, lr: float, raw_model: GPT):
        # Use raw_model to satisfy type checker regardless of compile() wrapping
        losses = estimate_loss(
            raw_model, self.batches, self.runtime_cfg.eval_iters, self.ctx
        )
        self.logger.info(
            f"step {self.iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )
        if self.writer:
            self.writer.add_scalar("Loss/train", losses["train"], self.iter_num)
            self.writer.add_scalar("Loss/val", losses["val"], self.iter_num)
            self.writer.add_scalar("LR", lr, self.iter_num)

        if losses["val"] < self.best_val_loss:
            self.best_val_loss = losses["val"]
            if self.iter_num > 0:
                self._save_checkpoint(raw_model, is_best=True)

    def _train_step(self, X, Y):
        loss_tensor = torch.tensor(0.0)
        for micro_step in range(self.data_cfg.grad_accum_steps):
            with self.ctx:
                logits, loss_tensor = self.model(X, Y)
                loss_tensor = loss_tensor / self.data_cfg.grad_accum_steps
            self.scaler.scale(loss_tensor).backward()

        if self.optim_cfg.grad_clip != 0.0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.optim_cfg.grad_clip
            )

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        if self.ema:
            self.ema.update(self.model)
        return loss_tensor

    def _save_checkpoint(self, raw_model, is_best: bool):
        checkpoint = Checkpoint(
            model=raw_model.state_dict(),
            optimizer=self.optimizer.state_dict(),
            model_args=self.model_cfg.model_dump(),
            iter_num=self.iter_num,
            best_val_loss=self.best_val_loss,
            config=self.cfg.model_dump(),
            ema=self.ema.shadow if self.ema else None,
        )
        base_filename = "ckpt_best.pt" if is_best else "ckpt_last.pt"
        self.ckpt_mgr.save_checkpoint(
            checkpoint,
            base_filename=base_filename,
            metric=self.best_val_loss,
            iter_num=self.iter_num,
            logger=self.logger,
            is_best=is_best,
        )

    def _propagate_meta(self):
        try:
            meta_src = self.data_cfg.meta_path(self.shared.dataset_dir)
            if meta_src and meta_src.exists():
                meta_dst = self.out_dir / meta_src.name
                import shutil

                shutil.copy2(meta_src, meta_dst)
        except Exception:
            pass


def train(cfg: TrainerConfig, shared: SharedConfig | None = None) -> tuple[int, float]:
    """Main training loop."""
    # Backward-compat: synthesize a minimal SharedConfig when not provided
    if shared is None:
        out_dir = cfg.runtime.out_dir
        shared = SharedConfig(
            experiment="unknown",
            config_path=out_dir / "cfg.toml",
            project_home=out_dir.parent if out_dir.parent else out_dir,
            dataset_dir=out_dir,
            train_out_dir=out_dir,
            sample_out_dir=out_dir,
        )
    trainer = Trainer(cfg, shared)
    return trainer.run()


# Explicit public API for this module
__all__ = ["get_lr", "Trainer", "train"]
