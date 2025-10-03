"""Primary training loop orchestration."""

from __future__ import annotations

import time
from contextlib import AbstractContextManager
from typing import Optional, Tuple, cast

import torch
from torch.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter

from ml_playground.configuration.models import TrainerConfig, SharedConfig
from ml_playground.ema import EMA
from ml_playground.models.core.model import GPT

from ml_playground.core.error_handling import CheckpointError
from ml_playground.training.checkpointing.service import (
    apply_checkpoint,
    create_manager,
    load_checkpoint,
    propagate_metadata,
    save_checkpoint,
)
from ml_playground.training.hooks.components import initialize_components
from ml_playground.training.hooks.data import initialize_batches
from ml_playground.training.hooks.evaluation import run_evaluation
from ml_playground.training.hooks.logging import log_training_step
from ml_playground.training.hooks.model import initialize_model
from ml_playground.training.hooks.runtime import RuntimeContext, setup_runtime
from ml_playground.training.loop.scheduler import get_lr


__all__ = ["Trainer", "train", "get_lr"]


class Trainer:
    """Coordinate the end-to-end training loop for a configured experiment."""

    def __init__(self, cfg: TrainerConfig, shared: SharedConfig):
        self.cfg = cfg
        self.shared = shared
        self.logger = cfg.logger

        self.runtime: RuntimeContext = setup_runtime(cfg)
        self.ctx: AbstractContextManager[object] = self.runtime.autocast_context
        self.device_type = self.runtime.device_type

        self.batches = initialize_batches(cfg, shared)
        self.model, self.optimizer = initialize_model(cfg, self.logger)

        self.out_dir = shared.train_out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_mgr = create_manager(cfg, shared)

        self.model, self.scaler, self.ema, self.writer = initialize_components(
            self.model,
            cfg,
            self.runtime,
            log_dir=str(self.out_dir),
        )

        self.iter_num = 0
        self.best_val_loss = 1e9

        checkpoint = load_checkpoint(self.ckpt_mgr, cfg, logger=self.logger)
        if checkpoint:
            self.iter_num, self.best_val_loss = apply_checkpoint(
                checkpoint,
                model=cast(GPT, getattr(self.model, "_orig_mod", self.model)),
                optimizer=self.optimizer,
                ema=self.ema,
            )

    @property
    def scaler(self) -> GradScaler:
        return self._scaler

    @scaler.setter
    def scaler(self, value: GradScaler) -> None:
        self._scaler = value

    @property
    def ema(self) -> Optional[EMA]:
        return self._ema

    @ema.setter
    def ema(self, value: Optional[EMA]) -> None:
        self._ema = value

    @property
    def writer(self) -> Optional[SummaryWriter]:
        return self._writer

    @writer.setter
    def writer(self, value: Optional[SummaryWriter]) -> None:
        self._writer = value

    def run(self) -> Tuple[int, float]:
        """Execute the main training loop until reaching the maximum iteration count."""
        self.logger.info("Starting training loop")
        X, Y = self.batches.get_batch("train")
        t0 = time.time()
        local_iter_num = 0
        raw_model = cast(GPT, getattr(self.model, "_orig_mod", self.model))
        running_mfu = -1.0

        should_save_checkpoint = True

        try:
            while True:
                lr = get_lr(self.iter_num, self.cfg.schedule, self.cfg.optim)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

                if (
                    self.iter_num % self.cfg.runtime.eval_interval == 0
                    and self.cfg.runtime.eval_iters > 0
                ):
                    losses = run_evaluation(
                        self.cfg,
                        logger=self.logger,
                        iter_num=self.iter_num,
                        lr=lr,
                        raw_model=raw_model,
                        batches=self.batches,
                        ctx=self.ctx,
                        writer=self.writer,
                    )
                    if losses["val"] < self.best_val_loss:
                        self.best_val_loss = losses["val"]
                        if self.iter_num > 0:
                            save_checkpoint(
                                self.ckpt_mgr,
                                self.cfg,
                                model=raw_model,
                                optimizer=self.optimizer,
                                ema=self.ema,
                                iter_num=self.iter_num,
                                best_val_loss=self.best_val_loss,
                                logger=self.logger,
                                is_best=True,
                            )

                    if self.iter_num == 0 and self.cfg.runtime.eval_only:
                        break

                loss = self._train_step(X, Y)
                X, Y = self.batches.get_batch("train")

                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                if self.iter_num % self.cfg.runtime.log_interval == 0:
                    running_mfu = log_training_step(
                        self.logger,
                        iter_num=self.iter_num,
                        loss_value=loss.item(),
                        dt=dt,
                        local_iter_num=local_iter_num,
                        raw_model=raw_model,
                        running_mfu=running_mfu,
                        batch_size=self.cfg.data.batch_size,
                        grad_accum_steps=self.cfg.data.grad_accum_steps,
                    )
                    # TensorBoard logging if update mode is 'log'
                    try:
                        if (
                            self.writer
                            and getattr(
                                self.cfg.runtime, "tensorboard_update_mode", "eval"
                            )
                            == "log"
                        ):
                            scaled_loss = loss.item() * self.cfg.data.grad_accum_steps
                            self.writer.add_scalar(
                                "Loss/train", scaled_loss, self.iter_num
                            )
                            self.writer.add_scalar("LR", lr, self.iter_num)
                    except (ValueError, RuntimeError, OSError) as exc:
                        self.logger.debug(
                            "TensorBoard logging skipped due to writer error: %s", exc
                        )

                self.iter_num += 1
                local_iter_num += 1

                if self.iter_num > self.cfg.runtime.max_iters:
                    break

        except KeyboardInterrupt:
            should_save_checkpoint = False
            self.logger.info(
                "Training loop interrupted; skipping final checkpoint save"
            )
            raise
        except BaseException:
            should_save_checkpoint = False
            raise
        finally:
            try:
                if should_save_checkpoint:
                    save_checkpoint(
                        self.ckpt_mgr,
                        self.cfg,
                        model=raw_model,
                        optimizer=self.optimizer,
                        ema=self.ema,
                        iter_num=self.iter_num,
                        best_val_loss=self.best_val_loss,
                        logger=self.logger,
                        is_best=False,
                    )
            except (CheckpointError, RuntimeError, OSError) as exc:
                self.logger.warning(f"Failed to save final checkpoint: {exc}")

            try:
                propagate_metadata(self.cfg, self.shared, logger=self.logger)
            except (OSError, RuntimeError) as exc:
                self.logger.warning(f"Failed to propagate meta file: {exc}")

            if self.writer:
                self.writer.close()

        return self.iter_num, self.best_val_loss

    def _train_step(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Perform a gradient accumulation step and update EMA if configured."""
        loss_tensor = torch.tensor(0.0, device=X.device)
        for _ in range(self.cfg.data.grad_accum_steps):
            with self.ctx:
                logits, loss_tensor = self.model(X, Y)
                loss_tensor = loss_tensor / self.cfg.data.grad_accum_steps
            self.scaler.scale(loss_tensor).backward()

        if self.cfg.optim.grad_clip != 0.0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.optim.grad_clip
            )

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        if self.ema:
            self.ema.update(self.model)
        return loss_tensor


def train(cfg: TrainerConfig, shared: SharedConfig | None = None) -> Tuple[int, float]:
    """Run the strict trainer with optional shared metadata fallback."""
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
