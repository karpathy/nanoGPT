from __future__ import annotations

from pathlib import Path

import pytest

from ml_playground.config import (
    DataConfig,
    LRSchedule,
    ModelConfig,
    OptimConfig,
    RuntimeConfig,
    TrainerConfig,
)
from ml_playground.training.hooks import evaluation


class _Logger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, msg: str) -> None:
        self.messages.append(msg)


class _Writer:
    def __init__(self) -> None:
        self.entries: list[tuple[str, float, int]] = []

    def add_scalar(self, name: str, value: float, step: int) -> None:
        self.entries.append((name, value, step))


def _cfg() -> TrainerConfig:
    return TrainerConfig(
        model=ModelConfig(n_layer=1, n_head=1, n_embd=4, block_size=4, dropout=0.0),
        data=DataConfig(batch_size=2, block_size=4, grad_accum_steps=1),
        optim=OptimConfig(learning_rate=0.01),
        schedule=LRSchedule(
            decay_lr=False, warmup_iters=0, lr_decay_iters=0, min_lr=0.0
        ),
        runtime=RuntimeConfig(
            out_dir=Path("."),
            max_iters=1,
            eval_interval=1,
            eval_iters=1,
            log_interval=1,
            eval_only=False,
            seed=1,
            device="cpu",
            dtype="float32",
            compile=False,
            tensorboard_enabled=True,
            ema_decay=0.0,
        ),
        hf_model=TrainerConfig.HFModelConfig(
            model_name="hf/model",
            gradient_checkpointing=False,
            block_size=128,
        ),
        peft=TrainerConfig.PeftConfig(enabled=False),
    )


def test_run_evaluation_records_scalars(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg()
    logger = _Logger()
    monkeypatch.setattr(
        evaluation,
        "estimate_loss",
        lambda model, batches, eval_iters, ctx: {"train": 0.5, "val": 0.4},
    )

    writer = _Writer()
    losses = evaluation.run_evaluation(
        cfg,
        logger=logger,
        iter_num=1,
        lr=0.01,
        raw_model=None,
        batches=None,
        ctx=None,
        writer=writer,
    )

    assert losses == {"train": 0.5, "val": 0.4}
    assert any("train loss" in msg for msg in logger.messages)
    assert ("Loss/train", 0.5, 1) in writer.entries
    assert ("Loss/val", 0.4, 1) in writer.entries
    assert ("LR", 0.01, 1) in writer.entries
