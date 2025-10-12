from __future__ import annotations

import logging
from pathlib import Path

import torch

from ml_playground.configuration.models import (
    DataConfig,
    LRSchedule,
    ModelConfig,
    OptimConfig,
    RuntimeConfig,
    TrainerConfig,
)
from ml_playground.models.core.model import GPT
from ml_playground.training.hooks.model import initialize_model


def test_initialize_model_creates_gpt_and_optimizer(tmp_path: Path) -> None:
    """initialize_model should create a GPT model and optimizer."""
    cfg = TrainerConfig(
        model=ModelConfig(
            n_layer=1, n_head=1, n_embd=4, block_size=4, dropout=0.0, vocab_size=50
        ),
        data=DataConfig(batch_size=2, block_size=4, grad_accum_steps=1),
        optim=OptimConfig(learning_rate=0.01),
        schedule=LRSchedule(
            decay_lr=False, warmup_iters=0, lr_decay_iters=0, min_lr=0.0
        ),
        runtime=RuntimeConfig(
            out_dir=tmp_path,
            max_iters=1,
            eval_interval=1,
            eval_iters=1,
            log_interval=1,
            eval_only=False,
            seed=1,
            device="cpu",
            dtype="float32",
            compile=False,
        ),
        hf_model=TrainerConfig.HFModelConfig(
            model_name="hf/model",
            gradient_checkpointing=False,
            block_size=128,
        ),
        peft=TrainerConfig.PeftConfig(enabled=False),
    )
    logger = logging.getLogger(__name__)

    model, optimizer = initialize_model(cfg, logger)

    # Should return a GPT model and optimizer
    assert isinstance(model, GPT)
    assert isinstance(optimizer, torch.optim.Optimizer)


def test_initialize_model_moves_model_to_correct_device(tmp_path: Path) -> None:
    """initialize_model should move the model to the configured device."""
    cfg = TrainerConfig(
        model=ModelConfig(
            n_layer=1, n_head=1, n_embd=4, block_size=4, dropout=0.0, vocab_size=50
        ),
        data=DataConfig(batch_size=2, block_size=4, grad_accum_steps=1),
        optim=OptimConfig(learning_rate=0.01),
        schedule=LRSchedule(
            decay_lr=False, warmup_iters=0, lr_decay_iters=0, min_lr=0.0
        ),
        runtime=RuntimeConfig(
            out_dir=tmp_path,
            max_iters=1,
            eval_interval=1,
            eval_iters=1,
            log_interval=1,
            eval_only=False,
            seed=1,
            device="cpu",
            dtype="float32",
            compile=False,
        ),
        hf_model=TrainerConfig.HFModelConfig(
            model_name="hf/model",
            gradient_checkpointing=False,
            block_size=128,
        ),
        peft=TrainerConfig.PeftConfig(enabled=False),
    )
    logger = logging.getLogger(__name__)

    model, optimizer = initialize_model(cfg, logger)

    # Model should be on CPU
    assert next(model.parameters()).device.type == "cpu"
