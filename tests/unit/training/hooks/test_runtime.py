from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ml_playground.configuration.models import (
    DataConfig,
    LRSchedule,
    ModelConfig,
    OptimConfig,
    RuntimeConfig,
    TrainerConfig,
)
from ml_playground.training.hooks.runtime import setup_runtime


def _make_config(
    device: str = "cpu", dtype: str = "float32", seed: int = 42
) -> TrainerConfig:
    """Create a TrainerConfig for testing."""
    return TrainerConfig(
        model=ModelConfig(
            n_layer=1, n_head=1, n_embd=4, block_size=4, dropout=0.0, vocab_size=50
        ),
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
            seed=seed,
            device=device,
            dtype=dtype,
            compile=False,
            tensorboard_enabled=False,
            ema_decay=0.0,
        ),
        hf_model=TrainerConfig.HFModelConfig(
            model_name="hf/model",
            gradient_checkpointing=False,
            block_size=128,
        ),
        peft=TrainerConfig.PeftConfig(enabled=False),
    )


def test_setup_runtime_cpu() -> None:
    """setup_runtime should configure CPU runtime context."""
    cfg = _make_config(device="cpu", dtype="float32")

    runtime = setup_runtime(cfg)

    assert runtime.device_type == "cpu"
    assert runtime.autocast_context is not None


def test_setup_runtime_seeds_torch() -> None:
    """setup_runtime should seed torch RNG."""
    cfg = _make_config(seed=123)

    setup_runtime(cfg)

    # Verify seed was set by generating a random number
    val1 = torch.rand(1).item()

    # Reset and verify reproducibility
    setup_runtime(cfg)
    val2 = torch.rand(1).item()

    assert val1 == val2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_setup_runtime_cuda() -> None:
    """setup_runtime should configure CUDA runtime context."""
    cfg = _make_config(device="cuda", dtype="float16")

    runtime = setup_runtime(cfg)

    assert runtime.device_type == "cuda"
    assert runtime.autocast_context is not None


def test_setup_runtime_bfloat16() -> None:
    """setup_runtime should handle bfloat16 dtype."""
    cfg = _make_config(device="cpu", dtype="bfloat16")

    runtime = setup_runtime(cfg)

    assert runtime.device_type == "cpu"
    assert runtime.autocast_context is not None


def test_setup_runtime_float16() -> None:
    """setup_runtime should handle float16 dtype."""
    cfg = _make_config(device="cpu", dtype="float16")

    runtime = setup_runtime(cfg)

    assert runtime.device_type == "cpu"
    assert runtime.autocast_context is not None
