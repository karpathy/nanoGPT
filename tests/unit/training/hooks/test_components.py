from __future__ import annotations

import logging
from contextlib import nullcontext
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
from ml_playground.models.core.model import GPT
from ml_playground.training.hooks.components import initialize_components
from ml_playground.training.hooks.runtime import RuntimeContext


def _make_model() -> GPT:
    """Create a minimal GPT model for testing."""
    cfg = ModelConfig(
        n_layer=1,
        n_head=1,
        n_embd=4,
        block_size=4,
        dropout=0.0,
        vocab_size=50,
    )
    logger = logging.getLogger(__name__)
    return GPT(cfg, logger)


def _make_config(
    *,
    compile: bool = False,
    ema_decay: float = 0.0,
    tensorboard_enabled: bool = False,
    device: str = "cpu",
    dtype: str = "float32",
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
            seed=1,
            device=device,
            dtype=dtype,
            compile=compile,
            tensorboard_enabled=tensorboard_enabled,
            ema_decay=ema_decay,
        ),
        hf_model=TrainerConfig.HFModelConfig(
            model_name="hf/model",
            gradient_checkpointing=False,
            block_size=128,
        ),
        peft=TrainerConfig.PeftConfig(enabled=False),
    )


def test_initialize_components_without_optional_features(tmp_path: Path) -> None:
    """initialize_components should work without compile, EMA, or TensorBoard."""
    model = _make_model()
    cfg = _make_config()
    runtime = RuntimeContext(device_type="cpu", autocast_context=nullcontext())

    compiled_model, scaler, ema, writer = initialize_components(
        model, cfg, runtime, log_dir=str(tmp_path)
    )

    # Model should be returned as-is (not compiled)
    assert compiled_model is model
    # Scaler should be created but disabled for CPU + float32
    assert scaler is not None
    # EMA should be None (ema_decay=0.0)
    assert ema is None
    # Writer should be None (tensorboard_enabled=False)
    assert writer is None


def test_initialize_components_with_ema(tmp_path: Path) -> None:
    """initialize_components should create EMA when ema_decay > 0."""
    model = _make_model()
    cfg = _make_config(ema_decay=0.999)
    runtime = RuntimeContext(device_type="cpu", autocast_context=nullcontext())

    compiled_model, scaler, ema, writer = initialize_components(
        model, cfg, runtime, log_dir=str(tmp_path)
    )

    # EMA should be created
    assert ema is not None
    assert ema.decay == 0.999


def test_initialize_components_with_tensorboard(tmp_path: Path) -> None:
    """initialize_components should create TensorBoard writer when enabled."""
    model = _make_model()
    cfg = _make_config(tensorboard_enabled=True)
    runtime = RuntimeContext(device_type="cpu", autocast_context=nullcontext())

    log_dir = tmp_path / "logs"
    compiled_model, scaler, ema, writer = initialize_components(
        model, cfg, runtime, log_dir=str(log_dir)
    )

    # Writer should be created
    assert writer is not None
    # Clean up
    if writer:
        writer.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_initialize_components_scaler_enabled_for_cuda_float16(tmp_path: Path) -> None:
    """initialize_components should enable GradScaler for CUDA + float16."""
    model = _make_model()
    cfg = _make_config(device="cuda", dtype="float16")
    runtime = RuntimeContext(device_type="cuda", autocast_context=nullcontext())

    compiled_model, scaler, ema, writer = initialize_components(
        model, cfg, runtime, log_dir=str(tmp_path)
    )

    # Scaler should be enabled for CUDA + float16
    assert scaler is not None
    assert scaler.is_enabled()


def test_initialize_components_scaler_disabled_for_cpu(tmp_path: Path) -> None:
    """initialize_components should disable GradScaler for CPU."""
    model = _make_model()
    cfg = _make_config(device="cpu", dtype="float32")
    runtime = RuntimeContext(device_type="cpu", autocast_context=nullcontext())

    compiled_model, scaler, ema, writer = initialize_components(
        model, cfg, runtime, log_dir=str(tmp_path)
    )

    # Scaler should be disabled for CPU
    assert scaler is not None
    assert not scaler.is_enabled()


def test_initialize_components_with_all_features(tmp_path: Path) -> None:
    """initialize_components should handle all features enabled together."""
    model = _make_model()
    cfg = _make_config(
        compile=False,  # Can't test actual compilation easily
        ema_decay=0.95,
        tensorboard_enabled=True,
        device="cpu",
        dtype="float32",
    )
    runtime = RuntimeContext(device_type="cpu", autocast_context=nullcontext())

    log_dir = tmp_path / "logs"
    compiled_model, scaler, ema, writer = initialize_components(
        model, cfg, runtime, log_dir=str(log_dir)
    )

    # All components should be created
    assert compiled_model is not None
    assert scaler is not None
    assert ema is not None
    assert ema.decay == 0.95
    assert writer is not None

    # Clean up
    if writer:
        writer.close()


def test_initialize_components_with_compile(tmp_path: Path) -> None:
    """initialize_components should attempt to compile model when enabled."""
    model = _make_model()
    cfg = _make_config(compile=True)
    runtime = RuntimeContext(device_type="cpu", autocast_context=nullcontext())

    compiled_calls: list[object] = []

    def _fake_compile(module: GPT) -> GPT:
        compiled_calls.append(module)
        return module

    compiled_model, scaler, ema, writer = initialize_components(
        model, cfg, runtime, log_dir=str(tmp_path), compile_fn=_fake_compile
    )

    # Model should be returned (compiled or not, depending on torch version)
    assert compiled_model is not None
    # Should have attempted compilation exactly once
    assert compiled_calls == [model]
