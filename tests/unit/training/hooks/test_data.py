from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from ml_playground.configuration.models import (
    DataConfig,
    LRSchedule,
    ModelConfig,
    OptimConfig,
    RuntimeConfig,
    SharedConfig,
    TrainerConfig,
)
from ml_playground.training.hooks.data import initialize_batches


def test_initialize_batches_creates_simple_batches() -> None:
    """initialize_batches should create a SimpleBatches instance with correct dataset directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        dataset_dir = tmp_path / "datasets"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Create mock dataset files
        train_data = np.random.randint(0, 50, size=(100,), dtype=np.uint16)
        val_data = np.random.randint(0, 50, size=(50,), dtype=np.uint16)

        train_path = dataset_dir / "train.bin"
        val_path = dataset_dir / "val.bin"

        train_data.tofile(train_path)
        val_data.tofile(val_path)

        # Create a proper TrainerConfig like other tests do
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

        shared = SharedConfig(
            experiment="test",
            config_path=tmp_path / "config.toml",
            project_home=tmp_path,
            dataset_dir=dataset_dir,
            train_out_dir=tmp_path,
            sample_out_dir=tmp_path,
        )

        batches = initialize_batches(cfg, shared)

        # Should return a SimpleBatches instance
        assert batches is not None
        # The dataset_dir should be used from shared config
        assert hasattr(batches, "get_batch")  # Should have the get_batch method

        # Should be able to get a batch
        x, y = batches.get_batch("train")
        assert x.shape == (2, 4)  # batch_size=2, block_size=4
        assert y.shape == (2, 4)
