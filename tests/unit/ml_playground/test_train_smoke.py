from __future__ import annotations
from pathlib import Path
import numpy as np
from ml_playground.config import (
    TrainExperiment,
    ModelConfig,
    DataConfig,
    OptimConfig,
    LRSchedule,
    RuntimeConfig,
)
from ml_playground.trainer import train


def test_train_smoke(tmp_path: Path) -> None:
    # Create tiny dataset
    arr = (np.arange(1024) % 256).astype("uint16")
    (tmp_path / "train.bin").write_bytes(arr.tobytes())
    (tmp_path / "val.bin").write_bytes(arr.tobytes())

    exp = TrainExperiment(
        model=ModelConfig(
            n_layer=1,
            n_head=1,
            n_embd=32,
            block_size=16,
            dropout=0.0,
            bias=False,
            vocab_size=256,
        ),
        data=DataConfig(
            dataset_dir=tmp_path,
            batch_size=2,
            block_size=16,
            grad_accum_steps=1,
            train_bin="train.bin",
            val_bin="val.bin",
            meta_pkl=None,
        ),
        optim=OptimConfig(
            learning_rate=1e-3, weight_decay=0.0, grad_clip=0.0, beta1=0.9, beta2=0.95
        ),
        schedule=LRSchedule(
            decay_lr=False, warmup_iters=0, lr_decay_iters=1, min_lr=1e-3
        ),
        runtime=RuntimeConfig(
            out_dir=tmp_path / "out",
            max_iters=2,
            eval_interval=1,
            eval_iters=1,
            log_interval=1,
            eval_only=False,
            always_save_checkpoint=False,
            seed=123,
            device="cpu",
            dtype="float32",
            compile=False,
        ),
    )
    iters, best = train(exp)
    assert iters >= 2
