from __future__ import annotations
from pathlib import Path
import numpy as np
from _next.config import TrainExperiment, ModelConfig, DataConfig, OptimConfig, LRSchedule, RuntimeConfig
from _next.trainer import train


def test_resume_smoke(tmp_path: Path) -> None:
    # Create tiny dataset
    arr = (np.arange(2048) % 256).astype("uint16")
    (tmp_path / "train.bin").write_bytes(arr.tobytes())
    (tmp_path / "val.bin").write_bytes(arr.tobytes())

    out_dir = tmp_path / "out"

    base_train = TrainExperiment(
        model=ModelConfig(n_layer=1, n_head=1, n_embd=32, block_size=16, dropout=0.0, bias=False, vocab_size=256),
        data=DataConfig(dataset_dir=tmp_path, batch_size=2, block_size=16, grad_accum_steps=1,
                        train_bin="train.bin", val_bin="val.bin", meta_pkl=None),
        optim=OptimConfig(learning_rate=1e-3, weight_decay=0.0, grad_clip=0.0, beta1=0.9, beta2=0.95),
        schedule=LRSchedule(decay_lr=False, warmup_iters=0, lr_decay_iters=1, min_lr=1e-3),
        runtime=RuntimeConfig(out_dir=out_dir, max_iters=2, eval_interval=1, eval_iters=1, log_interval=10,
                              eval_only=False, always_save_checkpoint=True, seed=123, device="cpu", dtype="float32", compile=False),
    )

    iters1, best1 = train(base_train)
    assert (out_dir / "ckpt_last.pt").exists(), "Expected last checkpoint after first training run"

    # Resume: increase max_iters and rerun with identical configs except runtime.max_iters
    resumed_train = TrainExperiment(
        model=base_train.model,
        data=base_train.data,
        optim=base_train.optim,
        schedule=base_train.schedule,
        runtime=RuntimeConfig(out_dir=out_dir, max_iters=4, eval_interval=1, eval_iters=1, log_interval=10,
                              eval_only=False, always_save_checkpoint=True, seed=123, device="cpu", dtype="float32", compile=False),
    )

    iters2, best2 = train(resumed_train)
    assert iters2 > iters1, f"Expected resume to continue training (iters2={iters2} > iters1={iters1})"
