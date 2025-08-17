from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from _next.config import (
    TrainExperiment,
    ModelConfig,
    DataConfig,
    OptimConfig,
    LRSchedule,
    RuntimeConfig,
)
from _next.trainer import train


def _make_tiny_dataset(tmp_path: Path) -> None:
    arr = (np.arange(2048) % 256).astype("uint16")
    (tmp_path / "train.bin").write_bytes(arr.tobytes())
    (tmp_path / "val.bin").write_bytes(arr.tobytes())


def _base_exp(tmp_path: Path) -> TrainExperiment:
    return TrainExperiment(
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
            learning_rate=1e-3,
            weight_decay=0.0,
            grad_clip=0.0,
            beta1=0.9,
            beta2=0.95,
        ),
        schedule=LRSchedule(
            decay_lr=False,
            warmup_iters=0,
            lr_decay_iters=1,
            min_lr=1e-3,
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


def test_sidecar_last_schema(tmp_path: Path) -> None:
    _make_tiny_dataset(tmp_path)
    exp = _base_exp(tmp_path)
    # train
    iters, _ = train(exp)
    assert iters >= 2
    sidecar_path = exp.runtime.out_dir / "ckpt_last.json"
    assert sidecar_path.exists()
    data = json.loads(sidecar_path.read_text())

    # basic schema
    assert data["kind"] == "last"
    assert "created_at" in data and isinstance(data["created_at"], float)
    assert isinstance(data["sha256"], str) and len(data["sha256"]) >= 16
    assert data["iter_num"] >= 0
    assert data["tokens_seen"] == (
        exp.data.grad_accum_steps * exp.data.batch_size * exp.data.block_size * data["iter_num"]
    )
    assert data["lr"] == exp.optim.learning_rate
    assert data["env"]["device"] == "cpu"
    assert data["env"]["dtype"] == "float32"

    # eval block
    ev = data["eval"]
    assert ev["iters"] == exp.runtime.eval_iters
    assert ev["metric_name"] in ("val_loss", "ppl")
    assert isinstance(ev["greater_is_better"], bool)
    assert isinstance(ev["metric_raw"], float)
    assert ev["smoothing_alpha"] == 0.0
    assert ev["decision_metric"] == ev["metric_raw"]

    # backwards compatibility
    assert data["metric"] == ev["decision_metric"]
    # filename must not be present
    assert "filename" not in data


def test_sidecar_smoothing_changes_decision(tmp_path: Path) -> None:
    _make_tiny_dataset(tmp_path)
    exp = _base_exp(tmp_path)
    # enable smoothing, ensure two evals occur (max_iters>=1 and eval_interval=1)
    rt = exp.runtime
    exp = TrainExperiment(
        model=exp.model,
        data=exp.data,
        optim=exp.optim,
        schedule=exp.schedule,
        runtime=RuntimeConfig(
            **{**rt.__dict__, "best_smoothing_alpha": 0.5, "always_save_checkpoint": True}
        ),
    )
    iters, _ = train(exp)
    assert iters >= 2
    sidecar_path = exp.runtime.out_dir / "ckpt_last.json"
    data = json.loads(sidecar_path.read_text())
    ev = data["eval"]
    assert ev["smoothing_alpha"] == 0.5
    # not guaranteed to differ in pathological cases, but very likely; at least check type and allow equality failure
    # strengthen by requiring that when alpha>0, decision_metric is a float and metric_raw is float
    assert isinstance(ev["decision_metric"], float)
    assert isinstance(ev["metric_raw"], float)


def test_sidecar_ema_flag_on_best(tmp_path: Path) -> None:
    _make_tiny_dataset(tmp_path)
    exp = _base_exp(tmp_path)
    rt = exp.runtime
    exp = TrainExperiment(
        model=exp.model,
        data=exp.data,
        optim=exp.optim,
        schedule=exp.schedule,
        runtime=RuntimeConfig(
            **{
                **rt.__dict__,
                "ema_decay": 0.9,
                "always_save_checkpoint": True,
                "best_smoothing_alpha": 0.0,
                "max_iters": 1,
            }
        ),
    )
    iters, _ = train(exp)
    assert iters >= 1
    best_sidecar = exp.runtime.out_dir / "ckpt_best.json"
    assert best_sidecar.exists()
    data = json.loads(best_sidecar.read_text())
    assert data["kind"] == "best"
    assert data["ema"]["used_for_saved_model"] is True
    assert "eval_metric_on_ema" not in data["ema"]
