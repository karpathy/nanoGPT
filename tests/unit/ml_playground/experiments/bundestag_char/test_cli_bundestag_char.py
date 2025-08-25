from __future__ import annotations

from pathlib import Path
import json as _json

import pytest

from ml_playground.cli import main


@pytest.fixture()
def prepared_dataset() -> None:
    # Idempotent prepare; use experiment's own preparer directly (new CLI uses generic preparer)
    from ml_playground.experiments.bundestag_char.prepare import main as exp_prepare

    exp_prepare()


def _train_overrides(out_dir: Path) -> str:
    ov = {
        "runtime": {
            "out_dir": str(out_dir),
            "max_iters": 2,
            "eval_interval": 1,
            "eval_iters": 1,
            "log_interval": 1,
            "eval_only": False,
            "always_save_checkpoint": True,
            "ckpt_time_interval_minutes": 0,
            "device": "cpu",
            "dtype": "float32",
            "compile": False,
        },
        # Keep training tiny
        "data": {
            "batch_size": 4,
            "block_size": 64,
            "grad_accum_steps": 1,
        },
        "model": {
            "n_layer": 1,
            "n_head": 2,
            "n_embd": 64,
            "block_size": 64,
            "dropout": 0.0,
            "bias": False,
        },
        "schedule": {
            "decay_lr": True,
            "warmup_iters": 1,
            "lr_decay_iters": 2,
            "min_lr": 0.00008,
        },
        "optim": {
            "learning_rate": 0.0008,
            "weight_decay": 0.0,
            "beta1": 0.9,
            "beta2": 0.95,
            "grad_clip": 1.0,
        },
    }
    return _json.dumps(ov)


def _sample_overrides(out_dir: Path) -> str:
    ov = {
        "runtime": {
            "out_dir": str(out_dir),
            "device": "cpu",
            "dtype": "float32",
            "compile": False,
        },
        "sample": {
            "num_samples": 1,
            "max_new_tokens": 8,
            "temperature": 0.9,
            "top_k": 50,
        },
    }
    return _json.dumps(ov)


def test_prepare_bundestag_char(prepared_dataset: None) -> None:
    ds = Path("ml_playground/experiments/bundestag_char/datasets")
    assert (ds / "train.bin").exists()
    assert (ds / "val.bin").exists()
    assert (ds / "meta.pkl").exists()


def test_train_bundestag_char_quick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prepared_dataset: None
) -> None:
    out_dir = Path("ml_playground/experiments/bundestag_char/out")
    main(["train", "bundestag_char"])  # should run few iterations and save checkpoints
    # Check for expected artifacts
    assert (out_dir / "ckpt_last.pt").exists() or (out_dir / "ckpt.pt").exists()
    assert (
        out_dir / "ckpt_best.pt"
    ).exists()  # best should be saved at iter 1 with our settings
    # meta.pkl should be propagated for sampling
    assert (out_dir / "meta.pkl").exists()


def test_sample_bundestag_char_quick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prepared_dataset: None
) -> None:
    out_dir = tmp_path / "out_sample"
    # Ensure a tiny checkpoint exists by training quickly first
    monkeypatch.setenv("ML_PLAYGROUND_TRAIN_OVERRIDES", _train_overrides(out_dir))
    main(["train", "bundestag_char"])  # produce checkpoint
    # Now sample with small settings
    monkeypatch.setenv("ML_PLAYGROUND_SAMPLE_OVERRIDES", _sample_overrides(out_dir))
    main(["sample", "bundestag_char"])  # should not raise and print some text


def test_loop_bundestag_char_quick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "out_loop"
    monkeypatch.setenv("ML_PLAYGROUND_TRAIN_OVERRIDES", _train_overrides(out_dir))
    monkeypatch.setenv("ML_PLAYGROUND_SAMPLE_OVERRIDES", _sample_overrides(out_dir))
    main(["loop", "bundestag_char"])  # end-to-end pipeline
    # Check that training produced checkpoints in the designated directory
    assert (out_dir / "ckpt_best.pt").exists()


def test_train_ckpt_filename_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prepared_dataset: None
) -> None:
    out_dir = tmp_path / "out_ckpt_names"
    # Override checkpoint filenames to verify config changes take effect
    ov = {
        "runtime": {
            "out_dir": str(out_dir),
            "max_iters": 2,
            "eval_interval": 1,
            "eval_iters": 1,
            "log_interval": 1,
            "always_save_checkpoint": True,
            "ckpt_last_filename": "last_custom.pt",
            "ckpt_best_filename": "best_custom.pt",
            "device": "cpu",
            "dtype": "float32",
        },
        "data": {"batch_size": 2, "block_size": 32, "grad_accum_steps": 1},
        "model": {
            "n_layer": 1,
            "n_head": 2,
            "n_embd": 64,
            "block_size": 32,
            "bias": False,
            "dropout": 0.0,
        },
        "schedule": {
            "decay_lr": True,
            "warmup_iters": 1,
            "lr_decay_iters": 2,
            "min_lr": 0.00008,
        },
        "optim": {
            "learning_rate": 0.0008,
            "weight_decay": 0.0,
            "beta1": 0.9,
            "beta2": 0.95,
            "grad_clip": 1.0,
        },
    }
    monkeypatch.setenv("ML_PLAYGROUND_TRAIN_OVERRIDES", _json.dumps(ov))
    main(["train", "bundestag_char"])  # produce checkpoint with custom names
    # Assert custom-named checkpoints exist
    assert (out_dir / "last_custom.pt").exists()
    assert (out_dir / "best_custom.pt").exists()
    # And default names are absent (proving the override was effective)
    assert not (out_dir / "ckpt_last.pt").exists()
    assert not (out_dir / "ckpt_best.pt").exists()
