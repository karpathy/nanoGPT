from __future__ import annotations

from pathlib import Path
import json as _json

import numpy as np
import pytest

from ml_playground.cli import main


@pytest.fixture()
def tmp_dataset(tmp_path: Path) -> Path:
    # create a minimal dataset in a temporary directory
    ds = tmp_path / "dataset"
    ds.mkdir(parents=True, exist_ok=True)
    arr = (np.arange(512) % 256).astype("uint16")
    (ds / "train.bin").write_bytes(arr.tobytes())
    (ds / "val.bin").write_bytes(arr.tobytes())
    meta = {
        "meta_version": 1,
        "kind": "char",
        "dtype": "uint16",
        "stoi": {chr(i): i for i in range(256)},
        "itos": {i: chr(i) for i in range(256)},
    }
    import pickle

    with (ds / "meta.pkl").open("wb") as f:
        pickle.dump(meta, f)
    return ds


def _train_overrides(out_dir: Path, dataset_dir: Path) -> str:
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
            "dataset_dir": str(dataset_dir),
            "train_bin": "train.bin",
            "val_bin": "val.bin",
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


def test_train_bundestag_char_quick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tmp_dataset: Path
) -> None:
    out_dir = tmp_path / "out_train"
    monkeypatch.setenv(
        "ML_PLAYGROUND_TRAIN_OVERRIDES", _train_overrides(out_dir, tmp_dataset)
    )
    main(["train", "bundestag_char"])  # should run few iterations and save checkpoints
    # Check for expected artifacts
    assert (out_dir / "ckpt_last.pt").exists() or (out_dir / "ckpt.pt").exists()
    assert (out_dir / "ckpt_best.pt").exists()
    # meta.pkl should be propagated for sampling
    assert (out_dir / "meta.pkl").exists()


def test_sample_bundestag_char_quick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tmp_dataset: Path
) -> None:
    out_dir = tmp_path / "out_sample"
    # Ensure a tiny checkpoint exists by training quickly first
    monkeypatch.setenv(
        "ML_PLAYGROUND_TRAIN_OVERRIDES", _train_overrides(out_dir, tmp_dataset)
    )
    main(["train", "bundestag_char"])  # produce checkpoint
    # Now sample with small settings
    monkeypatch.setenv("ML_PLAYGROUND_SAMPLE_OVERRIDES", _sample_overrides(out_dir))
    main(["sample", "bundestag_char"])  # should not raise and print some text


def test_loop_bundestag_char_quick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tmp_dataset: Path
) -> None:
    out_dir = tmp_path / "out_loop"
    # Both train and sample should pick up the out_dir via overrides
    monkeypatch.setenv(
        "ML_PLAYGROUND_TRAIN_OVERRIDES", _train_overrides(out_dir, tmp_dataset)
    )
    monkeypatch.setenv("ML_PLAYGROUND_SAMPLE_OVERRIDES", _sample_overrides(out_dir))
    main(["loop", "bundestag_char"])  # end-to-end pipeline
    # Check that training produced checkpoints in the designated directory
    assert (out_dir / "ckpt_best.pt").exists()
