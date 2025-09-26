from __future__ import annotations

from pathlib import Path

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
        "tokenizer_type": "char",
        "dtype": "uint16",
        "stoi": {chr(i): i for i in range(256)},
        "itos": {i: chr(i) for i in range(256)},
    }
    import pickle

    with (ds / "meta.pkl").open("wb") as f:
        pickle.dump(meta, f)
    return ds


def _write_exp_config(tmp_dir: Path, out_dir: Path, dataset_dir: Path) -> Path:
    """Create a minimal, strict experiment config TOML pointing to tmp paths."""
    cfg = f'''
[prepare]
dataset_dir = "{dataset_dir}"

[train.model]
n_layer = 1
n_head = 2
n_embd = 64
block_size = 64
dropout = 0.0
bias = false
vocab_size = 256

[train.data]
train_bin = "train.bin"
val_bin = "val.bin"
batch_size = 4
block_size = 64
grad_accum_steps = 1

[train.optim]
learning_rate = 0.0005
weight_decay = 0.0
beta1 = 0.9
beta2 = 0.95
grad_clip = 0.0

[train.schedule]
decay_lr = true
warmup_iters = 1
lr_decay_iters = 2
min_lr = 0.00008

[train.runtime]
out_dir = "{out_dir}"
max_iters = 2
eval_interval = 1
eval_iters = 1
log_interval = 1
eval_only = false
device = "cpu"
dtype = "float32"
compile = false
ckpt_time_interval_minutes = 0

[train.runtime.checkpointing.keep]
last = 1
best = 1

[sample.runtime]
out_dir = "{out_dir}"
device = "cpu"
dtype = "float32"
compile = false
eval_only = false
max_iters = 0
eval_interval = 1
eval_iters = 1
log_interval = 1

[sample.sample]
start = "Hi"
num_samples = 1
max_new_tokens = 8
temperature = 0.9
top_k = 50
'''
    p = tmp_dir / "config.toml"
    p.write_text(cfg)
    return p


def _write_sample_only_config(tmp_dir: Path, out_dir: Path) -> Path:
    cfg = f'''
[sample.runtime]
out_dir = "{out_dir}"
device = "cpu"
dtype = "float32"
compile = false
eval_only = false
max_iters = 0
eval_interval = 1
eval_iters = 1
log_interval = 1

[sample.sample]
start = "Hi"
num_samples = 1
max_new_tokens = 8
temperature = 0.9
top_k = 50
'''
    p = tmp_dir / "config.toml"
    p.write_text(cfg)
    return p


def test_train_bundestag_char_quick(tmp_path: Path, tmp_dataset: Path) -> None:
    out_dir = tmp_path / "out_train"
    cfg = _write_exp_config(tmp_path, out_dir, tmp_dataset)
    main(["--exp-config", str(cfg), "train", "bundestag_char"])  # run quickly
    # Check for expected rotated checkpoint artifacts
    assert any(out_dir.glob("ckpt_last_*.pt")), "no rotated last checkpoint found"
    assert any(out_dir.glob("ckpt_best_*.pt")), "no rotated best checkpoint found"
    # meta.pkl should be propagated for sampling
    assert (out_dir / "meta.pkl").exists()


def test_sample_bundestag_char_quick(tmp_path: Path, tmp_dataset: Path) -> None:
    out_dir = tmp_path / "out_sample"
    # Train small to produce a checkpoint
    train_cfg = _write_exp_config(tmp_path, out_dir, tmp_dataset)
    main(["--exp-config", str(train_cfg), "train", "bundestag_char"])
    # Sample with minimal config pointing to same out_dir
    sample_cfg = _write_sample_only_config(tmp_path, out_dir)
    main(["--exp-config", str(sample_cfg), "sample", "bundestag_char"])
