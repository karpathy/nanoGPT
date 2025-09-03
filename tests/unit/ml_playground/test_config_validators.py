from __future__ import annotations

from pathlib import Path
import pytest
from pydantic import ValidationError

from ml_playground.config import (
    DataConfig,
    SampleConfig,
    LRSchedule,
    OptimConfig,
    ModelConfig,
    RuntimeConfig,
    load_toml,
)


def test_dataconfig_positive_ints(tmp_path: Path):
    with pytest.raises(ValidationError):
        DataConfig(dataset_dir=tmp_path, batch_size=0)
    with pytest.raises(ValidationError):
        DataConfig(dataset_dir=tmp_path, block_size=-1)
    with pytest.raises(ValidationError):
        DataConfig(dataset_dir=tmp_path, grad_accum_steps=0)
    with pytest.raises(ValidationError):
        DataConfig(dataset_dir=tmp_path, ngram_size=0)
    # happy path
    cfg = DataConfig(
        dataset_dir=tmp_path,
        batch_size=1,
        block_size=1,
        grad_accum_steps=1,
        ngram_size=1,
    )
    assert cfg.batch_size == 1


def test_sampleconfig_ranges():
    with pytest.raises(ValidationError):
        SampleConfig(temperature=0.0)
    with pytest.raises(ValidationError):
        SampleConfig(top_k=-1)
    with pytest.raises(ValidationError):
        SampleConfig(top_p=0.0)
    with pytest.raises(ValidationError):
        SampleConfig(top_p=1.5)
    # ok
    SampleConfig(temperature=0.1, top_k=0, top_p=0.5)


def test_lrschedule_validations():
    with pytest.raises(ValidationError):
        LRSchedule(warmup_iters=-1)
    with pytest.raises(ValidationError):
        LRSchedule(lr_decay_iters=-5)
    with pytest.raises(ValidationError):
        LRSchedule(warmup_iters=10, lr_decay_iters=5)
    with pytest.raises(ValidationError):
        LRSchedule(min_lr=-1e-5)
    # ok
    LRSchedule(warmup_iters=1, lr_decay_iters=2, min_lr=0)


def test_optimconfig_non_negative():
    with pytest.raises(ValidationError):
        OptimConfig(learning_rate=-1e-3)
    with pytest.raises(ValidationError):
        OptimConfig(weight_decay=-1e-1)
    with pytest.raises(ValidationError):
        OptimConfig(beta1=-0.1)
    with pytest.raises(ValidationError):
        OptimConfig(beta2=-0.1)
    with pytest.raises(ValidationError):
        OptimConfig(grad_clip=-1)
    # ok
    OptimConfig()


def test_modelconfig_ranges():
    with pytest.raises(ValidationError):
        ModelConfig(n_layer=0)
    with pytest.raises(ValidationError):
        ModelConfig(n_head=0)
    with pytest.raises(ValidationError):
        ModelConfig(n_embd=0)
    with pytest.raises(ValidationError):
        ModelConfig(block_size=0)
    with pytest.raises(ValidationError):
        ModelConfig(dropout=1.5)
    with pytest.raises(ValidationError):
        ModelConfig(vocab_size=0)
    # ok
    ModelConfig()


def test_runtime_checkpointing_keep_non_negative(tmp_path: Path):
    with pytest.raises(ValidationError):
        RuntimeConfig(
            out_dir=tmp_path,
            checkpointing=RuntimeConfig.Checkpointing(
                keep=RuntimeConfig.Checkpointing.Keep(last=-1)
            ),
        )
    with pytest.raises(ValidationError):
        RuntimeConfig(
            out_dir=tmp_path,
            checkpointing=RuntimeConfig.Checkpointing(
                keep=RuntimeConfig.Checkpointing.Keep(best=-2)
            ),
        )
    # ok
    RuntimeConfig(out_dir=tmp_path)


def test_load_toml_filters_extras(tmp_path: Path):
    toml_text = """
[train.model]
 n_layer=1
 n_head=1
 n_embd=32
 block_size=16

[train.data]
 dataset_dir = "data/shakespeare"

[train.optim]
 learning_rate = 0.001

[train.schedule]

[train.runtime]
 out_dir = "out/test"

[export]
 some = "value"
"""
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text(toml_text)
    # Should not raise; extras are filtered out at top-level
    _ = load_toml(cfg_path)


def test_load_toml_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_toml(tmp_path / "absent.toml")
