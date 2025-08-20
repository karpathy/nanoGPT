from __future__ import annotations
from pathlib import Path
import pytest
from pydantic import ValidationError
from ml_playground.config import load_toml, AppConfig


def test_load_toml_roundtrip(tmp_path: Path) -> None:
    toml_text = """
[train.model]
n_layer=1
n_head=1
n_embd=32
block_size=16
bias=false

[train.data]
dataset_dir = "data/shakespeare"
block_size = 16
batch_size = 2
grad_accum_steps = 1

[train.optim]
learning_rate = 0.001

[train.schedule]

[train.runtime]
out_dir = "out/test_next"
max_iters = 1

[sample.runtime]
out_dir = "out/test_next"

[sample.sample]
"""
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text(toml_text)

    cfg: AppConfig = load_toml(cfg_path)
    assert cfg.train is not None
    assert cfg.sample is not None
    assert isinstance(cfg.train.runtime.out_dir, Path)
    assert isinstance(cfg.train.data.dataset_dir, Path)


def test_load_toml_empty_config(tmp_path: Path) -> None:
    """Test loading empty TOML config returns AppConfig with None values."""
    toml_text = ""
    cfg_path = tmp_path / "empty.toml"
    cfg_path.write_text(toml_text)

    cfg: AppConfig = load_toml(cfg_path)
    assert cfg.train is None
    assert cfg.sample is None


def test_load_toml_incomplete_train_config(tmp_path: Path) -> None:
    """Strict: incomplete [train] should raise ValidationError."""
    # Missing required sections like model, data, optim, schedule, runtime
    toml_text = """
[train.model]
n_layer=1

# Missing other required sections
"""
    cfg_path = tmp_path / "incomplete.toml"
    cfg_path.write_text(toml_text)

    with pytest.raises(ValidationError):
        load_toml(cfg_path)


def test_load_toml_incomplete_sample_config(tmp_path: Path) -> None:
    """Strict: incomplete [sample] should raise ValidationError."""
    toml_text = """
[sample.runtime]
out_dir = "out/test"
# Missing sample.sample section
"""
    cfg_path = tmp_path / "incomplete_sample.toml"
    cfg_path.write_text(toml_text)

    with pytest.raises(ValidationError):
        load_toml(cfg_path)


def test_load_toml_no_train_section(tmp_path: Path) -> None:
    """Test loading TOML without train section."""
    toml_text = """
[sample.runtime]
out_dir = "out/test"

[sample.sample]
"""
    cfg_path = tmp_path / "no_train.toml"
    cfg_path.write_text(toml_text)

    cfg: AppConfig = load_toml(cfg_path)
    assert cfg.train is None
    assert cfg.sample is not None


def test_load_toml_no_sample_section(tmp_path: Path) -> None:
    """Test loading TOML without sample section."""
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
"""
    cfg_path = tmp_path / "no_sample.toml"
    cfg_path.write_text(toml_text)

    cfg: AppConfig = load_toml(cfg_path)
    assert cfg.train is not None
    assert cfg.sample is None


def test_load_toml_train_missing_data_section(tmp_path: Path) -> None:
    """Strict: [train] missing required data subsection must raise ValidationError."""
    toml_text = """
[train.model]
 n_layer=1
 n_head=1
 n_embd=32
 block_size=16
 
 [train.optim]
 learning_rate = 0.001
 
 [train.schedule]
 
 [train.runtime]
 out_dir = "out/test"
 # Missing [train.data] section
 """
    cfg_path = tmp_path / "missing_data.toml"
    cfg_path.write_text(toml_text)

    with pytest.raises(ValidationError):
        load_toml(cfg_path)


def test_load_toml_train_missing_runtime_section(tmp_path: Path) -> None:
    """Strict: [train] missing required runtime subsection must raise ValidationError."""
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
 # Missing [train.runtime] section
 """
    cfg_path = tmp_path / "missing_runtime.toml"
    cfg_path.write_text(toml_text)

    with pytest.raises(ValidationError):
        load_toml(cfg_path)


def test_load_toml_sample_missing_runtime_section(tmp_path: Path) -> None:
    """Strict: [sample] missing required runtime subsection must raise ValidationError."""
    toml_text = """
[sample.sample]
# Missing [sample.runtime] section
"""
    cfg_path = tmp_path / "sample_missing_runtime.toml"
    cfg_path.write_text(toml_text)

    with pytest.raises(ValidationError):
        load_toml(cfg_path)
