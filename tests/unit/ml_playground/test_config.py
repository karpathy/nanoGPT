from __future__ import annotations
from pathlib import Path
import pytest
from pydantic import ValidationError
from ml_playground.config import (
    load_toml,
    AppConfig,
    DataConfig,
    ExperimentConfig,
    load_experiment_toml,
    validate_config_field,
    validate_path_exists,
    SampleConfig,
    LRSchedule,
    OptimConfig,
    ModelConfig,
    RuntimeConfig,
)


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


# Consolidated: validators and util tests previously in fragmented files


def test_validate_config_field_success_and_errors() -> None:
    # Success cases
    validate_config_field(3, "k", int)
    validate_config_field(3.5, "k", float, min_value=0.0, max_value=10.0)
    validate_config_field(None, "k", int, required=False)

    # Required missing
    with pytest.raises(ValueError):
        validate_config_field(None, "k", int, required=True)

    # Type mismatch
    with pytest.raises(ValueError):
        validate_config_field("x", "k", int)

    # Range violations
    with pytest.raises(ValueError):
        validate_config_field(-1, "k", int, min_value=0)
    with pytest.raises(ValueError):
        validate_config_field(11, "k", int, max_value=10)


def test_validate_path_exists_file_and_dir(tmp_path: Path) -> None:
    # Missing
    with pytest.raises(ValueError):
        validate_path_exists(tmp_path / "missing", "x")

    # File case
    f = tmp_path / "a.txt"
    f.write_text("hi")
    validate_path_exists(f, "x", must_be_file=True)
    with pytest.raises(ValueError):
        validate_path_exists(f, "x", must_be_dir=True)

    # Dir case
    d = tmp_path / "d"
    d.mkdir()
    validate_path_exists(d, "x", must_be_dir=True)
    with pytest.raises(ValueError):
        validate_path_exists(d, "x", must_be_file=True)


def test_load_toml_filters_sections(tmp_path: Path) -> None:
    p = tmp_path / "cfg.toml"
    p.write_text(
        """
[train]
[train.model]

[train.data]
dataset_dir = "./data"

[train.optim]

[train.schedule]

[train.runtime]
out_dir = "./out"

[sample]
runtime_ref = "train.runtime"
[sample.sample]
start = "\\n"

[export]
foo = 1
"""
    )
    app = load_toml(p)
    assert isinstance(app, AppConfig)
    # Only known sections should be retained
    assert app.train is not None and app.sample is not None


def test_load_experiment_toml_and_reference_resolution(tmp_path: Path) -> None:
    p = tmp_path / "exp.toml"
    p.write_text(
        """
[prepare]
# minimal preparer config

[train]
[train.model]

[train.data]
dataset_dir = "./data"

[train.optim]

[train.schedule]

[train.runtime]
out_dir = "./out"
log_interval = 2

[sample]
    [sample.runtime]
    out_dir = "./out"
    log_interval = 2
    [sample.sample]
    start = "\\n"
"""
    )
    exp = load_experiment_toml(p)
    assert isinstance(exp, ExperimentConfig)
    # Parsed runtime present and matches provided values
    assert exp.sample.runtime is not None
    assert exp.sample.runtime.out_dir == Path("./out")
    assert exp.sample.runtime.log_interval == 2


def test_data_config_tokenizer_choices(tmp_path: Path) -> None:
    # Ensure DataConfig accepts tokenizer variants
    DataConfig(dataset_dir=tmp_path, tokenizer="char")
    DataConfig(dataset_dir=tmp_path, tokenizer="word")
    DataConfig(dataset_dir=tmp_path, tokenizer="tiktoken")


def test_dataconfig_positive_ints(tmp_path: Path) -> None:
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


def test_sampleconfig_ranges() -> None:
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


def test_lrschedule_validations() -> None:
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


def test_optimconfig_non_negative() -> None:
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


def test_modelconfig_ranges() -> None:
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


def test_runtime_checkpointing_keep_non_negative(tmp_path: Path) -> None:
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


def test_config_models_shim_exports() -> None:
    # Consolidated: coverage for config_models shim
    from ml_playground import config_models as shim

    assert hasattr(shim, "TrainerConfig")
    assert hasattr(shim, "SamplerConfig")
    assert hasattr(shim, "DataConfig")
    assert hasattr(shim, "RuntimeConfig")
    assert hasattr(shim, "load_toml")


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
