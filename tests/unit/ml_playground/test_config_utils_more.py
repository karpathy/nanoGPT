from __future__ import annotations

from pathlib import Path

import pytest

from ml_playground.config import (
    AppConfig,
    DataConfig,
    ExperimentConfig,
    load_toml,
    load_experiment_toml,
    validate_config_field,
    validate_path_exists,
)


def test_validate_config_field_success_and_errors():
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


def test_validate_path_exists_file_and_dir(tmp_path: Path):
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


def test_load_toml_filters_sections(tmp_path: Path):
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


def test_load_experiment_toml_and_reference_resolution(tmp_path: Path):
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


def test_data_config_tokenizer_choices(tmp_path: Path):
    # Ensure DataConfig accepts all tokenizer variants from the tokenizer protocol work
    DataConfig(dataset_dir=tmp_path, tokenizer="char")
    DataConfig(dataset_dir=tmp_path, tokenizer="word")
    DataConfig(dataset_dir=tmp_path, tokenizer="tiktoken")
