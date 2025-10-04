from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import hypothesis.strategies as st
from hypothesis import example, given, settings

from ml_playground.configuration import loading

_DEFAULT_CONFIG = """
[train.model]
n_layer = 1
n_head = 1
n_embd = 64
block_size = 128
bias = true

[train.data]
train_bin = "train.bin"
val_bin = "val.bin"
meta_pkl = "meta.pkl"
batch_size = 1
block_size = 128
grad_accum_steps = 1
tokenizer = "char"
ngram_size = 1
sampler = "random"

[train.optim]
learning_rate = 0.001
weight_decay = 0.0
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

[train.schedule]
decay_lr = false
warmup_iters = 0
lr_decay_iters = 100
min_lr = 0.0

[train.runtime]
out_dir = "out"
max_iters = 1
eval_interval = 1
eval_iters = 1
log_interval = 1
seed = 1

[sample]
[sample.runtime]
out_dir = "sample_out"

[sample.sample]
max_new_tokens = 1
num_samples = 1
start = "x"

[prepare]
raw_dir = "raw"
"""


def _write_default_config(tmp_path: Path) -> Path:
    defaults_path = tmp_path / "default_config.toml"
    defaults_path.write_text(_DEFAULT_CONFIG, encoding="utf-8")
    return defaults_path


_PATH_PARTS = st.lists(
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=4),
    min_size=1,
    max_size=3,
)


@given(path_parts=_PATH_PARTS)
@example(path_parts=["runs", "outputs"])
@settings(max_examples=20, deadline=None, derandomize=True)
def test_load_train_config_resolves_relative_out_dir(path_parts: list[str]) -> None:
    """`load_train_config` must resolve relative runtime.out_dir against the config directory."""

    rel_path = Path(*path_parts)
    rel_out_dir = rel_path.as_posix()

    with TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        defaults_path = _write_default_config(base)
        config_path = base / "config.toml"
        config_path.write_text(
            f"""
[train.runtime]
out_dir = "{rel_out_dir}"
""",
            encoding="utf-8",
        )

        cfg = loading.load_train_config(config_path, default_config_path=defaults_path)

        expected = (config_path.parent / rel_path).resolve()
        assert cfg.runtime.out_dir == expected


def test_load_train_config_rejects_non_mapping_sections(tmp_path: Path) -> None:
    """`load_train_config` must raise when sections are not mappings."""

    defaults_path = _write_default_config(tmp_path)
    bad_config = tmp_path / "bad.toml"
    bad_config.write_text(
        """
train = "not-a-table"
""",
        encoding="utf-8",
    )

    with pytest.raises(TypeError):
        loading.load_train_config(bad_config, default_config_path=defaults_path)


def test_load_sample_config_requires_sample_section(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match=r"\[sample\]"):
        loading.load_sample_config(config_path)


def test_load_train_config_merges_defaults(tmp_path: Path) -> None:
    defaults_path = _write_default_config(tmp_path)
    config_path = tmp_path / "override.toml"
    config_path.write_text(
        """
[train.model]
n_layer = 3

[train.runtime]
out_dir = "./outputs"
""",
        encoding="utf-8",
    )

    cfg = loading.load_train_config(config_path, default_config_path=defaults_path)

    assert cfg.model.n_layer == 3
    assert cfg.model.n_head == 1  # inherited from defaults
    assert cfg.runtime.out_dir == (config_path.parent / "outputs").resolve()


@given(path_parts=_PATH_PARTS)
@example(path_parts=["artifacts", "samples"])
@settings(max_examples=20, deadline=None, derandomize=True)
def test_load_sample_config_resolves_relative_out_dir(path_parts: list[str]) -> None:
    """`load_sample_config` must resolve relative runtime.out_dir against the config directory."""

    rel_path = Path(*path_parts)
    rel_out_dir = rel_path.as_posix()

    with TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        defaults_path = _write_default_config(base)
        config_path = base / "config.toml"
        config_path.write_text(
            f"""
[sample]
[sample.runtime]
out_dir = "{rel_out_dir}"
""",
            encoding="utf-8",
        )

        cfg = loading.load_sample_config(config_path, default_config_path=defaults_path)

        expected = (config_path.parent / rel_path).resolve()
        assert cfg.runtime.out_dir == expected
