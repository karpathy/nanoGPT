from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from ml_playground.config import RuntimeConfig
from ml_playground.config_loader import (
    deep_merge_dicts,
    get_cfg_path,
    read_toml_dict,
    get_default_config_path,
    load_train_config,
    load_sample_config,
    load_sample_config_from_raw,
    load_train_config_from_raw,
    load_prepare_config,
)


def test_deep_merge_basic_and_nested() -> None:
    base = {"a": 1, "b": {"x": 1, "y": 2}, "c": {"z": 3}}
    override = {"b": {"y": 20, "k": 9}, "c": 5, "d": 6}
    merged = deep_merge_dicts(base, override)
    assert merged == {"a": 1, "b": {"x": 1, "y": 20, "k": 9}, "c": 5, "d": 6}


def test_get_cfg_path_with_explicit_and_default(tmp_path: Path) -> None:
    explicit = tmp_path / "conf.toml"
    explicit.write_text("[train]\n[ sample ]\n")
    p1 = get_cfg_path("unused_exp", explicit)
    assert p1 == explicit

    # default path resolution just returns path under package; we only assert it ends as expected
    p2 = get_cfg_path("bundestag_char", None)
    assert p2.name == "config.toml"
    assert "experiments" in str(p2)


def test_read_toml_dict_and_missing(tmp_path: Path) -> None:
    good = tmp_path / "a.toml"
    good.write_text("[x]\ny=1\n")
    d = read_toml_dict(good)
    assert d == {"x": {"y": 1}}

    with pytest.raises(FileNotFoundError):
        read_toml_dict(tmp_path / "missing.toml")


def test_get_default_config_path(tmp_path: Path) -> None:
    exp_dir = tmp_path / "experiments" / "exp1"
    exp_dir.mkdir(parents=True)
    cp = exp_dir / "config.toml"
    cp.write_text("")
    default = get_default_config_path(cp)
    # Expected at experiments/../default_config.toml
    assert default.parent == exp_dir.parent
    assert default.name == "default_config.toml"


def test_load_train_config_merges_defaults_and_sets_provenance(tmp_path: Path) -> None:
    # Create default_config.toml one level above experiments/<exp>/config.toml
    root = tmp_path
    experiments = root / "experiments" / "exp"
    experiments.mkdir(parents=True)
    # Place defaults next to the experiments/ directory (expected by loader)
    defaults = experiments.parent / "default_config.toml"
    defaults.write_text(
        textwrap.dedent(
            """
            [train.runtime]
            out_dir = "./out"
            eval_interval = 5

            [train.model]
            n_layer = 1
            n_head = 1
            n_embd = 8
            block_size = 4

            [train.data]
            dataset_dir = "./data"
            batch_size = 2
            block_size = 4
            grad_accum_steps = 1

            [train.optim]
            learning_rate = 0.001

            [train.schedule]
            decay_lr = false
            warmup_iters = 0
            lr_decay_iters = 10
            min_lr = 1e-5
            """
        ).strip()
    )
    exp_cfg = experiments / "config.toml"
    exp_cfg.write_text(
        textwrap.dedent(
            """
            [train.runtime]
            log_interval = 7

            [train.optim]
            weight_decay = 0.01
            """
        ).strip()
    )

    cfg = load_train_config(exp_cfg)
    # Overrides applied
    assert cfg.runtime.log_interval == 7
    assert cfg.optim.weight_decay == 0.01
    # Defaults preserved
    assert cfg.runtime.eval_interval == 5
    # Provenance recorded
    assert cfg.extras["provenance"]["context"]["config_path"].endswith("config.toml")


def test_load_sample_config_requires_section_and_runtime_ref_merge(
    tmp_path: Path,
) -> None:
    root = tmp_path
    experiments = root / "experiments" / "exp"
    experiments.mkdir(parents=True)
    # Place defaults next to the experiments/ directory (expected by loader)
    defaults = experiments.parent / "default_config.toml"
    defaults.write_text(
        textwrap.dedent(
            """
            [train.runtime]
            out_dir = "./out"
            device = "cpu"
            dtype = "float32"

            [sample]
            runtime_ref = "train.runtime"
            [sample.sample]
            max_new_tokens = 7
            """
        ).strip()
    )

    # Missing [sample] in experiment -> use defaults only; still valid
    exp_cfg = experiments / "config.toml"
    exp_cfg.write_text(
        textwrap.dedent("""
        [train.runtime]
        device = "cpu"
        """)
    )

    # Should load and use defaults.sample
    s = load_sample_config(exp_cfg)
    assert s.sample.max_new_tokens == 7

    # Now explicit [sample] with runtime_ref and overrides
    exp_cfg.write_text(
        textwrap.dedent("""
        [train.runtime]
        out_dir = "./exp_out"
        device = "cpu"

        [sample]
        runtime_ref = "train.runtime"
        [sample.runtime]
        tensorboard_enabled = false
        [sample.sample]
        max_new_tokens = 9
    """)
    )
    s2 = load_sample_config(exp_cfg)
    assert isinstance(s2.runtime, RuntimeConfig)
    # runtime = train.runtime merged with sample.runtime overrides
    assert s2.runtime.out_dir.as_posix().endswith("exp_out")
    assert s2.runtime.tensorboard_enabled is False
    assert s2.sample.max_new_tokens == 9


def test_load_sample_config_from_raw_and_errors() -> None:
    raw = {
        "train": {"runtime": {"out_dir": "./out", "device": "cpu"}},
        "sample": {
            "runtime_ref": "train.runtime",
            "runtime": {"tensorboard_enabled": False},
            "sample": {"max_new_tokens": 11},
        },
    }
    cfg = load_sample_config_from_raw(raw, defaults_raw={})
    assert cfg.runtime and cfg.runtime.tensorboard_enabled is False
    assert cfg.sample.max_new_tokens == 11

    # Error path: missing [train] section referenced by runtime_ref should simply not merge, not raise
    raw2 = {"sample": {"runtime_ref": "train.runtime", "sample": {}}}
    cfg2 = load_sample_config_from_raw(raw2, defaults_raw={})
    # runtime remains None because no train.runtime exists and no direct runtime provided
    assert cfg2.runtime is None


def test_runtime_ref_merge_overrides_and_preserves_out_dir_relative() -> None:
    # Consolidated from test_config_loader_strict.py
    raw = {
        "train": {"runtime": {"out_dir": "./out_train", "seed": 123}},
        "sample": {
            "runtime_ref": "train.runtime",
            # Override seed in sample.runtime; out_dir should be preserved from train.runtime
            "runtime": {"seed": 999},
            "sample": {"start": "hi", "max_new_tokens": 1},
        },
    }
    cfg = load_sample_config_from_raw(raw, defaults_raw={})
    rt = cfg.runtime
    assert rt is not None
    assert rt.seed == 999  # override applied
    # Ensure path remains relative and retains trailing directory name
    assert not rt.out_dir.is_absolute()
    assert rt.out_dir.name == "out_train"


def test_loader_preserves_relative_out_dir_string_exactly() -> None:
    # Consolidated from test_config_loader_strict.py
    raw = {
        "sample": {
            "runtime": {
                "out_dir": "relative/out",
                "device": "cpu",
                "dtype": "float32",
                "seed": 1,
            },
            "sample": {"start": "hi", "max_new_tokens": 1},
        }
    }
    cfg = load_sample_config_from_raw(raw, defaults_raw={})
    assert cfg.runtime is not None
    # ensure the string is preserved exactly (no absolute resolution)
    assert str(cfg.runtime.out_dir) == "relative/out"


def test_load_train_config_from_raw_errors_when_missing_train() -> None:
    with pytest.raises(ValueError):
        load_train_config_from_raw({}, {})


def test_load_prepare_config_and_missing(tmp_path: Path) -> None:
    p = tmp_path / "c.toml"
    # missing [prepare]
    p.write_text("[other]\n")
    with pytest.raises(ValueError):
        load_prepare_config(p)

    # minimal valid prepare
    p.write_text(
        textwrap.dedent("""
        [prepare]
        dataset_dir = "./d"
    """)
    )
    cfg = load_prepare_config(p)
    assert cfg.dataset_dir is not None
