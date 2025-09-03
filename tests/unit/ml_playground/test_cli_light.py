from __future__ import annotations

from pathlib import Path

import pytest

import ml_playground.cli as cli


def test_complete_experiments_lists_valid_dirs(tmp_path: Path):
    # Create two experiment dirs; only one has config.toml
    exp1 = tmp_path / "exp_one"
    exp2 = tmp_path / "exp_two"
    exp1.mkdir()
    exp2.mkdir()
    (exp1 / "config.toml").write_text("[train]\n")

    # Monkeypatch root to tmp_path
    def _root() -> Path:
        return tmp_path

    orig = cli._experiments_root
    cli._experiments_root = _root  # type: ignore[assignment]
    try:
        out = cli._complete_experiments(None, "exp_")  # type: ignore[arg-type]
        assert out == ["exp_one"]
    finally:
        cli._experiments_root = orig  # type: ignore[assignment]


def test_complete_experiments_handles_iterdir_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # Ensure root exists but iterdir raises
    def _root() -> Path:
        return tmp_path

    monkeypatch.setattr(cli, "_experiments_root", _root)
    monkeypatch.setattr(
        Path, "iterdir", lambda self: (_ for _ in ()).throw(OSError("boom"))
    )

    out = cli._complete_experiments(None, "")  # type: ignore[arg-type]
    assert out == []


def test_deep_merge_dicts_nested():
    base = {"a": {"b": 1, "c": 2}, "x": 5}
    override = {"a": {"b": 9}, "y": 7}
    merged = cli._deep_merge_dicts(base, override)
    assert merged == {"a": {"b": 9, "c": 2}, "x": 5, "y": 7}


def test_load_train_config_missing_sections(tmp_path: Path):
    toml_text = """
[train]
# Missing required subsections
"""
    p = tmp_path / "cfg.toml"
    p.write_text(toml_text)
    # Defaults are merged, so loader should succeed and fill required sections
    cfg = cli._load_train_config(p)
    assert cfg.model.n_layer > 0
    assert cfg.data.batch_size > 0


def test_load_train_config_unknown_train_data_key(tmp_path: Path):
    toml_text = """
[train.model]
 n_layer=1
 n_head=1
 n_embd=32
 block_size=16

[train.data]
 dataset_dir = "data/x"
 unknown_field = 123

[train.optim]
 learning_rate = 0.001

[train.schedule]

[train.runtime]
 out_dir = "out/x"
"""
    p = tmp_path / "cfg.toml"
    p.write_text(toml_text)
    with pytest.raises(ValueError, match=r"Unknown key\(s\) in \[train\.data\]"):
        cli._load_train_config(p)
