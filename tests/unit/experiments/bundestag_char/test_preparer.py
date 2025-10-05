from __future__ import annotations

import logging
from pathlib import Path

import pytest

from ml_playground.configuration.models import PreparerConfig
from ml_playground.experiments.bundestag_char.preparer import (
    BundestagCharPreparer,
    _artifacts_look_valid,
)


def test_bundestag_char_preparer_creates_dataset(tmp_path: Path) -> None:
    """BundestagCharPreparer should create train/val/meta files."""
    exp_dir = tmp_path / "bundestag_char"
    exp_dir.mkdir()
    ds_dir = exp_dir / "datasets"
    ds_dir.mkdir()

    # Create input file
    input_file = ds_dir / "input.txt"
    input_file.write_text("Hello world. This is test data.", encoding="utf-8")

    preparer = BundestagCharPreparer()

    import ml_playground.experiments.bundestag_char.preparer as preparer_module

    original_file = preparer_module.__file__
    preparer_module.__file__ = str(exp_dir / "preparer.py")

    try:
        cfg = PreparerConfig(
            tokenizer_type="char",
            logger=logging.getLogger(__name__),
        )

        report = preparer.prepare(cfg)

        # Check that files were created
        assert (ds_dir / "train.bin").exists()
        assert (ds_dir / "val.bin").exists()
        assert (ds_dir / "meta.pkl").exists()

        # Check report
        assert len(report.messages) > 0
        assert any("bundestag_char" in msg for msg in report.messages)

    finally:
        preparer_module.__file__ = original_file


def test_bundestag_char_preparer_skips_if_valid(tmp_path: Path) -> None:
    """BundestagCharPreparer should skip if valid artifacts exist."""
    exp_dir = tmp_path / "bundestag_char"
    exp_dir.mkdir()
    ds_dir = exp_dir / "datasets"
    ds_dir.mkdir()

    # Create valid artifacts
    (ds_dir / "train.bin").write_bytes(b"train data")
    (ds_dir / "val.bin").write_bytes(b"val data")
    (ds_dir / "meta.pkl").write_bytes(b"meta data")

    preparer = BundestagCharPreparer()

    import ml_playground.experiments.bundestag_char.preparer as preparer_module

    original_file = preparer_module.__file__
    preparer_module.__file__ = str(exp_dir / "preparer.py")

    try:
        cfg = PreparerConfig(
            tokenizer_type="char",
            logger=logging.getLogger(__name__),
        )

        report = preparer.prepare(cfg)

        # Should skip
        assert len(report.skipped_files) > 0
        assert len(report.created_files) == 0
        assert any("skipping" in msg for msg in report.messages)

    finally:
        preparer_module.__file__ = original_file


def test_bundestag_char_preparer_with_dataset_dir_override(tmp_path: Path) -> None:
    """BundestagCharPreparer should respect dataset_dir_override."""
    override_dir = tmp_path / "override"
    override_dir.mkdir()
    ds_dir = override_dir / "datasets"
    ds_dir.mkdir()

    input_file = ds_dir / "input.txt"
    input_file.write_text("Test data for override.", encoding="utf-8")

    preparer = BundestagCharPreparer()

    # Create config with extras using model_copy
    cfg = PreparerConfig(
        tokenizer_type="char",
        logger=logging.getLogger(__name__),
        extras={"dataset_dir_override": str(override_dir)},
    )

    preparer.prepare(cfg)

    # Should create files in override directory
    assert (ds_dir / "train.bin").exists()
    assert (ds_dir / "val.bin").exists()
    assert (ds_dir / "meta.pkl").exists()


def test_bundestag_char_preparer_raises_on_wrong_tokenizer(tmp_path: Path) -> None:
    """BundestagCharPreparer should raise ValueError for non-char tokenizer."""
    exp_dir = tmp_path / "bundestag_char"
    exp_dir.mkdir()
    ds_dir = exp_dir / "datasets"
    ds_dir.mkdir()

    input_file = ds_dir / "input.txt"
    input_file.write_text("Test data.", encoding="utf-8")

    preparer = BundestagCharPreparer()

    import ml_playground.experiments.bundestag_char.preparer as preparer_module

    original_file = preparer_module.__file__
    preparer_module.__file__ = str(exp_dir / "preparer.py")

    try:
        cfg = PreparerConfig(
            tokenizer_type="tiktoken",  # Wrong tokenizer
            logger=logging.getLogger(__name__),
        )

        with pytest.raises(ValueError, match="only supports char tokenizer"):
            preparer.prepare(cfg)

    finally:
        preparer_module.__file__ = original_file


def test_artifacts_look_valid_returns_true_for_valid_files(tmp_path: Path) -> None:
    """_artifacts_look_valid should return True for valid files."""
    file1 = tmp_path / "file1.bin"
    file2 = tmp_path / "file2.bin"
    file1.write_bytes(b"data")
    file2.write_bytes(b"more data")

    assert _artifacts_look_valid([file1, file2]) is True


def test_artifacts_look_valid_returns_false_for_missing_file(tmp_path: Path) -> None:
    """_artifacts_look_valid should return False if any file is missing."""
    file1 = tmp_path / "file1.bin"
    file2 = tmp_path / "missing.bin"
    file1.write_bytes(b"data")

    assert _artifacts_look_valid([file1, file2]) is False


def test_artifacts_look_valid_returns_false_for_empty_file(tmp_path: Path) -> None:
    """_artifacts_look_valid should return False if any file is empty."""
    file1 = tmp_path / "file1.bin"
    file2 = tmp_path / "empty.bin"
    file1.write_bytes(b"data")
    file2.write_bytes(b"")  # Empty file

    assert _artifacts_look_valid([file1, file2]) is False
