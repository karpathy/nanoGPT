from __future__ import annotations

import logging
from pathlib import Path

from ml_playground.configuration.models import PreparerConfig
from ml_playground.experiments.bundestag_tiktoken.preparer import (
    BundestagTiktokenPreparer,
)


def test_bundestag_tiktoken_preparer_creates_dataset(tmp_path: Path) -> None:
    """BundestagTiktokenPreparer should create train/val/meta files."""
    # Create a temporary experiment directory structure
    exp_dir = tmp_path / "bundestag_tiktoken"
    exp_dir.mkdir()
    ds_dir = exp_dir / "datasets"
    ds_dir.mkdir()

    # Create input file
    input_file = ds_dir / "input.txt"
    input_file.write_text(
        "Hello world. This is test data for tokenization.", encoding="utf-8"
    )

    # Mock the preparer to use our temp directory
    preparer = BundestagTiktokenPreparer()

    # Patch __file__ to point to our temp directory
    import ml_playground.experiments.bundestag_tiktoken.preparer as preparer_module

    original_file = preparer_module.__file__
    preparer_module.__file__ = str(exp_dir / "preparer.py")

    try:
        cfg = PreparerConfig(
            tokenizer_type="tiktoken",
            logger=logging.getLogger(__name__),
        )

        report = preparer.prepare(cfg)

        # Check that files were created
        assert (ds_dir / "train.bin").exists()
        assert (ds_dir / "val.bin").exists()
        assert (ds_dir / "meta.pkl").exists()

        # Check report
        assert len(report.messages) > 0
        assert any("bundestag_tiktoken" in msg for msg in report.messages)

    finally:
        preparer_module.__file__ = original_file


def test_bundestag_tiktoken_preparer_handles_existing_files(tmp_path: Path) -> None:
    """BundestagTiktokenPreparer should handle existing output files."""
    import pickle

    exp_dir = tmp_path / "bundestag_tiktoken"
    exp_dir.mkdir()
    ds_dir = exp_dir / "datasets"
    ds_dir.mkdir()

    input_file = ds_dir / "input.txt"
    input_file.write_text("Sample text for testing.", encoding="utf-8")

    # Pre-create output files with valid data
    (ds_dir / "train.bin").write_bytes(b"old")
    (ds_dir / "val.bin").write_bytes(b"old")
    with (ds_dir / "meta.pkl").open("wb") as f:
        pickle.dump({"meta_version": 1}, f)

    preparer = BundestagTiktokenPreparer()

    import ml_playground.experiments.bundestag_tiktoken.preparer as preparer_module

    original_file = preparer_module.__file__
    preparer_module.__file__ = str(exp_dir / "preparer.py")

    try:
        cfg = PreparerConfig(
            tokenizer_type="tiktoken",
            logger=logging.getLogger(__name__),
        )

        report = preparer.prepare(cfg)

        # Files should be updated
        assert (ds_dir / "train.bin").exists()
        assert (ds_dir / "val.bin").exists()
        assert (ds_dir / "meta.pkl").exists()

        # Check that report indicates updates or skips
        assert (
            len(report.updated_files) > 0
            or len(report.created_files) > 0
            or len(report.skipped_files) > 0
        )

    finally:
        preparer_module.__file__ = original_file


def test_bundestag_tiktoken_preparer_with_bundled_input(tmp_path: Path) -> None:
    """BundestagTiktokenPreparer should use bundled input.txt if available."""
    exp_dir = tmp_path / "bundestag_tiktoken"
    exp_dir.mkdir()
    ds_dir = exp_dir / "datasets"
    ds_dir.mkdir()

    # Create bundled input.txt in experiment directory
    bundled_input = exp_dir / "input.txt"
    bundled_input.write_text("Bundled input text for testing.", encoding="utf-8")

    preparer = BundestagTiktokenPreparer()

    import ml_playground.experiments.bundestag_tiktoken.preparer as preparer_module

    original_file = preparer_module.__file__
    preparer_module.__file__ = str(exp_dir / "preparer.py")

    try:
        cfg = PreparerConfig(
            tokenizer_type="tiktoken",
            logger=logging.getLogger(__name__),
        )

        preparer.prepare(cfg)

        # Should successfully create dataset from bundled input
        assert (ds_dir / "train.bin").exists()
        assert (ds_dir / "val.bin").exists()
        assert (ds_dir / "meta.pkl").exists()

    finally:
        preparer_module.__file__ = original_file
