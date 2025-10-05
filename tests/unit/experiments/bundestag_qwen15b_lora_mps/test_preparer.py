from __future__ import annotations

import logging
from pathlib import Path

from ml_playground.configuration.models import PreparerConfig
from ml_playground.experiments.bundestag_qwen15b_lora_mps.preparer import (
    BundestagQwen15bLoraMpsPreparer,
)


def test_bundestag_qwen15b_preparer_creates_dataset_dir(tmp_path: Path) -> None:
    """BundestagQwen15bLoraMpsPreparer should ensure dataset directory exists."""
    exp_dir = tmp_path / "bundestag_qwen15b_lora_mps"
    exp_dir.mkdir()

    preparer = BundestagQwen15bLoraMpsPreparer()

    import ml_playground.experiments.bundestag_qwen15b_lora_mps.preparer as preparer_module

    original_file = preparer_module.__file__
    preparer_module.__file__ = str(exp_dir / "preparer.py")

    try:
        cfg = PreparerConfig(
            tokenizer_type="tiktoken",
            logger=logging.getLogger(__name__),
        )

        report = preparer.prepare(cfg)

        # Check that dataset directory was created
        ds_dir = exp_dir / "datasets"
        assert ds_dir.exists()
        assert ds_dir.is_dir()

        # Check report
        assert len(report.messages) > 0
        assert any("bundestag_qwen15b_lora_mps" in msg for msg in report.messages)
        assert len(report.created_files) > 0 or len(report.skipped_files) > 0

    finally:
        preparer_module.__file__ = original_file


def test_bundestag_qwen15b_preparer_handles_existing_dir(tmp_path: Path) -> None:
    """BundestagQwen15bLoraMpsPreparer should handle existing dataset directory."""
    exp_dir = tmp_path / "bundestag_qwen15b_lora_mps"
    exp_dir.mkdir()
    ds_dir = exp_dir / "datasets"
    ds_dir.mkdir()

    preparer = BundestagQwen15bLoraMpsPreparer()

    import ml_playground.experiments.bundestag_qwen15b_lora_mps.preparer as preparer_module

    original_file = preparer_module.__file__
    preparer_module.__file__ = str(exp_dir / "preparer.py")

    try:
        cfg = PreparerConfig(
            tokenizer_type="tiktoken",
            logger=logging.getLogger(__name__),
        )

        report = preparer.prepare(cfg)

        # Directory should still exist
        assert ds_dir.exists()
        assert ds_dir.is_dir()

        # Should report as skipped since it already existed
        assert len(report.skipped_files) > 0

    finally:
        preparer_module.__file__ = original_file


def test_bundestag_qwen15b_preparer_snapshot_handles_oserror() -> None:
    """_snapshot should handle OSError when accessing file stats."""
    from ml_playground.experiments.bundestag_qwen15b_lora_mps.preparer import _snapshot
    from pathlib import Path

    # Create a path that doesn't exist
    nonexistent = Path("/nonexistent/path/file.txt")

    result = _snapshot([nonexistent])

    # Should return False for existence
    assert nonexistent in result
    assert result[nonexistent] == (False, 0.0, 0)


def test_bundestag_qwen15b_preparer_diff_handles_oserror() -> None:
    """_diff should handle OSError when checking file existence."""
    from ml_playground.experiments.bundestag_qwen15b_lora_mps.preparer import _diff
    from pathlib import Path

    # Create a path that might cause OSError
    problematic_path = Path("/nonexistent/path/file.txt")
    before = {problematic_path: (False, 0.0, 0)}

    created, updated, skipped = _diff([problematic_path], before)

    # Should handle gracefully
    assert isinstance(created, list)
    assert isinstance(updated, list)
    assert isinstance(skipped, list)


def test_bundestag_qwen15b_preparer_detects_file_updates(tmp_path: Path) -> None:
    """BundestagQwen15bLoraMpsPreparer should detect when files are updated."""
    from ml_playground.experiments.bundestag_qwen15b_lora_mps.preparer import (
        _snapshot,
        _diff,
    )

    # Create a file
    test_file = tmp_path / "test.txt"
    test_file.write_text("original")

    # Take snapshot
    before = _snapshot([test_file])

    # Modify the file
    import time

    time.sleep(0.01)  # Ensure mtime changes
    test_file.write_text("modified content")

    # Check diff
    created, updated, skipped = _diff([test_file], before)

    # Should detect as updated
    assert test_file in updated or test_file in created
