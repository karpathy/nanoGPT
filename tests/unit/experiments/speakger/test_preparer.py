from __future__ import annotations

import logging
from pathlib import Path

from ml_playground.configuration.models import PreparerConfig
from ml_playground.experiments.speakger.preparer import SpeakGerPreparer, _config_path


def test_speakger_preparer_creates_dataset_dir(tmp_path: Path) -> None:
    """SpeakGerPreparer should ensure dataset directory exists."""
    exp_dir = tmp_path / "speakger"
    exp_dir.mkdir()

    preparer = SpeakGerPreparer()

    cfg = PreparerConfig(
        tokenizer_type="char",
        logger=logging.getLogger(__name__),
        extras={"dataset_dir_override": str(exp_dir)},
    )

    report = preparer.prepare(cfg)

    # Check that dataset directory was created
    ds_dir = exp_dir / "datasets"
    assert ds_dir.exists()
    assert ds_dir.is_dir()

    # Check report
    assert len(report.messages) > 0
    assert any("speakger" in msg for msg in report.messages)
    assert ds_dir in report.skipped_files


def test_speakger_preparer_handles_existing_dir(tmp_path: Path) -> None:
    """SpeakGerPreparer should handle existing dataset directory."""
    exp_dir = tmp_path / "speakger"
    exp_dir.mkdir()
    ds_dir = exp_dir / "datasets"
    ds_dir.mkdir()

    preparer = SpeakGerPreparer()

    cfg = PreparerConfig(
        tokenizer_type="char",
        logger=logging.getLogger(__name__),
        extras={"dataset_dir_override": str(exp_dir)},
    )

    report = preparer.prepare(cfg)

    # Directory should still exist
    assert ds_dir.exists()
    assert ds_dir.is_dir()

    # Should report as skipped
    assert ds_dir in report.skipped_files
    assert len(report.created_files) == 0
    assert len(report.updated_files) == 0


def test_config_path_returns_valid_path() -> None:
    """_config_path should return path to config.toml."""
    path = _config_path()

    assert isinstance(path, Path)
    assert path.name == "config.toml"
    assert "speakger" in str(path)
