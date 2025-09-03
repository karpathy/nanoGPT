from __future__ import annotations
import pytest
from pathlib import Path
from pytest_mock import MockerFixture
from typer.testing import CliRunner

from ml_playground.cli import app
from ml_playground.config import (
    TrainerConfig,
    SamplerConfig,
    RuntimeConfig,
    SampleConfig,
)
from ml_playground.prepare import PreparerConfig

runner = CliRunner()


def test_main_prepare_shakespeare_success(mocker: MockerFixture) -> None:
    """Test prepare command with shakespeare dataset succeeds."""
    mock_run = mocker.patch("ml_playground.cli._run_prepare")
    mocker.patch(
        "ml_playground.cli.def_load_effective_prepare",
        return_value=(Path("cfg.toml"), PreparerConfig()),
    )
    result = runner.invoke(app, ["prepare", "shakespeare"])
    assert result.exit_code == 0
    mock_run.assert_called_once()


def test_main_prepare_bundestag_char_success(mocker: MockerFixture) -> None:
    """Test prepare command with bundestag_char dataset succeeds."""
    mock_run = mocker.patch("ml_playground.cli._run_prepare")
    mocker.patch(
        "ml_playground.cli.def_load_effective_prepare",
        return_value=(Path("cfg.toml"), PreparerConfig()),
    )
    result = runner.invoke(app, ["prepare", "bundestag_char"])
    assert result.exit_code == 0
    mock_run.assert_called_once()


def test_main_prepare_unknown_dataset_fails(
    mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """Unknown experiment should surface as a CLI error exit."""
    mocker.patch(
        "ml_playground.cli.def_load_effective_prepare",
        side_effect=FileNotFoundError("Config not found"),
    )
    result = runner.invoke(app, ["prepare", "unknown"])
    assert result.exit_code == 1
    assert "Config not found" in result.stdout


def test_main_train_success(tmp_path: Path, mocker: MockerFixture) -> None:
    """Test train command auto-resolves config for experiment and calls train (strict loader)."""
    mock_train_cfg = mocker.Mock(spec=TrainerConfig)
    mock_run = mocker.patch("ml_playground.cli._run_train")
    mocker.patch(
        "ml_playground.cli.def_load_effective_train",
        return_value=(Path("cfg.toml"), mock_train_cfg),
    )
    result = runner.invoke(app, ["train", "shakespeare"])
    assert result.exit_code == 0
    mock_run.assert_called_once()


def test_main_train_no_train_block_fails(
    tmp_path: Path, mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test train command fails when strict loader raises."""
    mocker.patch(
        "ml_playground.cli.def_load_effective_train",
        side_effect=ValueError("Missing train config"),
    )
    result = runner.invoke(app, ["train", "shakespeare"])
    assert result.exit_code == 1
    assert "Missing train config" in result.stdout


def test_main_sample_success(tmp_path: Path, mocker: MockerFixture) -> None:
    """Test sample command auto-resolves config and calls sample function (strict loader)."""
    mock_sample_cfg = SamplerConfig(
        runtime=RuntimeConfig(out_dir=Path("out")),
        sample=SampleConfig(start="x"),
    )
    mock_run = mocker.patch("ml_playground.cli._run_sample")
    mocker.patch(
        "ml_playground.cli.def_load_effective_sample",
        return_value=(Path("cfg.toml"), mock_sample_cfg),
    )
    result = runner.invoke(app, ["sample", "shakespeare"])
    assert result.exit_code == 0
    mock_run.assert_called_once()


def test_main_sample_no_sample_block_fails(
    tmp_path: Path, mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test sample command fails when strict loader raises."""
    mocker.patch(
        "ml_playground.cli.def_load_effective_sample",
        side_effect=ValueError("Missing sample config"),
    )
    result = runner.invoke(app, ["sample", "shakespeare"])
    assert result.exit_code == 1
    assert "Missing sample config" in result.stdout


def test_main_loop_success(tmp_path: Path, mocker: MockerFixture) -> None:
    """Test loop command executes via _run_loop with loaded configs."""
    mock_train_config = mocker.Mock(spec=TrainerConfig)
    mock_sample_config = SamplerConfig(
        runtime=RuntimeConfig(out_dir=Path("out")),
        sample=SampleConfig(start="x"),
    )
    mock_run = mocker.patch("ml_playground.cli._run_loop")
    mocker.patch(
        "ml_playground.cli.def_load_effective_prepare",
        return_value=(Path("cfg.toml"), PreparerConfig()),
    )
    mocker.patch(
        "ml_playground.cli.def_load_effective_train",
        return_value=(Path("cfg.toml"), mock_train_config),
    )
    mocker.patch(
        "ml_playground.cli.def_load_effective_sample",
        return_value=(Path("cfg.toml"), mock_sample_config),
    )

    result = runner.invoke(app, ["loop", "shakespeare"])
    assert result.exit_code == 0
    mock_run.assert_called_once()


def test_main_loop_unknown_dataset_fails(tmp_path: Path, mocker: MockerFixture) -> None:
    """Unknown experiment should bubble up as CLI error exit."""
    mocker.patch(
        "ml_playground.cli.def_load_effective_prepare",
        side_effect=FileNotFoundError("Config not found"),
    )
    result = runner.invoke(app, ["loop", "shakespeare"])
    assert result.exit_code == 1
    assert "Config not found" in result.stdout


def test_main_loop_missing_train_block_fails(
    tmp_path: Path, mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test loop command fails when strict train loader raises."""
    mocker.patch(
        "ml_playground.cli.def_load_effective_prepare",
        return_value=(Path("cfg.toml"), PreparerConfig()),
    )
    mocker.patch(
        "ml_playground.cli.def_load_effective_train",
        side_effect=ValueError("Missing train config"),
    )
    result = runner.invoke(app, ["loop", "shakespeare"])
    assert result.exit_code == 1
    assert "Missing train config" in result.stdout


def test_main_loop_missing_sample_block_fails(
    tmp_path: Path, mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test loop command fails when strict sample loader raises."""
    mocker.patch(
        "ml_playground.cli.def_load_effective_prepare",
        return_value=(Path("cfg.toml"), PreparerConfig()),
    )
    mocker.patch(
        "ml_playground.cli.def_load_effective_train",
        return_value=(Path("cfg.toml"), mocker.Mock(spec=TrainerConfig)),
    )
    mocker.patch(
        "ml_playground.cli.def_load_effective_sample",
        side_effect=ValueError("Missing sample config"),
    )
    result = runner.invoke(app, ["loop", "shakespeare"])
    assert result.exit_code == 1
    assert "Missing sample config" in result.stdout


# Tests for removed functionality (legacy registry usage, direct train/sample calls)
# have been updated to reflect the new CLI architecture.
