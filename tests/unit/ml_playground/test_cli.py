from __future__ import annotations
import pytest
from pathlib import Path
from pytest_mock import MockerFixture
from ml_playground.cli import main
from ml_playground.config import TrainerConfig, SamplerConfig
from ml_playground.cli import _load_train_config, _load_sample_config


def test_main_prepare_shakespeare_success(mocker: MockerFixture) -> None:
    """Test prepare command with shakespeare dataset succeeds."""
    mock_instance = mocker.Mock()
    mocker.patch("ml_playground.cli.make_preparer", return_value=mock_instance)
    # Registry membership validates the experiment name
    mocker.patch("ml_playground.datasets.PREPARERS", {"shakespeare": object()})
    main(["prepare", "shakespeare"])
    mock_instance.assert_called_once()


def test_main_prepare_bundestag_char_success(mocker: MockerFixture) -> None:
    """Test prepare command with bundestag_char dataset succeeds."""
    mock_instance = mocker.Mock()
    mocker.patch("ml_playground.cli.make_preparer", return_value=mock_instance)
    mocker.patch("ml_playground.datasets.PREPARERS", {"bundestag_char": object()})
    main(["prepare", "bundestag_char"])
    mock_instance.assert_called_once()


def test_main_prepare_unknown_dataset_fails(mocker: MockerFixture) -> None:
    """Test prepare command with an unknown experiment raises SystemExit."""
    with pytest.raises(SystemExit, match="Unknown experiment: unknown"):
        main(["prepare", "unknown"])


def test_main_train_success(tmp_path: Path, mocker: MockerFixture) -> None:
    """Test train command auto-resolves config for experiment and calls train (strict loader)."""
    mock_train_cfg = mocker.Mock(spec=TrainerConfig)

    mock_load = mocker.patch(
        "ml_playground.cli.load_train_config", return_value=mock_train_cfg
    )
    mock_train = mocker.patch("ml_playground.cli.train")
    main(["train", "shakespeare"])

    mock_load.assert_called_once()
    mock_train.assert_called_once_with(mock_train_cfg)


def test_main_train_no_train_block_fails(tmp_path: Path, mocker: MockerFixture) -> None:
    """Test train command fails when strict loader raises."""
    mocker.patch(
        "ml_playground.cli._load_train_config_from_raw",
        side_effect=Exception("Config must contain [train] block"),
    )
    with pytest.raises(SystemExit, match="Config must contain \\[train\\] block"):
        main(["train", "shakespeare"])


def test_main_sample_success(tmp_path: Path, mocker: MockerFixture) -> None:
    """Test sample command auto-resolves config and calls sample function (strict loader)."""
    mock_sample_cfg = mocker.Mock(spec=SamplerConfig)

    mock_load = mocker.patch(
        "ml_playground.cli._load_sample_config_from_raw", return_value=mock_sample_cfg
    )
    mock_sample = mocker.patch("ml_playground.cli.sample")
    main(["sample", "shakespeare"])

    mock_load.assert_called_once()
    mock_sample.assert_called_once_with(mock_sample_cfg)


def test_main_sample_no_sample_block_fails(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    """Test sample command fails when strict loader raises."""
    mocker.patch(
        "ml_playground.cli._load_sample_config_from_raw",
        side_effect=Exception("Config must contain [sample] block"),
    )
    with pytest.raises(SystemExit, match="Config must contain \\[sample\\] block"):
        main(["sample", "shakespeare"])


def test_main_loop_success(tmp_path: Path, mocker: MockerFixture) -> None:
    """Test loop command executes prepare, train, and sample successfully (strict loaders)."""
    mock_preparer = mocker.Mock()
    mock_train_config = mocker.Mock(spec=TrainerConfig)
    mock_sample_config = mocker.Mock(spec=SamplerConfig)

    # Create mock data config with meta_pkl
    mock_data_config = mocker.Mock()
    mock_data_config.meta_pkl = "meta.pkl"
    mock_data_config.dataset_dir = tmp_path / "dataset"
    mock_train_config.data = mock_data_config

    # Create mock runtime config
    mock_runtime_config = mocker.Mock()
    mock_runtime_config.out_dir = tmp_path / "out"
    mock_train_config.runtime = mock_runtime_config

    # Create source meta.pkl file
    src_meta = mock_data_config.dataset_dir / "meta.pkl"
    src_meta.parent.mkdir(exist_ok=True)
    src_meta.write_text("mock meta data")

    mocker.patch("ml_playground.datasets.PREPARERS", {"shakespeare": mock_preparer})
    mocker.patch("ml_playground.cli.load_train_config", return_value=mock_train_config)
    mocker.patch(
        "ml_playground.cli.load_sample_config", return_value=mock_sample_config
    )
    mock_train = mocker.patch("ml_playground.cli.train")
    mock_sample = mocker.patch("ml_playground.cli.sample")
    mock_copy = mocker.patch("shutil.copy2")

    main(["loop", "shakespeare"])

    mock_preparer.assert_called_once()
    mock_train.assert_called_once_with(mock_train_config)
    mock_sample.assert_called_once_with(mock_sample_config)
    mock_copy.assert_called_once()


def test_main_loop_unknown_dataset_fails(tmp_path: Path, mocker: MockerFixture) -> None:
    """Test loop command fails with valid choice but missing preparer."""
    mocker.patch("ml_playground.datasets.PREPARERS", {})
    with pytest.raises(SystemExit, match="Unknown experiment: shakespeare"):
        main(["loop", "shakespeare"])


def test_main_loop_missing_train_block_fails(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    """Test loop command fails when strict train loader raises."""
    mock_preparer = mocker.Mock()

    mocker.patch("ml_playground.datasets.PREPARERS", {"shakespeare": mock_preparer})
    mocker.patch(
        "ml_playground.cli.load_train_config", side_effect=Exception("bad train")
    )
    with pytest.raises(SystemExit):
        main(["loop", "shakespeare"])


def test_main_loop_missing_sample_block_fails(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    """Test loop command fails when strict sample loader raises."""
    mock_preparer = mocker.Mock()

    mocker.patch("ml_playground.datasets.PREPARERS", {"shakespeare": mock_preparer})
    mocker.patch(
        "ml_playground.cli.load_train_config",
        return_value=mocker.Mock(spec=TrainerConfig),
    )
    mocker.patch(
        "ml_playground.cli.load_sample_config", side_effect=Exception("bad sample")
    )
    with pytest.raises(SystemExit):
        main(["loop", "shakespeare"])

# Tests for removed functionality (meta.pkl copying, manual dispatcher) have been removed
# as they are no longer relevant after the refactoring
