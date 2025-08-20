from __future__ import annotations
import pytest
from pathlib import Path
from pytest_mock import MockerFixture
from ml_playground.cli import main
from ml_playground.config import AppConfig, TrainExperiment, SampleExperiment


def test_main_prepare_shakespeare_success(mocker: MockerFixture) -> None:
    """Test prepare command with shakespeare dataset succeeds."""
    mock_preparer = mocker.Mock()
    mocker.patch("ml_playground.datasets.PREPARERS", {"shakespeare": mock_preparer})
    main(["prepare", "shakespeare"])
    mock_preparer.assert_called_once()


def test_main_prepare_bundestag_char_success(mocker: MockerFixture) -> None:
    """Test prepare command with bundestag_char dataset succeeds."""
    mock_preparer = mocker.Mock()
    mocker.patch("ml_playground.datasets.PREPARERS", {"bundestag_char": mock_preparer})
    main(["prepare", "bundestag_char"])
    mock_preparer.assert_called_once()


def test_main_prepare_unknown_dataset_fails(mocker: MockerFixture) -> None:
    """Test prepare command with valid choice but missing preparer raises SystemExit."""
    # Use a valid argparse choice but empty PREPARERS to test our custom logic
    mocker.patch("ml_playground.datasets.PREPARERS", {})
    with pytest.raises(SystemExit, match="Unknown experiment: shakespeare"):
        main(["prepare", "shakespeare"])


def test_main_train_success(tmp_path: Path, mocker: MockerFixture) -> None:
    """Test train command auto-resolves config for experiment and calls train."""
    mock_config = AppConfig(train=mocker.Mock(spec=TrainExperiment))

    mock_load = mocker.patch("ml_playground.cli.load_toml", return_value=mock_config)
    mock_train = mocker.patch("ml_playground.cli.train")
    main(["train", "shakespeare"]) 

    mock_load.assert_called_once()
    mock_train.assert_called_once_with(mock_config.train)


def test_main_train_no_train_block_fails(tmp_path: Path, mocker: MockerFixture) -> None:
    """Test train command fails when config has no train block."""
    config_path = tmp_path / "config.toml"
    mock_config = AppConfig(train=None)

    mocker.patch("ml_playground.cli.load_toml", return_value=mock_config)
    with pytest.raises(SystemExit, match="Config must contain \\[train\\] block"):
        main(["train", "shakespeare"])


def test_main_sample_success(tmp_path: Path, mocker: MockerFixture) -> None:
    """Test sample command auto-resolves config and calls sample function."""
    mock_config = AppConfig(sample=mocker.Mock(spec=SampleExperiment))

    mock_load = mocker.patch("ml_playground.cli.load_toml", return_value=mock_config)
    mock_sample = mocker.patch("ml_playground.cli.sample")
    main(["sample", "shakespeare"]) 

    mock_load.assert_called_once()
    mock_sample.assert_called_once_with(mock_config.sample)


def test_main_sample_no_sample_block_fails(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    """Test sample command fails when config has no sample block."""
    config_path = tmp_path / "config.toml"
    mock_config = AppConfig(sample=None)

    mocker.patch("ml_playground.cli.load_toml", return_value=mock_config)
    with pytest.raises(SystemExit, match="Config must contain \\[sample\\] block"):
        main(["sample", "shakespeare"])


def test_main_loop_success(tmp_path: Path, mocker: MockerFixture) -> None:
    """Test loop command executes prepare, train, and sample successfully."""
    config_path = tmp_path / "config.toml"
    mock_preparer = mocker.Mock()
    mock_train_config = mocker.Mock(spec=TrainExperiment)
    mock_sample_config = mocker.Mock(spec=SampleExperiment)

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

    mock_config = AppConfig(train=mock_train_config, sample=mock_sample_config)

    mocker.patch("ml_playground.datasets.PREPARERS", {"shakespeare": mock_preparer})
    mocker.patch("ml_playground.cli.load_toml", return_value=mock_config)
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
    config_path = tmp_path / "config.toml"

    mocker.patch("ml_playground.datasets.PREPARERS", {})
    with pytest.raises(SystemExit, match="Unknown experiment: shakespeare"):
        main(["loop", "shakespeare"])


def test_main_loop_missing_train_block_fails(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    """Test loop command fails when config missing train block."""
    config_path = tmp_path / "config.toml"
    mock_preparer = mocker.Mock()
    mock_config = AppConfig(train=None, sample=mocker.Mock(spec=SampleExperiment))

    mocker.patch("ml_playground.datasets.PREPARERS", {"shakespeare": mock_preparer})
    mocker.patch("ml_playground.cli.load_toml", return_value=mock_config)
    with pytest.raises(SystemExit, match="Config for loop must contain both"):
        main(["loop", "shakespeare"])


def test_main_loop_missing_sample_block_fails(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    """Test loop command fails when config missing sample block."""
    config_path = tmp_path / "config.toml"
    mock_preparer = mocker.Mock()
    mock_config = AppConfig(train=mocker.Mock(spec=TrainExperiment), sample=None)

    mocker.patch("ml_playground.datasets.PREPARERS", {"shakespeare": mock_preparer})
    mocker.patch("ml_playground.cli.load_toml", return_value=mock_config)
    with pytest.raises(SystemExit, match="Config for loop must contain both"):
        main(["loop", "shakespeare"])


def test_main_loop_meta_copy_exception_handled(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    """Test loop command handles meta.pkl copy exceptions gracefully."""
    config_path = tmp_path / "config.toml"
    mock_preparer = mocker.Mock()
    mock_train_config = mocker.Mock(spec=TrainExperiment)
    mock_sample_config = mocker.Mock(spec=SampleExperiment)

    # Mock data config with meta_pkl
    mock_data_config = mocker.Mock()
    mock_data_config.meta_pkl = "meta.pkl"
    mock_data_config.dataset_dir = tmp_path / "dataset"
    mock_train_config.data = mock_data_config

    # Create mock runtime config
    mock_runtime_config = mocker.Mock()
    mock_runtime_config.out_dir = tmp_path / "out"
    mock_train_config.runtime = mock_runtime_config

    # Create source meta.pkl file so src_meta.exists() returns True
    src_meta = mock_data_config.dataset_dir / "meta.pkl"
    src_meta.parent.mkdir(exist_ok=True)
    src_meta.write_text("mock meta data")

    mock_config = AppConfig(train=mock_train_config, sample=mock_sample_config)

    mocker.patch("ml_playground.datasets.PREPARERS", {"shakespeare": mock_preparer})
    mocker.patch("ml_playground.cli.load_toml", return_value=mock_config)
    mock_train = mocker.patch("ml_playground.cli.train")
    mock_sample = mocker.patch("ml_playground.cli.sample")
    mocker.patch("shutil.copy2", side_effect=Exception("Copy failed"))
    mock_print = mocker.patch("builtins.print")

    main(["loop", "shakespeare"])

    mock_preparer.assert_called_once()
    mock_train.assert_called_once_with(mock_train_config)
    mock_sample.assert_called_once_with(mock_sample_config)
    # Should print warning about meta.pkl copy failure
    mock_print.assert_called()


def test_main_loop_no_meta_pkl_skips_copy(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    """Test loop command skips meta.pkl copy when meta_pkl is None."""
    config_path = tmp_path / "config.toml"
    mock_preparer = mocker.Mock()
    mock_train_config = mocker.Mock(spec=TrainExperiment)
    mock_sample_config = mocker.Mock(spec=SampleExperiment)

    # Mock data config with no meta_pkl
    mock_data_config = mocker.Mock()
    mock_data_config.meta_pkl = None
    mock_train_config.data = mock_data_config

    mock_config = AppConfig(train=mock_train_config, sample=mock_sample_config)

    mocker.patch("ml_playground.datasets.PREPARERS", {"shakespeare": mock_preparer})
    mocker.patch("ml_playground.cli.load_toml", return_value=mock_config)
    mock_train = mocker.patch("ml_playground.cli.train")
    mock_sample = mocker.patch("ml_playground.cli.sample")
    mock_copy = mocker.patch("shutil.copy2")

    main(["loop", "shakespeare"])

    mock_preparer.assert_called_once()
    mock_train.assert_called_once_with(mock_train_config)
    mock_sample.assert_called_once_with(mock_sample_config)
    # Should not attempt to copy meta.pkl
    mock_copy.assert_not_called()
