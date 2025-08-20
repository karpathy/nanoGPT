from __future__ import annotations
import pytest
from pathlib import Path
from pytest_mock import MockerFixture
from ml_playground.cli import main
from ml_playground.config import TrainExperiment, SampleExperiment
from ml_playground.cli import load_train_config, load_sample_config


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
    """Test train command auto-resolves config for experiment and calls train (strict loader)."""
    mock_train_cfg = mocker.Mock(spec=TrainExperiment)

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
        "ml_playground.cli.load_train_config",
        side_effect=Exception("Config must contain [train] block"),
    )
    with pytest.raises(SystemExit, match="Config must contain \\[train\\] block"):
        main(["train", "shakespeare"])


def test_main_sample_success(tmp_path: Path, mocker: MockerFixture) -> None:
    """Test sample command auto-resolves config and calls sample function (strict loader)."""
    mock_sample_cfg = mocker.Mock(spec=SampleExperiment)

    mock_load = mocker.patch(
        "ml_playground.cli.load_sample_config", return_value=mock_sample_cfg
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
        "ml_playground.cli.load_sample_config",
        side_effect=Exception("Config must contain [sample] block"),
    )
    with pytest.raises(SystemExit, match="Config must contain \\[sample\\] block"):
        main(["sample", "shakespeare"])


def test_main_loop_success(tmp_path: Path, mocker: MockerFixture) -> None:
    """Test loop command executes prepare, train, and sample successfully (strict loaders)."""
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
        return_value=mocker.Mock(spec=TrainExperiment),
    )
    mocker.patch(
        "ml_playground.cli.load_sample_config", side_effect=Exception("bad sample")
    )
    with pytest.raises(SystemExit):
        main(["loop", "shakespeare"])


def test_main_loop_meta_copy_exception_handled(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    """Test loop command handles meta.pkl copy exceptions gracefully."""
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

    mocker.patch("ml_playground.datasets.PREPARERS", {"shakespeare": mock_preparer})
    mocker.patch("ml_playground.cli.load_train_config", return_value=mock_train_config)
    mocker.patch(
        "ml_playground.cli.load_sample_config", return_value=mock_sample_config
    )
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
    mock_preparer = mocker.Mock()
    mock_train_config = mocker.Mock(spec=TrainExperiment)
    mock_sample_config = mocker.Mock(spec=SampleExperiment)

    # Mock data config with no meta_pkl
    mock_data_config = mocker.Mock()
    mock_data_config.meta_pkl = None
    mock_train_config.data = mock_data_config

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
    # Should not attempt to copy meta.pkl
    mock_copy.assert_not_called()


def _minimal_train_toml(extra: str = "") -> str:
    return (
        """
[train.model]

[train.data]
dataset_dir = "data"

[train.optim]

[train.schedule]

[train.runtime]
out_dir = "out"
"""
        + extra
    )


def _minimal_sample_toml(extra: str = "") -> str:
    return (
        """
[sample.runtime]
out_dir = "out"

[sample.sample]
"""
        + extra
    )


def test_train_missing_runtime_section_strict(tmp_path: Path) -> None:
    toml_text = """
[train.model]

[train.data]
dataset_dir = "data"

[train.optim]

[train.schedule]
"""
    cfg_dir = tmp_path / "exp"
    cfg_dir.mkdir()
    cfg_path = cfg_dir / "config.toml"
    cfg_path.write_text(toml_text)

    with pytest.raises(ValueError) as e:
        load_train_config(cfg_path)
    assert "Missing required section [runtime]" in str(e.value)


def test_unknown_key_in_train_data_strict(tmp_path: Path) -> None:
    toml_text = """
[train.model]

[train.data]
dataset_dir = "data"
unknown_key = 123

[train.optim]

[train.schedule]

[train.runtime]
out_dir = "out"
"""
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.write_text(toml_text)

    with pytest.raises(ValueError) as e:
        load_train_config(cfg_path)
    assert "Unknown key(s) in [train.data]" in str(e.value)


def test_relative_path_resolution_train_strict(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "subdir"
    cfg_dir.mkdir()
    cfg_path = cfg_dir / "config.toml"
    cfg_path.write_text(_minimal_train_toml())

    exp = load_train_config(cfg_path)
    assert exp.data.dataset_dir == (cfg_dir / "data").resolve()
    assert exp.runtime.out_dir == (cfg_dir / "out").resolve()


def test_sanity_check_batch_size_strict(tmp_path: Path) -> None:
    toml_text = """
[train.model]

[train.data]
dataset_dir = "data"
batch_size = 0

[train.optim]

[train.schedule]

[train.runtime]
out_dir = "out"
"""
    cfg_path = tmp_path / "bad.toml"
    cfg_path.write_text(toml_text)

    with pytest.raises(ValueError) as e:
        load_train_config(cfg_path)
    assert "batch_size" in str(e.value)


def test_sample_missing_runtime_strict(tmp_path: Path) -> None:
    toml_text = """
[sample.sample]
"""
    cfg_path = tmp_path / "sample_missing_runtime.toml"
    cfg_path.write_text(toml_text)

    with pytest.raises(ValueError) as e:
        load_sample_config(cfg_path)
    assert "Missing required section [runtime]" in str(e.value)


def test_sample_relative_out_dir_resolution_strict(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "dir"
    cfg_dir.mkdir()
    cfg_path = cfg_dir / "cfg.toml"
    cfg_path.write_text(_minimal_sample_toml())

    exp = load_sample_config(cfg_path)
    assert exp.runtime.out_dir == (cfg_dir / "out").resolve()
