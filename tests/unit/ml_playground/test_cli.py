from __future__ import annotations
import pytest
from pathlib import Path
from pytest_mock import MockerFixture
from ml_playground.cli import main
from ml_playground.config import TrainerConfig, SamplerConfig, AppConfig
from ml_playground.prepare import PreparerConfig


def test_main_prepare_shakespeare_success(mocker: MockerFixture) -> None:
    """Test prepare command with shakespeare dataset succeeds."""
    # Simulate successful config load and call into _run_prepare
    mock_run = mocker.patch("ml_playground.cli._run_prepare")
    mocker.patch(
        "ml_playground.cli.ensure_loaded",
        return_value=(
            Path("cfg.toml"),
            AppConfig(train=None, sample=None),
            PreparerConfig(),
        ),
    )
    main(["prepare", "shakespeare"])
    mock_run.assert_called_once()


def test_main_prepare_bundestag_char_success(mocker: MockerFixture) -> None:
    """Test prepare command with bundestag_char dataset succeeds."""
    mock_run = mocker.patch("ml_playground.cli._run_prepare")
    mocker.patch(
        "ml_playground.cli.ensure_loaded",
        return_value=(
            Path("cfg.toml"),
            AppConfig(train=None, sample=None),
            PreparerConfig(),
        ),
    )
    main(["prepare", "bundestag_char"])
    mock_run.assert_called_once()


def test_main_prepare_unknown_dataset_fails(
    mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """Unknown experiment should surface as a CLI error exit."""
    mocker.patch("ml_playground.cli.ensure_loaded", side_effect=SystemExit(2))
    with pytest.raises(SystemExit) as e:
        main(["prepare", "unknown"])
    assert e.value.code == 2


def test_main_train_success(tmp_path: Path, mocker: MockerFixture) -> None:
    """Test train command auto-resolves config for experiment and calls train (strict loader)."""
    mock_train_cfg = mocker.Mock(spec=TrainerConfig)
    mock_run = mocker.patch("ml_playground.cli._run_train")
    mocker.patch(
        "ml_playground.cli.ensure_loaded",
        return_value=(
            Path("cfg.toml"),
            AppConfig(train=mock_train_cfg, sample=None),
            PreparerConfig(),
        ),
    )
    main(["train", "shakespeare"])
    mock_run.assert_called_once()


def test_main_train_no_train_block_fails(
    tmp_path: Path, mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test train command fails when strict loader raises."""
    # Simulate missing train config by returning AppConfig with train=None
    mocker.patch(
        "ml_playground.cli.ensure_loaded",
        return_value=(
            Path("cfg.toml"),
            AppConfig(train=None, sample=None),
            PreparerConfig(),
        ),
    )
    with pytest.raises(SystemExit) as e:
        main(["train", "shakespeare"])
    assert e.value.code == 2


def test_main_sample_success(tmp_path: Path, mocker: MockerFixture) -> None:
    """Test sample command auto-resolves config and calls sample function (strict loader)."""
    mock_sample_cfg = SamplerConfig.model_validate(
        {"sample": {"start": "x"}, "runtime": {"out_dir": Path("out")}}
    )
    mock_run = mocker.patch("ml_playground.cli._run_sample")
    mocker.patch(
        "ml_playground.cli.ensure_loaded",
        return_value=(
            Path("cfg.toml"),
            AppConfig(train=None, sample=mock_sample_cfg),
            PreparerConfig(),
        ),
    )
    main(["sample", "shakespeare"])
    mock_run.assert_called_once()


def test_main_sample_no_sample_block_fails(
    tmp_path: Path, mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test sample command fails when strict loader raises."""
    mocker.patch(
        "ml_playground.cli.ensure_loaded",
        return_value=(
            Path("cfg.toml"),
            AppConfig(train=None, sample=None),
            PreparerConfig(),
        ),
    )
    with pytest.raises(SystemExit) as e:
        main(["sample", "shakespeare"])
    assert e.value.code == 2


def test_main_loop_success(tmp_path: Path, mocker: MockerFixture) -> None:
    """Test loop command executes via _run_loop with loaded configs."""
    mock_train_config = mocker.Mock(spec=TrainerConfig)
    mock_sample_config = SamplerConfig.model_validate(
        {"sample": {"start": "x"}, "runtime": {"out_dir": Path("out")}}
    )
    mock_run = mocker.patch("ml_playground.cli._run_loop")
    mocker.patch(
        "ml_playground.cli.ensure_loaded",
        return_value=(
            Path("cfg.toml"),
            AppConfig(train=mock_train_config, sample=mock_sample_config),
            PreparerConfig(),
        ),
    )

    main(["loop", "shakespeare"])
    mock_run.assert_called_once()


def test_main_loop_unknown_dataset_fails(tmp_path: Path, mocker: MockerFixture) -> None:
    """Unknown experiment should bubble up as CLI error exit."""
    mocker.patch("ml_playground.cli.ensure_loaded", side_effect=SystemExit(2))
    with pytest.raises(SystemExit) as e:
        main(["loop", "shakespeare"])
    assert e.value.code == 2


def test_main_loop_missing_train_block_fails(
    tmp_path: Path, mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test loop command fails when strict train loader raises."""
    mocker.patch(
        "ml_playground.cli.ensure_loaded",
        return_value=(
            Path("cfg.toml"),
            AppConfig(
                train=None,
                sample=SamplerConfig.model_validate(
                    {"sample": {"start": "x"}, "runtime": {"out_dir": Path("out")}}
                ),
            ),
            PreparerConfig(),
        ),
    )
    with pytest.raises(SystemExit) as e:
        main(["loop", "shakespeare"])
    assert e.value.code == 2


def test_main_loop_missing_sample_block_fails(
    tmp_path: Path, mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test loop command fails when strict sample loader raises."""
    mocker.patch(
        "ml_playground.cli.ensure_loaded",
        return_value=(
            Path("cfg.toml"),
            AppConfig(
                train=TrainerConfig.model_validate(
                    {
                        "model": {
                            "n_layer": 1,
                            "n_head": 1,
                            "n_embd": 8,
                            "block_size": 8,
                            "dropout": 0.0,
                            "bias": False,
                        },
                        "data": {
                            "dataset_dir": Path("dataset"),
                            "train_bin": "train.bin",
                            "val_bin": "val.bin",
                            "meta_pkl": "meta.pkl",
                            "batch_size": 1,
                            "block_size": 8,
                            "ngram_size": 1,
                        },
                        "optim": {
                            "learning_rate": 1e-3,
                            "beta1": 0.9,
                            "beta2": 0.95,
                            "weight_decay": 0.1,
                            "grad_clip": 1.0,
                        },
                        "schedule": {
                            "decay_lr": False,
                            "warmup_iters": 0,
                            "lr_decay_iters": 10_000,
                            "min_lr": 1e-4,
                        },
                        "runtime": {
                            "out_dir": Path("out"),
                            "device": "cpu",
                            "dtype": "float32",
                            "seed": 1,
                            "compile": False,
                            "checkpointing": {
                                "keep": {
                                    "last": 1,
                                    "best": 1,
                                }
                            },
                            "eval_interval": 1,
                            "eval_iters": 1,
                            "log_interval": 1,
                            "max_iters": 0,
                        },
                    }
                ),
                sample=None,
            ),
            PreparerConfig(),
        ),
    )
    with pytest.raises(SystemExit) as e:
        main(["loop", "shakespeare"])
    assert e.value.code == 2


# Tests for removed functionality (legacy registry usage, direct train/sample calls)
# have been updated to reflect the new CLI architecture.
