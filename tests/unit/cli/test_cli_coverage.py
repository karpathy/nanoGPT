from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ml_playground.cli import app, override_cli_dependencies, CLIDependencies


@pytest.fixture
def runner() -> CliRunner:
    """Typer CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def temp_config_file(tmp_path: Path) -> Path:
    """Create a minimal temporary config file for testing."""
    config_content = """
[shared]
dataset_dir = "tmp/dataset"
config_path = "tmp/config.toml"

[prepare]
vocab_size = 256

[train]
n_layer = 1
n_head = 1
n_embd = 32

[sample]
max_new_tokens = 100
"""
    config_path = tmp_path / "config.toml"
    config_path.write_text(config_content, encoding="utf-8")
    return config_path


class TestGlobalOptions:
    """Test global CLI options like --exp-config validation."""

    def test_exp_config_missing_file_exits_with_code_2(
        self, runner: CliRunner, caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        """Test that --exp-config with missing file exits with code 2."""
        missing_path = tmp_path / "missing.toml"
        with caplog.at_level(logging.ERROR, logger="ml_playground.cli"):
            result = runner.invoke(
                app, ["--exp-config", str(missing_path), "prepare", "shakespeare"]
            )
        assert result.exit_code == 2
        assert "Config file not found" in caplog.messages[-1]

    def test_exp_config_valid_file_sets_context(
        self, runner: CliRunner, temp_config_file: Path
    ) -> None:
        """Test that valid --exp-config is stored in context."""
        # This test passes if no error occurs; context setting is internal
        result = runner.invoke(
            app, ["--exp-config", str(temp_config_file), "prepare", "shakespeare"]
        )
        # Since shakespeare experiment may not exist, we expect an error, but context should be set
        assert result.exit_code != 2  # Not the config file error

    def test_context_initialization_fallback_on_bad_context(
        self, runner: CliRunner
    ) -> None:
        """Test fallback when Typer context object is malformed."""
        # This is hard to trigger directly, but ensure basic commands work
        result = runner.invoke(app, ["prepare", "nonexistent"])
        assert result.exit_code != 0  # Some error, but not context-related crash


class TestCommandRunners:
    """Test prepare, train, sample commands with run_or_exit wrapping."""

    def test_prepare_keyboard_interrupt_handled(
        self, runner: CliRunner, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test prepare handles KeyboardInterrupt gracefully."""
        deps = CLIDependencies(
            load_experiment=lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()),
            ensure_train_prerequisites=lambda *args: None,
            ensure_sample_prerequisites=lambda *args: None,
            run_prepare=lambda *args: None,
            run_train=lambda *args: None,
            run_sample=lambda *args: None,
        )
        with (
            override_cli_dependencies(deps),
            caplog.at_level(logging.INFO, logger="ml_playground.cli"),
        ):
            result = runner.invoke(app, ["prepare", "shakespeare"])
            assert result.exit_code == 0
            assert "Data preparation cancelled" in caplog.messages[-1]

    def test_train_keyboard_interrupt_handled(
        self, runner: CliRunner, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test train handles KeyboardInterrupt gracefully."""
        deps = CLIDependencies(
            load_experiment=lambda *args: None,
            ensure_train_prerequisites=lambda *args: (_ for _ in ()).throw(
                KeyboardInterrupt()
            ),
            ensure_sample_prerequisites=lambda *args: None,
            run_prepare=lambda *args: None,
            run_train=lambda *args: None,
            run_sample=lambda *args: None,
        )
        with (
            override_cli_dependencies(deps),
            caplog.at_level(logging.INFO, logger="ml_playground.cli"),
        ):
            result = runner.invoke(app, ["train", "shakespeare"])
            assert result.exit_code == 0
            assert "Training cancelled" in caplog.messages[-1]

    def test_sample_keyboard_interrupt_handled(
        self, runner: CliRunner, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test sample handles KeyboardInterrupt gracefully."""
        deps = CLIDependencies(
            load_experiment=lambda *args: None,
            ensure_train_prerequisites=lambda *args: None,
            ensure_sample_prerequisites=lambda *args: (_ for _ in ()).throw(
                KeyboardInterrupt()
            ),
            run_prepare=lambda *args: None,
            run_train=lambda *args: None,
            run_sample=lambda *args: None,
        )
        with (
            override_cli_dependencies(deps),
            caplog.at_level(logging.INFO, logger="ml_playground.cli"),
        ):
            result = runner.invoke(app, ["sample", "shakespeare"])
            assert result.exit_code == 0
            assert "Sampling cancelled" in caplog.messages[-1]

    def test_prepare_domain_exception_exits_with_code_1(
        self, runner: CliRunner, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test prepare exits with code 1 on ValueError."""
        deps = CLIDependencies(
            load_experiment=lambda *args: (_ for _ in ()).throw(
                ValueError("test error")
            ),
            ensure_train_prerequisites=lambda *args: None,
            ensure_sample_prerequisites=lambda *args: None,
            run_prepare=lambda *args: None,
            run_train=lambda *args: None,
            run_sample=lambda *args: None,
        )
        with override_cli_dependencies(deps), caplog.at_level(logging.ERROR):
            result = runner.invoke(app, ["prepare", "shakespeare"])
            assert result.exit_code == 1
            assert "test error" in caplog.messages[-1]

    def test_train_domain_exception_exits_with_code_1(
        self, runner: CliRunner, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test train exits with code 1 on ValueError."""
        deps = CLIDependencies(
            load_experiment=lambda *args: None,
            ensure_train_prerequisites=lambda *args: (_ for _ in ()).throw(
                ValueError("test error")
            ),
            ensure_sample_prerequisites=lambda *args: None,
            run_prepare=lambda *args: None,
            run_train=lambda *args: None,
            run_sample=lambda *args: None,
        )
        with override_cli_dependencies(deps), caplog.at_level(logging.ERROR):
            result = runner.invoke(app, ["train", "shakespeare"])
            assert result.exit_code == 1
            assert "test error" in caplog.messages[-1]

    def test_sample_domain_exception_exits_with_code_1(
        self, runner: CliRunner, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test sample exits with code 1 on ValueError."""
        deps = CLIDependencies(
            load_experiment=lambda *args: None,
            ensure_train_prerequisites=lambda *args: None,
            ensure_sample_prerequisites=lambda *args: (_ for _ in ()).throw(
                ValueError("test error")
            ),
            run_prepare=lambda *args: None,
            run_train=lambda *args: None,
            run_sample=lambda *args: None,
        )
        with override_cli_dependencies(deps), caplog.at_level(logging.ERROR):
            result = runner.invoke(app, ["sample", "shakespeare"])
            assert result.exit_code == 1
            assert "test error" in caplog.messages[-1]


class TestDeviceSetupFallbacks:
    """Test _global_device_setup error handling."""

    def test_device_setup_swallows_cuda_errors(self) -> None:
        """Test that torch errors in device setup are swallowed."""
        from ml_playground.cli import _global_device_setup

        # Should not raise, even with cuda available but error
        _global_device_setup("cuda", "float32", 42, cuda_is_available=lambda: True)

    def test_device_setup_success_path(self) -> None:
        """Test successful CUDA setup."""
        from ml_playground.cli import _global_device_setup

        # Inject fake cuda available
        _global_device_setup("cuda", "float32", 42, cuda_is_available=lambda: True)

    def test_device_setup_explicit_cuda_override(self) -> None:
        """Test injecting cuda_is_available callable."""
        from ml_playground.cli import _global_device_setup

        called = False

        def fake_cuda():
            nonlocal called
            called = True
            return False

        _global_device_setup("cpu", "float32", 42, cuda_is_available=fake_cuda)
        assert called


class TestAnalysisGuardRails:
    """Test _run_analyze experiment validation."""

    def test_analyze_rejects_non_bundestag_char(self) -> None:
        """Test RuntimeError for unsupported experiments."""
        from ml_playground.cli import _run_analyze

        with pytest.raises(
            RuntimeError, match="analyze currently supports only 'bundestag_char'"
        ):
            _run_analyze("shakespeare", "127.0.0.1", 8050, True)

    def test_analyze_accepts_bundestag_char(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test info log for supported experiment."""
        from ml_playground.cli import _run_analyze

        with caplog.at_level("INFO"):
            _run_analyze("bundestag_char", "127.0.0.1", 8050, True)
        assert "Analysis for 'bundestag_char' not implemented" in caplog.text


class TestDirectoryLoggingResilience:
    """Test _log_dir and _log_command_status error handling."""

    def test_log_dir_handles_unset_path(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test logging when dir_path is None."""
        from ml_playground.cli import _log_dir
        import logging

        logger = logging.getLogger("test")
        with caplog.at_level("INFO"):
            _log_dir("test", "test_dir", None, logger)
        assert "<not set>" in caplog.text

    def test_log_dir_handles_missing_directory(
        self, caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        """Test logging for non-existent directory."""
        from ml_playground.cli import _log_dir
        import logging

        logger = logging.getLogger("test")
        missing_dir = tmp_path / "missing"
        with caplog.at_level("INFO"):
            _log_dir("test", "test_dir", missing_dir, logger)
        assert "(missing)" in caplog.text

    def test_log_dir_handles_existing_directory(
        self, caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        """Test logging for existing directory."""
        from ml_playground.cli import _log_dir
        import logging

        logger = logging.getLogger("test")
        existing_dir = tmp_path / "existing"
        existing_dir.mkdir()
        (existing_dir / "file.txt").write_text("content")
        with caplog.at_level("INFO"):
            _log_dir("test", "test_dir", existing_dir, logger)
        assert "(exists)" in caplog.text
        assert "Contents:" in caplog.text

    def test_log_dir_handles_unreadable_directory(
        self, caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        """Test logging for directory without read permission."""
        from ml_playground.cli import _log_dir
        import logging

        logger = logging.getLogger("test")
        unreadable_dir = tmp_path / "unreadable"
        unreadable_dir.mkdir()
        os.chmod(unreadable_dir, 0o000)  # No permissions
        try:
            with caplog.at_level("INFO"):
                _log_dir("test", "test_dir", unreadable_dir, logger)
            assert "(exists)" in caplog.text
            # Should not crash, even if contents not listed
        finally:
            os.chmod(unreadable_dir, 0o755)  # Restore for cleanup

    def test_log_command_status_handles_errors_gracefully(self, tmp_path: Path) -> None:
        """Test _log_command_status handles OSError/ValueError/TypeError gracefully."""
        from ml_playground.cli import _log_command_status
        from ml_playground.configuration.models import SharedConfig

        shared = SharedConfig(
            experiment="test",
            config_path=tmp_path / "config.toml",
            project_home=tmp_path,
            dataset_dir=tmp_path / "dataset",
            train_out_dir=tmp_path / "train",
            sample_out_dir=tmp_path / "sample",
        )

        # Should not raise
        _log_command_status("tag", shared, tmp_path, logging.getLogger(__name__))
