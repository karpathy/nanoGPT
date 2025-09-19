from __future__ import annotations

import io
import logging
from pathlib import Path

import pytest

from ml_playground.error_handling import (
    setup_logging,
    safe_call,
    safe_file_operation,
    validate_file_exists,
    validate_directory_exists,
    validate_config_value,
    format_error_message,
    ProgressReporter,
    log_operation_start,
    log_operation_progress,
    log_operation_complete,
    log_operation_error,
    DataError,
    FileOperationError,
    ValidationError,
)


def test_setup_logging_idempotent_and_level(monkeypatch: pytest.MonkeyPatch):
    name = "ml_pg_test_logger"
    # capture log output
    stream = io.StringIO()

    class _Stream(logging.StreamHandler):
        def __init__(self):
            super().__init__(stream)

    monkeypatch.setattr(logging, "StreamHandler", _Stream)

    logger = setup_logging(name, level=logging.DEBUG)
    logger.debug("hello")
    # re-call should not add duplicate handlers
    logger2 = setup_logging(name, level=logging.INFO)
    logger2.info("world")

    out = stream.getvalue()
    assert "hello" in out
    assert "world" in out
    assert logger is logger2


def test_safe_call_success_and_defaults():
    def ok(x):
        return x + 1

    def bad(_):
        raise RuntimeError("boom")

    import logging

    logger = logging.getLogger("ml_pg_test")
    assert safe_call(ok, 1, logger=logger) == 2
    assert safe_call(bad, 0, default=42, logger=logger) == 42
    with pytest.raises(RuntimeError):
        safe_call(bad, 0, logger=logger)


def test_safe_file_operation_wraps_ioerror():
    def bad_io():
        raise OSError("disk full")

    import logging

    logger = logging.getLogger("ml_pg_test")
    with pytest.raises(FileOperationError, match="disk full"):
        safe_file_operation(bad_io, logger=logger)


def test_validate_file_and_directory(tmp_path: Path):
    f = tmp_path / "file.txt"
    d = tmp_path / "dir"
    f.write_text("x", encoding="utf-8")
    d.mkdir()

    # success cases
    validate_file_exists(f, "TestFile")
    validate_directory_exists(d, "TestDir")

    # file does not exist
    with pytest.raises(DataError):
        validate_file_exists(tmp_path / "missing.txt")
    # path exists but is not file
    with pytest.raises(DataError):
        validate_file_exists(d)

    # dir does not exist
    with pytest.raises(DataError):
        validate_directory_exists(tmp_path / "missing_dir")
    # path exists but is not directory
    with pytest.raises(DataError):
        validate_directory_exists(f)


def test_validate_config_value():
    with pytest.raises(ValidationError):
        validate_config_value(None, "x", int, required=True)
    with pytest.raises(ValidationError):
        validate_config_value("hi", "x", int)
    # ok
    validate_config_value(3, "x", int)


def test_format_error_message():
    msg = format_error_message(ValueError("nope"))
    assert "nope" in msg
    msg2 = format_error_message(ValueError("nope"), context="ctx")
    assert "ctx" in msg2 and "nope" in msg2


def test_progress_reporter_and_log_helpers(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO)
    logger = logging.getLogger("ml_pg_progress")
    pr = ProgressReporter(logger=logger, total_steps=5)
    pr.start("Start")
    pr.update(step=2, message="doing")
    pr.update(step=3)
    pr.finish("Done")

    log_operation_start(logger, "op", details="d")
    log_operation_progress(logger, "op", progress="50%")
    log_operation_complete(logger, "op", result="ok")
    log_operation_error(logger, "op", error=RuntimeError("x"))

    # ensure messages appeared; not asserting exact text to keep stable
    assert caplog.records
