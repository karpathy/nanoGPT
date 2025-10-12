from __future__ import annotations

import io
import logging
from pathlib import Path

import pytest

from ml_playground.core.error_handling import (
    DetailedException,
    MLPlaygroundError,
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


def test_ml_playground_error_reports_reason_and_rationale() -> None:
    err = MLPlaygroundError(
        "Boom",
        reason="Demonstration failure",
        rationale="Test ensures rich diagnostics are surfaced",
    )
    assert err.message == "Boom"
    assert err.reason == "Demonstration failure"
    assert err.rationale == "Test ensures rich diagnostics are surfaced"
    rendered = str(err)
    assert "Reason: Demonstration failure" in rendered
    assert "Rationale: Test ensures rich diagnostics are surfaced" in rendered
    assert isinstance(err, DetailedException)


def test_setup_logging_idempotent_and_level():
    name = "ml_pg_test_logger"
    # capture log output
    stream = io.StringIO()

    class _Stream(logging.StreamHandler):
        def __init__(self):
            super().__init__(stream)

    logger = setup_logging(
        name,
        level=logging.DEBUG,
        stream_handler_factory=lambda: _Stream(),
    )
    logger.debug("hello")
    # re-call should not add duplicate handlers
    logger2 = setup_logging(
        name,
        level=logging.INFO,
        stream_handler_factory=lambda: _Stream(),
    )
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
    with pytest.raises(FileOperationError, match="disk full") as exc:
        safe_file_operation(bad_io, logger=logger)
    assert exc.value.reason == "OSError raised during file operation"
    assert "Filesystem paths must be reachable" in exc.value.rationale


def test_validate_file_and_directory(tmp_path: Path):
    f = tmp_path / "file.txt"
    d = tmp_path / "dir"
    f.write_text("x", encoding="utf-8")
    d.mkdir()

    # success cases
    validate_file_exists(f, "TestFile")
    validate_directory_exists(d, "TestDir")

    # file does not exist
    with pytest.raises(DataError) as missing_file:
        validate_file_exists(tmp_path / "missing.txt")
    assert missing_file.value.reason == "Path does not exist on disk"
    # path exists but is not file
    with pytest.raises(DataError) as wrong_kind:
        validate_file_exists(d)
    assert wrong_kind.value.reason == "Path refers to a non-file entry"

    # dir does not exist
    with pytest.raises(DataError) as missing_dir:
        validate_directory_exists(tmp_path / "missing_dir")
    assert missing_dir.value.reason == "Path does not exist on disk"
    # path exists but is not directory
    with pytest.raises(DataError) as wrong_dir_kind:
        validate_directory_exists(f)
    assert wrong_dir_kind.value.reason == "Path refers to a non-directory entry"


def test_validate_config_value():
    with pytest.raises(ValidationError) as missing:
        validate_config_value(None, "x", int, required=True)
    assert missing.value.reason == "Configuration entry absent"
    assert "required keys" in missing.value.rationale
    with pytest.raises(ValidationError) as wrong_type:
        validate_config_value("hi", "x", int)
    assert wrong_type.value.reason == "Type mismatch for configuration entry"
    assert "deterministic" in wrong_type.value.rationale
    # ok
    validate_config_value(3, "x", int)


def test_format_error_message():
    msg = format_error_message(ValueError("nope"))
    assert "nope" in msg
    msg2 = format_error_message(ValueError("nope"), context="ctx")
    assert "ctx" in msg2 and "nope" in msg2


def test_progress_reporter_percentages(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    logger = logging.getLogger("ml_pg_progress_percent")
    pr = ProgressReporter(logger=logger, total_steps=5)

    pr.start("Start")
    pr.update(step=1, message="warming up")  # 20%
    pr.update(step=1)  # 40%
    pr.update(step=2, message="main work")  # 80%
    pr.finish("Done")  # should emit 100% because we are short of total

    messages = [record.getMessage() for record in caplog.records]
    assert "Start..." in messages[0]
    assert "Progress: 20% (1/5) - warming up" in messages
    assert "Progress: 40% (2/5)" in messages
    assert "Progress: 80% (4/5) - main work" in messages
    assert "Progress: 100% (5/5)" in messages
    assert messages[-1] == "Done"


def test_handle_exception_keyboard_interrupt() -> None:
    """Ensure KeyboardInterrupt is handled gracefully."""
    logger = logging.getLogger("ml_pg_test_kb")
    stream = io.StringIO()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(stream))

    from ml_playground.core.error_handling import handle_exception

    try:
        raise KeyboardInterrupt
    except KeyboardInterrupt as e:
        handle_exception(
            type(e),
            e,
            e.__traceback__,
            logger,
            excepthook=lambda *a, **kw: None,
        )

    assert "Received keyboard interrupt" in stream.getvalue()


def test_safe_file_operation_unexpected_error() -> None:
    """Ensure non-IO errors are wrapped correctly."""

    def bad_logic():
        raise ValueError("logic error")

    logger = logging.getLogger("ml_pg_test_unexpected")
    with pytest.raises(FileOperationError, match="Unexpected error") as unexpected:
        safe_file_operation(bad_logic, logger=logger)
    assert unexpected.value.reason == "ValueError raised outside expected IO failures"
    assert "investigate the triggering logic" in unexpected.value.rationale


def test_progress_reporter_no_total_and_small_updates(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test progress reporting without a total and with small increments."""
    caplog.set_level(logging.INFO)
    logger = logging.getLogger("ml_pg_progress_no_total")

    # Test without total_steps
    pr_no_total = ProgressReporter(logger=logger)
    pr_no_total.update(message="Step 1")
    assert "Step 1: Step 1" in caplog.text

    # Test with small increments that don't trigger a log
    pr_small_steps = ProgressReporter(logger=logger, total_steps=100)
    pr_small_steps.update(step=5)  # 5% < 10%, should not log
    assert "Progress: 5%" not in caplog.text


def test_log_helpers_no_details(caplog: pytest.LogCaptureFixture) -> None:
    """Ensure log helpers work without optional details."""
    caplog.set_level(logging.INFO)
    logger = logging.getLogger("ml_pg_log_helpers")
    log_operation_start(logger, "op_simple")
    log_operation_complete(logger, "op_simple")
    assert "Starting op_simple" in caplog.text
    assert "Completed op_simple" in caplog.text


def test_progress_reporter_clamps_and_log_helpers(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    logger = logging.getLogger("ml_pg_progress_clamp")
    pr = ProgressReporter(logger=logger, total_steps=4)

    pr.start()
    pr.update(step=6)  # exceeds total -> clamp at 100%
    pr.finish("Finished")  # Should not duplicate 100% log

    messages = [record.getMessage() for record in caplog.records]
    progress_msgs = [m for m in messages if "Progress" in m]
    assert progress_msgs.count("Progress: 100% (4/4)") == 1

    log_operation_start(logger, "op", details="d")
    log_operation_progress(logger, "op", progress="50%")
    log_operation_complete(logger, "op", result="ok")
    log_operation_error(logger, "op", error=RuntimeError("x"))

    assert "Starting op: d" in messages or any("Starting op" in m for m in messages)
