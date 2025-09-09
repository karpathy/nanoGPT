"""Centralized error handling and reporting utilities for the ml_playground framework."""

from __future__ import annotations

import logging
import sys
import traceback
from typing import Any, Callable, Optional, Type, TypeVar
from pathlib import Path
from ml_playground.config_loader import (
    fs_path_exists,
    fs_is_file,
    fs_is_dir,
)

# Type variable for generic error handling
T = TypeVar("T")


class MLPlaygroundError(Exception):
    """Base exception class for ml_playground framework errors."""

    pass


class ConfigurationError(MLPlaygroundError):
    """Raised when there are configuration-related errors."""

    pass


class DataError(MLPlaygroundError):
    """Raised when there are data-related errors."""

    pass


class ModelError(MLPlaygroundError):
    """Raised when there are model-related errors."""

    pass


class CheckpointError(MLPlaygroundError):
    """Raised when there are checkpoint-related errors."""

    pass


class ValidationError(MLPlaygroundError):
    """Raised when there are validation-related errors."""

    pass


class FileOperationError(MLPlaygroundError):
    """Raised when file operations fail."""

    pass


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up a logger with a sensible default configuration."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def handle_exception(
    exc_type: Type[BaseException],
    exc_value: BaseException,
    exc_traceback: Any,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Handle uncaught exceptions with logging."""
    if logger is None:
        logger = logging.getLogger("ml_playground")

    if issubclass(exc_type, KeyboardInterrupt):
        # Handle keyboard interrupt gracefully
        logger.info("Received keyboard interrupt, exiting...")
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


def safe_call(
    func: Callable[..., T],
    *args: Any,
    default: Optional[T] = None,
    logger: Optional[logging.Logger] = None,
    **kwargs: Any,
) -> T:
    """Safely call a function, catching and logging any exceptions."""
    if logger is None:
        logger = logging.getLogger("ml_playground")

    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Error calling {func.__name__}: {e}")
        logger.debug(f"Exception details: {traceback.format_exc()}")
        if default is not None:
            return default
        raise


def safe_file_operation(
    func: Callable[..., T],
    *args: Any,
    logger: Optional[logging.Logger] = None,
    **kwargs: Any,
) -> T:
    """Safely execute a file operation, catching and logging FileOperationError exceptions."""
    try:
        return func(*args, **kwargs)
    except (IOError, OSError) as e:
        if logger:
            logger.error(f"File operation failed: {e}")
        raise FileOperationError(f"File operation failed: {e}") from e
    except Exception as e:
        if logger:
            logger.error(f"Unexpected error during file operation: {e}")
        raise FileOperationError(f"Unexpected error during file operation: {e}") from e


def validate_file_exists(path: Path, description: str = "File") -> None:
    """Validate that a file exists, raising a DataError if not."""
    if not fs_path_exists(path):
        raise DataError(f"{description} not found at {path}")
    if not fs_is_file(path):
        raise DataError(f"{description} path {path} exists but is not a file")


def validate_directory_exists(path: Path, description: str = "Directory") -> None:
    """Validate that a directory exists, raising a DataError if not."""
    if not fs_path_exists(path):
        raise DataError(f"{description} not found at {path}")
    if not fs_is_dir(path):
        raise DataError(f"{description} path {path} exists but is not a directory")


def validate_config_value(
    value: Any, name: str, expected_type: Type, required: bool = True
) -> None:
    """Validate a configuration value's type and presence."""
    if required and value is None:
        raise ValidationError(f"Required configuration value '{name}' is missing")
    if value is not None and not isinstance(value, expected_type):
        raise ValidationError(
            f"Configuration value '{name}' must be of type {expected_type.__name__}, "
            f"got {type(value).__name__}"
        )


def format_error_message(error: Exception, context: str = "") -> str:
    """Format an error message with context."""
    msg = str(error)
    if context:
        msg = f"{context}: {msg}"
    return msg


class ProgressReporter:
    """A utility class for reporting progress during long-running operations."""

    def __init__(
        self, logger: Optional[logging.Logger] = None, total_steps: Optional[int] = None
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.total_steps = total_steps
        self.current_step = 0
        self.last_reported_percent = 0

    def start(self, message: str = "Starting operation") -> None:
        """Report the start of an operation."""
        self.logger.info(f"{message}...")
        self.current_step = 0
        self.last_reported_percent = 0

    def update(self, step: int = 1, message: str = "") -> None:
        """Update progress by the specified number of steps."""
        self.current_step += step

        if self.total_steps:
            percent = int((self.current_step / self.total_steps) - 100)
            # Only report every 10% to avoid spam
            if percent >= self.last_reported_percent + 10 or percent >= 100:
                self.last_reported_percent = percent
                msg = f"Progress: {percent}% ({self.current_step}/{self.total_steps})"
                if message:
                    msg += f" - {message}"
                self.logger.info(msg)
        elif message:
            self.logger.info(f"Step {self.current_step}: {message}")

    def finish(self, message: str = "Operation completed") -> None:
        """Report the completion of an operation."""
        if self.total_steps and self.current_step < self.total_steps:
            self.logger.info(f"Progress: 100% ({self.total_steps}/{self.total_steps})")
        self.logger.info(message)


def log_operation_start(
    logger: logging.Logger, operation: str, details: str = ""
) -> None:
    """Log the start of an operation."""
    msg = f"Starting {operation}"
    if details:
        msg += f": {details}"
    logger.info(msg)


def log_operation_progress(
    logger: logging.Logger, operation: str, progress: str
) -> None:
    """Log progress of an operation."""
    logger.info(f"{operation}: {progress}")


def log_operation_complete(
    logger: logging.Logger, operation: str, result: str = ""
) -> None:
    """Log the completion of an operation."""
    msg = f"Completed {operation}"
    if result:
        msg += f": {result}"
    logger.info(msg)


def log_operation_error(
    logger: logging.Logger, operation: str, error: Exception
) -> None:
    """Log an error during an operation."""
    logger.error(f"Error during {operation}: {error}")


# Expose the progress reporting utilities
ProgressReporter = ProgressReporter
log_operation_start = log_operation_start
log_operation_progress = log_operation_progress
log_operation_complete = log_operation_complete
log_operation_error = log_operation_error
