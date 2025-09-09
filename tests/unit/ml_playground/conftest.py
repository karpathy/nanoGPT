from __future__ import annotations

import logging
from typing import Callable
import pytest


class ListLogger(logging.Logger):
    """A lightweight logger that also captures messages in lists for assertions."""

    def __init__(self) -> None:
        super().__init__("test-list-logger")
        self.infos: list[str] = []
        self.warnings: list[str] = []

    def info(self, msg: str, *args, **kwargs) -> None:  # noqa: D401
        self.infos.append(str(msg))
        super().info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:  # noqa: D401
        self.warnings.append(str(msg))
        super().warning(msg, *args, **kwargs)


@pytest.fixture()
def list_logger() -> ListLogger:
    """Provide a fresh ListLogger instance for tests that need a single logger."""
    return ListLogger()


@pytest.fixture()
def list_logger_factory() -> Callable[[], ListLogger]:
    """Provide a factory to create multiple independent ListLogger instances."""

    def _factory() -> ListLogger:
        return ListLogger()

    return _factory
