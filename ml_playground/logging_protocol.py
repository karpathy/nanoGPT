from __future__ import annotations

from typing import Protocol, Any, runtime_checkable


@runtime_checkable
class LoggerLike(Protocol):
    """Structural protocol for loggers used in this project.

    Allows using standard logging.Logger as well as lightweight test doubles
    that implement these methods.
    """

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None: ...

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None: ...

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None: ...

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None: ...
