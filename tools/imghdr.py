"""Backport of the deprecated stdlib `imghdr` module required by third-party packages.

This minimal stub provides the `what` function used by `pybadges` to determine
image formats. It currently only recognizes SVG badges and returns ``None`` for
other inputs, which is sufficient for our usage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

SVG_SIGNATURE = b"<svg"


def what(filename: Optional[str] = None, h: Optional[bytes] = None) -> Optional[str]:
    """Return the image type for the given file or byte buffer.

    Parameters
    ----------
    filename:
        Path to the image file on disk.
    h:
        Optional bytes buffer to inspect directly.
    """

    data: Optional[bytes] = h
    if data is None and filename is not None:
        path = Path(filename)
        try:
            with path.open("rb") as fh:
                data = fh.read(5)
        except FileNotFoundError:
            data = None

    if data is None:
        return None

    sample = data.strip().lower()
    if sample.startswith(SVG_SIGNATURE):
        return "svg"

    return None
