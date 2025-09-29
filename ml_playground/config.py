from __future__ import annotations

"""Backward-compatible shim for configuration models.

All configuration schemas now live under ``ml_playground.configuration.models``.
This module re-exports the public API so existing imports continue to work until
callers migrate to the new package structure.
"""

from ml_playground.configuration.models import *  # noqa: F401,F403,E402
