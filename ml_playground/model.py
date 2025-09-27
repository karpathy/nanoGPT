"""Deprecated import shim for `ml_playground.model`.

The original monolithic module was split into the modular `ml_playground.models`
hierarchy. Import from the new modules instead of this shim.
"""

from __future__ import annotations


def _raise_deprecated_import() -> None:  # pragma: no cover
    raise ImportError(
        "`ml_playground.model` has been removed. Use the modular `ml_playground.models` "
        "package (e.g. `ml_playground.models.core.model`) instead."
    )


_raise_deprecated_import()
