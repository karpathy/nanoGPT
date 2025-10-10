from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any


def merge_mappings(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
    *,
    override_only: bool = False,
) -> dict[str, Any]:
    """Deep merge two mapping objects into a new ``dict``.

    Values from ``override`` replace those in ``base``. Nested mappings are merged
    recursively, and non-mapping values are deep-copied to avoid mutating the
    override input. When ``override_only`` is ``True`` the result contains only keys
    present in ``override`` (and their descendants), while still using ``base`` for
    recursive defaults.
    """

    merged: dict[str, Any] = {} if override_only else deepcopy(dict(base))
    for key, override_value in override.items():
        base_value = base.get(key) if isinstance(base, Mapping) else None
        if isinstance(base_value, Mapping) and isinstance(override_value, Mapping):
            merged[key] = merge_mappings(
                base_value,
                override_value,
                override_only=override_only,
            )
        else:
            merged[key] = deepcopy(override_value)
    return merged
