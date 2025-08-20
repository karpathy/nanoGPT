from __future__ import annotations
from pathlib import Path
from typing import Any
import tomllib
from pydantic import ValidationError
from .config_models import AppConfig


def _prune_incomplete_sections(raw: dict[str, Any]) -> dict[str, Any]:
    """Mimic legacy behavior: if [train] or [sample] is present but incomplete,
    drop the section so AppConfig validates with None for that section.
    """
    data = dict(raw)
    t = data.get("train")
    if isinstance(t, dict):
        needed = {"model", "data", "optim", "schedule", "runtime"}
        if not needed.issubset(set(t.keys())):
            data.pop("train", None)
    s = data.get("sample")
    if isinstance(s, dict):
        needed = {"runtime", "sample"}
        if not needed.issubset(set(s.keys())):
            data.pop("sample", None)
    return data


def load_config(path: Path) -> AppConfig:
    """Load and validate a TOML config file into AppConfig using Pydantic v2.

    - Unknown keys are rejected by the models (extra="forbid").
    - Incomplete [train]/[sample] sections are pruned (become None) to preserve
      legacy behavior used by some tests/utilities.
    - This function is the single TOML loader used by the CLI; other modules
      should accept already-validated models.
    """
    with path.open("rb") as f:
        raw = tomllib.load(f)
    try:
        pruned = _prune_incomplete_sections(raw if isinstance(raw, dict) else {})
        return AppConfig.model_validate(pruned)
    except ValidationError:
        # Re-raise to preserve structured error info for callers/tests
        raise
