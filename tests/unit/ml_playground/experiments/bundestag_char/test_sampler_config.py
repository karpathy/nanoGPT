from __future__ import annotations

from pathlib import Path
import tomllib

import pytest

from ml_playground.config import SECTION_TRAIN


def test_bundestag_char_sampler_is_sequential() -> None:
    # Strict: use repository-relative config only; if missing, skip
    cfg_path = Path("ml_playground/experiments/bundestag_char/config.toml")
    if not cfg_path.exists():
        pytest.skip("bundestag_char config.toml not found; skipping")
    # Read TOML directly to avoid strict AppConfig validation rejecting extras
    with cfg_path.open("rb") as f:
        raw = tomllib.load(f)
    if SECTION_TRAIN not in raw or not isinstance(raw[SECTION_TRAIN], dict):
        pytest.skip("train section missing; skipping")
    train = raw[SECTION_TRAIN]
    data = train.get("data", {}) if isinstance(train, dict) else {}
    sampler = data.get("sampler") if isinstance(data, dict) else None
    # Strict: assert exact current repo config value
    assert sampler == "random"
