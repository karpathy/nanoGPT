from __future__ import annotations

from pathlib import Path

from ml_playground.config import load_toml


def test_bundestag_char_sampler_is_sequential() -> None:
    cfg_path = Path("ml_playground/experiments/bundestag_char/config.toml")
    app = load_toml(cfg_path)
    assert app.train is not None
    assert app.train.data.sampler == "sequential"
