"""Command-line overrides may contain '=' in strings (e.g. sample prompts)."""

import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("override,expected", [
    ("--start=E=mc^2", "E=mc^2"),
    ("--start=token=a=b", "token=a=b"),
    ("--start=", ""),
    ("--start=ordinary", "ordinary"),
])
def test_configurator_preserves_equals_in_string_values(monkeypatch, override, expected):
    path = Path(__file__).resolve().parents[1] / "configurator.py"
    namespace = {"start": "before"}
    monkeypatch.setattr(sys, "argv", [str(path), override])
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
    assert namespace["start"] == expected


def test_configurator_still_rejects_unknown_options(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "configurator.py"
    monkeypatch.setattr(sys, "argv", [str(path), "--does_not_exist=42"])
    with pytest.raises(ValueError, match="Unknown config key"):
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), {"start": ""})
