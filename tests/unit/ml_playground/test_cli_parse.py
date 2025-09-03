from __future__ import annotations
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "args, expect",
    [
        (["--help"], "usage:"),
        (["prepare", "--help"], "prepare"),
        (["train", "--help"], "train"),
        (["sample", "--help"], "sample"),
        (["loop", "--help"], "loop"),
        (["analyze", "--help"], "analyze"),
    ],
)
def test_cli_help_and_version(args, expect):
    # Run via python -m to exercise entry-point wiring without performing heavy work
    cmd = [sys.executable, "-m", "ml_playground.cli"] + args
    cp = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    out = cp.stdout
    if args == ["--help"]:
        assert "usage" in out.lower()
    else:
        assert expect in out.lower()
    assert cp.returncode == 0


def test_cli_global_exp_config_missing_exits(tmp_path):
    # Point to a definitely missing path
    missing = tmp_path / "nope.toml"
    # Use a real subcommand to ensure the callback runs (help short-circuits)
    cmd = [
        sys.executable,
        "-m",
        "ml_playground.cli",
        "--exp-config",
        str(missing),
        "prepare",
        "shakespeare",
    ]
    cp = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    # Typer exits with code 2 for callback-triggered validation failures
    assert cp.returncode == 2
    assert "config file not found" in cp.stdout.lower()
