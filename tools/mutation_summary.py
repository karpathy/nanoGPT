#!/usr/bin/env python3
"""Emit a quick summary of the current Cosmic Ray mutation configuration."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from cosmic_ray.config import load_config
from cosmic_ray.modules import find_modules


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rem = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {rem:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m"


def _count_modules(module_path: Path) -> int:
    modules: Iterable[Path] = find_modules([module_path])
    return sum(1 for _ in modules)


def summarize(config_path: Path) -> None:
    cfg = load_config(str(config_path))

    module_path = Path(cfg["module-path"])
    timeout = float(cfg.get("timeout", 10.0))
    session_cfg = cfg.get("session", {}) or {}
    session_file = session_cfg.get("file", ".cache/cosmic-ray/session.sqlite")
    test_command = cfg["test-command"]

    module_count = _count_modules(module_path)
    optimistic_total = module_count * timeout

    print(f"[mutation] config: {config_path}")
    print(f"[mutation] session file: {session_file}")
    print(
        f"[mutation] module path: {module_path} ({module_count} module(s) discovered)"
    )
    print(f"[mutation] pytest command: {test_command}")
    print(f"[mutation] timeout per mutant: {timeout:.1f}s")
    print(
        "[mutation] baseline+exec worst-case: "
        f"{_format_duration(optimistic_total)} (~{module_count} mutants)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Display the active Cosmic Ray mutation configuration summary."
    )
    parser.add_argument(
        "--config",
        default="pyproject.toml",
        type=Path,
        help="Path to the Cosmic Ray configuration file (default: pyproject.toml)",
    )
    args = parser.parse_args()
    summarize(args.config)


if __name__ == "__main__":
    main()
