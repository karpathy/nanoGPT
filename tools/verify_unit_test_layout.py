#!/usr/bin/env python3
"""
Verify unit test layout and naming convention.

Rules (one per logical unit):
- Under tests/unit/ml_playground/, each logical unit should have a single test file.
- Logical units can be either:
  * a top-level module in ml_playground/ (e.g. trainer.py -> test_trainer.py), or
  * a subpackage unit (e.g. analysis/ -> test_lit_integration.py, test_sample_quality.py), where each file covers a distinct unit.
- File naming must follow: test_*.py

This script enforces:
- All unit test files match test_*.py
- For top-level modules in ml_playground/ (excluding directories), at most one corresponding test file exists: test_<module>.py
- Reports duplicates and naming deviations with a non-zero exit code.

Usage:
  uv run python tools/verify_unit_test_layout.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "ml_playground"
UNIT_DIR = PROJECT_ROOT / "tests" / "unit" / "ml_playground"

# Subpackages allowed to host multiple logical unit tests
ALLOWED_SUBPACKAGES = {"analysis", "experiments"}


def fail(msg: str) -> None:
    print(f"[unit-layout] ERROR: {msg}")
    sys.exit(1)


def warn(msg: str) -> None:
    print(f"[unit-layout] WARNING: {msg}")


def main() -> int:
    if not SRC_DIR.is_dir():
        fail(f"missing source dir: {SRC_DIR}")
    if not UNIT_DIR.is_dir():
        fail(f"missing unit test dir: {UNIT_DIR}")

    # 1) Enforce naming: all files in unit tree must be test_*.py
    bad_names: list[Path] = []
    for p in UNIT_DIR.rglob("*.py"):
        # allow package configs like conftest.py
        if p.name == "conftest.py":
            continue
        if not p.name.startswith("test_"):
            bad_names.append(p)
    if bad_names:
        lines = "\n".join(
            f" - {p.relative_to(PROJECT_ROOT)}" for p in sorted(bad_names)
        )
        fail(
            f"Non-conforming unit test names found (must start with 'test_'):\n{lines}"
        )

    # 2) Map top-level src modules to expected test file names
    src_modules = [p.stem for p in SRC_DIR.glob("*.py") if p.name != "__init__.py"]
    # Build actual test file map
    test_files = list(UNIT_DIR.glob("test_*.py"))

    # Group test files by their implied module name when matching test_<module>.py
    from collections import defaultdict

    by_module: dict[str, list[Path]] = defaultdict(list)
    for tf in test_files:
        # Only consider files directly under tests/unit/ml_playground/ for this mapping
        if tf.parent != UNIT_DIR:
            continue
        name = tf.stem  # test_<module>
        if name.startswith("test_"):
            mod = name[len("test_") :]
            by_module[mod].append(tf)

    # 3) Enforce at most one test_<module>.py per top-level src module
    dup_modules = {
        m: files
        for m, files in by_module.items()
        if len(files) > 1 and m in src_modules
    }
    if dup_modules:
        lines = []
        for m, files in sorted(dup_modules.items()):
            for f in files:
                lines.append(f" - {m}: {f.relative_to(PROJECT_ROOT)}")
        fail(
            "Multiple unit test files found for the same logical unit (top-level modules):\n"
            + "\n".join(lines)
        )

    # 4) Optionally, warn if a top-level module has no corresponding test file
    missing = [m for m in src_modules if (UNIT_DIR / f"test_{m}.py").exists() is False]
    if missing:
        warn(
            "Top-level modules without a corresponding unit test file: "
            + ", ".join(sorted(missing))
        )

    print("[unit-layout] OK: naming and one-file-per-unit checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
