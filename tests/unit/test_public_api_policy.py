from __future__ import annotations

import re
from pathlib import Path

# Enforce that unit tests only import public APIs from ml_playground
# This is a lightweight guard to avoid regressions.

# Point to tests/unit/ml_playground
UNIT_ROOT = Path(__file__).resolve().parent
TESTS_ROOT = UNIT_ROOT / "ml_playground"

# Patterns considered violations in test sources
FORBIDDEN_IMPORT_PATTERN = re.compile(
    r"^\s*from\s+ml_playground\.[^\n]+\s+import\s+.*_\b"
)
FORBIDDEN_ATTR_PATTERN = re.compile(
    r"\bml_playground\.[A-Za-z0-9_.]*\._[A-Za-z0-9_]+\b"
)
FORBIDDEN_ANALYSIS_IMPORT = re.compile(
    r"^\s*from\s+ml_playground\.analysis\.lit_integration\s+import\s+"
)

# Allowlist specific files (none by default)
ALLOWLIST: set[Path] = set()


def _file_violations(p: Path) -> list[str]:
    text = p.read_text(encoding="utf-8", errors="ignore")
    errors: list[str] = []
    for i, line in enumerate(text.splitlines(), start=1):
        # Skip mocking lines that intentionally reference private names in strings
        if "mocker.patch(" in line or "patch(" in line:
            continue
        # Check forbidden imports (public API only)
        if FORBIDDEN_IMPORT_PATTERN.search(line):
            errors.append(f"L{i}: Forbidden private import: {line.strip()}")
        if FORBIDDEN_ANALYSIS_IMPORT.search(line):
            errors.append(f"L{i}: Use public analysis re-export: {line.strip()}")
        # Heuristic: avoid matches that are inside quotes
        if FORBIDDEN_ATTR_PATTERN.search(line):
            dq = line.count('"')
            sq = line.count("'")
            if (dq % 2 == 0) and (sq % 2 == 0):
                errors.append(
                    f"L{i}: Forbidden private attribute access: {line.strip()}"
                )
    return errors


def test_unit_tests_use_public_api_only() -> None:
    # Scan only unit tests under tests/unit/ml_playground/
    assert TESTS_ROOT.exists(), f"Missing tests folder: {TESTS_ROOT}"
    offenders: list[str] = []
    for p in TESTS_ROOT.rglob("test_*.py"):
        if p in ALLOWLIST:
            continue
        errs = _file_violations(p)
        if errs:
            offenders.append(f"{p}:\n  - " + "\n  - ".join(errs))
    assert not offenders, (
        "Found unit tests importing private/internal APIs.\n" + "\n\n".join(offenders)
    )
