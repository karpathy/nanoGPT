#!/usr/bin/env bash
set -euo pipefail

# Quality gate script: lint, type-check, and tests with coverage
# Usage: uv run bash tools/quality_gate.sh

echo "[1/4] Ruff (fix + format)"
uv run ruff check --fix . && uv run ruff format .

echo "[2/4] Pyright"
uv run pyright

echo "[3/4] Mypy (ml_playground package)"
uv run mypy ml_playground

echo "[4/4] Pytest (strict flags, coverage temporarily disabled)"
uv run pytest -n auto -W error --strict-markers --strict-config -v

echo "\nAll quality gates passed ✅"
