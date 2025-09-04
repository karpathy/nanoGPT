#!/usr/bin/env bash
set -euo pipefail

# Quality gate script: dead-code, lint, type-check, and tests
# Usage: uv run bash tools/quality_gate.sh

echo "[1/5] Vulture (dead code scanning)"
# Scan only project package; ignore tests and common artifact dirs via path selection
uv run vulture ml_playground --min-confidence 90

echo "[2/5] Ruff (fix + format)"
uv run ruff check --fix . && uv run ruff format .

echo "[3/5] Pyright"
uv run pyright

echo "[4/5] Mypy (ml_playground package)"
uv run mypy ml_playground

echo "[5/5] Pytest (strict flags, coverage temporarily disabled)"
uv run pytest -n auto -W error --strict-markers --strict-config -v

echo "\nAll quality gates passed ✅"
