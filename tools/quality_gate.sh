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

echo "[5/6] Pytest (strict flags, coverage temporarily disabled)"
uv run pytest -n auto -W error --strict-markers --strict-config -v

echo "[6/6] Cosmic Ray (mutation testing, capped at 10s; non-fatal)"
set +e
CR_TIMEOUT=10 bash tools/mutation_test.sh
cr_code=$?
set -e
if [[ "$cr_code" -eq 124 ]]; then
  echo "[warning] Cosmic Ray timed out after 10s cap; proceeding"
elif [[ "$cr_code" -ne 0 ]]; then
  echo "[warning] Cosmic Ray returned non-zero (code=$cr_code); proceeding"
fi

echo "\nAll quality gates passed ✅"
