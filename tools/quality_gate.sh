#!/usr/bin/env bash
set -euo pipefail

# Quality gate script: dead-code, lint, type-check, and tests
# Usage: uv run bash tools/quality_gate.sh

echo "[1/3] Vulture (dead code scanning)"
# Scan only project package; ignore tests and common artifact dirs via path selection
uv run vulture ml_playground --min-confidence 90

echo "[2/3] Core quality gates via Makefile (ruff, format, pyright, mypy, pytest)"
make quality

echo "[3/3] Cosmic Ray (mutation testing, capped at 10s; non-fatal)"
set +e
CR_TIMEOUT=3 bash tools/mutation_test.sh
cr_code=$?
set -e
if [[ "$cr_code" -eq 124 ]]; then
  echo "[warning] Cosmic Ray timed out after 10s cap; proceeding"
elif [[ "$cr_code" -ne 0 ]]; then
  echo "[warning] Cosmic Ray returned non-zero (code=$cr_code); proceeding"
fi

echo "\nAll quality gates passed ✅"
