#!/usr/bin/env bash
set -euo pipefail

# Simple wrapper to run Cosmic Ray mutation testing with uv when available.
# Uses the init + exec flow compatible with modern Cosmic Ray versions.
# Usage: tools/mutation_test.sh [extra-args]
# Env:
#   CR_TIMEOUT=<seconds>       Apply a timeout wrapper to exec
#   CR_FORCE=1                 Force reset the session DB (delete and re-init)
#   CR_SESSION_SUFFIX=<label>  Use out/cosmic-ray/session-<label>.sqlite
# Extra args are forwarded to `cosmic-ray exec`.

ROOT_DIR="$(dirname "$0")/.."
cd "$ROOT_DIR"

mkdir -p out/cosmic-ray

# Support custom session DB naming for matrix/parallel runs
if [[ -n "${CR_SESSION_SUFFIX:-}" ]]; then
  SESSION_DB="out/cosmic-ray/session-${CR_SESSION_SUFFIX}.sqlite"
else
  SESSION_DB="out/cosmic-ray/session.sqlite"
fi

USE_UV=0
if command -v uv >/dev/null 2>&1; then
  USE_UV=1
fi

run_cmd() {
  if [[ "$USE_UV" -eq 1 ]]; then
    uv run "$@"
  else
    "$@"
  fi
}

# Step 1: initialize session (preserve by default)
echo "[mut] session DB: $SESSION_DB"
if [[ "${CR_FORCE:-0}" == "1" ]]; then
  echo "[mut] CR_FORCE=1 → resetting session"
  rm -f "$SESSION_DB"
fi

if [[ ! -f "$SESSION_DB" ]]; then
  echo "[mut] cosmic-ray init (creating new session)"
  run_cmd cosmic-ray init pyproject.toml "$SESSION_DB" || true
else
  echo "[mut] Reusing existing session (set CR_FORCE=1 to reset)"
fi

# Step 2: execute mutations (apply timeout if requested)
echo "[mut] cosmic-ray exec (timeout: ${CR_TIMEOUT:-none})"
if [[ -n "${CR_TIMEOUT:-}" ]]; then
  run_cmd python tools/with_timeout.py "$CR_TIMEOUT" cosmic-ray exec pyproject.toml "$SESSION_DB" "$@"
else
  run_cmd cosmic-ray exec pyproject.toml "$SESSION_DB" "$@"
fi
