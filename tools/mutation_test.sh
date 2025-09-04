#!/usr/bin/env bash
set -euo pipefail

# Simple wrapper to run Cosmic Ray mutation testing with uv when available.
# Uses the init + exec flow compatible with modern Cosmic Ray versions.
# Usage: tools/mutation_test.sh [extra-args]
# Extra args are forwarded to `cosmic-ray exec`.

ROOT_DIR="$(dirname "$0")/.."
cd "$ROOT_DIR"

mkdir -p out/cosmic-ray
SESSION_DB="out/cosmic-ray/session.sqlite"

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

# Step 1: initialize session (idempotent)
echo "[mut] cosmic-ray init"
run_cmd cosmic-ray init cosmic-ray.toml "$SESSION_DB" || true

# Step 2: execute mutations (apply timeout if requested)
echo "[mut] cosmic-ray exec (timeout: ${CR_TIMEOUT:-none})"
if [[ -n "${CR_TIMEOUT:-}" ]]; then
  run_cmd python tools/with_timeout.py "$CR_TIMEOUT" cosmic-ray exec cosmic-ray.toml "$SESSION_DB" "$@"
else
  run_cmd cosmic-ray exec cosmic-ray.toml "$SESSION_DB" "$@"
fi
