#!/usr/bin/env bash
set -euo pipefail

# Simple wrapper to run Cosmic Ray mutation testing with uv when available.
# Usage: tools/mutation_test.sh [extra-args]
# Example: tools/mutation_test.sh --baseline

ROOT_DIR="$(dirname "$0")/.."
cd "$ROOT_DIR"

mkdir -p out/cosmic-ray

CMD=(cosmic-ray run -v -c cosmic-ray.toml)

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

if [[ -n "${CR_TIMEOUT:-}" ]]; then
  echo "[info] Enforcing timeout: ${CR_TIMEOUT}s"
  run_cmd python tools/with_timeout.py "$CR_TIMEOUT" "${CMD[@]}" "$@"
else
  run_cmd "${CMD[@]}" "$@"
fi
