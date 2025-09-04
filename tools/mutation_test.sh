#!/usr/bin/env bash
set -euo pipefail

# Simple wrapper to run Cosmic Ray mutation testing with uv when available.
# Usage: tools/mutation_test.sh [extra-args]
# Example: tools/mutation_test.sh --baseline

ROOT_DIR="$(dirname "$0")/.."
cd "$ROOT_DIR"

CMD=(cosmic-ray run -v -c cosmic-ray.toml)

if command -v uv >/dev/null 2>&1; then
  echo "[info] Using uv to execute cosmic-ray"
  exec uv run "${CMD[@]}" "$@"
else
  echo "[info] Using system python to execute cosmic-ray"
  exec "${CMD[@]}" "$@"
fi
