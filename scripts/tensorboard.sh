#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PORT="${PORT:-6006}"
LOGDIR="${LOGDIR:-runs}"

mkdir -p "$LOGDIR"

uv run tensorboard \
    --logdir="$LOGDIR" \
    --host=0.0.0.0 \
    --port="$PORT"
