#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PORT="${PORT:-8888}"

uv run jupyter lab \
    --ip=0.0.0.0 \
    --port="$PORT" \
    --no-browser \
    --ServerApp.token='' \
    --ServerApp.password='' \
    --ServerApp.disable_check_xsrf=True
