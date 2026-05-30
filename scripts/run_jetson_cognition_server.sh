#!/usr/bin/env bash
set -euo pipefail

CONNECTION="${CONNECTION:-udpin://0.0.0.0:14540}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

python -m src.api.cognition_server \
  --host "${HOST}" \
  --port "${PORT}" \
  --connection "${CONNECTION}"
