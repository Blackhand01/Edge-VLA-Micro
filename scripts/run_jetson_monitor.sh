#!/usr/bin/env bash
set -euo pipefail

cd "${JETSON_REPO:-$HOME/Edge-VLA-Micro}"
source "${JETSON_VENV:-.venv-jetson/bin/activate}"

python -m src.tools.jetson_monitor \
  --interval-ms "${JETSON_MONITOR_INTERVAL_MS:-1000}" \
  --output "${JETSON_MONITOR_OUTPUT:-logs/jetson_telemetry.csv}" \
  "$@"
