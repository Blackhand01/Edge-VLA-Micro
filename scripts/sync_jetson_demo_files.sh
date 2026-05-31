#!/usr/bin/env bash
set -euo pipefail

JETSON_HOST="${JETSON_HOST:-ste@192.168.55.1}"
JETSON_REPO="${JETSON_REPO:-~/Edge-VLA-Micro}"

FILES=(
  src/api/cognition_server.py
  src/core/drone_snapshot.py
  src/monitoring/__init__.py
  src/monitoring/telemetry_logger.py
  src/perception/guardrails.py
  src/perception/smolvlm_runtime.py
  src/tools/jetson_monitor.py
  scripts/run_jetson_cognition_server.sh
  scripts/run_jetson_monitor.sh
  Makefile
)

rsync -avhR "${FILES[@]}" "${JETSON_HOST}:${JETSON_REPO}/"
