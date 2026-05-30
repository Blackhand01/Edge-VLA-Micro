#!/usr/bin/env bash
set -euo pipefail

JETSON_HOST="${JETSON_HOST:-ste@192.168.55.1}"
JETSON_REPO="${JETSON_REPO:-~/Edge-VLA-Micro}"

rsync -avh src/api/cognition_server.py "${JETSON_HOST}:${JETSON_REPO}/src/api/cognition_server.py"
rsync -avh src/core/drone_snapshot.py "${JETSON_HOST}:${JETSON_REPO}/src/core/drone_snapshot.py"
rsync -avh src/perception/guardrails.py "${JETSON_HOST}:${JETSON_REPO}/src/perception/guardrails.py"
rsync -avh src/perception/smolvlm_runtime.py "${JETSON_HOST}:${JETSON_REPO}/src/perception/smolvlm_runtime.py"
rsync -avh scripts/run_jetson_cognition_server.sh "${JETSON_HOST}:${JETSON_REPO}/scripts/run_jetson_cognition_server.sh"
