#!/usr/bin/env bash
set -euo pipefail

JETSON_HOST="${JETSON_HOST:-ste@192.168.55.1}"
JETSON_REPO="${JETSON_REPO:-~/Edge-VLA-Micro}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-logs}"

mkdir -p "${LOCAL_LOG_DIR}"

rsync -avh \
  "${JETSON_HOST}:${JETSON_REPO}/logs/jetson_telemetry.csv" \
  "${LOCAL_LOG_DIR}/jetson_telemetry.csv"

rsync -avh \
  "${JETSON_HOST}:${JETSON_REPO}/logs/telemetry.csv" \
  "${LOCAL_LOG_DIR}/jetson_edge_telemetry.csv" || true

rsync -avh \
  "${JETSON_HOST}:${JETSON_REPO}/logs/cognition_server.log" \
  "${LOCAL_LOG_DIR}/jetson_cognition_server.log" || true

echo "Pulled Jetson telemetry into ${LOCAL_LOG_DIR}/"
