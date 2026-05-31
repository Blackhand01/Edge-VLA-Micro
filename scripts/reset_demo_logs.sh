#!/usr/bin/env bash
set -euo pipefail

JETSON_HOST="${JETSON_HOST:-ste@192.168.55.1}"
JETSON_REPO="${JETSON_REPO:-~/Edge-VLA-Micro}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-logs}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-${LOCAL_LOG_DIR}/archive}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
ARCHIVE_DIR="${ARCHIVE_DIR:-${ARCHIVE_ROOT}/${STAMP}}"
RESET_REMOTE="${RESET_REMOTE:-1}"

LOCAL_FILES=(
  "telemetry.csv"
  "jetson_edge_telemetry.csv"
  "jetson_telemetry.csv"
  "jetson_telemetry_summary.json"
  "jetson_cognition_server.log"
  "cognition.log"
  "performance.csv"
)

LOCAL_DIRS=(
  "sessions"
)

GENERATED_CHARTS=(
  "docs/imgs/jetson_memory_timeseries.png"
  "docs/imgs/jetson_compute_thermal_timeseries.png"
)

mkdir -p "${LOCAL_LOG_DIR}" "${ARCHIVE_DIR}"

for file in "${LOCAL_FILES[@]}"; do
  if [[ -f "${LOCAL_LOG_DIR}/${file}" ]]; then
    mv "${LOCAL_LOG_DIR}/${file}" "${ARCHIVE_DIR}/${file}"
  fi
done

for dir in "${LOCAL_DIRS[@]}"; do
  if [[ -d "${LOCAL_LOG_DIR}/${dir}" ]]; then
    mv "${LOCAL_LOG_DIR}/${dir}" "${ARCHIVE_DIR}/${dir}"
  fi
  mkdir -p "${LOCAL_LOG_DIR}/${dir}"
done

mkdir -p "${ARCHIVE_DIR}/docs-imgs"
for chart in "${GENERATED_CHARTS[@]}"; do
  if [[ -f "${chart}" ]]; then
    mv "${chart}" "${ARCHIVE_DIR}/docs-imgs/$(basename "${chart}")"
  fi
done

if [[ "${RESET_REMOTE}" == "1" ]]; then
  ssh "${JETSON_HOST}" "JETSON_REPO=${JETSON_REPO} STAMP=${STAMP} bash -s" <<'REMOTE'
set -euo pipefail
cd "${JETSON_REPO}"
mkdir -p logs "logs/archive/${STAMP}"
for file in telemetry.csv jetson_telemetry.csv cognition_server.log; do
  if [[ -f "logs/${file}" ]]; then
    mv "logs/${file}" "logs/archive/${STAMP}/${file}"
  fi
done
REMOTE
fi

echo "Local logs archived in ${ARCHIVE_DIR}"
if [[ "${RESET_REMOTE}" == "1" ]]; then
  echo "Remote Jetson logs archived in ${JETSON_REPO}/logs/archive/${STAMP}"
else
  echo "Remote Jetson logs were not touched because RESET_REMOTE=0"
fi
echo "Next demo will write fresh telemetry files."
