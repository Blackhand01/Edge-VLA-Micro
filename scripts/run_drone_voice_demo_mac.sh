#!/usr/bin/env bash
set -euo pipefail

SERVER_URL="${SERVER_URL:-http://192.168.55.1:8000/process_intent}"
CAMERA_INDEX="${CAMERA_INDEX:-1}"
ASR_BACKEND="${ASR_BACKEND:-mlx-whisper}"
WHISPER_MODEL="${WHISPER_MODEL:-mlx-community/whisper-small.en-mlx}"
WHISPER_LANGUAGE="${WHISPER_LANGUAGE:-en}"
RECORD_SECONDS="${RECORD_SECONDS:-5}"
ASR_TIMEOUT="${ASR_TIMEOUT:-180}"
IMAGE_MODE="${IMAGE_MODE:-auto}"

python -m src.tools.drone_voice_app \
  --server-url "${SERVER_URL}" \
  --interactive \
  --asr-backend "${ASR_BACKEND}" \
  --whisper-model "${WHISPER_MODEL}" \
  --whisper-language "${WHISPER_LANGUAGE}" \
  --record-seconds "${RECORD_SECONDS}" \
  --asr-timeout "${ASR_TIMEOUT}" \
  --camera-index "${CAMERA_INDEX}" \
  --image-mode "${IMAGE_MODE}" \
  --telemetry-log "${TELEMETRY_LOG:-logs/telemetry.csv}" \
  --timeout 180
