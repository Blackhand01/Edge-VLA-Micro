#!/usr/bin/env bash
set -euo pipefail

SERVER_URL="${SERVER_URL:-http://192.168.55.1:8000/process_intent}"
CAMERA_INDEX="${CAMERA_INDEX:-1}"
WHISPER_MODEL="${WHISPER_MODEL:-mlx-community/whisper-small.en-mlx}"
RECORD_SECONDS="${RECORD_SECONDS:-3}"

python -m src.tools.drone_voice_app \
  --server-url "${SERVER_URL}" \
  --interactive \
  --asr-backend mlx-whisper \
  --whisper-model "${WHISPER_MODEL}" \
  --whisper-language en \
  --record-seconds "${RECORD_SECONDS}" \
  --camera-index "${CAMERA_INDEX}" \
  --image-mode auto \
  --timeout 180
