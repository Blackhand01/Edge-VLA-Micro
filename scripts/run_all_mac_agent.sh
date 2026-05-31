#!/usr/bin/env bash
set -euo pipefail

CONNECTION="${CONNECTION:-udpin://0.0.0.0:14540}"
CAMERA_INDEX="${CAMERA_INDEX:-1}"
WHISPER_MODEL="${WHISPER_MODEL:-base.en}"
WHISPER_LANGUAGE="${WHISPER_LANGUAGE:-en}"
VLM_BACKEND="${VLM_BACKEND:-mlx}"
COGNITION_MODEL="${COGNITION_MODEL:-mlx-community/Qwen2-VL-2B-Instruct-4bit}"

python -m src.core.cli \
  --connection "${CONNECTION}" \
  --camera-index "${CAMERA_INDEX}" \
  --whisper-model "${WHISPER_MODEL}" \
  --whisper-language "${WHISPER_LANGUAGE}" \
  --silence-threshold "${SILENCE_THRESHOLD:-0.015}" \
  --trailing-silence "${TRAILING_SILENCE:-0.6}" \
  --max-record "${MAX_RECORD:-3.0}" \
  --vlm-backend "${VLM_BACKEND}" \
  --cognition-model "${COGNITION_MODEL}" \
  --cognition-backend "${COGNITION_BACKEND:-process}" \
  --cognition-timeout "${COGNITION_TIMEOUT:-75}" \
  --blackbox-dir "${BLACKBOX_DIR:-logs/sessions}" \
  --performance-log "${PERFORMANCE_LOG:-logs/performance.csv}" \
  --telemetry-log "${TELEMETRY_LOG:-logs/telemetry.csv}"
