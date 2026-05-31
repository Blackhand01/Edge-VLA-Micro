SHELL := /bin/bash
.DEFAULT_GOAL := help

PYTHON ?= python3
MAC_VENV ?= .venv
MAC_PYTHON ?= $(MAC_VENV)/bin/python
JETSON_VENV ?= .venv-jetson
JETSON_PYTHON ?= $(JETSON_VENV)/bin/python
SERVER_URL ?= http://192.168.55.1:8000/process_intent
CONNECTION ?= udpin://0.0.0.0:14540
HOST ?= 0.0.0.0
PORT ?= 8000
CAMERA_INDEX ?= 1
ASR_BACKEND ?= mlx-whisper
WHISPER_MODEL ?= mlx-community/whisper-small.en-mlx
WHISPER_LANGUAGE ?= en
RECORD_SECONDS ?= 5
ASR_TIMEOUT ?= 180
IMAGE_MODE ?= auto
COGNITION_MODEL ?= mlx-community/Qwen2-VL-2B-Instruct-4bit
TELEMETRY_LOG ?= logs/telemetry.csv
JETSON_MONITOR_OUTPUT ?= logs/jetson_telemetry.csv

.PHONY: help setup-mac setup-jetson run-local run-edge-sensor run-edge-brain run-sitl run-qgc monitor-jetson pull-jetson-logs charts-jetson charts-edge charts-local reset-demo-logs sync-jetson test

help:
	@echo "Edge-VLA-Micro targets"
	@echo ""
	@echo "Setup:"
	@echo "  make setup-mac          Install macOS local/sensor dependencies into $(MAC_VENV)"
	@echo "  make setup-jetson       Install Jetson runtime dependencies into $(JETSON_VENV)"
	@echo ""
	@echo "Runtime profiles:"
	@echo "  make run-local          Run the all-in-Mac MLX profile"
	@echo "  make run-edge-sensor    Run Mac smart sensor HTTP client"
	@echo "  make run-edge-brain     Run Jetson FastAPI/SmolVLM/MAVSDK server"
	@echo ""
	@echo "SITL and tooling:"
	@echo "  make run-sitl           Start PX4 jMAVSim"
	@echo "  make run-qgc            Start QGroundControl"
	@echo "  make monitor-jetson     Record tegrastats to $(JETSON_MONITOR_OUTPUT)"
	@echo "  make pull-jetson-logs   Copy Jetson logs into local ./logs"
	@echo "  make charts-jetson      Generate Jetson telemetry summary and charts"
	@echo "  make charts-edge        Generate Mac + Jetson demo latency and TPS charts"
	@echo "  make charts-local       Generate local latency charts from logs/performance.csv"
	@echo "  make reset-demo-logs    Archive local and Jetson logs before a clean demo run"
	@echo "  make sync-jetson        Rsync demo/server files to the Jetson"
	@echo "  make test               Run focused unit tests"

setup-mac:
	$(PYTHON) -m venv $(MAC_VENV)
	$(MAC_PYTHON) -m pip install --upgrade pip
	$(MAC_PYTHON) -m pip install -r requirements.txt
	$(MAC_PYTHON) -m pip install -r requirements-mac-sensor.txt

setup-jetson:
	$(JETSON_PYTHON) -m pip install --upgrade pip
	$(JETSON_PYTHON) -m pip install -r requirements-jetson.txt

run-local:
	COGNITION_MODEL="$(COGNITION_MODEL)" VLM_BACKEND="mlx" TELEMETRY_LOG="$(TELEMETRY_LOG)" ./scripts/run_all_mac_agent.sh

run-edge-sensor:
	SERVER_URL="$(SERVER_URL)" CAMERA_INDEX="$(CAMERA_INDEX)" ASR_BACKEND="$(ASR_BACKEND)" WHISPER_MODEL="$(WHISPER_MODEL)" WHISPER_LANGUAGE="$(WHISPER_LANGUAGE)" RECORD_SECONDS="$(RECORD_SECONDS)" ASR_TIMEOUT="$(ASR_TIMEOUT)" IMAGE_MODE="$(IMAGE_MODE)" TELEMETRY_LOG="$(TELEMETRY_LOG)" ./scripts/run_drone_voice_demo_mac.sh

run-edge-brain:
	CONNECTION="$(CONNECTION)" HOST="$(HOST)" PORT="$(PORT)" TELEMETRY_LOG="$(TELEMETRY_LOG)" ./scripts/run_jetson_cognition_server.sh

run-sitl:
	./scripts/run_jmavsim.sh

run-qgc:
	./scripts/run_qgc.sh

monitor-jetson:
	JETSON_MONITOR_OUTPUT="$(JETSON_MONITOR_OUTPUT)" ./scripts/run_jetson_monitor.sh

pull-jetson-logs:
	./scripts/pull_jetson_logs.sh

charts-jetson:
	$(MAC_PYTHON) scripts/generate_jetson_monitor_charts.py \
		--input logs/jetson_telemetry.csv \
		--summary-output logs/jetson_telemetry_summary.json \
		--output-dir docs/imgs

charts-edge:
	$(MAC_PYTHON) scripts/generate_edge_demo_charts.py \
		--input logs/telemetry.csv \
		--output-dir docs/imgs

charts-local:
	$(MAC_PYTHON) scripts/generate_charts.py \
		--input logs/performance.csv \
		--output-dir docs/imgs

reset-demo-logs:
	./scripts/reset_demo_logs.sh

sync-jetson:
	./scripts/sync_jetson_demo_files.sh

test:
	$(MAC_PYTHON) -m unittest \
		tests.test_validator \
		tests.test_telemetry_logger \
		tests.test_jetson_monitor \
		tests.test_jetson_monitor_charts
