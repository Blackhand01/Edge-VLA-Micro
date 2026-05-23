#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PX4_DIR="${ROOT_DIR}/third_party/PX4-Autopilot"

export JAVA_HOME="/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
export PATH="/opt/homebrew/opt/openjdk@17/bin:/opt/homebrew/opt/arm-gcc-bin@13/bin:/opt/homebrew/opt/ccache/libexec:${PATH}"
# jMAVSim's Java3D/JOGL GUI bundles x86_64 native libraries in this PX4 tag.
# Keep the simulator physics running headless on Apple Silicon and use QGC as UI.
export HEADLESS="${HEADLESS:-1}"

cd "${PX4_DIR}"
source .venv/bin/activate
exec make px4_sitl_default jmavsim
