# Demo Commands

Operational runbook for running Edge-VLA-Micro in two modes:

1. **Mac + Jetson distributed edge**: Mac as smart sensor node, Jetson as VLA/control brain.
2. **Mac-only local prototype**: full stack on the Mac for quick tests and functional regression.

Reference video:

| Mode | Video |
| --- | --- |
| Mac-only local prototype | [Open `vla-demo.mov`](docs/imgs/vla-demo.mov) |

![QGroundControl demo](docs/imgs/qgroundcontrol.png)

---

## Mode 1: Mac + Jetson

This is the primary deployment mode: ASR and camera capture run on the Mac, while SmolVLM, MAVSDK, and validation run on the Jetson.

### 0. Mac: Sync the Jetson

```bash
cd /Users/stefanoroybisignano/Desktop/Edge-VLA-Micro
source .venv/bin/activate

./scripts/sync_jetson_demo_files.sh
```

Verify that the updated Makefile reached the Jetson:

```bash
ssh ste@192.168.55.1 'cd ~/Edge-VLA-Micro && make help'
```

For a clean demo, archive previous runtime data before starting:

```bash
RESET_REMOTE=1 make reset-demo-logs
```

### 1. Mac: Start PX4 jMAVSim

Mac terminal 1:

```bash
cd /Users/stefanoroybisignano/Desktop/Edge-VLA-Micro
source .venv/bin/activate

make run-sitl
```

In the PX4 `pxh>` prompt:

```bash
mavlink stop -u 14580
mavlink start -x -u 14580 -r 4000000 -m onboard -o 14540 -t 192.168.55.1
```

### 2. Mac: Start QGroundControl

Mac terminal 2:

```bash
cd /Users/stefanoroybisignano/Desktop/Edge-VLA-Micro
source .venv/bin/activate

make run-qgc
```

Wait until QGroundControl detects the SITL vehicle.

### 3. Jetson: Verify MAVSDK heartbeat

Jetson terminal:

```bash
cd ~/Edge-VLA-Micro
source .venv-jetson/bin/activate

python -m src.tools.heartbeat_monitor \
  --connection udpin://0.0.0.0:14540 \
  --cycles 3 \
  --timeout 30
```

Expected output:

```text
Heartbeat received from PX4/MAVLink system.
Connected to drone.
```

### 4. Jetson: Start hardware monitoring

Jetson terminal 2:

```bash
cd ~/Edge-VLA-Micro
source .venv-jetson/bin/activate

make monitor-jetson
```

Leave the terminal open during the demo. The monitor records Jetson hardware telemetry for chart generation.

The monitor can generate charts such as:

![Jetson memory chart](docs/imgs/jetson_memory_timeseries.png)

### 5. Jetson: Start the VLA brain/server

Jetson terminal 3:

```bash
cd ~/Edge-VLA-Micro
source .venv-jetson/bin/activate

make run-edge-brain
```

Equivalent fallback if `make` is unavailable:

```bash
cd ~/Edge-VLA-Micro
source .venv-jetson/bin/activate

CONNECTION=udpin://0.0.0.0:14540 \
HOST=0.0.0.0 \
PORT=8000 \
./scripts/run_jetson_cognition_server.sh
```

### 6. Mac: Start the voice/camera sensor node

Mac terminal 3:

```bash
cd /Users/stefanoroybisignano/Desktop/Edge-VLA-Micro
source .venv/bin/activate

make run-edge-sensor
```

The target uses these defaults:

```text
ASR_BACKEND=mlx-whisper
WHISPER_MODEL=mlx-community/whisper-small.en-mlx
WHISPER_LANGUAGE=en
RECORD_SECONDS=5
ASR_TIMEOUT=180
CAMERA_INDEX=1
IMAGE_MODE=auto
```

More explicit fallback:

```bash
python -m src.tools.drone_voice_app \
  --server-url http://192.168.55.1:8000/process_intent \
  --interactive \
  --asr-backend mlx-whisper \
  --whisper-model mlx-community/whisper-small.en-mlx \
  --whisper-language en \
  --record-seconds 5 \
  --camera-index 1 \
  --image-mode auto \
  --timeout 180
```

### 7. Demo commands

```text
Arm the drone.
Take off.
Move toward the red object.
Move one meter per second.
Move left.
Hold position.
Land.
```

### 8. Visual debugging

After a visual command:

```bash
open tmp/mac_sensor_frame.jpg
open tmp/last_detection_debug.jpg
```

Example saved overlay:

![Red target debug overlay](docs/imgs/red_object_detected.png)

### 9. End of demo: Pull telemetry and generate charts

Stop the Jetson server and monitor with `Ctrl+C`, then run on the Mac:

```bash
cd /Users/stefanoroybisignano/Desktop/Edge-VLA-Micro
source .venv/bin/activate

make pull-jetson-logs
make charts-jetson
```

Expected output: updated charts under `docs/imgs/`.

Latest clean measured demo:

| Command | Result | Server latency |
| --- | --- | ---: |
| `Arm the drone.` | `arm` accepted | 1.05 s |
| `Take off!` | `takeoff` accepted | 1.72 s |
| `Move toward the red object.` | visual `move_velocity` | 17.52 s |
| `Move toward the red object.` | visual `move_velocity` | 11.81 s |
| `Move 1 meter per second.` | `move_velocity` fast path | 3.4 ms |

![Jetson compute and thermal chart](docs/imgs/jetson_compute_thermal_timeseries.png)

---

## Mode 2: Mac-only

This mode runs ASR, camera capture, VLM inference, validation, and MAVSDK control on the Mac. Use it for fast tests without the Jetson.

### 1. Mac: Start PX4 jMAVSim

Mac terminal 1:

```bash
cd /Users/stefanoroybisignano/Desktop/Edge-VLA-Micro
source .venv/bin/activate

make run-sitl
```

### 2. Mac: Start QGroundControl

Mac terminal 2:

```bash
cd /Users/stefanoroybisignano/Desktop/Edge-VLA-Micro
source .venv/bin/activate

make run-qgc
```

### 3. Mac: Start the local pipeline

Mac terminal 3:

```bash
cd /Users/stefanoroybisignano/Desktop/Edge-VLA-Micro
source .venv/bin/activate

make run-local
```

Equivalent fallback:

```bash
./scripts/run_all_mac_agent.sh
```

### 4. Mac-only performance charts

After a local session:

```bash
cd /Users/stefanoroybisignano/Desktop/Edge-VLA-Micro
source .venv/bin/activate

make charts-local
```

Primary outputs:

```text
docs/imgs/latency_pie_chart.png
docs/imgs/tps_bar_chart.png
```

---

## Supported Commands

```text
Arm the drone.
Take off.
Hold position.
Land.
Disarm the drone.
Move forward.
Move backward.
Move left.
Move right.
Move forward one meter per second.
Move one meter per second.
Move toward the red object.
Move toward the blue object.
Follow the red object.
Approach the red object.
Stop.
Hover.
Stay.
```

## Primary Documents

```text
README.md
ARCHITECTURE.md
DEVELOPER_JOURNAL.md
```
