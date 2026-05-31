# Commands

## Setup

```bash
make setup-mac
```

```bash
make setup-jetson
```

```bash
make test
```

## Sync Mac To Jetson

```bash
cd /Users/stefanoroybisignano/Desktop/Edge-VLA-Micro
source .venv/bin/activate

make sync-jetson
```

```bash
ssh ste@192.168.55.1 'cd ~/Edge-VLA-Micro && make help'
```

## Reset Demo Data

```bash
cd /Users/stefanoroybisignano/Desktop/Edge-VLA-Micro
source .venv/bin/activate

RESET_REMOTE=1 make reset-demo-logs
```

## Distributed Edge Demo

### Terminal 1: Mac PX4 SITL

```bash
cd /Users/stefanoroybisignano/Desktop/Edge-VLA-Micro
source .venv/bin/activate

make run-sitl
```

```bash
mavlink stop -u 14580
mavlink start -x -u 14580 -r 4000000 -m onboard -o 14540 -t 192.168.55.1
```

### Terminal 2: Mac QGroundControl

```bash
cd /Users/stefanoroybisignano/Desktop/Edge-VLA-Micro
source .venv/bin/activate

make run-qgc
```

### Terminal 3: Jetson MAVSDK Heartbeat Check

```bash
cd ~/Edge-VLA-Micro
source .venv-jetson/bin/activate

python -m src.tools.heartbeat_monitor \
  --connection udpin://0.0.0.0:14540 \
  --cycles 3 \
  --timeout 30
```

### Terminal 4: Jetson Monitoring

```bash
cd ~/Edge-VLA-Micro
source .venv-jetson/bin/activate

make monitor-jetson
```

### Terminal 5: Jetson VLA Brain

```bash
cd ~/Edge-VLA-Micro
source .venv-jetson/bin/activate

make run-edge-brain
```

```bash
cd ~/Edge-VLA-Micro
source .venv-jetson/bin/activate

CONNECTION=udpin://0.0.0.0:14540 \
HOST=0.0.0.0 \
PORT=8000 \
./scripts/run_jetson_cognition_server.sh
```

### Terminal 6: Mac Smart Sensor

```bash
cd /Users/stefanoroybisignano/Desktop/Edge-VLA-Micro
source /tmp/edge-vla-demo-venv/bin/activate

make run-edge-sensor
```

```bash
cd /Users/stefanoroybisignano/Desktop/Edge-VLA-Micro
source /tmp/edge-vla-demo-venv/bin/activate

SERVER_URL=http://192.168.55.1:8000/process_intent \
CAMERA_INDEX=1 \
ASR_BACKEND=mlx-whisper \
WHISPER_MODEL=mlx-community/whisper-small.en-mlx \
WHISPER_LANGUAGE=en \
RECORD_SECONDS=5 \
ASR_TIMEOUT=180 \
IMAGE_MODE=auto \
TELEMETRY_LOG=logs/telemetry.csv \
./scripts/run_drone_voice_demo_mac.sh
```

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

## Voice Commands

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

## Visual Debugging

```bash
open tmp/mac_sensor_frame.jpg
open tmp/last_detection_debug.jpg
```

## Pull Jetson Logs And Generate Charts

```bash
cd /Users/stefanoroybisignano/Desktop/Edge-VLA-Micro
source /tmp/edge-vla-demo-venv/bin/activate

make pull-jetson-logs
make charts-jetson
```

## Mac-Only Demo

### Terminal 1: Mac PX4 SITL

```bash
cd /Users/stefanoroybisignano/Desktop/Edge-VLA-Micro
source .venv/bin/activate

make run-sitl
```

### Terminal 2: Mac QGroundControl

```bash
cd /Users/stefanoroybisignano/Desktop/Edge-VLA-Micro
source .venv/bin/activate

make run-qgc
```

### Terminal 3: Mac Local Pipeline

```bash
cd /Users/stefanoroybisignano/Desktop/Edge-VLA-Micro
source .venv/bin/activate

make run-local
```

```bash
cd /Users/stefanoroybisignano/Desktop/Edge-VLA-Micro
source .venv/bin/activate

COGNITION_MODEL=mlx-community/Qwen2-VL-2B-Instruct-4bit make run-local
```

```bash
cd /Users/stefanoroybisignano/Desktop/Edge-VLA-Micro
source .venv/bin/activate

./scripts/run_all_mac_agent.sh
```

## Mac-Only Charts

```bash
cd /Users/stefanoroybisignano/Desktop/Edge-VLA-Micro
source .venv/bin/activate

make charts-local
```

## Latency Profiler

```bash
open docs/index.html
```
