# Demo Commands

Runbook operativo per avviare Edge-VLA-Micro in due modalita':

1. **Mac + Jetson distributed edge**: Mac come smart sensor node, Jetson come VLA/control brain.
2. **Mac-only local prototype**: tutto sul Mac, utile per test rapidi e regressione funzionale.

---

## Modalita' 1: Mac + Jetson

Questa e' la modalita' principale: ASR e camera sul Mac, SmolVLM/MAVSDK/validator sulla Jetson.

### 0. Mac: sincronizza la Jetson

```bash
cd /Users/stefanoroybisignano/Desktop/Edge-VLA-Micro
source .venv/bin/activate

./scripts/sync_jetson_demo_files.sh
```

Verifica che sulla Jetson sia arrivato il Makefile aggiornato:

```bash
ssh ste@192.168.55.1 'cd ~/Edge-VLA-Micro && make help'
```

### 1. Mac: avvia PX4 jMAVSim

Terminale Mac 1:

```bash
cd /Users/stefanoroybisignano/Desktop/Edge-VLA-Micro
source .venv/bin/activate

make run-sitl
```

Nel prompt PX4 `pxh>`:

```bash
mavlink stop -u 14580
mavlink start -x -u 14580 -r 4000000 -m onboard -o 14540 -t 192.168.55.1
```

### 2. Mac: avvia QGroundControl

Terminale Mac 2:

```bash
cd /Users/stefanoroybisignano/Desktop/Edge-VLA-Micro
source .venv/bin/activate

make run-qgc
```

Aspetta che QGroundControl veda il veicolo SITL.

### 3. Jetson: verifica heartbeat MAVSDK

Terminale Jetson:

```bash
cd ~/Edge-VLA-Micro
source .venv-jetson/bin/activate

python -m src.tools.heartbeat_monitor \
  --connection udpin://0.0.0.0:14540 \
  --cycles 3 \
  --timeout 30
```

Output atteso:

```text
Heartbeat received from PX4/MAVLink system.
Connected to drone.
```

### 4. Jetson: avvia monitor hardware

Terminale Jetson 2:

```bash
cd ~/Edge-VLA-Micro
source .venv-jetson/bin/activate

make monitor-jetson
```

Lascia il terminale aperto. Il CSV viene scritto in:

```text
logs/jetson_telemetry.csv
```

### 5. Jetson: avvia VLA brain/server

Terminale Jetson 3:

```bash
cd ~/Edge-VLA-Micro
source .venv-jetson/bin/activate

make run-edge-brain
```

Fallback equivalente, se `make` non fosse disponibile:

```bash
cd ~/Edge-VLA-Micro
source .venv-jetson/bin/activate

TELEMETRY_LOG=logs/telemetry.csv \
CONNECTION=udpin://0.0.0.0:14540 \
HOST=0.0.0.0 \
PORT=8000 \
./scripts/run_jetson_cognition_server.sh
```

### 6. Mac: avvia voice/camera sensor node

Terminale Mac 3:

```bash
cd /Users/stefanoroybisignano/Desktop/Edge-VLA-Micro
source .venv/bin/activate

make run-edge-sensor
```

Fallback equivalente, piu' esplicito:

```bash
python -m src.tools.drone_voice_app \
  --server-url http://192.168.55.1:8000/process_intent \
  --interactive \
  --asr-backend mlx-whisper \
  --whisper-model mlx-community/whisper-small.en-mlx \
  --whisper-language en \
  --record-seconds 3 \
  --camera-index 1 \
  --image-mode auto \
  --telemetry-log logs/telemetry.csv \
  --timeout 180
```

### 7. Comandi demo

```text
Arm the drone.
Take off.
Move toward the red object.
Move one meter per second.
Move left.
Hold position.
Land.
```

### 8. Debug visuale

Dopo un comando visuale:

```bash
open tmp/mac_sensor_frame.jpg
open tmp/last_detection_debug.jpg
```

### 9. Fine demo: recupera log e genera grafici

Ferma server/monitor Jetson con `Ctrl+C`, poi sul Mac:

```bash
cd /Users/stefanoroybisignano/Desktop/Edge-VLA-Micro
source .venv/bin/activate

make pull-jetson-logs
make charts-jetson
```

Output atteso:

```text
logs/telemetry.csv
logs/jetson_edge_telemetry.csv
logs/jetson_telemetry.csv
logs/jetson_telemetry_summary.json
docs/imgs/jetson_memory_timeseries.png
docs/imgs/jetson_compute_thermal_timeseries.png
```

---

## Modalita' 2: solo Mac

Questa modalita' esegue ASR, camera, VLM, validator e MAVSDK sul Mac. Serve per test rapidi senza Jetson.

### 1. Mac: avvia PX4 jMAVSim

Terminale Mac 1:

```bash
cd /Users/stefanoroybisignano/Desktop/Edge-VLA-Micro
source .venv/bin/activate

make run-sitl
```

### 2. Mac: avvia QGroundControl

Terminale Mac 2:

```bash
cd /Users/stefanoroybisignano/Desktop/Edge-VLA-Micro
source .venv/bin/activate

make run-qgc
```

### 3. Mac: avvia pipeline locale

Terminale Mac 3:

```bash
cd /Users/stefanoroybisignano/Desktop/Edge-VLA-Micro
source .venv/bin/activate

make run-local
```

Fallback equivalente:

```bash
./scripts/run_all_mac_agent.sh
```

### 4. Grafici performance Mac-only

Dopo una sessione locale:

```bash
cd /Users/stefanoroybisignano/Desktop/Edge-VLA-Micro
source .venv/bin/activate

make charts-local
```

Output principali:

```text
logs/performance.csv
docs/imgs/latency_pie_chart.png
docs/imgs/tps_bar_chart.png
```

---

## Comandi supportati

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

## Documenti principali

```text
README.md
ARCHITECTURE.md
DEVELOPER_JOURNAL.md
```
