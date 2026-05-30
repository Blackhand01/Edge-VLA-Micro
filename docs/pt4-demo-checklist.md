# PT4 - Demo checklist

Questa checklist serve per preparare la demo senza improvvisare.

## 30 minuti prima

- Alimentare Jetson Orin Nano.
- Collegare Mac e Jetson sulla rete USB/Ethernet `192.168.55.x`.
- Aprire QGroundControl.
- Avviare PX4 jMAVSim.
- Verificare camera MacBook:

```bash
python -m src.tools.vision_probe \
  --camera-index 1 \
  --frame-path tmp/macbook_camera_test.jpg

open tmp/macbook_camera_test.jpg
```

- Cache ASR Mac:

```bash
python -m src.tools.drone_voice_app \
  --server-url http://192.168.55.1:8000/process_intent \
  --text "warmup" \
  --no-image \
  --asr-backend mlx-whisper \
  --whisper-model mlx-community/whisper-small.en-mlx \
  --whisper-language en \
  --timeout 180
```

- Cache SmolVLM Jetson:

```bash
python scripts/jetson_smolvlm_baseline.py \
  --max-new-tokens 8 \
  --image-size 384
```

## 10 minuti prima

Sul Mac, in PX4 `pxh>`:

```bash
mavlink stop -u 14580
mavlink start -x -u 14580 -r 4000000 -m onboard -o 14540 -t 192.168.55.1
```

Sulla Jetson:

```bash
python -m src.tools.heartbeat_monitor \
  --connection udpin://0.0.0.0:14540 \
  --cycles 3 \
  --timeout 30
```

Risultato atteso:

```text
Heartbeat received from PX4/MAVLink system.
Connected to drone.
```

Sincronizzare il codice demo:

```bash
./scripts/sync_jetson_demo_files.sh
```

## Avvio demo

Jetson:

```bash
cd ~/Edge-VLA-Micro
source .venv-jetson/bin/activate

python -m src.api.cognition_server \
  --host 0.0.0.0 \
  --port 8000 \
  --connection udpin://0.0.0.0:14540
```

Equivalente:

```bash
./scripts/run_jetson_cognition_server.sh
```

Mac:

```bash
cd /Users/stefanoroybisignano/Desktop/Edge-VLA-Micro
source .venv/bin/activate

python -m src.tools.drone_voice_app \
  --server-url http://192.168.55.1:8000/process_intent \
  --interactive \
  --asr-backend mlx-whisper \
  --whisper-model mlx-community/whisper-small.en-mlx \
  --whisper-language en \
  --record-seconds 3 \
  --camera-index 1 \
  --image-mode auto \
  --timeout 180
```

Equivalente:

```bash
./scripts/run_drone_voice_demo_mac.sh
```

## Demo script principale

```text
Arm the drone.
Take off.
Move toward the red object.
Move one meter per second.
Move left.
Hold position.
Land.
```

## Cosa mostrare sullo schermo

- QGroundControl: stato drone e movimento.
- Terminale Mac: transcript, frame capture, accepted/rejected action.
- Terminale Jetson: server VLA, MAVSDK, SmolVLM, guardrail logs.
- Dopo target rosso:

```bash
open tmp/mac_sensor_frame.jpg
open tmp/last_detection_debug.jpg
```

## Fallback demo se ASR sbaglia

Scrivere direttamente i comandi nel prompt:

```text
arm the drone
take off
move toward the red object
move one meter per second
hold position
land
```

Questo dimostra comunque il sistema distribuito Mac camera -> Jetson VLA -> PX4.

## Fallback demo se target rosso non viene visto

Mostrare che il sistema fallisce in sicurezza:

```text
[REJECTED] or command=hold target_found=false
```

Poi mostrare `tmp/last_detection_debug.jpg` e spiegare:

```text
The VLM is not trusted alone. A deterministic OpenCV guardrail must confirm
visual evidence before motion toward a target is allowed.
```

## Fallback demo se MAVLink cade

Rieseguire nel prompt PX4:

```bash
mavlink stop -u 14580
mavlink start -x -u 14580 -r 4000000 -m onboard -o 14540 -t 192.168.55.1
```

Poi riavviare il server Jetson.

## Frasi tecniche pronte

```text
This is an edge intelligence split: Mac as a smart sensor node, Jetson as the VLA control core.
```

```text
The Jetson never runs Whisper. It keeps UMA memory for SmolVLM, PX4 telemetry, and MAVSDK control.
```

```text
The VLM output is only a proposal. The deterministic validator and visual guardrail are the authority.
```

```text
The system rejects invalid state transitions, for example arm while airborne or land while grounded.
```
