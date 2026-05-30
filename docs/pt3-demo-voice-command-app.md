# PT3 - Demo: applicazione vocale Edge-VLA

Questa e' la demo finale del progetto: il Mac funziona da nodo sensoriale, la Jetson Orin Nano esegue la cognizione VLA e PX4/QGroundControl mostra l'effetto dei comandi sul drone.

![Edge-VLA demo flow](imgs/demo-flow.svg)

## Obiettivo della demo

Mostrare un ciclo completo di edge autonomy:

1. L'operatore parla in inglese.
2. Il Mac trascrive localmente con MLX Whisper.
3. Il Mac cattura un frame camera solo quando il comando richiede visione.
4. Il Mac invia testo + immagine via HTTP alla Jetson.
5. La Jetson esegue SmolVLM su CUDA, applica guardrail deterministici e valida lo stato PX4.
6. Solo se il comando e' valido, la Jetson invia azioni MAVSDK a PX4.
7. QGroundControl mostra arm, takeoff, offboard movement, hold e land.

## Vincoli rispettati

- Jetson Orin Nano 8GB, JetPack 6.x bare-metal.
- Nessun ASR/Whisper sulla Jetson: l'ASR resta sul Mac.
- Nessun offload CPU/GPU sulla Jetson: niente `device_map="auto"`, niente `max_memory`, niente partizionamento artificiale.
- Inferenza VLA single-device CUDA con SmolVLM.
- Validazione comandi sempre lato Jetson prima di inviare MAVSDK.
- Fallback sicuro: target non trovato, stato non valido o JSON malformato diventano reject/hold.

## Architettura runtime

```text
MacBook
  src.tools.drone_voice_app
  - microfono
  - MLX Whisper small.en
  - camera MacBook
  - HTTP POST /process_intent

Jetson Orin Nano
  src.api.cognition_server
  - SmolVLM su CUDA
  - OpenCV HSV + RGB guardrail
  - CommandValidator
  - MAVSDK dispatch

PX4/QGroundControl
  - jMAVSim SITL
  - MAVLink onboard stream verso Jetson
  - stato reale usato dal validator
```

## Pre-demo: install/cache

### Mac: cache ASR

Usare `small.en` per accento italiano che parla inglese. `tiny` e' piu' veloce ma troppo fragile per comandi brevi.

```bash
cd /Users/stefanoroybisignano/Desktop/Edge-VLA-Micro
source .venv/bin/activate

python -m src.tools.drone_voice_app \
  --server-url http://192.168.55.1:8000/process_intent \
  --text "warmup" \
  --no-image \
  --asr-backend mlx-whisper \
  --whisper-model mlx-community/whisper-small.en-mlx \
  --whisper-language en \
  --timeout 180
```

Il primo run scarica circa 481 MB da Hugging Face. Farlo prima della demo evita download e rate limit live.

### Jetson: cache SmolVLM

Sulla Jetson:

```bash
cd ~/Edge-VLA-Micro
source .venv-jetson/bin/activate

python scripts/jetson_smolvlm_baseline.py \
  --max-new-tokens 8 \
  --image-size 384
```

## File da sincronizzare sulla Jetson

Dopo modifiche locali, sincronizzare almeno:

```bash
cd /Users/stefanoroybisignano/Desktop/Edge-VLA-Micro

./scripts/sync_jetson_demo_files.sh
```

## Terminal layout per la demo

### Terminale 1 - PX4 jMAVSim sul Mac

```bash
cd /Users/stefanoroybisignano/Desktop/Edge-VLA-Micro
./scripts/run_jmavsim.sh
```

Nel prompt `pxh>`:

```bash
mavlink stop -u 14580
mavlink start -x -u 14580 -r 4000000 -m onboard -o 14540 -t 192.168.55.1
```

Verifica con `mavlink status`: l'istanza onboard deve mostrare UDP local port `14580`, remote port `14540`.

### Terminale 2 - QGroundControl sul Mac

```bash
cd /Users/stefanoroybisignano/Desktop/Edge-VLA-Micro
./scripts/run_qgc.sh
```

QGroundControl deve vedere il veicolo SITL prima di iniziare la demo.

### Terminale 3 - Jetson VLA server

```bash
cd ~/Edge-VLA-Micro
source .venv-jetson/bin/activate

python -m src.api.cognition_server \
  --host 0.0.0.0 \
  --port 8000 \
  --connection udpin://0.0.0.0:14540
```

Equivalente breve:

```bash
./scripts/run_jetson_cognition_server.sh
```

Non usare `--dry-run` per la demo live.

### Terminale 4 - Mac voice app

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

Equivalente breve:

```bash
./scripts/run_drone_voice_demo_mac.sh
```

Nel prompt puoi:

- premere Invio e parlare;
- scrivere direttamente il comando per bypassare ASR;
- scrivere `q` per uscire senza traceback.

## Camera

Su questo Mac:

- `camera-index 0`: OBS virtual camera, da evitare nella demo.
- `camera-index 1`: camera MacBook, raccomandata.
- `camera-index 2`: disponibile solo in alcune enumerazioni OpenCV.

Verifica:

```bash
python -m src.tools.vision_probe \
  --camera-index 1 \
  --frame-path tmp/macbook_camera_test.jpg

open tmp/macbook_camera_test.jpg
```

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

## Script consigliato per la demo lineare

La console non impone un ordine: il validator accetta o rifiuta in base allo stato PX4. Per una demo lineare:

```text
Arm the drone.
Take off.
Move toward the red object.
Move one meter per second.
Move left.
Hold position.
Land.
```

Output atteso:

```text
Arm the drone.              -> [ACCEPTED] action=arm state=ARMED
Take off.                   -> [ACCEPTED] action=takeoff state=AIRBORNE
Move toward the red object. -> [INFO] frame=tmp/mac_sensor_frame.jpg
                            -> [INFO] detection_debug=tmp/last_detection_debug.jpg
                            -> [ACCEPTED] action=move_velocity state=OFFBOARD
Move one meter per second.  -> [ACCEPTED] action=move_velocity state=OFFBOARD
Move left.                  -> [ACCEPTED] action=move_velocity state=OFFBOARD
Hold position.              -> [ACCEPTED] action=hold state=AIRBORNE
Land.                       -> [ACCEPTED] action=land state=LANDED/GROUNDED
```

## Vision debug

Il comando visuale salva due immagini sul Mac:

```bash
open tmp/mac_sensor_frame.jpg
open tmp/last_detection_debug.jpg
```

- `tmp/mac_sensor_frame.jpg`: frame 384x384 inviato alla Jetson.
- `tmp/last_detection_debug.jpg`: overlay ritornato dalla Jetson con maschera colore, bounding box e conteggio pixel.

![Red target debug overlay](imgs/red-target-debug-overlay.svg)

Il guardrail rosso usa due condizioni:

1. HSV rosso stretto.
2. Dominanza RGB: il canale R deve essere nettamente maggiore di G e B.

Questo riduce falsi positivi su pelle, labbra e luce calda.

## Stato PX4 e validazione

La Jetson legge lo stato reale PX4 e blocca comandi non coerenti:

- `arm` accettato da `GROUNDED`.
- `takeoff` accettato da `ARMED`.
- `move_velocity` accettato da `AIRBORNE` o `OFFBOARD`.
- `land` accettato da `AIRBORNE`, `OFFBOARD`, `LANDING`.
- `hold` accettato in stati sicuri; se non airborne puo' essere saltato come `skipped_hold_not_airborne`.

Esempio utile da spiegare:

```text
arm the drone while AIRBORNE -> rejected
land while GROUNDED          -> rejected
red target absent            -> hold, target_found=false
```

## Troubleshooting rapido

### `drone is not connected`

La Jetson non sta ricevendo telemetria MAVLink. Controllare:

```bash
mavlink stop -u 14580
mavlink start -x -u 14580 -r 4000000 -m onboard -o 14540 -t 192.168.55.1
```

Poi dalla Jetson:

```bash
python -m src.tools.heartbeat_monitor \
  --connection udpin://0.0.0.0:14540 \
  --cycles 3 \
  --timeout 30
```

### `No speech text captured`

Parlare dopo il prompt, alzare il volume o scrivere direttamente il comando nel prompt per bypassare ASR.

Per demo, `mlx-community/whisper-small.en-mlx` e' piu' robusto di `tiny`.

### `target_found=false`

Il target rosso/blu non e' abbastanza visibile nel frame o il detector e' troppo conservativo. Aprire:

```bash
open tmp/mac_sensor_frame.jpg
open tmp/last_detection_debug.jpg
```

### Non vedo `detection_debug`

Sincronizzare e riavviare Jetson:

```bash
rsync -avh src/perception/guardrails.py \
  ste@192.168.55.1:~/Edge-VLA-Micro/src/perception/guardrails.py

rsync -avh src/api/cognition_server.py \
  ste@192.168.55.1:~/Edge-VLA-Micro/src/api/cognition_server.py
```

### Dopo `land`, `takeoff` viene rifiutato come `LANDED`

Sincronizzare `src/core/drone_snapshot.py` e riavviare il server Jetson.

### ASR capisce male `red object`

Ripetere con frase piu' netta:

```text
Move toward the red object.
```

Se serve, scriverla direttamente nel prompt per dimostrare la parte VLA/vision/control senza ASR.

