# Edge-VLA-Micro

## Live demo — Esecuzione locale

Apri 3 terminali nella root del repository:

```bash
cd /Users/stefanoroybisignano/Desktop/Edge-VLA-Micro
```

Terminale 1: PX4 + jMAVSim

```bash
./scripts/run_jmavsim.sh
```

Attendi righe come:

```text
Simulator connected on TCP port 4560
mavlink ... udp port 14580 remote port 14540
Ready for takeoff
```

Terminale 2: QGroundControl

```bash
./scripts/run_qgc.sh
```

Attendi che QGC mostri il drone.

Terminale 3: main agent con log live (usa la runtime esterna)

```bash
/tmp/edge-vla-live-venv/bin/python main_agent.py \
	--whisper-model base.en \
	--whisper-language en \
	--silence-threshold 0.004 \
	--trailing-silence 1.2 \
	--max-record 7.0
```

Parla in frasi brevi, per esempio:

```text
Arm the drone.
Take off.
Move toward the red object.
Move forward one meter per second.
```

Vedrai log tipo:

```text
[STATE] AIRBORNE | [INPUT] Take off. | [ACTION] EXECUTED:takeoff | [AUDIO_MS] ... | [VISION_MS] ... | [VLM_MS] ... | [TOTAL_LATENCY] ...
```

Per chiudere in sicurezza:

```text
Land.
```

oppure premi `CTRL+C`; dopo `CTRL+C` verifica stato con:

```bash
/tmp/edge-vla-live-venv/bin/python heartbeat.py --cycles 1 --timeout 10 --interval 1
```

Per log persistente VLM/validator:

```bash
tail -f logs/cognition.log
```
