# Edge-VLA-Micro

## Live Demo

Open 3 terminals from the repository root:

```bash
cd /Users/stefanoroybisignano/Desktop/Edge-VLA-Micro
```

Terminal 1: PX4 + jMAVSim

```bash
./scripts/run_jmavsim.sh
```

Wait for lines such as:

```text
Simulator connected on TCP port 4560
mavlink ... udp port 14580 remote port 14540
Ready for takeoff
```

Terminal 2: QGroundControl

```bash
./scripts/run_qgc.sh
```

Wait until QGroundControl shows the drone.

Terminal 3: main agent with live logs, camera index 1, isolated VLM process, and blackbox logging

```bash
/tmp/edge-vla-live-venv/bin/python main_agent.py \
  --camera-index 1 \
  --whisper-model base.en \
  --whisper-language en \
  --silence-threshold 0.015 \
  --trailing-silence 0.6 \
  --max-record 3.0 \
  --cognition-backend process \
  --cognition-timeout 75 \
  --blackbox-dir logs/sessions \
  --performance-log logs/performance.csv
```

Use short voice commands, for example:

```text
Arm the drone.
Take off.
Move toward the red object.
Move forward one meter per second.
```

You should see logs such as:

```text
[STATE] AIRBORNE | [INPUT] Take off. | [ACTION] EXECUTED:takeoff | [AUDIO_MS] ... | [VISION_MS] ... | [VLM_MS] ... | [TOTAL_LATENCY] ...
```

To shut down safely:

```text
Land.
```

or press `CTRL+C`; after `CTRL+C`, verify the state with:

```bash
/tmp/edge-vla-live-venv/bin/python heartbeat.py --cycles 1 --timeout 10 --interval 1
```

For persistent VLM/validator logs:

```bash
tail -f logs/cognition.log
```

To verify the camera before the demo:

```bash
/tmp/edge-vla-live-venv/bin/python vision_probe.py --camera-index 1 --frame-path tmp/probe_index1.jpg
open tmp/probe_index1.jpg
```

Blackbox bundles are saved in:

```text
logs/sessions/
```

Per-run latency metrics are appended to:

```text
logs/performance.csv
```

Generate whitepaper charts after a run with:

```bash
/tmp/edge-vla-live-venv/bin/python scripts/generate_charts.py --input logs/performance.csv --output-dir docs
```

The complete 3-terminal command reference is also available in [docs/commands.md](docs/commands.md).
