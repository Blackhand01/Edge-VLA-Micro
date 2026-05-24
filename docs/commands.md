# Demo Commands - 3 Terminals

Open three separate terminals and move to the repository root:

```bash
cd /Users/stefanoroybisignano/Desktop/Edge-VLA-Micro
```

## Terminal 1 - PX4 + jMAVSim

```bash
./scripts/run_jmavsim.sh
```

Wait until PX4 has started SITL and MAVSDK/QGroundControl can connect.

## Terminal 2 - QGroundControl

```bash
./scripts/run_qgc.sh
```

Wait until QGroundControl shows the vehicle.

## Terminal 3 - Edge-VLA Agent

Before the demo, verify that camera index 1 is the correct camera:

```bash
/tmp/edge-vla-live-venv/bin/python vision_probe.py --camera-index 1 --frame-path tmp/probe_index1.jpg
open tmp/probe_index1.jpg
```

Then start the agent:

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

Recommended voice commands:

```text
Arm the drone.
Take off.
Move toward the red object.
Move forward one meter per second.
Hold position.
Land.
```

Useful logs:

```bash
tail -f logs/cognition.log
```

Output blackbox:

```text
logs/sessions/
```

Performance CSV:

```text
logs/performance.csv
```

Generate whitepaper charts:

```bash
/tmp/edge-vla-live-venv/bin/python scripts/generate_charts.py --input logs/performance.csv --output-dir docs
```

Note: do not use `...` in the command line. It is descriptive text only, and argparse treats it as an invalid argument.
