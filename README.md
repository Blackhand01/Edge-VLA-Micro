# Edge-VLA-Micro

Distributed Vision-Language-Action stack for PX4 drones running on Jetson Orin Nano.

Voice -> VLM -> Safety Layer -> PX4

Built for edge robotics under an 8GB memory budget.

[Demo Video](docs/imgs/vla-demo.mov) | [Architecture](ARCHITECTURE.md) | [Developer Journal](DEVELOPER_JOURNAL.md)

![Edge-VLA Action Demo](docs/imgs/action_demo.gif)

## Why it matters

Most Vision-Language-Action systems require cloud inference or workstation-class GPUs. Edge-VLA-Micro demonstrates that a complete Voice-to-Action robotics pipeline can run on a Jetson Orin Nano while maintaining deterministic safety boundaries and PX4 integration.

## Key Results

| Metric | Result |
| --- | --- |
| Platform | Jetson Orin Nano 8GB |
| VLM | SmolVLM-256M |
| Peak RAM | 3.5 GB |
| Swap | 0 MB |
| Voice-to-Action | demonstrated |
| PX4 Integration | MAVSDK |
| Safety Layer | Pydantic + CV + State Machine |

## The Core Rule

The VLM is not a control authority. Perception proposes intent -> Safety authorizes -> Control executes.

## Technical Profile

Edge-VLA-Micro has two runtime profiles:

| Profile | Hardware | Purpose |
| --- | --- | --- |
| Local Mac-only | Apple Silicon Mac | Research-class VLM validation with Qwen2-VL over MLX |
| Edge distributed | Mac + Jetson Orin Nano | SWaP-constrained deployment with Mac as sensor node and Jetson as CUDA VLA/control engine |

The Mac profile can run larger Qwen2-VL MLX models for local spatial-reasoning validation. The Jetson profile runs SmolVLM to stay within the Orin Nano 8GB Unified Memory Architecture budget while preserving PX4/MAVLink control integration.

## Prerequisites

- Apple Silicon Mac for ASR, camera capture, PX4 SITL, QGroundControl, and local MLX tests.
- NVIDIA Jetson Orin Nano 8GB with JetPack 6.x and CUDA-capable NVIDIA PyTorch.
- PX4 SITL with jMAVSim and QGroundControl available from this repository.
- USB-C or routed local network between Mac and Jetson. The default Jetson USB device-mode address is `192.168.55.1`.

## Quickstart

Install dependencies:

```bash
make setup-mac
make setup-jetson
```

Run the Mac-only profile:

```bash
make run-sitl
make run-qgc
make run-local
```

Run the Mac-only profile with explicit Qwen2-VL MLX:

```bash
COGNITION_MODEL=mlx-community/Qwen2-VL-2B-Instruct-4bit make run-local
```

Run the distributed edge profile:

| Terminal | Device | Command |
| --- | --- | --- |
| 1 | Mac | `make run-sitl` |
| 2 | Mac | `make run-qgc` |
| 3 | Jetson | `make monitor-jetson` |
| 4 | Jetson | `make run-edge-brain` |
| 5 | Mac | `make run-edge-sensor` |

In the PX4 `pxh>` shell, route onboard MAVLink traffic to the Jetson:

```bash
mavlink stop -u 14580
mavlink start -x -u 14580 -r 4000000 -m onboard -o 14540 -t 192.168.55.1
```

## Supported Commands

```text
Arm the drone.
Take off.
Move toward the red object.
Move one meter per second.
Move left.
Move right.
Hold position.
Land.
```

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md): system design, deployment profiles, data flow, model selection rationale, telemetry interpretation.
- [DEVELOPER_JOURNAL.md](DEVELOPER_JOURNAL.md): Jetson hardware setup, flashing guide, and chronological engineering problem log.
- [commands.md](commands.md): operational runbook for live demos, SITL routing, monitoring, and supported commands.

## Reporting

Generate charts after a run:

```bash
make pull-jetson-logs
make charts-jetson
make charts-local
```

The generated visual reports are written under `docs/imgs/` and are discussed in [ARCHITECTURE.md](ARCHITECTURE.md).
