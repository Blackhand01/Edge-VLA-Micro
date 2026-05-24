# Phase 0: PX4 SITL MAVSDK Link

Pinned setup:

- PX4: `third_party/PX4-Autopilot`, tag `v1.16.2` (ignored by git).
- Python control venv: `.venv`, Python 3.12, `mavsdk==3.15.3`.
- PX4 build venv: `third_party/PX4-Autopilot/.venv`.
- jMAVSim Java: Homebrew `openjdk@17`.
- jMAVSim build: Homebrew `ant`.
- Local PX4 build patch for macOS 26 / AppleClang 21: add `-Wno-vla-cxx-extension` in `third_party/PX4-Autopilot/cmake/px4_add_common_flags.cmake`.

Terminal 1:

```sh
./scripts/run_jmavsim.sh
```

On Apple Silicon this runs jMAVSim headless by default because the bundled Java3D/JOGL GUI native library is x86_64-only in this PX4 tag. QGroundControl is the GUI for this phase.

Terminal 2:

```sh
./scripts/run_qgc.sh
```

Terminal 3:

```sh
.venv/bin/python heartbeat.py
```

Stability gate before any flight-control logic:

```sh
param set COM_DISARM_PRFLT -1
commander arm
.venv/bin/python heartbeat.py --require-armed --cycles 50
```

The heartbeat node never arms, moves, takes off, lands, or sends setpoints. Arm using QGroundControl or the PX4 shell only when validating the armed-state monitor.

---

# Phase 1: Control Abstraction Layer

`drone_controller.py` exposes an importable `DroneController` around MAVSDK.

The mission harness is intentionally explicit:

```sh
.venv/bin/python mission.py
```

Expected preconditions:

- PX4 SITL is running.
- QGroundControl is connected.
- The vehicle is armable.
- For long armed-on-ground validation, disable PX4 preflight auto-disarm in the PX4 shell:

```sh
param set COM_DISARM_PRFLT -1
```

`move_velocity(vx, vy, vz, yaw_deg)` uses OFFBOARD NED velocity setpoints:

- `vx`: north velocity in m/s
- `vy`: east velocity in m/s
- `vz`: down velocity in m/s
- `yaw_deg`: absolute yaw in degrees

Do not run `mission.py` until the Phase 0 heartbeat gate is passing in the same SITL session.

## Safety Validator

`command_validator.py` validates raw LLM JSON before any command reaches `DroneController`.

Run the offline harness:

```sh
.venv/bin/python -m unittest -v test_validator.py
```

Example accepted command:

```json
{"command": "move_velocity", "velocity_x": 1.0, "velocity_y": 0.0, "velocity_z": 0.0, "yaw_deg": 0.0}
```

Example rejected command:

```json
{"command": "move_velocity", "velocity_x": 500, "velocity_y": 0.0, "velocity_z": 0.0, "yaw_deg": 0.0}
```
