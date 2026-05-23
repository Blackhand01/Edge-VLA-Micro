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
