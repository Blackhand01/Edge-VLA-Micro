#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable, Optional

try:
    from mavsdk import System
except Exception as exc:  # pragma: no cover - this is a boot-time diagnostic
    print(f"[ERROR] MAVSDK import failed: {exc}", file=sys.stderr)
    print("[HINT] Run: .venv/bin/python -m pip install -r requirements-phase0.txt", file=sys.stderr)
    raise


DEFAULT_CONNECTION = "udpin://0.0.0.0:14540"
DEFAULT_TIMEOUT_S = 20.0


@dataclass
class TelemetryState:
    connected: bool = False
    armed: Optional[bool] = None
    latitude_deg: Optional[float] = None
    longitude_deg: Optional[float] = None
    relative_altitude_m: Optional[float] = None
    absolute_altitude_m: Optional[float] = None
    health_all_ok: Optional[bool] = None


def _run(args: list[str], timeout: float = 2.0) -> str:
    try:
        result = subprocess.run(
            args,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"{type(exc).__name__}: {exc}"
    return result.stdout.strip()


def _mavsdk_version() -> str:
    try:
        return importlib.metadata.version("mavsdk")
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _udp_users(port: int) -> str:
    if shutil.which("lsof") is None:
        return "lsof not available"
    output = _run(["lsof", "-nP", f"-iUDP:{port}"])
    return output if output else "no process reported by lsof"


def _process_snapshot() -> str:
    if shutil.which("pgrep") is None:
        return "pgrep not available"
    output = _run(["pgrep", "-fl", "px4|jmavsim|mavsdk_server|QGroundControl"])
    return output if output else "no px4/jmavsim/mavsdk_server/QGroundControl process found"


def _print_failure_diagnostics(
    reason: str,
    connection: str,
    port: int,
    preflight_udp_users: str,
) -> None:
    print(f"[ERROR] {reason}", file=sys.stderr)
    print("[DIAG] Platform:", platform.platform(), platform.machine(), file=sys.stderr)
    print("[DIAG] Python:", sys.version.replace("\n", " "), file=sys.stderr)
    print("[DIAG] MAVSDK-Python:", _mavsdk_version(), file=sys.stderr)
    print("[DIAG] Connection string:", connection, file=sys.stderr)
    print(f"[DIAG] UDP:{port} users before MAVSDK start:\n{preflight_udp_users}", file=sys.stderr)
    print(f"[DIAG] UDP:{port} users now:\n{_udp_users(port)}", file=sys.stderr)
    print(f"[DIAG] Process snapshot:\n{_process_snapshot()}", file=sys.stderr)
    print("[HINT] If PX4 is not listed, start SITL first:", file=sys.stderr)
    print(
        "       cd third_party/PX4-Autopilot && "
        "source .venv/bin/activate && "
        "JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home "
        "PATH=/opt/homebrew/opt/openjdk@17/bin:/opt/homebrew/opt/arm-gcc-bin@13/bin:$PATH "
        "make px4_sitl_default jmavsim",
        file=sys.stderr,
    )
    print("[HINT] If UDP:14540 is already used before this script starts, stop the duplicate MAVSDK client.", file=sys.stderr)
    print("[HINT] If MAVSDK import/version is wrong, rebuild the local venv from requirements-phase0.txt.", file=sys.stderr)


async def _wait_until(predicate: Callable[[], bool], timeout_s: float, label: str) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.1)
    raise TimeoutError(label)


async def _watch_connection(drone: System, state: TelemetryState) -> None:
    async for connection_state in drone.core.connection_state():
        state.connected = connection_state.is_connected


async def _watch_armed(drone: System, state: TelemetryState) -> None:
    async for armed in drone.telemetry.armed():
        state.armed = armed


async def _watch_position(drone: System, state: TelemetryState) -> None:
    async for position in drone.telemetry.position():
        state.latitude_deg = position.latitude_deg
        state.longitude_deg = position.longitude_deg
        state.relative_altitude_m = position.relative_altitude_m
        state.absolute_altitude_m = position.absolute_altitude_m


async def _watch_health(drone: System, state: TelemetryState) -> None:
    async for health in drone.telemetry.health():
        state.health_all_ok = bool(
            health.is_global_position_ok
            and health.is_home_position_ok
            and health.is_local_position_ok
        )


async def monitor(args: argparse.Namespace) -> int:
    port = args.port
    preflight_udp_users = _udp_users(port)
    connection = args.connection or f"udpin://0.0.0.0:{port}"

    drone = System()

    try:
        await drone.connect(system_address=connection)
    except Exception as exc:
        _print_failure_diagnostics(f"MAVSDK could not open connection: {exc}", connection, port, preflight_udp_users)
        return 2

    state = TelemetryState()
    watchers = [
        asyncio.create_task(_watch_connection(drone, state)),
        asyncio.create_task(_watch_armed(drone, state)),
        asyncio.create_task(_watch_position(drone, state)),
        asyncio.create_task(_watch_health(drone, state)),
    ]

    try:
        await _wait_until(lambda: state.connected, args.timeout, "heartbeat timeout")
        print("[INFO] Heartbeat received from PX4/MAVLink system.")

        await _wait_until(
            lambda: state.armed is not None and state.latitude_deg is not None,
            args.timeout,
            "telemetry timeout",
        )

        if args.require_armed and not state.armed:
            _print_failure_diagnostics(
                "Vehicle is connected but not ARMED. Arm in QGroundControl or PX4 shell before this check.",
                connection,
                port,
                preflight_udp_users,
            )
            return 3

        cycle = 0
        while args.cycles == 0 or cycle < args.cycles:
            if not state.connected:
                _print_failure_diagnostics("Connection dropped after heartbeat.", connection, port, preflight_udp_users)
                return 4

            if args.require_armed and not state.armed:
                _print_failure_diagnostics("Vehicle disarmed during required-armed monitor window.", connection, port, preflight_udp_users)
                return 5

            status = "ARMED" if state.armed else "DISARMED"
            lat = state.latitude_deg if state.latitude_deg is not None else float("nan")
            lon = state.longitude_deg if state.longitude_deg is not None else float("nan")
            rel_alt = state.relative_altitude_m if state.relative_altitude_m is not None else float("nan")
            health = "OK" if state.health_all_ok else "PENDING"

            print(
                f"[INFO] Connected to drone. Status: {status}. "
                f"Lat: {lat:.7f} Lon: {lon:.7f} Altitude: {rel_alt:.1f}m Health: {health}",
                flush=True,
            )
            cycle += 1
            await asyncio.sleep(args.interval)

        return 0
    except TimeoutError as exc:
        _print_failure_diagnostics(str(exc), connection, port, preflight_udp_users)
        return 6
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
        return 130
    finally:
        for task in watchers:
            task.cancel()
        await asyncio.gather(*watchers, return_exceptions=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 0 PX4 SITL MAVSDK heartbeat/telemetry monitor.")
    parser.add_argument("--connection", default=DEFAULT_CONNECTION, help=f"MAVSDK connection string. Default: {DEFAULT_CONNECTION}")
    parser.add_argument("--port", type=int, default=14540, help="UDP port used for diagnostics. Default: 14540")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="Seconds to wait for heartbeat/telemetry.")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between telemetry log lines.")
    parser.add_argument("--cycles", type=int, default=0, help="Number of telemetry cycles to run. 0 means run forever.")
    parser.add_argument("--require-armed", action="store_true", help="Fail if the vehicle is not armed for the whole run.")
    return parser.parse_args()


def main() -> int:
    return asyncio.run(monitor(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
