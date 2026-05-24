from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional

from mavsdk import System
from mavsdk.action import ActionError
from mavsdk.offboard import OffboardError, VelocityNedYaw
from mavsdk.telemetry import Battery, FlightMode, Health, Position


DEFAULT_CONNECTION = "udpin://0.0.0.0:14540"
DEFAULT_TIMEOUT_S = 20.0
DEFAULT_BATTERY_MIN_REMAINING = 0.20


logger = logging.getLogger(__name__)


class DroneControllerError(RuntimeError):
    """Base exception for deterministic controller failures."""


class DroneStateError(DroneControllerError):
    """Raised when a command is not valid for the current vehicle state."""


class DroneHealthError(DroneControllerError):
    """Raised when health checks fail before a mission command."""


@dataclass(frozen=True)
class HealthReport:
    connected: bool
    armed: bool
    in_air: bool
    armable: bool
    local_position_ok: bool
    global_position_ok: bool
    home_position_ok: bool
    battery_remaining: Optional[float]
    battery_voltage_v: Optional[float]
    flight_mode: Optional[FlightMode]

    @property
    def gps_ok(self) -> bool:
        return self.global_position_ok and self.home_position_ok


@dataclass
class DroneState:
    connected: bool = False
    armed: Optional[bool] = None
    in_air: Optional[bool] = None
    health: Optional[Health] = None
    battery: Optional[Battery] = None
    position: Optional[Position] = None
    flight_mode: Optional[FlightMode] = None


class DroneController:
    """MAVSDK-backed control abstraction for future AI command validators.

    Velocity commands use NED axes:
    - vx: north velocity in m/s
    - vy: east velocity in m/s
    - vz: down velocity in m/s
    - yaw_deg: absolute yaw angle in degrees
    """

    def __init__(
        self,
        connection: str = DEFAULT_CONNECTION,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        battery_min_remaining: float = DEFAULT_BATTERY_MIN_REMAINING,
    ) -> None:
        self.connection = connection
        self.timeout_s = timeout_s
        self.battery_min_remaining = battery_min_remaining
        self.drone = System()
        self.state = DroneState()
        self._watchers: list[asyncio.Task[None]] = []
        self._offboard_active = False
        self._offboard_task: Optional[asyncio.Task[None]] = None
        self._offboard_setpoint = VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
        self._offboard_period_s = 0.1

    async def __aenter__(self) -> "DroneController":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.close()

    async def connect(self) -> None:
        logger.info("Connecting to drone via %s", self.connection)
        await self.drone.connect(system_address=self.connection)
        self._start_watchers()
        await self._wait_until(lambda: self.state.connected, self.timeout_s, "heartbeat timeout")
        await self._wait_until(
            lambda: self.state.armed is not None and self.state.health is not None,
            self.timeout_s,
            "telemetry state timeout",
        )
        logger.info("Drone connected")

    async def close(self) -> None:
        for task in self._watchers:
            task.cancel()

        if self._watchers:
            await asyncio.gather(*self._watchers, return_exceptions=True)

        self._watchers.clear()

    async def check_health(
        self,
        *,
        require_armable: bool = True,
        require_global_position: bool = True,
        require_battery: bool = True,
    ) -> HealthReport:
        await self._wait_until(lambda: self.state.health is not None, self.timeout_s, "health timeout")
        await self._wait_until(lambda: self.state.armed is not None, self.timeout_s, "arming-state timeout")

        if require_battery:
            await self._wait_until(lambda: self.state.battery is not None, self.timeout_s, "battery timeout")

        health = self.state.health
        battery = self.state.battery
        assert health is not None

        report = HealthReport(
            connected=self.state.connected,
            armed=bool(self.state.armed),
            in_air=bool(self.state.in_air),
            armable=health.is_armable,
            local_position_ok=health.is_local_position_ok,
            global_position_ok=health.is_global_position_ok,
            home_position_ok=health.is_home_position_ok,
            battery_remaining=self._normalise_battery_remaining(battery),
            battery_voltage_v=battery.voltage_v if battery is not None else None,
            flight_mode=self.state.flight_mode,
        )

        failures: list[str] = []

        if not report.connected:
            failures.append("not connected")

        if require_armable and not report.armable:
            failures.append("vehicle is not armable")

        if require_global_position and not report.gps_ok:
            failures.append("global/home position is not ready")

        if require_battery:
            if report.battery_remaining is None:
                failures.append("battery telemetry unavailable")
            elif report.battery_remaining < self.battery_min_remaining:
                failures.append(f"battery below threshold: {report.battery_remaining:.0%}")

        if failures:
            raise DroneHealthError("; ".join(failures))

        logger.info(
            "Health OK: armable=%s gps=%s battery=%s armed=%s in_air=%s mode=%s",
            report.armable,
            report.gps_ok,
            "n/a" if report.battery_remaining is None else f"{report.battery_remaining:.0%}",
            report.armed,
            report.in_air,
            report.flight_mode,
        )
        return report

    async def arm(self) -> None:
        await self._require_connected()

        if self.state.armed:
            logger.info("Arm skipped: vehicle is already armed")
            return

        await self.check_health(require_armable=True, require_global_position=True, require_battery=True)
        logger.info("Arming vehicle")

        try:
            await self.drone.action.arm()
        except ActionError as exc:
            raise DroneStateError(f"arm failed: {exc}") from exc

        await self._wait_until(lambda: self.state.armed is True, self.timeout_s, "arm confirmation timeout")
        logger.info("Vehicle armed")

    async def disarm(self) -> None:
        await self._require_connected()

        if not self.state.armed:
            logger.info("Disarm skipped: vehicle is already disarmed")
            return

        await self._wait_until(lambda: self.state.in_air is not None, self.timeout_s, "in-air state timeout")

        if self.state.in_air:
            raise DroneStateError("disarm requires a landed vehicle")

        logger.info("Disarming vehicle")

        try:
            await self.drone.action.disarm()
        except ActionError as exc:
            raise DroneStateError(f"disarm failed: {exc}") from exc

        await self._wait_until(lambda: self.state.armed is False, self.timeout_s, "disarm confirmation timeout")
        logger.info("Vehicle disarmed")

    async def takeoff(self) -> None:
        await self._require_connected()
        await self._require_armed("takeoff")

        if self.state.in_air:
            logger.info("Takeoff skipped: vehicle is already in air")
            return

        logger.info("Commanding takeoff")

        try:
            await self.drone.action.takeoff()
        except ActionError as exc:
            raise DroneStateError(f"takeoff failed: {exc}") from exc

        await self._wait_until(lambda: self.state.in_air is True, self.timeout_s, "takeoff confirmation timeout")
        logger.info("Vehicle is airborne")

    async def land(self) -> None:
        await self._require_connected()

        if not self.state.armed and not self.state.in_air:
            raise DroneStateError("land requires an armed or airborne vehicle")

        if self._offboard_active:
            await self._stop_offboard()

        logger.info("Commanding land")

        try:
            await self.drone.action.land()
        except ActionError as exc:
            raise DroneStateError(f"land failed: {exc}") from exc

        await self._wait_until(lambda: self.state.in_air is False, self.timeout_s * 3, "landing confirmation timeout")
        logger.info("Vehicle landed")

    async def hold(self) -> None:
        await self._require_connected()
        await self._require_armed("hold")
        await self._require_in_air("hold")

        if self._offboard_active:
            await self.drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))
            await self._stop_offboard()

        logger.info("Commanding hold")

        try:
            await self.drone.action.hold()
        except ActionError as exc:
            raise DroneStateError(f"hold failed: {exc}") from exc

    async def move_velocity(self, vx: float, vy: float, vz: float, yaw_deg: float) -> None:
        await self._require_connected()
        await self._require_armed("move_velocity")
        await self._require_in_air("move_velocity")

        setpoint = VelocityNedYaw(vx, vy, vz, yaw_deg)

        try:
            if not self._offboard_active:
                logger.info("Starting OFFBOARD velocity control")
                self._offboard_setpoint = VelocityNedYaw(0.0, 0.0, 0.0, yaw_deg)
                await self.drone.offboard.set_velocity_ned(self._offboard_setpoint)
                await self.drone.offboard.start()
                self._offboard_active = True
                self._offboard_task = asyncio.create_task(self._stream_offboard_setpoints())
                await self._wait_until(
                    lambda: self.state.flight_mode == FlightMode.OFFBOARD,
                    self.timeout_s,
                    "offboard mode confirmation timeout",
                )

            logger.info("Sending velocity setpoint vx=%.2f vy=%.2f vz=%.2f yaw=%.1f", vx, vy, vz, yaw_deg)
            self._offboard_setpoint = setpoint
            await self.drone.offboard.set_velocity_ned(self._offboard_setpoint)
        except OffboardError as exc:
            self._offboard_active = False
            raise DroneStateError(f"offboard velocity command failed: {exc}") from exc

    def _start_watchers(self) -> None:
        if self._watchers:
            return

        self._watchers = [
            asyncio.create_task(self._watch_connection()),
            asyncio.create_task(self._watch_armed()),
            asyncio.create_task(self._watch_in_air()),
            asyncio.create_task(self._watch_health()),
            asyncio.create_task(self._watch_battery()),
            asyncio.create_task(self._watch_position()),
            asyncio.create_task(self._watch_flight_mode()),
        ]

    async def _watch_connection(self) -> None:
        async for connection_state in self.drone.core.connection_state():
            self.state.connected = connection_state.is_connected

    async def _watch_armed(self) -> None:
        async for armed in self.drone.telemetry.armed():
            self.state.armed = armed

    async def _watch_in_air(self) -> None:
        async for in_air in self.drone.telemetry.in_air():
            self.state.in_air = in_air

    async def _watch_health(self) -> None:
        async for health in self.drone.telemetry.health():
            self.state.health = health

    async def _watch_battery(self) -> None:
        async for battery in self.drone.telemetry.battery():
            self.state.battery = battery

    async def _watch_position(self) -> None:
        async for position in self.drone.telemetry.position():
            self.state.position = position

    async def _watch_flight_mode(self) -> None:
        async for flight_mode in self.drone.telemetry.flight_mode():
            self.state.flight_mode = flight_mode

    async def _require_connected(self) -> None:
        if not self.state.connected:
            raise DroneStateError("command requires an active MAVSDK connection")

    async def _require_armed(self, command: str) -> None:
        await self._wait_until(lambda: self.state.armed is not None, self.timeout_s, "arming-state timeout")

        if not self.state.armed:
            raise DroneStateError(f"{command} requires an armed vehicle")

    async def _require_in_air(self, command: str) -> None:
        await self._wait_until(lambda: self.state.in_air is not None, self.timeout_s, "in-air state timeout")

        if not self.state.in_air:
            raise DroneStateError(f"{command} requires an airborne vehicle")

    async def _stop_offboard(self) -> None:
        if not self._offboard_active:
            return

        logger.info("Stopping OFFBOARD mode")
        self._offboard_active = False

        if self._offboard_task is not None:
            self._offboard_task.cancel()
            await asyncio.gather(self._offboard_task, return_exceptions=True)
            self._offboard_task = None

        try:
            await self.drone.offboard.stop()
        except OffboardError as exc:
            raise DroneStateError(f"offboard stop failed: {exc}") from exc

    async def _stream_offboard_setpoints(self) -> None:
        while self._offboard_active:
            try:
                await self.drone.offboard.set_velocity_ned(self._offboard_setpoint)
            except OffboardError:
                logger.exception("OFFBOARD setpoint stream failed")
                self._offboard_active = False
                raise

            await asyncio.sleep(self._offboard_period_s)

    @staticmethod
    def _normalise_battery_remaining(battery: Optional[Battery]) -> Optional[float]:
        if battery is None:
            return None

        remaining = battery.remaining_percent

        if remaining > 1.0:
            return remaining / 100.0

        return remaining

    async def _wait_until(self, predicate: Callable[[], bool], timeout_s: float, label: str) -> None:
        deadline = time.monotonic() + timeout_s

        while time.monotonic() < deadline:
            if predicate():
                return

            await asyncio.sleep(0.1)

        raise DroneStateError(label)
