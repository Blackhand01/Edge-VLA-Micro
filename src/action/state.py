from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from mavsdk.telemetry import Battery, FlightMode, Health, Position


class DroneControllerError(RuntimeError):
    pass


class DroneStateError(DroneControllerError):
    pass


class DroneHealthError(DroneControllerError):
    pass


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
