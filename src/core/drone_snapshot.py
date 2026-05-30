from __future__ import annotations

from mavsdk.telemetry import FlightMode

from src.safety.command_validator import DroneOperationalState, DroneStateSnapshot


class DroneSnapshotBuilder:
    def __init__(self) -> None:
        self.has_been_airborne = False
        self._previous_armed: bool | None = None

    def build(self, controller_state) -> DroneStateSnapshot:
        connected = bool(controller_state.connected)
        armed = bool(controller_state.armed)
        in_air = bool(controller_state.in_air)
        flight_mode = controller_state.flight_mode
        if in_air:
            self.has_been_airborne = True
        if armed and self._previous_armed is False and not in_air:
            self.has_been_airborne = False
        self._previous_armed = armed

        return DroneStateSnapshot(
            state=self._operational_state(connected, armed, in_air, flight_mode),
            connected=connected,
            battery_remaining=normalise_controller_battery(controller_state.battery),
        )

    def _operational_state(self, connected: bool, armed: bool, in_air: bool, flight_mode) -> DroneOperationalState:
        if not connected:
            return DroneOperationalState.GROUNDED
        if in_air and flight_mode == FlightMode.OFFBOARD:
            return DroneOperationalState.OFFBOARD
        if in_air and flight_mode == FlightMode.LAND:
            return DroneOperationalState.LANDING
        if in_air:
            return DroneOperationalState.AIRBORNE
        if armed and (flight_mode == FlightMode.LAND or self.has_been_airborne):
            return DroneOperationalState.LANDED
        if armed:
            return DroneOperationalState.ARMED
        return DroneOperationalState.GROUNDED


def normalise_controller_battery(battery) -> float | None:
    if battery is None:
        return None
    battery_remaining = float(battery.remaining_percent)
    if battery_remaining > 1.0:
        battery_remaining = battery_remaining / 100.0
    return max(0.0, min(1.0, battery_remaining))


def kinematic_state_dict(controller_state, drone_state: DroneStateSnapshot) -> dict:
    flight_mode = controller_state.flight_mode
    return {
        "operational_state": drone_state.state.value,
        "connected": bool(controller_state.connected),
        "armed": bool(controller_state.armed),
        "in_air": bool(controller_state.in_air),
        "flight_mode": flight_mode.name if flight_mode is not None else None,
        "battery_remaining": normalise_controller_battery(controller_state.battery),
    }
