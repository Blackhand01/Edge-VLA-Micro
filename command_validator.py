from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, RootModel, ValidationError


logger = logging.getLogger(__name__)


class SafetyViolationError(ValueError):
    """Raised when an LLM command fails schema or safety validation."""

    def __init__(self, reason: str, *, details: Optional[list[str]] = None) -> None:
        self.reason = reason
        self.details = details or []
        message = reason if not self.details else f"{reason}: {'; '.join(self.details)}"
        super().__init__(message)


class DroneOperationalState(str, Enum):
    GROUNDED = "GROUNDED"
    LANDED = "LANDED"
    ARMED = "ARMED"
    AIRBORNE = "AIRBORNE"
    OFFBOARD = "OFFBOARD"
    LANDING = "LANDING"


class DroneStateSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: DroneOperationalState
    connected: bool = True
    battery_remaining: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class CommandModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ArmModel(CommandModel):
    command: Literal["arm"]


class DisarmModel(CommandModel):
    command: Literal["disarm"]


class TakeoffModel(CommandModel):
    command: Literal["takeoff"]


class LandModel(CommandModel):
    command: Literal["land"]


class HoldModel(CommandModel):
    command: Literal["hold"]


class MoveVelocityModel(CommandModel):
    command: Literal["move_velocity"]
    velocity_x: float = Field(ge=-2.0, le=2.0, description="North velocity in m/s")
    velocity_y: float = Field(ge=-2.0, le=2.0, description="East velocity in m/s")
    velocity_z: float = Field(ge=-1.0, le=1.0, description="Down velocity in m/s")
    yaw_deg: float = Field(ge=-180.0, le=180.0, description="Absolute yaw in degrees")


CommandPayload = Annotated[
    Union[ArmModel, DisarmModel, TakeoffModel, LandModel, HoldModel, MoveVelocityModel],
    Field(discriminator="command"),
]


class ActionCommand(RootModel[CommandPayload]):
    pass


@dataclass(frozen=True)
class ValidatedCommand:
    command: CommandPayload
    raw: dict

    @property
    def name(self) -> str:
        return self.command.command

    @property
    def controller_method(self) -> str:
        if self.name == "move_velocity":
            return "move_velocity"

        return self.name

    def controller_kwargs(self) -> dict[str, float]:
        if isinstance(self.command, MoveVelocityModel):
            return {
                "vx": self.command.velocity_x,
                "vy": self.command.velocity_y,
                "vz": self.command.velocity_z,
                "yaw_deg": self.command.yaw_deg,
            }

        return {}


class CommandValidator:
    def __init__(
        self,
        drone_state: DroneStateSnapshot,
        *,
        min_battery_remaining: float = 0.20,
    ) -> None:
        self.drone_state = drone_state
        self.min_battery_remaining = min_battery_remaining

    def validate(self, raw_json: str, drone_state: Optional[DroneStateSnapshot] = None) -> ValidatedCommand:
        state = drone_state or self.drone_state
        raw = self._parse_json(raw_json)
        command = self._validate_schema(raw)
        self._check_business_rules(command, state)
        logger.info("Command accepted: %s in state %s", command.command, state.state.value)
        return ValidatedCommand(command=command, raw=raw)

    def _parse_json(self, raw_json: str) -> dict:
        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            self._reject("Malformed JSON", [str(exc)])

        if not isinstance(parsed, dict):
            self._reject("Command must be a JSON object", [f"got {type(parsed).__name__}"])

        return parsed

    def _validate_schema(self, raw: dict) -> CommandPayload:
        try:
            return ActionCommand.model_validate(raw).root
        except ValidationError as exc:
            details = [
                f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                for error in exc.errors()
            ]
            self._reject("Schema validation failed", details)

    def _check_business_rules(self, command: CommandPayload, state: DroneStateSnapshot) -> None:
        if not state.connected:
            self._reject("Business rule failed", ["drone is not connected"])

        if state.battery_remaining is not None and state.battery_remaining < self.min_battery_remaining:
            if not isinstance(command, (LandModel, DisarmModel)):
                self._reject(
                    "Business rule failed",
                    [f"battery below mission threshold: {state.battery_remaining:.0%}"],
                )

        allowed_states = self._allowed_states_for(command)

        if state.state not in allowed_states:
            allowed = ", ".join(sorted(item.value for item in allowed_states))
            self._reject(
                "Business rule failed",
                [f"{command.command} not allowed while {state.state.value}; allowed states: {allowed}"],
            )

    def _allowed_states_for(self, command: CommandPayload) -> set[DroneOperationalState]:
        if isinstance(command, ArmModel):
            return {DroneOperationalState.GROUNDED}

        if isinstance(command, DisarmModel):
            return {
                DroneOperationalState.GROUNDED,
                DroneOperationalState.LANDED,
            }

        if isinstance(command, TakeoffModel):
            return {DroneOperationalState.ARMED}

        if isinstance(command, LandModel):
            return {
                DroneOperationalState.AIRBORNE,
                DroneOperationalState.OFFBOARD,
                DroneOperationalState.LANDING,
            }

        if isinstance(command, HoldModel):
            return {
                DroneOperationalState.AIRBORNE,
                DroneOperationalState.OFFBOARD,
            }

        if isinstance(command, MoveVelocityModel):
            return {
                DroneOperationalState.AIRBORNE,
                DroneOperationalState.OFFBOARD,
            }

        self._reject("Business rule failed", [f"unsupported command type: {type(command).__name__}"])

    def _reject(self, reason: str, details: list[str]) -> None:
        logger.warning("Command rejected: %s | %s", reason, "; ".join(details))
        raise SafetyViolationError(reason, details=details)
