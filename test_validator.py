from __future__ import annotations

import json
import unittest

from command_validator import (
    CommandValidator,
    DroneOperationalState,
    DroneStateSnapshot,
    SafetyViolationError,
)


class CommandValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.airborne_validator = CommandValidator(
            DroneStateSnapshot(
                state=DroneOperationalState.AIRBORNE,
                connected=True,
                battery_remaining=0.95,
            )
        )

    def test_rejects_out_of_scale_velocity(self) -> None:
        raw = json.dumps(
            {
                "command": "move_velocity",
                "velocity_x": 500,
                "velocity_y": 0,
                "velocity_z": 0,
                "yaw_deg": 0,
            }
        )

        with self.assertRaises(SafetyViolationError) as ctx:
            self.airborne_validator.validate(raw)

        self.assertIn("Schema validation failed", str(ctx.exception))
        self.assertIn("velocity_x", str(ctx.exception))

    def test_rejects_missing_field(self) -> None:
        raw = json.dumps(
            {
                "command": "move_velocity",
                "velocity_x": 1.0,
                "velocity_y": 0.0,
                "yaw_deg": 0.0,
            }
        )

        with self.assertRaises(SafetyViolationError) as ctx:
            self.airborne_validator.validate(raw)

        self.assertIn("velocity_z", str(ctx.exception))

    def test_rejects_unknown_command(self) -> None:
        raw = json.dumps({"command": "barrel_roll", "aggressiveness": 10})

        with self.assertRaises(SafetyViolationError) as ctx:
            self.airborne_validator.validate(raw)

        self.assertIn("Schema validation failed", str(ctx.exception))
        self.assertIn("barrel_roll", str(ctx.exception))

    def test_rejects_grounded_move_velocity(self) -> None:
        validator = CommandValidator(
            DroneStateSnapshot(
                state=DroneOperationalState.GROUNDED,
                connected=True,
                battery_remaining=0.95,
            )
        )
        raw = json.dumps(
            {
                "command": "move_velocity",
                "velocity_x": 1.0,
                "velocity_y": 0.0,
                "velocity_z": 0.0,
                "yaw_deg": 0.0,
            }
        )

        with self.assertRaises(SafetyViolationError) as ctx:
            validator.validate(raw)

        self.assertIn("move_velocity not allowed while GROUNDED", str(ctx.exception))

    def test_accepts_coherent_move_velocity(self) -> None:
        raw = json.dumps(
            {
                "command": "move_velocity",
                "velocity_x": 1.0,
                "velocity_y": -0.5,
                "velocity_z": 0.0,
                "yaw_deg": 15.0,
            }
        )

        validated = self.airborne_validator.validate(raw)

        self.assertEqual(validated.name, "move_velocity")
        self.assertEqual(
            validated.controller_kwargs(),
            {"vx": 1.0, "vy": -0.5, "vz": 0.0, "yaw_deg": 15.0},
        )

    def test_accepts_takeoff_only_when_armed(self) -> None:
        validator = CommandValidator(
            DroneStateSnapshot(
                state=DroneOperationalState.ARMED,
                connected=True,
                battery_remaining=0.95,
            )
        )

        validated = validator.validate('{"command": "takeoff"}')

        self.assertEqual(validated.controller_method, "takeoff")

    def test_rejects_non_land_when_battery_low(self) -> None:
        validator = CommandValidator(
            DroneStateSnapshot(
                state=DroneOperationalState.AIRBORNE,
                connected=True,
                battery_remaining=0.10,
            )
        )

        with self.assertRaises(SafetyViolationError) as ctx:
            validator.validate('{"command": "hold"}')

        self.assertIn("battery below mission threshold", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
