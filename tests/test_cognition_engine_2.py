from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.perception import CognitionEngine, CognitionError, CognitionResult
from src.safety.command_validator import DroneOperationalState, DroneStateSnapshot, HoldModel
from tests.cognition_fixtures import _airborne_validator, _command_json, _process_quiet


class CognitionEngineTests(unittest.TestCase):
    def test_process_intent_accepts_live_drone_state_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = CognitionEngine(
                generator=lambda _, image_path: _command_json("hold"),
                log_path=Path(tmpdir) / "cognition.log",
            )

            result = _process_quiet(
                engine,
                "hold position",
                drone_state=DroneStateSnapshot(
                    state=DroneOperationalState.AIRBORNE,
                    connected=True,
                    battery_remaining=0.90,
                ),
            )

            self.assertIsInstance(result, CognitionResult)
            self.assertEqual(result.validated_command.name, "hold")

    def test_process_intent_passes_image_path_to_generator(self) -> None:
        observed: dict[str, str | None] = {}

        def generator(prompt: str, image_path: str | None) -> str:
            observed["prompt"] = prompt
            observed["image_path"] = image_path
            return _command_json("hold")

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = CognitionEngine(
                validator=_airborne_validator(),
                generator=generator,
                log_path=Path(tmpdir) / "cognition.log",
            )

            result = _process_quiet(engine, "hold position", image_path="/tmp/frame.jpg")

            self.assertIsInstance(result, CognitionResult)
            self.assertEqual(observed["image_path"], "/tmp/frame.jpg")
            self.assertIn("IMAGE_PATH", observed["prompt"] or "")

    def test_process_intent_includes_drone_state_in_prompt(self) -> None:
        observed: dict[str, str | None] = {}

        def generator(prompt: str, image_path: str | None) -> str:
            del image_path
            observed["prompt"] = prompt
            return _command_json("hold")

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = CognitionEngine(
                validator=_airborne_validator(),
                generator=generator,
                log_path=Path(tmpdir) / "cognition.log",
            )

            result = _process_quiet(
                engine,
                "hold position",
                drone_state=DroneStateSnapshot(
                    state=DroneOperationalState.AIRBORNE,
                    connected=True,
                    battery_remaining=0.90,
                ),
            )

            self.assertIsInstance(result, CognitionResult)
            self.assertIn("CURRENT_DRONE_STATE", observed["prompt"] or "")
            self.assertIn("AIRBORNE", observed["prompt"] or "")

    def test_process_intent_repairs_missing_arm_command_from_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = CognitionEngine(
                generator=lambda _, image_path: (
                    '{"target_found":true,'
                    '"reasoning":"The operator appears to request arming, but no command was emitted."}'
                ),
                log_path=Path(tmpdir) / "cognition.log",
            )

            result = _process_quiet(
                engine,
                "Ahem, arm the dead run.",
                drone_state=DroneStateSnapshot(
                    state=DroneOperationalState.GROUNDED,
                    connected=True,
                    battery_remaining=1.0,
                ),
            )

            self.assertIsInstance(result, CognitionResult)
            self.assertEqual(result.validated_command.name, "arm")
            self.assertEqual(result.parsed_json["command"], "arm")

    def test_process_intent_does_not_repair_missing_motion_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = CognitionEngine(
                validator=_airborne_validator(),
                generator=lambda _, image_path: (
                    '{"target_found":true,'
                    '"reasoning":"The operator wants movement, but no command was emitted."}'
                ),
                log_path=Path(tmpdir) / "cognition.log",
            )

            result = _process_quiet(engine, "Move toward the target.")

            self.assertIsInstance(result, CognitionError)
            self.assertEqual(result.reason, "VALIDATION_REJECTED")
            self.assertIn("Unable to extract tag", result.details or "")

    def test_process_intent_missing_target_fields_forces_hold(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = CognitionEngine(
                validator=_airborne_validator(),
                generator=lambda _, image_path: (
                    '{"command":"move_velocity","velocity_x":0.5,'
                    '"velocity_y":0.0,"velocity_z":0.0,"yaw_deg":0.0}'
                ),
                log_path=Path(tmpdir) / "cognition.log",
            )

            result = _process_quiet(engine, "Move forward.")

            self.assertIsInstance(result, CognitionResult)
            self.assertEqual(result.validated_command.name, "hold")
            self.assertFalse(result.parsed_json["target_found"])
            self.assertIn("VLM omitted required", result.parsed_json["reasoning"])

    def test_process_intent_missing_target_fields_on_arm_repairs_as_non_visual(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = CognitionEngine(
                generator=lambda _, image_path: '{"command":"arm"}',
                log_path=Path(tmpdir) / "cognition.log",
            )

            result = _process_quiet(
                engine,
                "Arm the drone.",
                drone_state=DroneStateSnapshot(
                    state=DroneOperationalState.GROUNDED,
                    connected=True,
                    battery_remaining=1.0,
                ),
            )

            self.assertIsInstance(result, CognitionResult)
            self.assertEqual(result.validated_command.name, "arm")
            self.assertTrue(result.parsed_json["target_found"])
            self.assertIn("non-visual command=arm", result.parsed_json["reasoning"])
