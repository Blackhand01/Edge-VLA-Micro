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
    def test_process_intent_drops_extra_telemetry_fields_before_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = CognitionEngine(
                generator=lambda _, image_path: (
                    '{"command":"takeoff","target_found":true,'
                    '"reasoning":"takeoff requested","battery_remaining":1.0}'
                ),
                log_path=Path(tmpdir) / "cognition.log",
            )

            result = _process_quiet(
                engine,
                "Take off.",
                drone_state=DroneStateSnapshot(
                    state=DroneOperationalState.ARMED,
                    connected=True,
                    battery_remaining=1.0,
                ),
            )

            self.assertIsInstance(result, CognitionResult)
            self.assertEqual(result.validated_command.name, "takeoff")
            self.assertNotIn("battery_remaining", result.parsed_json)

    def test_requested_color_treats_head_object_as_red_asr_alias(self) -> None:
        self.assertEqual(CognitionEngine._requested_color("move toward the head object"), "red")

    def test_hsv_guardrail_forces_hold_when_requested_red_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            frame_path = Path(tmpdir) / "frame.jpg"
            frame_path.write_bytes(b"mock-frame")

            engine = CognitionEngine(
                validator=_airborne_validator(),
                generator=lambda _, image_path: (
                    '{"command":"move_velocity","target_found":true,"reasoning":"vlm sees target",'
                    '"velocity_x":1.0,"velocity_y":0.0,"velocity_z":0.0,"yaw_deg":0.0}'
                ),
                log_path=Path(tmpdir) / "cognition.log",
            )

            with patch.object(CognitionEngine, "_frame_contains_hsv_color", return_value=False) as detector:
                result = _process_quiet(engine, "Move toward the red object.", image_path=str(frame_path))

            self.assertIsInstance(result, CognitionResult)
            detector.assert_called_once_with(str(frame_path), "red")
            self.assertIsInstance(result.validated_command.command, HoldModel)
            self.assertEqual(result.validated_command.name, "hold")
            self.assertFalse(result.parsed_json["target_found"])
            self.assertIn("requested red target is absent", result.parsed_json["reasoning"])

    def test_hsv_guardrail_allows_move_when_requested_red_is_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            frame_path = Path(tmpdir) / "frame.jpg"
            frame_path.write_bytes(b"mock-frame")

            engine = CognitionEngine(
                validator=_airborne_validator(),
                generator=lambda _, image_path: (
                    '{"command":"move_velocity","target_found":true,"reasoning":"red target visible",'
                    '"velocity_x":1.0,"velocity_y":0.0,"velocity_z":0.0,"yaw_deg":0.0}'
                ),
                log_path=Path(tmpdir) / "cognition.log",
            )

            with patch.object(CognitionEngine, "_frame_contains_hsv_color", return_value=True) as detector:
                result = _process_quiet(engine, "Move toward the red object.", image_path=str(frame_path))

            self.assertIsInstance(result, CognitionResult)
            detector.assert_called_once_with(str(frame_path), "red")
            self.assertEqual(result.validated_command.name, "move_velocity")
            self.assertTrue(result.parsed_json["target_found"])
            self.assertIn("requested red target is present", result.parsed_json["reasoning"])

    def test_hsv_guardrail_overrides_mismatched_vlm_reasoning(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            frame_path = Path(tmpdir) / "frame.jpg"
            frame_path.write_bytes(b"mock-frame")

            engine = CognitionEngine(
                validator=_airborne_validator(),
                generator=lambda _, image_path: (
                    '{"command":"move_velocity","target_found":true,"reasoning":"red object visible",'
                    '"velocity_x":1.0,"velocity_y":0.0,"velocity_z":0.0,"yaw_deg":0.0}'
                ),
                log_path=Path(tmpdir) / "cognition.log",
            )

            with patch.object(CognitionEngine, "_frame_contains_hsv_color", return_value=True) as detector:
                result = _process_quiet(engine, "Move toward the blue object.", image_path=str(frame_path))

            self.assertIsInstance(result, CognitionResult)
            detector.assert_called_once_with(str(frame_path), "blue")
            self.assertEqual(result.validated_command.name, "move_velocity")
            self.assertEqual(
                result.parsed_json["reasoning"],
                "OpenCV HSV guardrail: requested blue target is present in the frame.",
            )
