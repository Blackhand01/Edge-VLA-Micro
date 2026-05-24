from __future__ import annotations

import contextlib
import io
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cognition_engine import CognitionEngine, CognitionError, CognitionResult
from command_validator import CommandValidator, DroneOperationalState, DroneStateSnapshot, HoldModel


def _airborne_validator() -> CommandValidator:
    return CommandValidator(
        DroneStateSnapshot(
            state=DroneOperationalState.AIRBORNE,
            connected=True,
            battery_remaining=0.90,
        )
    )


def _process_quiet(engine: CognitionEngine, text: str, **kwargs) -> CognitionResult | CognitionError:
    with contextlib.redirect_stdout(io.StringIO()):
        return engine.process_intent(text, **kwargs)


def _command_json(command: str) -> str:
    return f'{{"command":"{command}","target_found":true,"reasoning":"test command"}}'


class CognitionEngineTests(unittest.TestCase):
    def test_warmup_runs_minimal_generator_call(self) -> None:
        observed: dict[str, object] = {}

        def generator(prompt: str, image_path: str | None) -> str:
            observed["prompt"] = prompt
            observed["image_path"] = image_path
            return "x"

        engine = CognitionEngine(generator=generator)

        engine.warmup()

        self.assertEqual(observed, {"prompt": "warmup", "image_path": None})

    def test_process_intent_accepts_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = CognitionEngine(
                validator=_airborne_validator(),
                generator=lambda _, image_path: _command_json("hold"),
                log_path=Path(tmpdir) / "cognition.log",
            )

            result = _process_quiet(engine, "hold position")

            self.assertIsInstance(result, CognitionResult)
            self.assertTrue(result.ok)
            self.assertEqual(result.validated_command.name, "hold")
            self.assertGreaterEqual(result.ttft_ms, 0.0)
            self.assertGreater(result.generated_tokens, 0)
            self.assertGreaterEqual(result.tps, 0.0)
            self.assertFalse(result.vlm_profile.used_streaming)

    def test_process_intent_profiles_streamed_generation(self) -> None:
        def stream_generator(*args, **kwargs):  # noqa: ANN002, ANN003
            del args, kwargs
            yield SimpleNamespace(text='{"command":"hold",', token=1)
            time.sleep(0.001)
            yield SimpleNamespace(text='"target_found":true,', token=2)
            time.sleep(0.001)
            yield SimpleNamespace(text='"reasoning":"test command"}', token=3, generation_tokens=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = CognitionEngine(
                validator=_airborne_validator(),
                log_path=Path(tmpdir) / "cognition.log",
            )
            engine._model = object()
            engine._processor = object()
            engine._mlx_stream_generate = stream_generator

            result = _process_quiet(engine, "hold position")

            self.assertIsInstance(result, CognitionResult)
            self.assertTrue(result.vlm_profile.used_streaming)
            self.assertEqual(result.generated_tokens, 3)
            self.assertGreaterEqual(result.ttft_ms, 0.0)
            self.assertGreater(result.decode_time_ms, 0.0)
            self.assertGreater(result.tps, 0.0)
            self.assertGreaterEqual(result.vlm_profile.safety_ms, 0.0)

    def test_process_intent_extracts_json_from_wrapped_response(self) -> None:
        response = (
            "Here is the command:\n"
            '{"command":"move_velocity","target_found":true,"reasoning":"test command",'
            '"velocity_x":1.0,"velocity_y":0.0,"velocity_z":0.0,"yaw_deg":0.0}\n'
            "fine"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = CognitionEngine(
                validator=_airborne_validator(),
                generator=lambda _, image_path: response,
                log_path=Path(tmpdir) / "cognition.log",
            )

            result = _process_quiet(engine, "move forward slowly")

            self.assertIsInstance(result, CognitionResult)
            self.assertEqual(
                result.validated_command.controller_kwargs(),
                {"vx": 1.0, "vy": 0.0, "vz": 0.0, "yaw_deg": 0.0},
            )

    def test_process_intent_returns_error_for_non_json_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = CognitionEngine(
                validator=_airborne_validator(),
                generator=lambda _, image_path: "I cannot help",
                log_path=Path(tmpdir) / "cognition.log",
            )

            result = _process_quiet(engine, "do something")

            self.assertIsInstance(result, CognitionError)
            self.assertFalse(result.ok)
            self.assertEqual(result.reason, "INVALID_JSON")

    def test_process_intent_returns_error_for_unsafe_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = CognitionEngine(
                validator=_airborne_validator(),
                generator=lambda _, image_path: (
                    '{"command":"move_velocity","target_found":true,"reasoning":"test command","velocity_x":500,'
                    '"velocity_y":0,"velocity_z":0,"yaw_deg":0}'
                ),
                log_path=Path(tmpdir) / "cognition.log",
            )

            result = _process_quiet(engine, "move forward very fast")

            self.assertIsInstance(result, CognitionError)
            self.assertEqual(result.reason, "VALIDATION_REJECTED")
            self.assertIn("velocity_x", result.details or "")

    def test_process_intent_writes_persistent_transaction_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "logs" / "cognition.log"
            engine = CognitionEngine(
                validator=_airborne_validator(),
                generator=lambda _, image_path: _command_json("land"),
                log_path=log_path,
            )

            _process_quiet(engine, "land")

            log_text = log_path.read_text(encoding="utf-8")
            self.assertIn("[PROMPT]", log_text)
            self.assertIn("[RAW_RESPONSE]", log_text)
            self.assertIn("[PARSED_JSON]", log_text)
            self.assertIn("[VALIDATION_STATUS]", log_text)
            self.assertIn("ACCEPTED | command=land", log_text)

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


if __name__ == "__main__":
    unittest.main()
