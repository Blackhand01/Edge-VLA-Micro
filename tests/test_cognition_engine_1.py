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
