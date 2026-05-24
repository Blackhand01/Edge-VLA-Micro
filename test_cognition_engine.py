from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from cognition_engine import CognitionEngine, CognitionError, CognitionResult
from command_validator import CommandValidator, DroneOperationalState, DroneStateSnapshot


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


class CognitionEngineTests(unittest.TestCase):
    def test_process_intent_accepts_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = CognitionEngine(
                validator=_airborne_validator(),
                generator=lambda _: '{"command":"hold"}',
                log_path=Path(tmpdir) / "cognition.log",
            )

            result = _process_quiet(engine, "mantieni posizione")

            self.assertIsInstance(result, CognitionResult)
            self.assertTrue(result.ok)
            self.assertEqual(result.validated_command.name, "hold")

    def test_process_intent_extracts_json_from_wrapped_response(self) -> None:
        response = (
            "Ecco il comando:\n"
            '{"command":"move_velocity","velocity_x":1.0,"velocity_y":0.0,'
            '"velocity_z":0.0,"yaw_deg":0.0}\n'
            "fine"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = CognitionEngine(
                validator=_airborne_validator(),
                generator=lambda _: response,
                log_path=Path(tmpdir) / "cognition.log",
            )

            result = _process_quiet(engine, "avanza lentamente")

            self.assertIsInstance(result, CognitionResult)
            self.assertEqual(
                result.validated_command.controller_kwargs(),
                {"vx": 1.0, "vy": 0.0, "vz": 0.0, "yaw_deg": 0.0},
            )

    def test_process_intent_returns_error_for_non_json_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = CognitionEngine(
                validator=_airborne_validator(),
                generator=lambda _: "non posso aiutarti",
                log_path=Path(tmpdir) / "cognition.log",
            )

            result = _process_quiet(engine, "fai qualcosa")

            self.assertIsInstance(result, CognitionError)
            self.assertFalse(result.ok)
            self.assertEqual(result.reason, "INVALID_JSON")

    def test_process_intent_returns_error_for_unsafe_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = CognitionEngine(
                validator=_airborne_validator(),
                generator=lambda _: (
                    '{"command":"move_velocity","velocity_x":500,'
                    '"velocity_y":0,"velocity_z":0,"yaw_deg":0}'
                ),
                log_path=Path(tmpdir) / "cognition.log",
            )

            result = _process_quiet(engine, "schizza in avanti")

            self.assertIsInstance(result, CognitionError)
            self.assertEqual(result.reason, "VALIDATION_REJECTED")
            self.assertIn("velocity_x", result.details or "")

    def test_process_intent_writes_persistent_transaction_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "logs" / "cognition.log"
            engine = CognitionEngine(
                validator=_airborne_validator(),
                generator=lambda _: '{"command":"land"}',
                log_path=log_path,
            )

            _process_quiet(engine, "atterra")

            log_text = log_path.read_text(encoding="utf-8")
            self.assertIn("[PROMPT]", log_text)
            self.assertIn("[RAW_RESPONSE]", log_text)
            self.assertIn("[PARSED_JSON]", log_text)
            self.assertIn("[VALIDATION_STATUS]", log_text)
            self.assertIn("ACCEPTED | command=land", log_text)

    def test_process_intent_accepts_live_drone_state_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = CognitionEngine(
                generator=lambda _: '{"command":"hold"}',
                log_path=Path(tmpdir) / "cognition.log",
            )

            result = _process_quiet(
                engine,
                "mantieni posizione",
                drone_state=DroneStateSnapshot(
                    state=DroneOperationalState.AIRBORNE,
                    connected=True,
                    battery_remaining=0.90,
                ),
            )

            self.assertIsInstance(result, CognitionResult)
            self.assertEqual(result.validated_command.name, "hold")


if __name__ == "__main__":
    unittest.main()
