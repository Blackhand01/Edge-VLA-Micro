from __future__ import annotations

import unittest

from mavsdk.telemetry import FlightMode

from src.safety.command_validator import DroneOperationalState
from src.audio import AudioModule
from src.core import AgentLoop
from src.core.cli import build_parser
from tests.agent_loop_fakes import (
    FakeAudioModule,
    FakeBlackboxLogger,
    FakeCognitionEngine,
    FakeDroneController,
    FakeVisionModule,
    ThreadRecordingCognitionEngine,
    _command_raw,
    ArmModel,
    CognitionError,
    CognitionResult,
    DEFAULT_WHISPER_LANGUAGE,
    HoldModel,
    LandModel,
    ValidatedCommand,
)


class AgentLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_iteration_emergency_keyword_triggers_hold(self) -> None:
        cognition = FakeCognitionEngine([])
        controller = FakeDroneController()
        audio = FakeAudioModule(["Emergency now"])
        vision = FakeVisionModule()
        agent = AgentLoop(
            drone_controller=controller,
            cognition_engine=cognition,
            audio_module=audio,
            vision_module=vision,
        )

        status = await agent.run_iteration()

        self.assertEqual(controller.calls, ["hold"])
        self.assertTrue(status.action.startswith("HOLD:"))
        self.assertEqual(cognition.calls, [])
        self.assertEqual(vision.capture_calls, 0)

    async def test_iteration_cognition_error_triggers_hold(self) -> None:
        cognition = FakeCognitionEngine(
            [
                CognitionError(
                    reason="INVALID_JSON",
                    raw_prompt="prompt",
                    raw_response="oops",
                    details="not json",
                )
            ]
        )
        controller = FakeDroneController()
        audio = FakeAudioModule(["move forward"])
        vision = FakeVisionModule()
        agent = AgentLoop(
            drone_controller=controller,
            cognition_engine=cognition,
            audio_module=audio,
            vision_module=vision,
        )

        status = await agent.run_iteration()

        self.assertIn("hold", controller.calls)
        self.assertTrue(status.action.startswith("HOLD:"))

    async def test_iteration_cognition_error_skips_hold_when_grounded(self) -> None:
        cognition = FakeCognitionEngine(
            [
                CognitionError(
                    reason="VALIDATION_REJECTED",
                    raw_prompt="prompt",
                    raw_response="{}",
                    details="missing command",
                )
            ]
        )
        controller = FakeDroneController()
        controller.state.armed = False
        controller.state.in_air = False
        audio = FakeAudioModule(["arm the drone"])
        vision = FakeVisionModule()
        agent = AgentLoop(
            drone_controller=controller,
            cognition_engine=cognition,
            audio_module=audio,
            vision_module=vision,
        )

        status = await agent.run_iteration()

        self.assertNotIn("hold", controller.calls)
        self.assertEqual(status.action, "HOLD_SKIPPED:VALIDATION_HOLD:NOT_AIRBORNE_SAFE")

    async def test_iteration_ignores_non_actionable_audio(self) -> None:
        cognition = FakeCognitionEngine([])
        controller = FakeDroneController()
        audio = FakeAudioModule(["and then maybe"])
        vision = FakeVisionModule()
        agent = AgentLoop(
            drone_controller=controller,
            cognition_engine=cognition,
            audio_module=audio,
            vision_module=vision,
        )

        status = await agent.run_iteration()

        self.assertEqual(status.action, "IGNORED:NON_ACTIONABLE_AUDIO")
        self.assertEqual(cognition.calls, [])
        self.assertEqual(vision.capture_calls, 0)

    async def test_iteration_executes_validated_vlm_command_without_transcript_evidence_gate(self) -> None:
        land_command = ValidatedCommand(
            command=LandModel(command="land", target_found=True, reasoning="test command"),
            raw=_command_raw("land"),
        )
        cognition = FakeCognitionEngine(
            [
                CognitionResult(
                    raw_prompt="",
                    raw_response='{"command":"land","target_found":true,"reasoning":"test command"}',
                    parsed_json=_command_raw("land"),
                    validated_command=land_command,
                )
            ]
        )
        controller = FakeDroneController()
        audio = FakeAudioModule(["move forward"])
        vision = FakeVisionModule()
        agent = AgentLoop(
            drone_controller=controller,
            cognition_engine=cognition,
            audio_module=audio,
            vision_module=vision,
        )

        status = await agent.run_iteration()

        self.assertEqual(controller.calls, ["land"])
        self.assertEqual(status.action, "EXECUTED:land")
        self.assertEqual(vision.capture_calls, 1)

    async def test_iteration_continues_text_only_when_vision_fails(self) -> None:
        command = ValidatedCommand(
            command=HoldModel(command="hold", target_found=True, reasoning="test command"),
            raw=_command_raw("hold"),
        )
        cognition = FakeCognitionEngine(
            [
                CognitionResult(
                    raw_prompt="",
                    raw_response='{"command":"hold","target_found":true,"reasoning":"test command"}',
                    parsed_json=_command_raw("hold"),
                    validated_command=command,
                )
            ]
        )
        controller = FakeDroneController()
        audio = FakeAudioModule(["hold position"])
        vision = FakeVisionModule(fail=True)
        agent = AgentLoop(
            drone_controller=controller,
            cognition_engine=cognition,
            audio_module=audio,
            vision_module=vision,
        )

        status = await agent.run_iteration()

        self.assertEqual(status.action, "EXECUTED:hold")
        self.assertEqual(cognition.calls[0][1], None)
