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
    async def test_audio_defaults_to_english_transcription(self) -> None:
        audio = AudioModule()

        self.assertEqual(audio.language, DEFAULT_WHISPER_LANGUAGE)
        self.assertEqual(audio.language, "en")

    async def test_parser_defaults_to_english_transcription(self) -> None:
        args = build_parser().parse_args([])

        self.assertEqual(args.whisper_language, DEFAULT_WHISPER_LANGUAGE)
        self.assertEqual(args.whisper_language, "en")

    async def test_iteration_executes_validated_command(self) -> None:
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
        vision = FakeVisionModule("/tmp/frame.jpg")
        agent = AgentLoop(
            drone_controller=controller,
            cognition_engine=cognition,
            audio_module=audio,
            vision_module=vision,
        )

        status = await agent.run_iteration()

        self.assertIn("hold", controller.calls)
        self.assertEqual(status.action, "EXECUTED:hold")
        self.assertEqual(vision.capture_calls, 0)
        self.assertEqual(cognition.calls[0][1], None)

    async def test_iteration_logs_blackbox_for_inference(self) -> None:
        command = ValidatedCommand(
            command=LandModel(command="land", target_found=True, reasoning="test command"),
            raw=_command_raw("land"),
        )
        cognition = FakeCognitionEngine(
            [
                CognitionResult(
                    raw_prompt="PROMPT",
                    raw_response='{"command":"land","target_found":true,"reasoning":"test command"}',
                    parsed_json=_command_raw("land"),
                    validated_command=command,
                )
            ]
        )
        controller = FakeDroneController()
        audio = FakeAudioModule(["move forward"])
        vision = FakeVisionModule("/tmp/frame.jpg")
        blackbox = FakeBlackboxLogger()
        agent = AgentLoop(
            drone_controller=controller,
            cognition_engine=cognition,
            audio_module=audio,
            vision_module=vision,
            blackbox_logger=blackbox,
        )

        await agent.run_iteration()

        self.assertEqual(len(blackbox.events), 1)
        image_path, payload = blackbox.events[0]
        self.assertEqual(image_path, "/tmp/frame.jpg")
        self.assertEqual(payload["prompt"], "PROMPT")
        self.assertEqual(payload["validation"]["status"], "ACCEPTED")
        self.assertEqual(payload["validation"]["name"], "land")

    async def test_warmup_and_iteration_use_same_cognition_thread(self) -> None:
        command = ValidatedCommand(
            command=LandModel(command="land", target_found=True, reasoning="test command"),
            raw=_command_raw("land"),
        )
        cognition = ThreadRecordingCognitionEngine(
            [
                CognitionResult(
                    raw_prompt="PROMPT",
                    raw_response='{"command":"land","target_found":true,"reasoning":"test command"}',
                    parsed_json=_command_raw("land"),
                    validated_command=command,
                )
            ]
        )
        controller = FakeDroneController()
        audio = FakeAudioModule(["move forward"])
        vision = FakeVisionModule("/tmp/frame.jpg")
        agent = AgentLoop(
            drone_controller=controller,
            cognition_engine=cognition,
            audio_module=audio,
            vision_module=vision,
        )

        await agent.warmup_cognition()
        await agent.run_iteration()
        await agent.shutdown()

        self.assertEqual([name for name, _ in cognition.thread_ids], ["warmup", "process"])
        self.assertEqual(cognition.thread_ids[0][1], cognition.thread_ids[1][1])

    async def test_iteration_executes_arm_without_vision_capture(self) -> None:
        command = ValidatedCommand(
            command=ArmModel(command="arm", target_found=True, reasoning="test command"),
            raw=_command_raw("arm"),
        )
        cognition = FakeCognitionEngine(
            [
                CognitionResult(
                    raw_prompt="",
                    raw_response='{"command":"arm","target_found":true,"reasoning":"test command"}',
                    parsed_json=_command_raw("arm"),
                    validated_command=command,
                )
            ]
        )
        controller = FakeDroneController()
        controller.state.armed = False
        controller.state.in_air = False
        audio = FakeAudioModule(["arm the drone"])
        vision = FakeVisionModule("/tmp/frame.jpg")
        agent = AgentLoop(
            drone_controller=controller,
            cognition_engine=cognition,
            audio_module=audio,
            vision_module=vision,
        )

        status = await agent.run_iteration()

        self.assertIn("arm", controller.calls)
        self.assertEqual(status.action, "EXECUTED:arm")
        self.assertEqual(vision.capture_calls, 0)
        self.assertEqual(cognition.calls[0][1], None)
