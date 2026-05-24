from __future__ import annotations

import unittest
from types import SimpleNamespace

from mavsdk.telemetry import FlightMode

from cognition_engine import CognitionError, CognitionResult
from command_validator import ArmModel, DroneOperationalState, HoldModel, LandModel, ValidatedCommand
from main_agent import AgentLoop, AudioModule, DEFAULT_WHISPER_LANGUAGE, build_parser
from vision_module import VisionError


class FakeAudioModule:
    def __init__(self, outputs: list[str | None]) -> None:
        self.outputs = list(outputs)
        self.closed = False

    async def transcribe_audio(self) -> str | None:
        if self.outputs:
            return self.outputs.pop(0)
        return None

    async def close(self) -> None:
        self.closed = True


class FakeCognitionEngine:
    def __init__(self, outputs: list[CognitionResult | CognitionError]) -> None:
        self.outputs = list(outputs)
        self.calls: list[tuple[str, object]] = []

    def process_intent(self, text: str, image_path: str | None = None, *, drone_state=None):
        self.calls.append((text, image_path, drone_state))
        if self.outputs:
            return self.outputs.pop(0)
        return CognitionError(reason="INFERENCE_ERROR", raw_prompt="", details="no output configured")


class FakeDroneController:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.state = SimpleNamespace(
            connected=True,
            armed=True,
            in_air=True,
            flight_mode=None,
            battery=SimpleNamespace(remaining_percent=0.8),
        )

    async def connect(self) -> None:
        self.calls.append("connect")

    async def check_health(self, **kwargs) -> None:
        del kwargs
        self.calls.append("check_health")

    async def hold(self) -> None:
        self.calls.append("hold")

    async def arm(self) -> None:
        self.calls.append("arm")

    async def land(self) -> None:
        self.calls.append("land")

    async def close(self) -> None:
        self.calls.append("close")


class FakeVisionModule:
    def __init__(self, image_path: str | None = "/tmp/frame.jpg", *, fail: bool = False) -> None:
        self.image_path = image_path
        self.fail = fail
        self.capture_calls = 0
        self.closed = False

    async def capture_single_frame(self):
        self.capture_calls += 1
        if self.fail:
            raise VisionError("camera unavailable")
        return SimpleNamespace(image_path=self.image_path)

    async def close(self) -> None:
        self.closed = True


def _command_raw(command: str) -> dict[str, object]:
    return {
        "command": command,
        "target_found": True,
        "reasoning": "test command",
    }


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
        audio = FakeAudioModule(["mantieni posizione"])
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

    async def test_iteration_emergency_keyword_triggers_hold(self) -> None:
        cognition = FakeCognitionEngine([])
        controller = FakeDroneController()
        audio = FakeAudioModule(["Emergency immediato"])
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
        audio = FakeAudioModule(["vai avanti"])
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
        audio = FakeAudioModule(["mantieni posizione"])
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

    async def test_iteration_skips_validated_hold_when_grounded(self) -> None:
        command = ValidatedCommand(
            command=HoldModel(command="hold", target_found=False, reasoning="target missing"),
            raw={"command": "hold", "target_found": False, "reasoning": "target missing"},
        )
        cognition = FakeCognitionEngine(
            [
                CognitionResult(
                    raw_prompt="",
                    raw_response='{"command":"hold","target_found":false,"reasoning":"target missing"}',
                    parsed_json={"command": "hold", "target_found": False, "reasoning": "target missing"},
                    validated_command=command,
                )
            ]
        )
        controller = FakeDroneController()
        controller.state.armed = False
        controller.state.in_air = False
        audio = FakeAudioModule(["hold position"])
        vision = FakeVisionModule()
        agent = AgentLoop(
            drone_controller=controller,
            cognition_engine=cognition,
            audio_module=audio,
            vision_module=vision,
        )

        status = await agent.run_iteration()

        self.assertNotIn("hold", controller.calls)
        self.assertEqual(status.action, "HOLD_SKIPPED:NOT_AIRBORNE_SAFE")

    async def test_iteration_skips_validated_hold_when_landed(self) -> None:
        command = ValidatedCommand(
            command=HoldModel(command="hold", target_found=False, reasoning="target missing"),
            raw={"command": "hold", "target_found": False, "reasoning": "target missing"},
        )
        cognition = FakeCognitionEngine(
            [
                CognitionResult(
                    raw_prompt="",
                    raw_response='{"command":"hold","target_found":false,"reasoning":"target missing"}',
                    parsed_json={"command": "hold", "target_found": False, "reasoning": "target missing"},
                    validated_command=command,
                )
            ]
        )
        controller = FakeDroneController()
        controller.state.armed = True
        controller.state.in_air = False
        controller.state.flight_mode = FlightMode.HOLD
        audio = FakeAudioModule(["move toward the red object"])
        vision = FakeVisionModule()
        agent = AgentLoop(
            drone_controller=controller,
            cognition_engine=cognition,
            audio_module=audio,
            vision_module=vision,
        )
        agent._has_been_airborne = True

        status = await agent.run_iteration()

        self.assertNotIn("hold", controller.calls)
        self.assertEqual(status.state, DroneOperationalState.LANDED.value)
        self.assertEqual(status.action, "HOLD_SKIPPED:NOT_AIRBORNE_SAFE")

    async def test_shutdown_lands_and_closes(self) -> None:
        cognition = FakeCognitionEngine([])
        controller = FakeDroneController()
        audio = FakeAudioModule([])
        vision = FakeVisionModule()
        agent = AgentLoop(
            drone_controller=controller,
            cognition_engine=cognition,
            audio_module=audio,
            vision_module=vision,
        )

        await agent.shutdown()

        self.assertEqual(controller.calls, ["land", "close"])
        self.assertTrue(audio.closed)
        self.assertTrue(vision.closed)

    async def test_snapshot_distinguishes_pre_takeoff_armed_from_landed(self) -> None:
        cognition = FakeCognitionEngine([])
        controller = FakeDroneController()
        controller.state.in_air = False
        controller.state.armed = True
        controller.state.flight_mode = FlightMode.HOLD
        audio = FakeAudioModule([])
        vision = FakeVisionModule()
        agent = AgentLoop(
            drone_controller=controller,
            cognition_engine=cognition,
            audio_module=audio,
            vision_module=vision,
        )

        self.assertEqual(agent._build_drone_snapshot().state, DroneOperationalState.ARMED)

        controller.state.in_air = True
        self.assertEqual(agent._build_drone_snapshot().state, DroneOperationalState.AIRBORNE)

        controller.state.in_air = False
        self.assertEqual(agent._build_drone_snapshot().state, DroneOperationalState.LANDED)


if __name__ == "__main__":
    unittest.main()
