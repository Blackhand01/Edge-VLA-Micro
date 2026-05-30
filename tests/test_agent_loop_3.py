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
        blackbox = FakeBlackboxLogger()
        agent = AgentLoop(
            drone_controller=controller,
            cognition_engine=cognition,
            audio_module=audio,
            vision_module=vision,
            blackbox_logger=blackbox,
        )

        await agent.shutdown()

        self.assertEqual(controller.calls, ["land", "close"])
        self.assertTrue(audio.closed)
        self.assertTrue(vision.closed)
        self.assertTrue(blackbox.closed)

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

        controller.state.armed = False
        self.assertEqual(agent._build_drone_snapshot().state, DroneOperationalState.GROUNDED)

        controller.state.armed = True
        self.assertEqual(agent._build_drone_snapshot().state, DroneOperationalState.ARMED)
