from __future__ import annotations

import unittest
from types import SimpleNamespace

from cognition_engine import CognitionError, CognitionResult
from command_validator import HoldModel, ValidatedCommand
from main_agent import AgentLoop


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

    def process_intent(self, text: str, *, drone_state=None):
        self.calls.append((text, drone_state))
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

    async def land(self) -> None:
        self.calls.append("land")

    async def close(self) -> None:
        self.calls.append("close")


class AgentLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_iteration_executes_validated_command(self) -> None:
        command = ValidatedCommand(
            command=HoldModel(command="hold"),
            raw={"command": "hold"},
        )
        cognition = FakeCognitionEngine(
            [
                CognitionResult(
                    raw_prompt="",
                    raw_response='{"command":"hold"}',
                    parsed_json={"command": "hold"},
                    validated_command=command,
                )
            ]
        )
        controller = FakeDroneController()
        audio = FakeAudioModule(["mantieni posizione"])
        agent = AgentLoop(
            drone_controller=controller,
            cognition_engine=cognition,
            audio_module=audio,
        )

        status = await agent.run_iteration()

        self.assertIn("hold", controller.calls)
        self.assertEqual(status.action, "EXECUTED:hold")

    async def test_iteration_emergency_keyword_triggers_hold(self) -> None:
        cognition = FakeCognitionEngine([])
        controller = FakeDroneController()
        audio = FakeAudioModule(["Emergency immediato"])
        agent = AgentLoop(
            drone_controller=controller,
            cognition_engine=cognition,
            audio_module=audio,
        )

        status = await agent.run_iteration()

        self.assertEqual(controller.calls, ["hold"])
        self.assertTrue(status.action.startswith("HOLD:"))
        self.assertEqual(cognition.calls, [])

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
        agent = AgentLoop(
            drone_controller=controller,
            cognition_engine=cognition,
            audio_module=audio,
        )

        status = await agent.run_iteration()

        self.assertIn("hold", controller.calls)
        self.assertTrue(status.action.startswith("HOLD:"))

    async def test_shutdown_lands_and_closes(self) -> None:
        cognition = FakeCognitionEngine([])
        controller = FakeDroneController()
        audio = FakeAudioModule([])
        agent = AgentLoop(
            drone_controller=controller,
            cognition_engine=cognition,
            audio_module=audio,
        )

        await agent.shutdown()

        self.assertEqual(controller.calls, ["land", "close"])
        self.assertTrue(audio.closed)


if __name__ == "__main__":
    unittest.main()
