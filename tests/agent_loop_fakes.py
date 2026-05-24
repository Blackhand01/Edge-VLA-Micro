from __future__ import annotations

import threading
from types import SimpleNamespace

from mavsdk.telemetry import FlightMode

from src.audio import AudioModule, DEFAULT_WHISPER_LANGUAGE
from src.perception import CognitionError, CognitionResult
from src.safety.command_validator import ArmModel, DroneOperationalState, HoldModel, LandModel, ValidatedCommand
from src.perception.vision_module import VisionError


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

    def warmup(self) -> None:
        return


class ThreadRecordingCognitionEngine(FakeCognitionEngine):
    def __init__(self, outputs: list[CognitionResult | CognitionError]) -> None:
        super().__init__(outputs)
        self.thread_ids: list[tuple[str, int]] = []

    def warmup(self) -> None:
        self.thread_ids.append(("warmup", threading.get_ident()))

    def process_intent(self, text: str, image_path: str | None = None, *, drone_state=None):
        self.thread_ids.append(("process", threading.get_ident()))
        return super().process_intent(text, image_path, drone_state=drone_state)


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


class FakeBlackboxLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str | None, dict]] = []
        self.closed = False

    def log_inference(self, *, image_path: str | None, payload: dict) -> None:
        self.events.append((image_path, payload))

    async def close(self) -> None:
        self.closed = True


def _command_raw(command: str) -> dict[str, object]:
    return {
        "command": command,
        "target_found": True,
        "reasoning": "test command",
    }
