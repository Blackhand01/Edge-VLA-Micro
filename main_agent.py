from __future__ import annotations

import argparse
import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from mavsdk.telemetry import FlightMode

from cognition_engine import CognitionEngine, CognitionError, CognitionResult, DEFAULT_MODEL_ID
from command_validator import (
    CommandValidator,
    DroneOperationalState,
    DroneStateSnapshot,
    ValidatedCommand,
)
from drone_controller import DEFAULT_CONNECTION, DroneController, DroneStateError
from vision_module import VisionError, VisionModule


logger = logging.getLogger(__name__)

EMERGENCY_KEYWORDS = ("stop", "emergency", "emergenza")
EMERGENCY_PATTERN = re.compile(r"\b(stop|emergency|emergenza)\b", re.IGNORECASE)
ACTIONABLE_PATTERN = re.compile(
    r"\b("
    r"arm|armed|disarm|take\s*off|takeoff|launch|land|hold|hover|"
    r"move|forward|backward|left|right|red|object|target|"
    r"mantieni|posizione|vai|avanti"
    r")\b",
    re.IGNORECASE,
)
COMMAND_EVIDENCE_PATTERNS = {
    "arm": re.compile(r"(?<!dis)\barm(?:ed|ing)?\b|\barma(?:re)?\b", re.IGNORECASE),
    "disarm": re.compile(r"\bdisarm(?:ed|ing)?\b|\bdisarma(?:re)?\b", re.IGNORECASE),
    "takeoff": re.compile(r"\btake\s*off\b|\btakeoff\b|\blaunch\b|\bdecol(?:la|lo|lare)\b", re.IGNORECASE),
    "land": re.compile(r"\bland(?:ing)?\b|\batterr(?:a|are|aggio)\b", re.IGNORECASE),
    "hold": re.compile(r"\bhold\b|\bhover\b|\bmantieni\b|\bposizione\b", re.IGNORECASE),
    "move_velocity": re.compile(
        r"\bmove\b|\bforward\b|\bbackward\b|\bleft\b|\bright\b|\bred\b|\bobject\b|\btarget\b|\bavanti\b",
        re.IGNORECASE,
    ),
}


@dataclass(frozen=True)
class LoopStatus:
    state: str
    input_text: str
    action: str
    audio_ms: float
    vision_ms: float
    vlm_ms: float
    total_latency_ms: float


class AudioModule:
    """Async audio capture + transcription with faster-whisper."""

    def __init__(
        self,
        *,
        model_size: str = "tiny",
        language: str = "it",
        sample_rate: int = 16_000,
        block_duration_s: float = 0.10,
        listen_timeout_s: float = 2.0,
        max_record_s: float = 5.0,
        min_record_s: float = 0.30,
        silence_threshold: float = 0.012,
        trailing_silence_s: float = 0.80,
        device: Optional[str] = None,
    ) -> None:
        self.model_size = model_size
        self.language = language
        self.sample_rate = sample_rate
        self.block_size = max(1, int(sample_rate * block_duration_s))
        self.listen_timeout_s = listen_timeout_s
        self.max_record_s = max_record_s
        self.min_record_s = min_record_s
        self.silence_threshold = silence_threshold
        self.trailing_silence_s = trailing_silence_s
        self.device = device

        self._whisper_model = None
        self._whisper_cls = None
        self._sounddevice = None
        self._deps_loaded = False

    async def transcribe_audio(self) -> Optional[str]:
        self._ensure_dependencies_loaded()
        samples = await self._capture_voice_chunk()
        if samples is None:
            return None

        await self._ensure_whisper_model()
        return await asyncio.to_thread(self._transcribe_samples, samples)

    async def close(self) -> None:
        return

    def _ensure_dependencies_loaded(self) -> None:
        if self._deps_loaded:
            return

        try:
            import sounddevice as sounddevice  # pylint: disable=import-outside-toplevel
            from faster_whisper import WhisperModel  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise RuntimeError(
                "Missing audio dependencies. Run: .venv/bin/python -m pip install -r requirements-phase3.txt"
            ) from exc

        self._sounddevice = sounddevice
        self._whisper_cls = WhisperModel
        self._deps_loaded = True

    async def _ensure_whisper_model(self) -> None:
        if self._whisper_model is not None:
            return

        whisper_cls = self._whisper_cls
        assert whisper_cls is not None
        self._whisper_model = await asyncio.to_thread(
            whisper_cls,
            self.model_size,
            device="cpu",
            compute_type="int8",
        )

    async def _capture_voice_chunk(self) -> Optional[np.ndarray]:
        sounddevice = self._sounddevice
        assert sounddevice is not None

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        capture_started_at = loop.time()
        speech_started = False
        speech_started_at = 0.0
        last_speech_at = 0.0
        chunks: list[np.ndarray] = []

        def callback(indata, frames, time_info, status) -> None:
            del frames, time_info
            if status:
                logger.debug("Audio callback status: %s", status)
            mono = np.asarray(indata, dtype=np.float32).reshape(-1).copy()
            loop.call_soon_threadsafe(queue.put_nowait, mono)

        with sounddevice.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.block_size,
            callback=callback,
            device=self.device,
        ):
            while True:
                now = loop.time()
                if not speech_started and (now - capture_started_at) >= self.listen_timeout_s:
                    return None

                if speech_started and (now - speech_started_at) >= self.max_record_s:
                    break

                if speech_started and (now - last_speech_at) >= self.trailing_silence_s:
                    break

                try:
                    chunk = await asyncio.wait_for(queue.get(), timeout=0.25)
                except asyncio.TimeoutError:
                    continue

                rms = float(np.sqrt(np.mean(np.square(chunk)) + 1e-12))
                has_voice = rms >= self.silence_threshold

                if has_voice and not speech_started:
                    speech_started = True
                    speech_started_at = loop.time()
                    last_speech_at = speech_started_at

                if not speech_started:
                    continue

                chunks.append(chunk)
                if has_voice:
                    last_speech_at = loop.time()

        if not chunks:
            return None

        audio = np.concatenate(chunks, axis=0)
        duration_s = audio.shape[0] / float(self.sample_rate)
        if duration_s < self.min_record_s:
            return None

        return audio.astype(np.float32, copy=False)

    def _transcribe_samples(self, samples: np.ndarray) -> Optional[str]:
        assert self._whisper_model is not None
        segments, _ = self._whisper_model.transcribe(
            samples,
            language=self.language,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            vad_filter=True,
        )
        text = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
        return text or None


async def transcribe_audio(audio_module: AudioModule) -> Optional[str]:
    return await audio_module.transcribe_audio()


class AgentLoop:
    def __init__(
        self,
        *,
        drone_controller: DroneController,
        cognition_engine: CognitionEngine,
        audio_module: AudioModule,
        vision_module: VisionModule,
        max_cognition_failures: int = 3,
        idle_sleep_s: float = 0.10,
    ) -> None:
        self.drone_controller = drone_controller
        self.cognition_engine = cognition_engine
        self.audio_module = audio_module
        self.vision_module = vision_module
        self.max_cognition_failures = max(1, max_cognition_failures)
        self.idle_sleep_s = idle_sleep_s
        self._consecutive_cognition_failures = 0
        self._shutdown_done = False
        self._has_been_airborne = False

    async def run(self) -> None:
        try:
            await self.drone_controller.connect()
            await self.drone_controller.check_health(
                require_armable=False,
                require_global_position=False,
                require_battery=False,
            )
            logger.info("Agentic loop started. Emergency keywords: %s", ", ".join(EMERGENCY_KEYWORDS))
            while True:
                await self.run_iteration()
        finally:
            await self.shutdown()

    async def run_iteration(self) -> LoopStatus:
        start = time.perf_counter()
        spoken_text = None
        action = "IDLE"
        audio_ms = 0.0
        vision_ms = 0.0
        vlm_ms = 0.0

        try:
            audio_started = time.perf_counter()
            spoken_text = await transcribe_audio(self.audio_module)
            audio_ms = self._elapsed_ms(audio_started)
            if spoken_text is None:
                await asyncio.sleep(self.idle_sleep_s)
                return self._emit_status(
                    input_text="<silence>",
                    action=action,
                    started_at=start,
                    audio_ms=audio_ms,
                    vision_ms=vision_ms,
                    vlm_ms=vlm_ms,
                )

            if EMERGENCY_PATTERN.search(spoken_text):
                self._consecutive_cognition_failures = 0
                action = await self._safe_hold("EMERGENCY_KEYWORD")
                return self._emit_status(
                    input_text=spoken_text,
                    action=action,
                    started_at=start,
                    audio_ms=audio_ms,
                    vision_ms=vision_ms,
                    vlm_ms=vlm_ms,
                )

            if not ACTIONABLE_PATTERN.search(spoken_text):
                return self._emit_status(
                    input_text=spoken_text,
                    action="IGNORED:NON_ACTIONABLE_AUDIO",
                    started_at=start,
                    audio_ms=audio_ms,
                    vision_ms=vision_ms,
                    vlm_ms=vlm_ms,
                )

            image_path = None
            vision_started = time.perf_counter()
            try:
                frame = await self.vision_module.capture_single_frame()
                image_path = frame.image_path
            except VisionError as exc:
                logger.warning("Vision capture failed; continuing text-only: %s", exc)
            finally:
                vision_ms = self._elapsed_ms(vision_started)

            drone_state = self._build_drone_snapshot()
            vlm_started = time.perf_counter()
            cognition_result = await asyncio.to_thread(
                self.cognition_engine.process_intent,
                spoken_text,
                image_path,
                drone_state=drone_state,
            )
            vlm_ms = self._elapsed_ms(vlm_started)

            if isinstance(cognition_result, CognitionError):
                self._consecutive_cognition_failures += 1
                logger.warning(
                    "Cognition rejected input: reason=%s details=%s",
                    cognition_result.reason,
                    cognition_result.details,
                )
                hold_reason = "VALIDATION_HOLD"
                if self._consecutive_cognition_failures >= self.max_cognition_failures:
                    hold_reason = "COGNITION_FAILSAFE_HOLD"
                    self._consecutive_cognition_failures = 0

                action = await self._safe_hold(hold_reason)
                return self._emit_status(
                    input_text=spoken_text,
                    action=action,
                    started_at=start,
                    audio_ms=audio_ms,
                    vision_ms=vision_ms,
                    vlm_ms=vlm_ms,
                )

            assert isinstance(cognition_result, CognitionResult)
            command = cognition_result.validated_command
            if not self._command_has_transcript_evidence(command.name, spoken_text):
                logger.warning(
                    "Command rejected after VLM: command=%s transcript=%r",
                    command.name,
                    spoken_text,
                )
                action = await self._safe_hold("TRANSCRIPT_COMMAND_MISMATCH")
                return self._emit_status(
                    input_text=spoken_text,
                    action=action,
                    started_at=start,
                    audio_ms=audio_ms,
                    vision_ms=vision_ms,
                    vlm_ms=vlm_ms,
                )

            self._consecutive_cognition_failures = 0
            action = await self._dispatch_command(command)
            return self._emit_status(
                input_text=spoken_text,
                action=action,
                started_at=start,
                audio_ms=audio_ms,
                vision_ms=vision_ms,
                vlm_ms=vlm_ms,
            )
        except Exception:
            logger.exception("Agent loop iteration failed")
            action = await self._safe_hold("LOOP_EXCEPTION_HOLD")
            return self._emit_status(
                input_text=spoken_text or "<error>",
                action=action,
                started_at=start,
                audio_ms=audio_ms,
                vision_ms=vision_ms,
                vlm_ms=vlm_ms,
            )

    async def shutdown(self) -> None:
        if self._shutdown_done:
            return

        self._shutdown_done = True
        logger.info("Shutting down main agent")

        try:
            if self.drone_controller.state.connected and (
                self.drone_controller.state.armed or self.drone_controller.state.in_air
            ):
                try:
                    await self.drone_controller.land()
                    logger.info("Safe shutdown: land command sent")
                except Exception as exc:  # noqa: BLE001 - shutdown must not mask the live-run result.
                    logger.warning("Safe shutdown landing skipped: %s", exc)
        finally:
            await self.drone_controller.close()
            await self.audio_module.close()
            await self.vision_module.close()

    async def _dispatch_command(self, command: ValidatedCommand) -> str:
        method_name = command.controller_method
        controller_method = getattr(self.drone_controller, method_name, None)
        if controller_method is None:
            raise RuntimeError(f"DroneController does not implement '{method_name}'")

        kwargs = command.controller_kwargs()
        await controller_method(**kwargs)
        return f"EXECUTED:{command.name}"

    async def _safe_hold(self, reason: str) -> str:
        try:
            await self.drone_controller.hold()
            logger.warning("Fallback HOLD applied: %s", reason)
            return f"HOLD:{reason}"
        except DroneStateError as exc:
            logger.warning("Fallback HOLD unavailable (%s): %s", reason, exc)
            return f"HOLD_UNAVAILABLE:{reason}"

    def _build_drone_snapshot(self) -> DroneStateSnapshot:
        connected = bool(self.drone_controller.state.connected)
        armed = bool(self.drone_controller.state.armed)
        in_air = bool(self.drone_controller.state.in_air)
        flight_mode = self.drone_controller.state.flight_mode
        battery = self.drone_controller.state.battery

        if in_air:
            self._has_been_airborne = True

        battery_remaining = None
        if battery is not None:
            battery_remaining = float(battery.remaining_percent)
            if battery_remaining > 1.0:
                battery_remaining = battery_remaining / 100.0
            battery_remaining = max(0.0, min(1.0, battery_remaining))

        if not connected:
            state = DroneOperationalState.GROUNDED
        elif in_air and flight_mode == FlightMode.OFFBOARD:
            state = DroneOperationalState.OFFBOARD
        elif in_air and flight_mode == FlightMode.LAND:
            state = DroneOperationalState.LANDING
        elif in_air:
            state = DroneOperationalState.AIRBORNE
        elif armed and (flight_mode == FlightMode.LAND or self._has_been_airborne):
            state = DroneOperationalState.LANDED
        elif armed:
            state = DroneOperationalState.ARMED
        else:
            state = DroneOperationalState.GROUNDED

        return DroneStateSnapshot(
            state=state,
            connected=connected,
            battery_remaining=battery_remaining,
        )

    def _emit_status(
        self,
        *,
        input_text: str,
        action: str,
        started_at: float,
        audio_ms: float,
        vision_ms: float,
        vlm_ms: float,
    ) -> LoopStatus:
        state = self._build_drone_snapshot().state.value
        total_latency_ms = self._elapsed_ms(started_at)
        compact_input = self._compact_text(input_text)
        line = (
            f"[STATE] {state} | "
            f"[INPUT] {compact_input} | "
            f"[ACTION] {action} | "
            f"[AUDIO_MS] {audio_ms:.1f} | "
            f"[VISION_MS] {vision_ms:.1f} | "
            f"[VLM_MS] {vlm_ms:.1f} | "
            f"[TOTAL_LATENCY] {total_latency_ms:.1f}ms"
        )
        print(line, flush=True)
        return LoopStatus(
            state=state,
            input_text=compact_input,
            action=action,
            audio_ms=audio_ms,
            vision_ms=vision_ms,
            vlm_ms=vlm_ms,
            total_latency_ms=total_latency_ms,
        )

    @staticmethod
    def _compact_text(text: str) -> str:
        compact = re.sub(r"\s+", " ", text).strip()
        if len(compact) > 120:
            return compact[:117] + "..."
        return compact

    @staticmethod
    def _command_has_transcript_evidence(command_name: str, transcript: str) -> bool:
        pattern = COMMAND_EVIDENCE_PATTERNS.get(command_name)
        if pattern is None:
            return False

        return pattern.search(transcript) is not None

    @staticmethod
    def _elapsed_ms(started_at: float) -> float:
        return (time.perf_counter() - started_at) * 1000.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Main async agent loop: audio -> cognition -> safety -> drone action.")
    parser.add_argument("--connection", default=DEFAULT_CONNECTION, help=f"MAVSDK connection string. Default: {DEFAULT_CONNECTION}")
    parser.add_argument("--whisper-model", default="tiny", help="faster-whisper model size (tiny, base, ...).")
    parser.add_argument("--whisper-language", default="it", help="Whisper language code.")
    parser.add_argument("--audio-device", default=None, help="Optional sounddevice input device.")
    parser.add_argument("--camera-index", default=0, type=int, help="OpenCV camera index.")
    parser.add_argument("--frame-path", default="tmp/frame.jpg", help="Path for the one-shot camera frame.")
    parser.add_argument("--sample-rate", default=16_000, type=int, help="Microphone sample rate.")
    parser.add_argument("--cognition-model", default=DEFAULT_MODEL_ID, help="MLX model id for CognitionEngine.")
    parser.add_argument("--max-cognition-failures", default=3, type=int, help="Consecutive cognition failures before emergency hold.")
    return parser


async def run_agent(args: argparse.Namespace) -> None:
    controller = DroneController(connection=args.connection)
    validator = CommandValidator(
        DroneStateSnapshot(
            state=DroneOperationalState.GROUNDED,
            connected=False,
            battery_remaining=None,
        )
    )
    cognition_engine = CognitionEngine(
        validator=validator,
        model_id=args.cognition_model,
        temperature=0.0,
    )
    audio_module = AudioModule(
        model_size=args.whisper_model,
        language=args.whisper_language,
        sample_rate=args.sample_rate,
        device=args.audio_device,
    )
    vision_module = VisionModule(
        camera_index=args.camera_index,
        output_path=args.frame_path,
    )
    agent = AgentLoop(
        drone_controller=controller,
        cognition_engine=cognition_engine,
        audio_module=audio_module,
        vision_module=vision_module,
        max_cognition_failures=args.max_cognition_failures,
    )
    await agent.run()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    args = build_parser().parse_args()
    try:
        asyncio.run(run_agent(args))
    except KeyboardInterrupt:
        logger.info("Manual interruption received")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
