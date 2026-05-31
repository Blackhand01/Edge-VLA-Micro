from __future__ import annotations

import asyncio
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from src.action import DroneController, DroneStateError
from src.audio import AudioModule, transcribe_audio
from src.core.drone_snapshot import DroneSnapshotBuilder, kinematic_state_dict
from src.core.intent_patterns import ACTIONABLE_PATTERN, EMERGENCY_KEYWORDS, EMERGENCY_PATTERN, VISION_REQUIRED_PATTERN
from src.core.status import LoopStatus
from src.monitoring import BlackboxLogger, PerformanceLogger, TelemetryCsvLogger
from src.perception import CognitionEngine, CognitionError, CognitionResult, VLMProfile
from src.perception.cognition_service import AsyncCognitionService, ThreadedCognitionService
from src.perception.vision_module import VisionError, VisionModule
from src.safety.command_validator import DroneStateSnapshot, ValidatedCommand


logger = logging.getLogger(__name__)


class AgentLoop:
    def __init__(
        self,
        *,
        drone_controller: DroneController,
        cognition_engine: CognitionEngine,
        audio_module: AudioModule,
        vision_module: VisionModule,
        blackbox_logger: Optional[BlackboxLogger] = None,
        performance_logger: Optional[PerformanceLogger] = None,
        telemetry_logger: Optional[TelemetryCsvLogger] = None,
        cognition_executor: Optional[ThreadPoolExecutor] = None,
        cognition_service: Optional[AsyncCognitionService] = None,
        max_cognition_failures: int = 3,
        idle_sleep_s: float = 0.10,
    ) -> None:
        self.drone_controller = drone_controller
        self.cognition_engine = cognition_engine
        self.audio_module = audio_module
        self.vision_module = vision_module
        self.blackbox_logger = blackbox_logger
        self.performance_logger = performance_logger
        self.telemetry_logger = telemetry_logger
        self.cognition_service = cognition_service or ThreadedCognitionService(cognition_engine, executor=cognition_executor)
        self.max_cognition_failures = max(1, max_cognition_failures)
        self.idle_sleep_s = idle_sleep_s
        self._snapshot_builder = DroneSnapshotBuilder()
        self._consecutive_cognition_failures = 0
        self._shutdown_done = False

    @property
    def _has_been_airborne(self) -> bool:
        return self._snapshot_builder.has_been_airborne

    @_has_been_airborne.setter
    def _has_been_airborne(self, value: bool) -> None:
        self._snapshot_builder.has_been_airborne = value

    async def warmup_cognition(self) -> None:
        await self.cognition_service.warmup()

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
        context = IterationContext(started_at=time.perf_counter())
        try:
            spoken_text = await self._capture_spoken_text(context)
            if spoken_text is None:
                await asyncio.sleep(self.idle_sleep_s)
                return self._emit_status(context, input_text="<silence>", action="IDLE")
            if EMERGENCY_PATTERN.search(spoken_text):
                self._consecutive_cognition_failures = 0
                return self._emit_status(context, input_text=spoken_text, action=await self._safe_hold("EMERGENCY_KEYWORD"))
            if not ACTIONABLE_PATTERN.search(spoken_text):
                return self._emit_status(context, input_text=spoken_text, action="IGNORED:NON_ACTIONABLE_AUDIO")
            return await self._process_actionable_text(context, spoken_text)
        except Exception:
            logger.exception("Agent loop iteration failed")
            action = await self._safe_hold("LOOP_EXCEPTION_HOLD")
            return self._emit_status(context, input_text=context.input_text or "<error>", action=action, record_performance=context.vlm_ms > 0.0)

    async def _capture_spoken_text(self, context) -> Optional[str]:
        audio_started = time.perf_counter()
        context.input_text = await transcribe_audio(self.audio_module)
        context.audio_ms = elapsed_ms_since(audio_started)
        return context.input_text

    async def _process_actionable_text(self, context, spoken_text: str) -> LoopStatus:
        image_path = await self._capture_frame_if_needed(context, spoken_text)
        drone_state = self._build_drone_snapshot()
        cognition_result = await self._run_cognition(context, spoken_text, image_path, drone_state)
        self._log_blackbox_inference(spoken_text, image_path, cognition_result, drone_state, context)

        if isinstance(cognition_result, CognitionError):
            self._consecutive_cognition_failures += 1
            logger.warning("Cognition rejected input: reason=%s details=%s", cognition_result.reason, cognition_result.details)
            hold_reason = self._hold_reason_after_cognition_error()
            action = await self._safe_hold(hold_reason)
            return self._emit_status(context, input_text=spoken_text, action=action, record_performance=True)

        self._consecutive_cognition_failures = 0
        action = await self._dispatch_command(cognition_result.validated_command)
        return self._emit_status(context, input_text=spoken_text, action=action, record_performance=True)

    async def _capture_frame_if_needed(self, context, spoken_text: str) -> Optional[str]:
        if not self._requires_vision(spoken_text):
            return None
        vision_started = time.perf_counter()
        try:
            return (await self.vision_module.capture_single_frame()).image_path
        except VisionError as exc:
            logger.warning("Vision capture failed; continuing text-only: %s", exc)
            return None
        finally:
            context.vision_ms = elapsed_ms_since(vision_started)

    async def _run_cognition(self, context, spoken_text: str, image_path: Optional[str], drone_state: DroneStateSnapshot):
        vlm_started = time.perf_counter()
        result = await self.cognition_service.process_intent(spoken_text, image_path, drone_state=drone_state)
        context.vlm_ms = elapsed_ms_since(vlm_started)
        context.vlm_profile = result.vlm_profile
        return result

    def _hold_reason_after_cognition_error(self) -> str:
        if self._consecutive_cognition_failures < self.max_cognition_failures:
            return "VALIDATION_HOLD"
        self._consecutive_cognition_failures = 0
        return "COGNITION_FAILSAFE_HOLD"

    async def shutdown(self) -> None:
        if self._shutdown_done:
            return
        self._shutdown_done = True
        logger.info("Shutting down main agent")
        try:
            if self.drone_controller.state.connected and (self.drone_controller.state.armed or self.drone_controller.state.in_air):
                await self.drone_controller.land()
                logger.info("Safe shutdown: land command sent")
        except Exception as exc:
            logger.warning("Safe shutdown landing skipped: %s", exc)
        finally:
            await self._close_components()

    async def _close_components(self) -> None:
        await self.drone_controller.close()
        await self.audio_module.close()
        await self.vision_module.close()
        if self.blackbox_logger is not None:
            await self.blackbox_logger.close()
        if self.performance_logger is not None:
            await self.performance_logger.close()
        await self.cognition_service.close()

    async def _dispatch_command(self, command: ValidatedCommand) -> str:
        if command.name == "hold" and not self.drone_controller.state.in_air:
            logger.warning("Command HOLD skipped: vehicle is not airborne")
            return "HOLD_SKIPPED:NOT_AIRBORNE_SAFE"
        controller_method = getattr(self.drone_controller, command.controller_method, None)
        if controller_method is None:
            raise RuntimeError(f"DroneController does not implement '{command.controller_method}'")
        await controller_method(**command.controller_kwargs())
        return f"EXECUTED:{command.name}"

    async def _safe_hold(self, reason: str) -> str:
        if not self.drone_controller.state.in_air:
            logger.warning("Fallback HOLD skipped (%s): vehicle is not airborne", reason)
            return f"HOLD_SKIPPED:{reason}:NOT_AIRBORNE_SAFE"
        try:
            await self.drone_controller.hold()
            logger.warning("Fallback HOLD applied: %s", reason)
            return f"HOLD:{reason}"
        except DroneStateError as exc:
            logger.warning("Fallback HOLD unavailable (%s): %s", reason, exc)
            return f"HOLD_UNAVAILABLE:{reason}"

    def _build_drone_snapshot(self) -> DroneStateSnapshot:
        return self._snapshot_builder.build(self.drone_controller.state)

    def _emit_status(self, context, *, input_text: str, action: str, record_performance: bool = False) -> LoopStatus:
        profile = context.vlm_profile
        total_latency_ms = elapsed_ms_since(context.started_at)
        state = self._build_drone_snapshot().state.value
        compact_input = compact_text(input_text)
        if record_performance and self.performance_logger is not None:
            self.performance_logger.log_run(
                audio_ms=context.audio_ms,
                vision_ms=context.vision_ms,
                ttft_ms=profile.ttft_ms,
                decode_time_ms=profile.decode_time_ms,
                tps=profile.tps,
                safety_ms=profile.safety_ms,
                total_latency_ms=total_latency_ms,
            )
        if record_performance and self.telemetry_logger is not None:
            self.telemetry_logger.log_event(
                profile="local_mac",
                component="agent_loop",
                ok=not action.startswith(("HOLD:", "HOLD_UNAVAILABLE", "HOLD_SKIPPED", "IGNORED")),
                action=action,
                state=state,
                input_text=compact_input,
                image_present=context.vision_ms > 0.0,
                audio_ms=context.audio_ms,
                vision_ms=context.vision_ms,
                cognition_ms=context.vlm_ms,
                total_ms=total_latency_ms,
                ttft_ms=profile.ttft_ms,
                decode_time_ms=profile.decode_time_ms,
                generated_tokens=profile.generated_tokens,
                tps=profile.tps,
                safety_ms=profile.safety_ms,
            )
        print(format_status_line(state, compact_input, action, context, total_latency_ms), flush=True)
        return LoopStatus(state, compact_input, action, context.audio_ms, context.vision_ms, context.vlm_ms, profile.ttft_ms, profile.decode_time_ms, profile.tps, profile.safety_ms, total_latency_ms)

    def _log_blackbox_inference(self, input_text, image_path, cognition_result, drone_state, context) -> None:
        if self.blackbox_logger is None:
            return
        payload = build_blackbox_payload(input_text, cognition_result, drone_state, self.drone_controller.state, context)
        self.blackbox_logger.log_inference(image_path=image_path, payload=payload)

    @staticmethod
    def _requires_vision(spoken_text: str) -> bool:
        return VISION_REQUIRED_PATTERN.search(spoken_text) is not None


class IterationContext:
    def __init__(self, *, started_at: float) -> None:
        self.started_at = started_at
        self.input_text: Optional[str] = None
        self.audio_ms = 0.0
        self.vision_ms = 0.0
        self.vlm_ms = 0.0
        self.vlm_profile = VLMProfile()


def build_blackbox_payload(input_text, cognition_result, drone_state, controller_state, context) -> dict:
    payload = {
        "input_text": input_text,
        "drone_state_at_inference": drone_state.model_dump(mode="json"),
        "drone_kinematics": kinematic_state_dict(controller_state, drone_state),
        "timing_ms": {
            "audio": context.audio_ms,
            "vision": context.vision_ms,
            "vlm": context.vlm_ms,
            "vlm_ttft": cognition_result.vlm_profile.ttft_ms,
            "vlm_decode": cognition_result.vlm_profile.decode_time_ms,
            "vlm_tps": cognition_result.vlm_profile.tps,
            "safety": cognition_result.vlm_profile.safety_ms,
        },
        "vlm_profile": cognition_result.vlm_profile.model_dump(),
        "prompt": cognition_result.raw_prompt,
        "raw_response": cognition_result.raw_response,
        "parsed_json": cognition_result.parsed_json,
    }
    payload["validation"] = validation_payload(cognition_result)
    return payload


def validation_payload(cognition_result) -> dict:
    if isinstance(cognition_result, CognitionResult):
        command = cognition_result.validated_command
        return {"status": "ACCEPTED", "name": command.name, "controller_method": command.controller_method, "controller_kwargs": command.controller_kwargs(), "raw": command.raw}
    return {"status": "REJECTED", "reason": cognition_result.reason, "details": cognition_result.details}


def format_status_line(state, compact_input, action, context, total_latency_ms) -> str:
    profile = context.vlm_profile
    return f"[STATE] {state} | [INPUT] {compact_input} | [ACTION] {action} | [AUDIO_MS] {context.audio_ms:.1f} | [VISION_MS] {context.vision_ms:.1f} | [VLM_TTFT] {profile.ttft_ms:.1f}ms | [VLM_TPS] {profile.tps:.2f} | [SAFETY_MS] {profile.safety_ms:.1f} | [TOTAL_LATENCY] {total_latency_ms:.1f}ms"


def compact_text(text: str) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact[:117] + "..." if len(compact) > 120 else compact


def elapsed_ms_since(started_at: float) -> float:
    return (time.perf_counter() - started_at) * 1000.0
