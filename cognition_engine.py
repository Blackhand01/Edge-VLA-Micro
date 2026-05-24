from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from command_validator import (
    ActionCommand,
    CommandValidator,
    DroneOperationalState,
    DroneStateSnapshot,
    SafetyViolationError,
    ValidatedCommand,
)


logger = logging.getLogger(__name__)


DEFAULT_MODEL_ID = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
JSON_OBJECT_START_RE = re.compile(r"\{")
COLOR_TARGET_RE = re.compile(r"\b(red|blue)\b|\bhead\s+object\b", re.IGNORECASE)
HSV_MIN_PIXELS = 50
HSV_MIN_RATIO = 0.0002
SAFE_COMMAND_PATTERNS = (
    ("disarm", re.compile(r"\bdisarm\b", re.IGNORECASE)),
    ("takeoff", re.compile(r"\b(take\s*off|takeoff|launch)\b", re.IGNORECASE)),
    ("arm", re.compile(r"\barm\b", re.IGNORECASE)),
    ("land", re.compile(r"\bland\b", re.IGNORECASE)),
    ("hold", re.compile(r"\b(hold|hover)\b", re.IGNORECASE)),
)
BASE_COMMAND_FIELDS = frozenset({"command", "target_found", "reasoning"})
COMMAND_ALLOWED_FIELDS = {
    "arm": BASE_COMMAND_FIELDS,
    "disarm": BASE_COMMAND_FIELDS,
    "takeoff": BASE_COMMAND_FIELDS,
    "land": BASE_COMMAND_FIELDS,
    "hold": BASE_COMMAND_FIELDS,
    "move_velocity": BASE_COMMAND_FIELDS
    | frozenset({"velocity_x", "velocity_y", "velocity_z", "yaw_deg"}),
}
SYSTEM_PROMPT = (
    "You are the onboard computer of a drone. Inspect the FPV camera image and "
    "read the operator's voice command. Your task is to extract the spatial intent "
    "and translate it into valid JSON that conforms to the schema. Every JSON object "
    "must include target_found and reasoning. If the requested object is not visible "
    "in the image, set target_found=false and explain why in reasoning. "
    "Return exactly one JSON command. The command field is required and must be one "
    "of: arm, disarm, takeoff, land, hold, move_velocity. If the voice command "
    "contains multiple actions, choose only the next valid action based on "
    "CURRENT_DRONE_STATE; if the state is GROUNDED and arm/takeoff/move is requested, "
    "respond with arm first. Valid move example: "
    '{"command":"move_velocity","target_found":true,"reasoning":"red object visible",'
    '"velocity_x":0.5,"velocity_y":0.0,"velocity_z":0.0,"yaw_deg":0.0}. '
    'Missing target example: {"command":"hold","target_found":false,'
    '"reasoning":"requested target not visible"}. '
    "Do not include telemetry fields such as battery_remaining, state, connected, "
    "altitude, or GPS data in the JSON command. "
    "Do not add extra text."
)


@dataclass
class VLMProfile:
    prompt_eval_ms: Optional[float] = None
    ttft_ms: float = 0.0
    decode_time_ms: float = 0.0
    generated_tokens: int = 0
    tps: float = 0.0
    safety_ms: float = 0.0
    used_streaming: bool = False
    fallback_reason: Optional[str] = None

    def model_dump(self) -> dict[str, Any]:
        return {
            "prompt_eval_ms": self.prompt_eval_ms,
            "ttft_ms": self.ttft_ms,
            "decode_time_ms": self.decode_time_ms,
            "generated_tokens": self.generated_tokens,
            "tps": self.tps,
            "safety_ms": self.safety_ms,
            "used_streaming": self.used_streaming,
            "fallback_reason": self.fallback_reason,
        }


def _estimate_token_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


@dataclass(frozen=True)
class CognitionResult:
    raw_prompt: str
    raw_response: str
    parsed_json: dict
    validated_command: ValidatedCommand
    vlm_profile: VLMProfile = field(default_factory=VLMProfile)

    @property
    def ttft_ms(self) -> float:
        return self.vlm_profile.ttft_ms

    @property
    def decode_time_ms(self) -> float:
        return self.vlm_profile.decode_time_ms

    @property
    def generated_tokens(self) -> int:
        return self.vlm_profile.generated_tokens

    @property
    def tps(self) -> float:
        return self.vlm_profile.tps

    @property
    def ok(self) -> bool:
        return True


@dataclass(frozen=True)
class CognitionError:
    reason: str
    raw_prompt: str
    raw_response: str = ""
    parsed_json: Optional[dict] = None
    details: Optional[str] = None
    vlm_profile: VLMProfile = field(default_factory=VLMProfile)

    @property
    def ok(self) -> bool:
        return False


GeneratorFn = Callable[[str, Optional[str]], str]


class CognitionEngine:
    def __init__(
        self,
        validator: Optional[CommandValidator] = None,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        max_tokens: int = 128,
        temperature: float = 0.0,
        log_path: str | Path = "logs/cognition.log",
        generator: Optional[GeneratorFn] = None,
    ) -> None:
        self.validator = validator or CommandValidator(
            # Fail closed when the caller does not provide a live drone state.
            DroneStateSnapshot(
                state=DroneOperationalState.GROUNDED,
                connected=True,
                battery_remaining=1.0,
            )
        )
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.log_path = Path(log_path)
        self._external_generator = generator
        self._model = None
        self._processor = None
        self._mlx_generate = None
        self._mlx_stream_generate = None

    def process_intent(
        self,
        text_input: str,
        image_path: Optional[str] = None,
        *,
        drone_state: Optional[DroneStateSnapshot] = None,
    ) -> CognitionResult | CognitionError:
        raw_prompt = self._build_prompt(text_input, image_path=image_path, drone_state=drone_state)
        raw_response = ""
        parsed_json: Optional[dict] = None
        vlm_profile = VLMProfile()
        safety_started: Optional[float] = None

        try:
            raw_response, vlm_profile = self._generate_profiled(raw_prompt, image_path=image_path)
            safety_started = time.perf_counter()
            parsed_json = self._extract_json_object(raw_response)
            self._apply_hsv_guardrail(text_input, image_path, parsed_json)
            self._repair_missing_command_from_text(text_input, parsed_json, drone_state)
            self._repair_missing_target_fields(parsed_json)
            self._drop_extra_command_fields(parsed_json)
            validated = self.validator.validate(
                json.dumps(parsed_json, ensure_ascii=False, separators=(",", ":")),
                drone_state=drone_state,
            )
            vlm_profile.safety_ms = self._elapsed_ms(safety_started)
        except SafetyViolationError as exc:
            if safety_started is not None:
                vlm_profile.safety_ms = self._elapsed_ms(safety_started)
            error = CognitionError(
                reason="VALIDATION_REJECTED",
                raw_prompt=raw_prompt,
                raw_response=raw_response,
                parsed_json=parsed_json,
                details=str(exc),
                vlm_profile=vlm_profile,
            )
            self._log_transaction(raw_prompt, raw_response, parsed_json, error)
            return error
        except (ValueError, json.JSONDecodeError) as exc:
            if safety_started is not None:
                vlm_profile.safety_ms = self._elapsed_ms(safety_started)
            error = CognitionError(
                reason="INVALID_JSON",
                raw_prompt=raw_prompt,
                raw_response=raw_response,
                parsed_json=parsed_json,
                details=str(exc),
                vlm_profile=vlm_profile,
            )
            self._log_transaction(raw_prompt, raw_response, parsed_json, error)
            return error
        except Exception as exc:  # noqa: BLE001 - local inference failures must not block control.
            logger.exception("Cognition inference failed")
            if safety_started is not None:
                vlm_profile.safety_ms = self._elapsed_ms(safety_started)
            error = CognitionError(
                reason="INFERENCE_ERROR",
                raw_prompt=raw_prompt,
                raw_response=raw_response,
                parsed_json=parsed_json,
                details=str(exc),
                vlm_profile=vlm_profile,
            )
            self._log_transaction(raw_prompt, raw_response, parsed_json, error)
            return error

        result = CognitionResult(
            raw_prompt=raw_prompt,
            raw_response=raw_response,
            parsed_json=parsed_json,
            validated_command=validated,
            vlm_profile=vlm_profile,
        )
        self._log_transaction(raw_prompt, raw_response, parsed_json, result)
        return result

    def warmup(self) -> None:
        logger.info("Pre-warming VLM models and compiling Metal shaders...")

        if self._external_generator is not None:
            self._external_generator("warmup", None)
            logger.info("VLM Warm-up complete.")
            return

        self._load_local_model()
        assert self._mlx_generate is not None
        self._mlx_generate(
            self._model,
            self._processor,
            prompt="warmup",
            image=None,
            max_tokens=1,
            temperature=self.temperature,
            verbose=False,
        )
        logger.info("VLM Warm-up complete.")

    def _build_prompt(
        self,
        intent_text: str,
        *,
        image_path: Optional[str] = None,
        drone_state: Optional[DroneStateSnapshot] = None,
    ) -> str:
        schema = json.dumps(ActionCommand.model_json_schema(), indent=2, ensure_ascii=False)
        image_reference = image_path if image_path is not None else "<NO_IMAGE_AVAILABLE>"
        state_reference = drone_state.model_dump(mode="json") if drone_state is not None else "<UNKNOWN>"
        return (
            f"{SYSTEM_PROMPT}\n\n"
            "JSON_SCHEMA:\n"
            f"{schema}\n\n"
            "CURRENT_DRONE_STATE:\n"
            f"{state_reference}\n\n"
            "IMAGE_PATH:\n"
            f"{image_reference}\n\n"
            "USER_INTENT:\n"
            f"{intent_text}\n\n"
            "JSON:"
        )

    def _generate(self, raw_prompt: str, *, image_path: Optional[str]) -> str:
        raw_response, _ = self._generate_profiled(raw_prompt, image_path=image_path)
        return raw_response

    def _generate_profiled(self, raw_prompt: str, *, image_path: Optional[str]) -> tuple[str, VLMProfile]:
        if self._external_generator is not None:
            return self._generate_external_profiled(raw_prompt, image_path=image_path)

        self._load_local_model()
        if self._mlx_stream_generate is not None:
            try:
                return self._stream_generate_profiled(raw_prompt, image_path=image_path)
            except Exception as exc:  # noqa: BLE001 - stream APIs vary across mlx-vlm releases.
                logger.warning("VLM streaming unavailable; falling back to batch generate: %s", exc)

        return self._batch_generate_profiled(
            raw_prompt,
            image_path=image_path,
            fallback_reason="stream_generate unavailable or failed",
        )

    def _generate_external_profiled(self, raw_prompt: str, *, image_path: Optional[str]) -> tuple[str, VLMProfile]:
        started = time.perf_counter()
        raw_response = self._external_generator(raw_prompt, image_path)
        elapsed_ms = self._elapsed_ms(started)
        generated_tokens = _estimate_token_count(raw_response)
        return raw_response, VLMProfile(
            ttft_ms=elapsed_ms,
            decode_time_ms=0.0,
            generated_tokens=generated_tokens,
            tps=(generated_tokens / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0.0,
            used_streaming=False,
            fallback_reason="external generator is non-streaming",
        )

    def _batch_generate_profiled(
        self,
        raw_prompt: str,
        *,
        image_path: Optional[str],
        fallback_reason: str,
    ) -> tuple[str, VLMProfile]:
        started = time.perf_counter()
        result = self._mlx_generate(
            self._model,
            self._processor,
            prompt=raw_prompt,
            image=image_path,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            verbose=False,
        )
        elapsed_ms = self._elapsed_ms(started)
        raw_response = result.text if hasattr(result, "text") else str(result)
        generated_tokens = self._result_token_count(result, raw_response)
        return raw_response, VLMProfile(
            ttft_ms=elapsed_ms,
            decode_time_ms=0.0,
            generated_tokens=generated_tokens,
            tps=(generated_tokens / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0.0,
            used_streaming=False,
            fallback_reason=fallback_reason,
        )

    def _stream_generate_profiled(self, raw_prompt: str, *, image_path: Optional[str]) -> tuple[str, VLMProfile]:
        started = time.perf_counter()
        first_token_at: Optional[float] = None
        ended_at = started
        parts: list[str] = []
        generated_tokens = 0
        final_token_count: Optional[int] = None
        prompt_eval_ms: Optional[float] = None

        for response in self._mlx_stream_generate(
            self._model,
            self._processor,
            prompt=raw_prompt,
            image=image_path,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            verbose=False,
        ):
            now = time.perf_counter()
            if first_token_at is None:
                first_token_at = now
            ended_at = now
            text = self._response_text(response)
            if text:
                parts.append(text)
            generated_tokens += self._response_token_increment(response, text)
            final_token_count = self._response_final_token_count(response, final_token_count)
            prompt_eval_ms = self._response_prompt_eval_ms(response, prompt_eval_ms)

        raw_response = "".join(parts)
        if final_token_count is not None:
            generated_tokens = max(generated_tokens, final_token_count)

        if first_token_at is None:
            elapsed_ms = self._elapsed_ms(started)
            return raw_response, VLMProfile(
                prompt_eval_ms=prompt_eval_ms,
                ttft_ms=elapsed_ms,
                decode_time_ms=0.0,
                generated_tokens=generated_tokens,
                tps=0.0,
                used_streaming=True,
            )

        ttft_ms = (first_token_at - started) * 1000.0
        decode_time_ms = max(0.0, (ended_at - first_token_at) * 1000.0)
        return raw_response, VLMProfile(
            prompt_eval_ms=prompt_eval_ms,
            ttft_ms=ttft_ms,
            decode_time_ms=decode_time_ms,
            generated_tokens=generated_tokens,
            tps=(generated_tokens / (decode_time_ms / 1000.0)) if decode_time_ms > 0 else 0.0,
            used_streaming=True,
        )

    def _load_local_model(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        try:
            from mlx_vlm import generate, load
        except ImportError as exc:
            raise RuntimeError(
                "mlx-vlm is not installed in this virtualenv. "
                "Run: .venv/bin/python -m pip install -r requirements-phase4.txt"
            ) from exc

        try:
            from mlx_vlm import stream_generate
        except ImportError:
            try:
                from mlx_vlm.generate import stream_generate  # type: ignore[no-redef]
            except ImportError:
                stream_generate = None

        logger.info("Loading local MLX-VLM model: %s", self.model_id)
        self._model, self._processor = load(self.model_id)
        self._mlx_generate = generate
        self._mlx_stream_generate = stream_generate

    @staticmethod
    def _response_text(response: Any) -> str:
        text = getattr(response, "text", None)
        if text is not None:
            return str(text)
        if isinstance(response, tuple) and response:
            return str(response[0])
        return str(response)

    @staticmethod
    def _response_token_increment(response: Any, text: str) -> int:
        if getattr(response, "token", None) is not None:
            return 1
        tokens = getattr(response, "tokens", None)
        if tokens is not None:
            try:
                return len(tokens)
            except TypeError:
                return 1
        return 1 if text else 0

    @staticmethod
    def _response_final_token_count(response: Any, current: Optional[int]) -> Optional[int]:
        for attr in ("generation_tokens", "generated_tokens", "total_generated_tokens"):
            value = getattr(response, attr, None)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return current

    @staticmethod
    def _response_prompt_eval_ms(response: Any, current: Optional[float]) -> Optional[float]:
        for attr in ("prompt_eval_ms", "prompt_time_ms", "prompt_processing_ms"):
            value = getattr(response, attr, None)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return current

    @staticmethod
    def _result_token_count(result: Any, raw_response: str) -> int:
        for attr in ("generation_tokens", "generated_tokens", "total_generated_tokens"):
            value = getattr(result, attr, None)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return _estimate_token_count(raw_response)

    @staticmethod
    def _elapsed_ms(started_at: float) -> float:
        return (time.perf_counter() - started_at) * 1000.0

    def _extract_json_object(self, raw_response: str) -> dict:
        cleaned = raw_response.strip()
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        match = JSON_OBJECT_START_RE.search(cleaned)
        if match is None:
            raise ValueError("LLM response does not contain a JSON object")

        candidate = self._balanced_json_slice(cleaned, match.start())
        parsed = json.loads(candidate)
        if not isinstance(parsed, dict):
            raise ValueError(f"LLM JSON must be an object, got {type(parsed).__name__}")

        return parsed

    def _apply_hsv_guardrail(self, text_input: str, image_path: Optional[str], parsed_json: dict) -> None:
        requested_color = self._requested_color(text_input)
        if requested_color is None:
            return

        if not image_path:
            self._force_target_not_found(
                parsed_json,
                f"OpenCV HSV guardrail: {requested_color} target requested but no frame is available.",
            )
            return

        if not self._frame_contains_hsv_color(image_path, requested_color):
            self._force_target_not_found(
                parsed_json,
                f"OpenCV HSV guardrail: requested {requested_color} target is absent from the frame.",
            )
            return

        self._force_target_found(
            parsed_json,
            f"OpenCV HSV guardrail: requested {requested_color} target is present in the frame.",
        )

    @staticmethod
    def _requested_color(text_input: str) -> Optional[str]:
        match = COLOR_TARGET_RE.search(text_input)
        if match is None:
            return None

        color = (match.group(1) or match.group(0)).lower()
        if color == "head object":
            logger.warning("ASR_COLOR_ALIAS: treating 'head object' as likely 'red object'")
            return "red"
        return color

    @staticmethod
    def _force_target_not_found(parsed_json: dict, reasoning: str) -> None:
        logger.warning("SAFETY_OVERRIDE: TARGET_NOT_FOUND | %s", reasoning)
        parsed_json["target_found"] = False
        parsed_json["reasoning"] = reasoning

    @staticmethod
    def _force_target_found(parsed_json: dict, reasoning: str) -> None:
        parsed_json["target_found"] = True
        parsed_json["reasoning"] = reasoning

    def _repair_missing_command_from_text(
        self,
        text_input: str,
        parsed_json: dict,
        drone_state: Optional[DroneStateSnapshot],
    ) -> None:
        if parsed_json.get("command"):
            return

        command = self._infer_safe_non_motion_command(text_input, drone_state)
        if command is None:
            return

        logger.warning("SCHEMA_REPAIR: INFERRED_COMMAND | command=%s", command)
        parsed_json["command"] = command
        parsed_json.setdefault("target_found", True)
        parsed_json.setdefault(
            "reasoning",
            f"Deterministic non-motion transcript fallback inferred command={command}.",
        )

    @staticmethod
    def _repair_missing_target_fields(parsed_json: dict) -> None:
        missing_fields = [field for field in ("target_found", "reasoning") if field not in parsed_json]
        if not missing_fields:
            return

        logger.warning("SCHEMA_REPAIR: MISSING_TARGET_FIELDS | fields=%s", ",".join(missing_fields))
        command = parsed_json.get("command")
        if command in {"arm", "disarm", "takeoff", "land", "hold"}:
            parsed_json.setdefault("target_found", True)
            parsed_json.setdefault(
                "reasoning",
                f"Deterministic schema repair: VLM omitted target metadata for non-visual command={command}.",
            )
            return

        parsed_json["target_found"] = False
        parsed_json["reasoning"] = (
            "SAFETY_OVERRIDE: TARGET_NOT_FOUND. VLM omitted required target_found/reasoning fields."
        )

    @staticmethod
    def _drop_extra_command_fields(parsed_json: dict) -> None:
        command = parsed_json.get("command")
        if not isinstance(command, str):
            return

        allowed_fields = COMMAND_ALLOWED_FIELDS.get(command)
        if allowed_fields is None:
            return

        extra_fields = sorted(field for field in parsed_json if field not in allowed_fields)
        if not extra_fields:
            return

        logger.warning("SCHEMA_REPAIR: DROPPED_EXTRA_FIELDS | fields=%s", ",".join(extra_fields))
        for field in extra_fields:
            parsed_json.pop(field, None)

    @staticmethod
    def _infer_safe_non_motion_command(
        text_input: str,
        drone_state: Optional[DroneStateSnapshot],
    ) -> Optional[str]:
        matched = {command for command, pattern in SAFE_COMMAND_PATTERNS if pattern.search(text_input)}
        if not matched:
            return None

        if drone_state is None:
            return CognitionEngine._first_matched_command(text_input, matched)

        state = drone_state.state
        if state == DroneOperationalState.GROUNDED:
            for command in ("arm", "disarm", "hold", "takeoff", "land"):
                if command in matched:
                    return command

        if state == DroneOperationalState.ARMED:
            for command in ("takeoff", "disarm", "hold", "arm", "land"):
                if command in matched:
                    return command

        if state in {
            DroneOperationalState.AIRBORNE,
            DroneOperationalState.OFFBOARD,
            DroneOperationalState.LANDING,
        }:
            for command in ("land", "hold", "takeoff", "disarm", "arm"):
                if command in matched:
                    return command

        return CognitionEngine._first_matched_command(text_input, matched)

    @staticmethod
    def _first_matched_command(text_input: str, matched: set[str]) -> Optional[str]:
        candidates: list[tuple[int, str]] = []
        for command, pattern in SAFE_COMMAND_PATTERNS:
            if command not in matched:
                continue
            match = pattern.search(text_input)
            if match is not None:
                candidates.append((match.start(), command))

        if not candidates:
            return None

        return min(candidates, key=lambda item: item[0])[1]

    @staticmethod
    def _frame_contains_hsv_color(image_path: str, color: str) -> bool:
        try:
            import cv2  # pylint: disable=import-outside-toplevel
            import numpy as np  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            logger.warning("SAFETY_OVERRIDE: TARGET_NOT_FOUND | OpenCV unavailable: %s", exc)
            return False

        frame = cv2.imread(image_path)
        if frame is None:
            logger.warning("SAFETY_OVERRIDE: TARGET_NOT_FOUND | OpenCV could not read frame: %s", image_path)
            return False

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if color == "red":
            lower_1 = np.array([0, 80, 50], dtype=np.uint8)
            upper_1 = np.array([10, 255, 255], dtype=np.uint8)
            lower_2 = np.array([170, 80, 50], dtype=np.uint8)
            upper_2 = np.array([180, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_1, upper_1) | cv2.inRange(hsv, lower_2, upper_2)
        elif color == "blue":
            lower = np.array([100, 80, 50], dtype=np.uint8)
            upper = np.array([130, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
        else:
            return False

        matching_pixels = int(cv2.countNonZero(mask))
        total_pixels = int(mask.shape[0] * mask.shape[1])
        if total_pixels <= 0:
            return False

        ratio = matching_pixels / total_pixels
        logger.info(
            "HSV target check: color=%s pixels=%d total=%d ratio=%.6f threshold_pixels=%d threshold_ratio=%.6f",
            color,
            matching_pixels,
            total_pixels,
            ratio,
            HSV_MIN_PIXELS,
            HSV_MIN_RATIO,
        )
        return matching_pixels >= HSV_MIN_PIXELS and ratio >= HSV_MIN_RATIO

    def _balanced_json_slice(self, text: str, start: int) -> str:
        depth = 0
        in_string = False
        escaped = False

        for index in range(start, len(text)):
            char = text[index]

            if escaped:
                escaped = False
                continue

            if char == "\\":
                escaped = True
                continue

            if char == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]

        raise ValueError("LLM response contains an unterminated JSON object")

    def _log_transaction(
        self,
        raw_prompt: str,
        raw_response: str,
        parsed_json: Optional[dict],
        validation_result: CognitionResult | CognitionError,
    ) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        parsed_text = (
            json.dumps(parsed_json, indent=2, ensure_ascii=False)
            if parsed_json is not None
            else "<NONE>"
        )
        status = self._validation_status(validation_result)
        block = (
            "\n"
            "================ COGNITION TRANSACTION ================\n"
            f"[UTC] {timestamp}\n"
            "[PROMPT]\n"
            f"{raw_prompt}\n"
            "[RAW_RESPONSE]\n"
            f"{raw_response or '<EMPTY>'}\n"
            "[PARSED_JSON]\n"
            f"{parsed_text}\n"
            "[VALIDATION_STATUS]\n"
            f"{status}\n"
            "=======================================================\n"
        )

        print(block)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(block)

    def _validation_status(self, validation_result: CognitionResult | CognitionError) -> str:
        if isinstance(validation_result, CognitionResult):
            command = validation_result.validated_command
            return (
                "ACCEPTED | "
                f"command={command.name} | "
                f"controller_method={command.controller_method} | "
                f"kwargs={command.controller_kwargs()}"
            )

        detail = f" | details={validation_result.details}" if validation_result.details else ""
        return f"REJECTED | reason={validation_result.reason}{detail}"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    engine = CognitionEngine()
    result = engine.process_intent("move forward at one meter per second while maintaining altitude and zero yaw")
    if not result.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
