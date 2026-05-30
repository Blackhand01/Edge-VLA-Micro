from __future__ import annotations

import json
import logging
import platform
import time
from pathlib import Path
from typing import Optional

from src.safety.command_validator import CommandValidator, DroneOperationalState, DroneStateSnapshot, SafetyViolationError
from src.perception.guardrails import (
    apply_hsv_guardrail,
    drop_extra_command_fields,
    frame_contains_hsv_color,
    repair_missing_command_from_text,
    repair_missing_target_fields,
    requested_target_color,
)
from src.perception.json_parsing import balanced_json_slice, extract_json_object
from src.perception.models import CognitionError, CognitionResult, GeneratorFn, VLMProfile
from src.perception.prompts import DEFAULT_MODEL_ID, build_cognition_prompt
from src.perception.transaction_log import validation_status, write_cognition_transaction
from src.perception.vlm_runtime import VLMRuntime, elapsed_ms_since


logger = logging.getLogger(__name__)


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
        vlm_backend: str = "mlx",
    ) -> None:
        self.validator = validator or CommandValidator(
            DroneStateSnapshot(
                state=DroneOperationalState.GROUNDED,
                connected=True,
                battery_remaining=1.0,
            )
        )
        self.log_path = Path(log_path)
        self.runtime = build_vlm_runtime(
            vlm_backend,
            model_id=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            external_generator=generator,
        )

    @property
    def _model(self):
        return self.runtime.model

    @_model.setter
    def _model(self, value) -> None:
        self.runtime.model = value

    @property
    def _processor(self):
        return self.runtime.processor

    @_processor.setter
    def _processor(self, value) -> None:
        self.runtime.processor = value

    @property
    def _mlx_stream_generate(self):
        return self.runtime.stream_generate

    @_mlx_stream_generate.setter
    def _mlx_stream_generate(self, value) -> None:
        self.runtime.stream_generate = value

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
            self._apply_safety_repairs(text_input, image_path, parsed_json, drone_state)
            validated = self.validator.validate(
                json.dumps(parsed_json, ensure_ascii=False, separators=(",", ":")),
                drone_state=drone_state,
            )
            vlm_profile.safety_ms = elapsed_ms_since(safety_started)
        except SafetyViolationError as exc:
            return self._error("VALIDATION_REJECTED", raw_prompt, raw_response, parsed_json, exc, vlm_profile, safety_started)
        except (ValueError, json.JSONDecodeError) as exc:
            return self._error("INVALID_JSON", raw_prompt, raw_response, parsed_json, exc, vlm_profile, safety_started)
        except Exception as exc:
            logger.exception("Cognition inference failed")
            return self._error("INFERENCE_ERROR", raw_prompt, raw_response, parsed_json, exc, vlm_profile, safety_started)

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
        self.runtime.warmup()

    def _build_prompt(self, intent_text: str, *, image_path=None, drone_state=None) -> str:
        return build_cognition_prompt(intent_text, image_path=image_path, drone_state=drone_state)

    def _generate(self, raw_prompt: str, *, image_path: Optional[str]) -> str:
        raw_response, _ = self._generate_profiled(raw_prompt, image_path=image_path)
        return raw_response

    def _generate_profiled(self, raw_prompt: str, *, image_path: Optional[str]) -> tuple[str, VLMProfile]:
        return self.runtime.generate_profiled(raw_prompt, image_path=image_path)

    def _apply_safety_repairs(self, text_input, image_path, parsed_json, drone_state) -> None:
        self._apply_hsv_guardrail(text_input, image_path, parsed_json)
        repair_missing_command_from_text(text_input, parsed_json, drone_state)
        repair_missing_target_fields(parsed_json)
        drop_extra_command_fields(parsed_json)

    def _apply_hsv_guardrail(self, text_input: str, image_path: Optional[str], parsed_json: dict) -> None:
        apply_hsv_guardrail(text_input, image_path, parsed_json, self._frame_contains_hsv_color)

    def _error(self, reason, raw_prompt, raw_response, parsed_json, exc, vlm_profile, safety_started):
        if safety_started is not None:
            vlm_profile.safety_ms = elapsed_ms_since(safety_started)
        error = CognitionError(
            reason=reason,
            raw_prompt=raw_prompt,
            raw_response=raw_response,
            parsed_json=parsed_json,
            details=str(exc),
            vlm_profile=vlm_profile,
        )
        self._log_transaction(raw_prompt, raw_response, parsed_json, error)
        return error

    @staticmethod
    def _extract_json_object(raw_response: str) -> dict:
        return extract_json_object(raw_response)

    @staticmethod
    def _balanced_json_slice(text: str, start: int) -> str:
        return balanced_json_slice(text, start)

    @staticmethod
    def _requested_color(text_input: str) -> Optional[str]:
        return requested_target_color(text_input)

    @staticmethod
    def _frame_contains_hsv_color(image_path: str, color: str) -> bool:
        return frame_contains_hsv_color(image_path, color)

    def _log_transaction(self, raw_prompt, raw_response, parsed_json, validation_result) -> None:
        write_cognition_transaction(
            log_path=self.log_path,
            raw_prompt=raw_prompt,
            raw_response=raw_response,
            parsed_json=parsed_json,
            validation_result=validation_result,
        )

    @staticmethod
    def _validation_status(validation_result: CognitionResult | CognitionError) -> str:
        return validation_status(validation_result)


def build_vlm_runtime(
    backend: str,
    *,
    model_id: str,
    max_tokens: int,
    temperature: float,
    external_generator: Optional[GeneratorFn] = None,
):
    if external_generator is not None:
        return VLMRuntime(
            model_id=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            external_generator=external_generator,
        )

    if backend == "auto":
        backend = "mlx" if platform.system() == "Darwin" else "dummy"
    if backend == "mlx":
        return VLMRuntime(model_id=model_id, max_tokens=max_tokens, temperature=temperature)
    if backend == "dummy":
        from src.perception.dummy_vlm_runtime import DummyVLMRuntime

        return DummyVLMRuntime()
    if backend == "tensorrt":
        from src.perception.trt_vlm_runtime import TRTVLMRuntime

        return TRTVLMRuntime()
    if backend == "smolvlm":
        from src.perception.smolvlm_runtime import SmolVLMRuntime

        return SmolVLMRuntime(model_id=model_id, max_tokens=max_tokens, temperature=temperature)
    raise ValueError(f"Unsupported VLM backend: {backend}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    result = CognitionEngine().process_intent("move forward at one meter per second")
    if not result.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
