from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

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
SYSTEM_PROMPT = (
    "Sei il computer di bordo di un drone. Guarda l'immagine della telecamera FPV "
    "e leggi il comando vocale dell'operatore. Il tuo compito è estrarre l'intento "
    "spaziale e tradurlo in un JSON valido conforme allo schema. Non aggiungere testo extra."
)


@dataclass(frozen=True)
class CognitionResult:
    raw_prompt: str
    raw_response: str
    parsed_json: dict
    validated_command: ValidatedCommand

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

        try:
            raw_response = self._generate(raw_prompt, image_path=image_path)
            parsed_json = self._extract_json_object(raw_response)
            validated = self.validator.validate(
                json.dumps(parsed_json, ensure_ascii=False, separators=(",", ":")),
                drone_state=drone_state,
            )
        except SafetyViolationError as exc:
            error = CognitionError(
                reason="VALIDATION_REJECTED",
                raw_prompt=raw_prompt,
                raw_response=raw_response,
                parsed_json=parsed_json,
                details=str(exc),
            )
            self._log_transaction(raw_prompt, raw_response, parsed_json, error)
            return error
        except (ValueError, json.JSONDecodeError) as exc:
            error = CognitionError(
                reason="INVALID_JSON",
                raw_prompt=raw_prompt,
                raw_response=raw_response,
                parsed_json=parsed_json,
                details=str(exc),
            )
            self._log_transaction(raw_prompt, raw_response, parsed_json, error)
            return error
        except Exception as exc:  # noqa: BLE001 - local inference failures must not block control.
            logger.exception("Cognition inference failed")
            error = CognitionError(
                reason="INFERENCE_ERROR",
                raw_prompt=raw_prompt,
                raw_response=raw_response,
                parsed_json=parsed_json,
                details=str(exc),
            )
            self._log_transaction(raw_prompt, raw_response, parsed_json, error)
            return error

        result = CognitionResult(
            raw_prompt=raw_prompt,
            raw_response=raw_response,
            parsed_json=parsed_json,
            validated_command=validated,
        )
        self._log_transaction(raw_prompt, raw_response, parsed_json, result)
        return result

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
        if self._external_generator is not None:
            return self._external_generator(raw_prompt, image_path)

        self._load_local_model()
        result = self._mlx_generate(
            self._model,
            self._processor,
            prompt=raw_prompt,
            image=image_path,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            verbose=False,
        )
        return result.text if hasattr(result, "text") else str(result)

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

        logger.info("Loading local MLX-VLM model: %s", self.model_id)
        self._model, self._processor = load(self.model_id)
        self._mlx_generate = generate

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
    result = engine.process_intent("vai avanti a 1 metro al secondo mantenendo quota e yaw zero")
    if not result.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
