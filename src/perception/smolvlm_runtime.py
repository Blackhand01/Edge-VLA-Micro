from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

from src.perception.models import VLMProfile, estimate_generated_token_count
from src.perception.prompts import DEFAULT_MODEL_ID
from src.perception.vlm_runtime import elapsed_ms_since, tokens_per_second


logger = logging.getLogger(__name__)

DEFAULT_SMOLVLM_MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
DEFAULT_IMAGE_SIZE = 384
SYNTHETIC_IMAGE_PATH = Path("tmp/smolvlm_no_image.jpg")
MOVE_INTENT_PHRASES = (
    "move",
    "moving",
    "go",
    "toward",
    "towards",
    "approach",
    "follow",
    "forward",
    "backward",
    "left",
    "right",
)


class SmolVLMRuntime:
    def __init__(
        self,
        *,
        model_id: str = DEFAULT_SMOLVLM_MODEL_ID,
        max_tokens: int = 128,
        temperature: float = 0.0,
        image_size: int = DEFAULT_IMAGE_SIZE,
    ) -> None:
        self.model_id = resolve_smolvlm_model_id(model_id)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.image_size = image_size
        self.model = None
        self.processor = None

    def warmup(self) -> None:
        self.load_model()
        self.generate_profiled("USER_INTENT:\nHold position.\n\nJSON:", image_path=None, max_tokens=1)

    def generate_profiled(
        self,
        raw_prompt: str,
        *,
        image_path: Optional[str],
        max_tokens: Optional[int] = None,
    ) -> tuple[str, VLMProfile]:
        self.load_model()
        image = self.load_image(image_path)
        prompt = build_smolvlm_prompt(raw_prompt)
        processor_input = self.build_processor_input(prompt, image)

        started = time.perf_counter()
        try:
            import torch
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Unable to import torch for SmolVLM runtime: {exc}") from exc

        disable_distributed_generation_checks()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        generate_kwargs = {
            "max_new_tokens": max_tokens or self.max_tokens,
            "do_sample": self.temperature > 0.0,
        }
        if self.temperature > 0.0:
            generate_kwargs["temperature"] = self.temperature
        with torch.inference_mode():
            generated_ids = self.model.generate(**processor_input, **generate_kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = elapsed_ms_since(started)

        input_len = processor_input["input_ids"].shape[1]
        generated_ids = generated_ids[:, input_len:]
        model_response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        raw_response = normalize_smolvlm_response(model_response, raw_prompt)
        generated_tokens = generated_ids.shape[1] if hasattr(generated_ids, "shape") else estimate_generated_token_count(raw_response)
        return raw_response, VLMProfile(
            ttft_ms=elapsed_ms,
            generated_tokens=int(generated_tokens),
            tps=tokens_per_second(int(generated_tokens), elapsed_ms),
            fallback_reason="smolvlm single-device cuda",
        )

    def load_model(self) -> None:
        if self.model is not None and self.processor is not None:
            return
        try:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Unable to import SmolVLM dependencies: {exc}") from exc

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for SmolVLMRuntime on Jetson.")

        logger.info("Loading SmolVLM model on CUDA: %s", self.model_id)
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            size={"longest_edge": self.image_size},
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            _attn_implementation="eager",
            low_cpu_mem_usage=True,
        ).to("cuda")

    def build_processor_input(self, prompt: str, image):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        return self.processor(text=text, images=[image], return_tensors="pt").to("cuda")

    def load_image(self, image_path: Optional[str]):
        try:
            from PIL import Image, ImageDraw
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Unable to import PIL for SmolVLM runtime: {exc}") from exc

        if image_path is None:
            image_path = str(ensure_synthetic_image(Image, ImageDraw, self.image_size))
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((self.image_size, self.image_size))
        canvas = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))
        x = (self.image_size - image.width) // 2
        y = (self.image_size - image.height) // 2
        canvas.paste(image, (x, y))
        return canvas


def resolve_smolvlm_model_id(model_id: str) -> str:
    if model_id == DEFAULT_MODEL_ID or model_id.startswith("mlx-community/"):
        return DEFAULT_SMOLVLM_MODEL_ID
    return model_id


def build_smolvlm_prompt(raw_prompt: str) -> str:
    user_intent = extract_section(raw_prompt, "USER_INTENT", "JSON") or raw_prompt
    drone_state = extract_section(raw_prompt, "CURRENT_DRONE_STATE", "IMAGE_PATH") or "<UNKNOWN>"
    return (
        "Return exactly one compact JSON object and no markdown.\n"
        "Allowed commands: arm, disarm, takeoff, land, hold, move_velocity.\n"
        'For arm: {"command":"arm","target_found":true,"reasoning":"..."}\n'
        'For takeoff: {"command":"takeoff","target_found":true,"reasoning":"..."}\n'
        'For land: {"command":"land","target_found":true,"reasoning":"..."}\n'
        'For hold: {"command":"hold","target_found":true,"reasoning":"..."}\n'
        'For move: {"command":"move_velocity","target_found":true,"reasoning":"...",'
        '"velocity_x":0.5,"velocity_y":0.0,"velocity_z":0.0,"yaw_deg":0.0}\n'
        "If CURRENT_DRONE_STATE is GROUNDED and the user asks to takeoff or move, return arm first.\n"
        f"CURRENT_DRONE_STATE:\n{drone_state}\n"
        f"USER_INTENT:\n{user_intent}\n"
        "JSON:"
    )


def normalize_smolvlm_response(model_response: str, raw_prompt: str) -> str:
    user_intent = extract_section(raw_prompt, "USER_INTENT", "JSON") or raw_prompt
    drone_state = extract_section(raw_prompt, "CURRENT_DRONE_STATE", "IMAGE_PATH")
    explicit_command = explicit_command_from_intent(user_intent, drone_state)
    parsed = parse_complete_json_object(model_response)
    if parsed is not None:
        normalized = command_from_parsed_json(parsed)
        if normalized is not None:
            if explicit_command is not None:
                if normalized.get("command") == explicit_command.get("command"):
                    return json.dumps(normalized, separators=(",", ":"))
                return json.dumps(explicit_command, separators=(",", ":"))
            if is_critical_command(str(normalized.get("command"))):
                return json.dumps(
                    {
                        "command": "hold",
                        "target_found": True,
                        "reasoning": (
                            "safety fallback: model proposed a critical command "
                            "without an explicit operator command"
                        ),
                    },
                    separators=(",", ":"),
                )
            return json.dumps(normalized, separators=(",", ":"))

    if explicit_command is not None:
        return json.dumps(explicit_command, separators=(",", ":"))

    fallback = command_from_intent(user_intent, drone_state)
    fallback["reasoning"] = f"smolvlm fallback after malformed output: {compact_reason(model_response)}"
    return json.dumps(fallback, separators=(",", ":"))


def parse_complete_json_object(text: str) -> Optional[dict]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def command_from_parsed_json(parsed: dict) -> Optional[dict[str, object]]:
    command = parsed.get("command")
    if command not in {"arm", "disarm", "takeoff", "land", "hold", "move_velocity"}:
        return None
    normalized: dict[str, object] = {
        "command": command,
        "target_found": bool(parsed.get("target_found", True)),
        "reasoning": str(parsed.get("reasoning") or "smolvlm parsed command"),
    }
    if command == "move_velocity":
        normalized.update(
            {
                "velocity_x": clamp_float(parsed.get("velocity_x", 0.5), -2.0, 2.0),
                "velocity_y": clamp_float(parsed.get("velocity_y", 0.0), -2.0, 2.0),
                "velocity_z": clamp_float(parsed.get("velocity_z", 0.0), -1.0, 1.0),
                "yaw_deg": clamp_float(parsed.get("yaw_deg", 0.0), -180.0, 180.0),
            }
        )
    return normalized


def explicit_command_from_intent(user_intent: str, drone_state: str) -> Optional[dict[str, object]]:
    lowered = user_intent.lower()
    state = drone_state.upper()
    base: dict[str, object] = {
        "target_found": True,
        "reasoning": "deterministic command from explicit operator text",
    }
    if has_any_phrase(lowered, ("hold position", "hold", "hover", "stay", "stop")):
        return {"command": "hold", **base}
    if has_any_phrase(lowered, ("land", "landing")):
        return {"command": "land", **base}
    if has_any_phrase(lowered, ("disarm", "power off")):
        return {"command": "disarm", **base}
    if has_any_phrase(lowered, ("take off", "takeoff", "launch")):
        if "GROUNDED" in state:
            return {"command": "arm", **base, "reasoning": "takeoff requested while grounded; arm is the next valid action"}
        return {"command": "takeoff", **base}
    if has_word(lowered, "arm"):
        return {"command": "arm", **base}
    if has_any_phrase(lowered, MOVE_INTENT_PHRASES):
        if "GROUNDED" in state:
            return {"command": "arm", **base, "reasoning": "movement requested while grounded; arm is the next valid action"}
        velocity_x = movement_speed_from_text(lowered)
        if "backward" in lowered:
            velocity_x = -velocity_x
        velocity_y = -0.5 if "left" in lowered else 0.5 if "right" in lowered else 0.0
        return {
            "command": "move_velocity",
            **base,
            "velocity_x": velocity_x,
            "velocity_y": velocity_y,
            "velocity_z": 0.0,
            "yaw_deg": 0.0,
        }
    return None


def is_critical_command(command: str) -> bool:
    return command in {"arm", "disarm", "takeoff", "land", "move_velocity"}


def command_from_intent(user_intent: str, drone_state: str) -> dict[str, object]:
    lowered = user_intent.lower()
    state = drone_state.upper()
    base: dict[str, object] = {"target_found": True}
    if ("take off" in lowered or "takeoff" in lowered or "launch" in lowered) and "GROUNDED" in state:
        return {"command": "arm", **base}
    if any(token in lowered for token in MOVE_INTENT_PHRASES) and "GROUNDED" in state:
        return {"command": "arm", **base}
    if "take off" in lowered or "takeoff" in lowered or "launch" in lowered:
        return {"command": "takeoff", **base}
    if "land" in lowered:
        return {"command": "land", **base}
    if "disarm" in lowered:
        return {"command": "disarm", **base}
    if "arm" in lowered:
        return {"command": "arm", **base}
    if any(token in lowered for token in MOVE_INTENT_PHRASES):
        velocity_x = movement_speed_from_text(lowered)
        if "backward" in lowered:
            velocity_x = -velocity_x
        velocity_y = -0.5 if "left" in lowered else 0.5 if "right" in lowered else 0.0
        return {
            "command": "move_velocity",
            **base,
            "velocity_x": velocity_x,
            "velocity_y": velocity_y,
            "velocity_z": 0.0,
            "yaw_deg": 0.0,
        }
    return {"command": "hold", **base}


def has_any_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    return any(has_word(text, phrase) for phrase in phrases)


def has_word(text: str, phrase: str) -> bool:
    escaped = re.escape(phrase).replace("\\ ", r"\s+")
    return re.search(rf"(?<![a-z0-9_]){escaped}(?![a-z0-9_])", text) is not None


def movement_speed_from_text(text: str) -> float:
    if has_any_phrase(text, ("one meter per second", "1 meter per second", "1 m/s", "one metre per second")):
        return 1.0
    return 0.5


def clamp_float(value, minimum: float, maximum: float) -> float:  # noqa: ANN001
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    return min(max(number, minimum), maximum)


def compact_reason(text: str) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact[:80] if compact else "empty response"


def extract_section(text: str, start_label: str, end_label: str) -> str:
    pattern = rf"{re.escape(start_label)}:\s*(.*?)\n{re.escape(end_label)}:"
    match = re.search(pattern, text, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def ensure_synthetic_image(Image, ImageDraw, image_size: int) -> Path:  # noqa: ANN001
    path = SYNTHETIC_IMAGE_PATH
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (image_size, image_size), (22, 28, 36))
    draw = ImageDraw.Draw(image)
    margin = max(16, image_size // 8)
    draw.rectangle(
        (margin, image_size // 2 - 24, image_size - margin, image_size // 2 + 24),
        outline=(80, 180, 255),
        width=3,
    )
    draw.rectangle(
        (image_size // 2 - 36, margin, image_size // 2 + 36, image_size - margin),
        outline=(255, 180, 70),
        width=3,
    )
    draw.text((margin, image_size - margin - 20), "no camera frame", fill=(230, 230, 230))
    image.save(path)
    return path


def disable_distributed_generation_checks() -> None:
    try:
        import transformers.generation.utils as generation_utils

        generation_utils.is_deepspeed_zero3_enabled = lambda: False
        generation_utils.is_fsdp_managed_module = lambda _module: False
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not patch distributed generation checks: %s", exc)
