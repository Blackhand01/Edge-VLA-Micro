from __future__ import annotations

import json
import time
from typing import Optional

from src.perception.models import VLMProfile, estimate_generated_token_count


class DummyVLMRuntime:
    def __init__(self, *, latency_s: float = 0.1) -> None:
        self.latency_s = latency_s

    def warmup(self) -> None:
        time.sleep(min(self.latency_s, 0.1))

    def generate_profiled(self, raw_prompt: str, *, image_path: Optional[str]) -> tuple[str, VLMProfile]:
        del image_path
        started = time.perf_counter()
        time.sleep(self.latency_s)
        raw_response = json.dumps(command_for_prompt(raw_prompt), separators=(",", ":"))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        generated_tokens = estimate_generated_token_count(raw_response)
        return raw_response, VLMProfile(
            ttft_ms=elapsed_ms,
            generated_tokens=generated_tokens,
            tps=generated_tokens / (elapsed_ms / 1000.0) if elapsed_ms > 0 else 0.0,
            fallback_reason="dummy runtime",
        )


def command_for_prompt(prompt: str) -> dict[str, object]:
    lowered = user_intent_from_prompt(prompt).lower()
    base = {"target_found": True}
    if "take off" in lowered or "takeoff" in lowered:
        return {"command": "takeoff", **base, "reasoning": "dummy takeoff command"}
    if "land" in lowered:
        return {"command": "land", **base, "reasoning": "dummy land command"}
    if "arm" in lowered and "disarm" not in lowered:
        return {"command": "arm", **base, "reasoning": "dummy arm command"}
    if "disarm" in lowered:
        return {"command": "disarm", **base, "reasoning": "dummy disarm command"}
    if any(token in lowered for token in ("move", "forward", "red", "blue", "object")):
        return {
            "command": "move_velocity",
            **base,
            "reasoning": "dummy movement proposal",
            "velocity_x": 0.5,
            "velocity_y": 0.0,
            "velocity_z": 0.0,
            "yaw_deg": 0.0,
        }
    return {"command": "hold", **base, "reasoning": "dummy hold command"}


def user_intent_from_prompt(prompt: str) -> str:
    marker = "USER_INTENT:"
    if marker not in prompt:
        return prompt
    tail = prompt.split(marker, 1)[1]
    return tail.split("\nJSON:", 1)[0].strip()
