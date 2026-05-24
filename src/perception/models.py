from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from src.safety.command_validator import ValidatedCommand


GeneratorFn = Callable[[str, Optional[str]], str]


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


def estimate_generated_token_count(text: str) -> int:
    return len(re.findall(r"\S+", text))
