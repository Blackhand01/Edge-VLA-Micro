from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LoopStatus:
    state: str
    input_text: str
    action: str
    audio_ms: float
    vision_ms: float
    vlm_ms: float
    vlm_ttft_ms: float
    vlm_decode_time_ms: float
    vlm_tps: float
    safety_ms: float
    total_latency_ms: float
