from __future__ import annotations

from typing import Optional

from src.perception.models import VLMProfile


class TRTVLMRuntime:
    def __init__(self, *, engine_path: Optional[str] = None) -> None:
        self.engine_path = engine_path

    def warmup(self) -> None:
        self._raise_not_implemented()

    def generate_profiled(self, raw_prompt: str, *, image_path: Optional[str]) -> tuple[str, VLMProfile]:
        del raw_prompt, image_path
        self._raise_not_implemented()

    def _raise_not_implemented(self) -> None:
        raise RuntimeError(
            "TensorRT VLM runtime is not implemented yet. Use `--vlm-backend dummy` "
            "for Jetson smoke tests or `--vlm-backend mlx` on macOS."
        )
