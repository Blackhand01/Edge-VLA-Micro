from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


class VisionError(RuntimeError):
    """Raised when one-shot camera capture fails."""


@dataclass(frozen=True)
class VisionFrame:
    image_path: str
    captured_at: datetime


class VisionModule:
    def __init__(
        self,
        *,
        camera_index: int = 0,
        output_path: str | Path = "tmp/frame.jpg",
        warmup_frames: int = 2,
    ) -> None:
        self.camera_index = camera_index
        self.output_path = Path(output_path)
        self.warmup_frames = max(0, warmup_frames)

    async def capture_single_frame(self) -> VisionFrame:
        await asyncio.sleep(0)
        return self._capture_single_frame_sync()

    async def close(self) -> None:
        return

    def _capture_single_frame_sync(self) -> VisionFrame:
        try:
            import cv2  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise VisionError(
                "OpenCV is not installed. Run: .venv/bin/python -m pip install -r requirements-phase4.txt"
            ) from exc

        capture = cv2.VideoCapture(self.camera_index)
        try:
            if not capture.isOpened():
                raise VisionError(f"camera device {self.camera_index} is not available")

            frame = None
            for _ in range(self.warmup_frames + 1):
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise VisionError(f"camera device {self.camera_index} did not return a frame")

            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            saved = cv2.imwrite(str(self.output_path), frame)
            if not saved:
                raise VisionError(f"failed to write frame to {self.output_path}")

            return VisionFrame(
                image_path=str(self.output_path),
                captured_at=datetime.now(timezone.utc),
            )
        finally:
            capture.release()


async def capture_single_frame(vision_module: Optional[VisionModule] = None) -> VisionFrame:
    module = vision_module or VisionModule()
    return await module.capture_single_frame()
