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
        debug_dir: str | Path | None = "tmp/frames",
        warmup_frames: int = 2,
        target_size: tuple[int, int] = (224, 224),
    ) -> None:
        self.camera_index = camera_index
        self.output_path = Path(output_path)
        self.debug_dir = Path(debug_dir) if debug_dir is not None else None
        self.warmup_frames = max(0, warmup_frames)
        self.target_size = target_size

    async def capture_single_frame(self) -> VisionFrame:
        return await asyncio.to_thread(self._capture_single_frame_sync)

    async def close(self) -> None:
        return

    def _capture_single_frame_sync(self) -> VisionFrame:
        try:
            import cv2  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise VisionError(
                "OpenCV is not installed. Run: .venv/bin/python -m pip install -r requirements-phase4.txt"
            ) from exc

        logger.info("Opening camera index=%s for one-shot frame capture", self.camera_index)
        capture = self._open_capture(cv2)
        try:
            if not capture.isOpened():
                raise VisionError(f"camera device {self.camera_index} is not available")

            frame = None
            for _ in range(self.warmup_frames + 1):
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise VisionError(f"camera device {self.camera_index} did not return a frame")

            height, width = frame.shape[:2]
            optimized_frame = capture_and_downsample(frame, target_size=self.target_size)
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            saved = cv2.imwrite(str(self.output_path), optimized_frame)
            if not saved:
                raise VisionError(f"failed to write frame to {self.output_path}")

            captured_at = datetime.now(timezone.utc)
            debug_path = self._write_debug_frame(cv2, frame, captured_at)
            logger.info(
                "Captured camera frame: index=%s size=%sx%s latest=%s debug=%s",
                self.camera_index,
                width,
                height,
                self.output_path,
                debug_path or "<disabled>",
            )
            return VisionFrame(
                image_path=str(self.output_path),
                captured_at=captured_at,
            )
        finally:
            capture.release()
            logger.info("Released camera index=%s", self.camera_index)

    def _open_capture(self, cv2):
        if hasattr(cv2, "CAP_AVFOUNDATION"):
            try:
                capture = cv2.VideoCapture(self.camera_index, cv2.CAP_AVFOUNDATION)
                if capture.isOpened():
                    return capture
                capture.release()
            except TypeError:
                pass

        return cv2.VideoCapture(self.camera_index)

    def _write_debug_frame(self, cv2, frame, captured_at: datetime) -> Optional[Path]:
        if self.debug_dir is None:
            return None

        self.debug_dir.mkdir(parents=True, exist_ok=True)
        timestamp = captured_at.strftime("%Y%m%dT%H%M%S%fZ")
        debug_path = self.debug_dir / f"frame_{timestamp}.jpg"
        if not cv2.imwrite(str(debug_path), frame):
            logger.warning("Failed to write debug frame to %s", debug_path)
            return None

        return debug_path


def capture_and_downsample(frame, target_size: tuple[int, int] = (224, 224)):
    import cv2  # pylint: disable=import-outside-toplevel

    if frame is None or len(frame.shape) < 2:
        raise VisionError("cannot downsample an empty camera frame")
    height, width = frame.shape[:2]
    crop_size = min(height, width)
    top = max(0, (height - crop_size) // 2)
    left = max(0, (width - crop_size) // 2)
    cropped = frame[top : top + crop_size, left : left + crop_size]
    return cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)


async def capture_single_frame(vision_module: Optional[VisionModule] = None) -> VisionFrame:
    module = vision_module or VisionModule()
    return await module.capture_single_frame()
