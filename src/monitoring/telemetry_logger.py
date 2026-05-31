from __future__ import annotations

import asyncio
import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


logger = logging.getLogger(__name__)

PERFORMANCE_COLUMNS = (
    "timestamp",
    "audio_ms",
    "vision_ms",
    "ttft_ms",
    "decode_time_ms",
    "tps",
    "safety_ms",
    "total_latency_ms",
)

TELEMETRY_COLUMNS = (
    "timestamp_utc",
    "profile",
    "component",
    "request_id",
    "source",
    "ok",
    "action",
    "command",
    "state",
    "input_text",
    "image_present",
    "audio_ms",
    "vision_ms",
    "http_ms",
    "cognition_ms",
    "action_ms",
    "total_ms",
    "ttft_ms",
    "decode_time_ms",
    "generated_tokens",
    "tps",
    "safety_ms",
    "reason",
    "details",
)

_FALLBACK_JPEG = bytes(
    [
        0xFF,
        0xD8,
        0xFF,
        0xDB,
        0x00,
        0x43,
        0x00,
        *([0x08] * 64),
        0xFF,
        0xC0,
        0x00,
        0x0B,
        0x08,
        0x00,
        0x01,
        0x00,
        0x01,
        0x01,
        0x01,
        0x11,
        0x00,
        0xFF,
        0xC4,
        0x00,
        0x14,
        0x00,
        0x01,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0xFF,
        0xC4,
        0x00,
        0x14,
        0x10,
        0x01,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0xFF,
        0xDA,
        0x00,
        0x08,
        0x01,
        0x01,
        0x00,
        0x00,
        0x3F,
        0x00,
        0xD2,
        0xCF,
        0x20,
        0xFF,
        0xD9,
    ]
)


class BlackboxLogger:
    def __init__(
        self,
        *,
        sessions_dir: str | Path = "logs/sessions",
        frame_width: int = 448,
        frame_height: int = 448,
    ) -> None:
        self.sessions_dir = Path(sessions_dir)
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._pending_tasks: set[asyncio.Task] = set()

    def log_inference(
        self,
        *,
        image_path: Optional[str],
        payload: dict[str, Any],
    ) -> None:
        timestamp = datetime.now(timezone.utc)
        stem = timestamp.strftime("%Y%m%dT%H%M%S%fZ")
        jpg_path = self.sessions_dir / f"{stem}.jpg"
        json_path = self.sessions_dir / f"{stem}.json"
        record = {
            "timestamp_utc": timestamp.isoformat(),
            "frame_file": jpg_path.name,
            "record_file": json_path.name,
            **payload,
        }

        task = asyncio.create_task(
            asyncio.to_thread(
                self._write_bundle,
                source_image_path=image_path,
                destination_image_path=jpg_path,
                destination_json_path=json_path,
                record=record,
            )
        )
        self._pending_tasks.add(task)
        task.add_done_callback(self._on_write_done)

    async def close(self) -> None:
        if not self._pending_tasks:
            return
        await asyncio.gather(*list(self._pending_tasks), return_exceptions=True)

    def _on_write_done(self, task: asyncio.Task) -> None:
        self._pending_tasks.discard(task)
        exc = task.exception()
        if exc is not None:
            logger.warning("Blackbox write failed: %s", exc)

    def _write_bundle(
        self,
        *,
        source_image_path: Optional[str],
        destination_image_path: Path,
        destination_json_path: Path,
        record: dict[str, Any],
    ) -> None:
        self._write_frame(source_image_path, destination_image_path)
        destination_json_path.write_text(
            json.dumps(record, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _write_frame(self, source_image_path: Optional[str], destination_image_path: Path) -> None:
        try:
            import cv2  # pylint: disable=import-outside-toplevel
            import numpy as np  # pylint: disable=import-outside-toplevel
        except ImportError:
            destination_image_path.write_bytes(_FALLBACK_JPEG)
            return

        frame = None
        if source_image_path:
            frame = cv2.imread(source_image_path)

        if frame is None:
            frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
            cv2.putText(
                frame,
                "NO_FRAME",
                (20, self.frame_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )
        else:
            frame = cv2.resize(
                frame,
                (self.frame_width, self.frame_height),
                interpolation=cv2.INTER_AREA,
            )

        if not cv2.imwrite(str(destination_image_path), frame):
            destination_image_path.write_bytes(_FALLBACK_JPEG)


class PerformanceLogger:
    def __init__(self, *, path: str | Path = "logs/performance.csv") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._pending_tasks: set[asyncio.Task] = set()

    def log_run(
        self,
        *,
        audio_ms: float,
        vision_ms: float,
        ttft_ms: float,
        decode_time_ms: float,
        tps: float,
        safety_ms: float,
        total_latency_ms: float,
    ) -> None:
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "audio_ms": f"{audio_ms:.3f}",
            "vision_ms": f"{vision_ms:.3f}",
            "ttft_ms": f"{ttft_ms:.3f}",
            "decode_time_ms": f"{decode_time_ms:.3f}",
            "tps": f"{tps:.6f}",
            "safety_ms": f"{safety_ms:.3f}",
            "total_latency_ms": f"{total_latency_ms:.3f}",
        }
        task = asyncio.create_task(asyncio.to_thread(self._append_row, row))
        self._pending_tasks.add(task)
        task.add_done_callback(self._on_write_done)

    async def close(self) -> None:
        if not self._pending_tasks:
            return
        await asyncio.gather(*list(self._pending_tasks), return_exceptions=True)

    def _on_write_done(self, task: asyncio.Task) -> None:
        self._pending_tasks.discard(task)
        exc = task.exception()
        if exc is not None:
            logger.warning("Performance CSV write failed: %s", exc)

    def _append_row(self, row: dict[str, str]) -> None:
        write_header = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=PERFORMANCE_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)


class TelemetryCsvLogger:
    def __init__(self, *, path: str | Path = "logs/telemetry.csv") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, **fields: Any) -> None:
        row = {column: "" for column in TELEMETRY_COLUMNS}
        row["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        for key, value in fields.items():
            if key not in row:
                continue
            row[key] = self._format_value(value)
        self._append_row(row)

    def _append_row(self, row: dict[str, str]) -> None:
        write_header = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=TELEMETRY_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    @staticmethod
    def _format_value(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, float):
            return f"{value:.3f}"
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, separators=(",", ":"), ensure_ascii=False)
        return str(value)
