from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.monitoring import BlackboxLogger, PerformanceLogger


class _FakeCv2:
    FONT_HERSHEY_SIMPLEX = 0
    INTER_AREA = 1

    @staticmethod
    def imread(path: str):
        if Path(path).exists():
            return np.ones((32, 64, 3), dtype=np.uint8) * 127
        return None

    @staticmethod
    def resize(frame, size, interpolation=1):  # noqa: ARG004 - signature mirrors cv2.resize.
        width, height = size
        return np.zeros((height, width, 3), dtype=np.uint8)

    @staticmethod
    def putText(frame, text, org, fontFace, fontScale, color, thickness):  # noqa: N803, ARG004
        del text, org, fontFace, fontScale, color, thickness
        return frame

    @staticmethod
    def imwrite(path: str, frame) -> bool:
        Path(path).write_bytes(frame.tobytes()[:64] or b"frame")
        return True


class BlackboxLoggerTests(unittest.IsolatedAsyncioTestCase):
    async def test_log_inference_writes_jpg_and_json_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sessions_dir = Path(tmpdir) / "logs" / "sessions"
            source_image = Path(tmpdir) / "frame.jpg"
            source_image.write_bytes(b"source")
            logger = BlackboxLogger(sessions_dir=sessions_dir, frame_width=448, frame_height=448)

            with patch.dict("sys.modules", {"cv2": _FakeCv2}):
                logger.log_inference(
                    image_path=str(source_image),
                    payload={"prompt": "P", "raw_response": "R", "validation": {"status": "ACCEPTED"}},
                )
                await logger.close()

            jpg_files = sorted(sessions_dir.glob("*.jpg"))
            json_files = sorted(sessions_dir.glob("*.json"))

            self.assertEqual(len(jpg_files), 1)
            self.assertEqual(len(json_files), 1)
            self.assertEqual(jpg_files[0].stem, json_files[0].stem)

            payload = json.loads(json_files[0].read_text(encoding="utf-8"))
            self.assertEqual(payload["prompt"], "P")
            self.assertEqual(payload["raw_response"], "R")
            self.assertEqual(payload["validation"]["status"], "ACCEPTED")
            self.assertEqual(payload["frame_file"], jpg_files[0].name)
            self.assertEqual(payload["record_file"], json_files[0].name)

    async def test_log_inference_writes_placeholder_when_frame_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sessions_dir = Path(tmpdir) / "logs" / "sessions"
            logger = BlackboxLogger(sessions_dir=sessions_dir)

            with patch.dict("sys.modules", {"cv2": _FakeCv2}):
                logger.log_inference(
                    image_path=None,
                    payload={"prompt": "P", "raw_response": "R", "validation": {"status": "REJECTED"}},
                )
                await logger.close()

            jpg_files = sorted(sessions_dir.glob("*.jpg"))
            json_files = sorted(sessions_dir.glob("*.json"))
            self.assertEqual(len(jpg_files), 1)
            self.assertEqual(len(json_files), 1)

    async def test_performance_logger_appends_csv_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "logs" / "performance.csv"
            logger = PerformanceLogger(path=csv_path)

            logger.log_run(
                audio_ms=10.0,
                vision_ms=20.0,
                ttft_ms=30.0,
                decode_time_ms=40.0,
                tps=12.5,
                safety_ms=5.0,
                total_latency_ms=120.0,
            )
            await logger.close()

            rows = csv_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(rows[0], "timestamp,audio_ms,vision_ms,ttft_ms,decode_time_ms,tps,safety_ms,total_latency_ms")
            self.assertIn("10.000,20.000,30.000,40.000,12.500000,5.000,120.000", rows[1])


if __name__ == "__main__":
    unittest.main()
