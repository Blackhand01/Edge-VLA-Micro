from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

from vision_module import VisionError, VisionModule


class FakeCapture:
    last_instance = None

    def __init__(self, camera_index: int, *, opened: bool = True) -> None:
        self.camera_index = camera_index
        self.opened = opened
        self.read_calls = 0
        self.released = False
        FakeCapture.last_instance = self

    def isOpened(self) -> bool:
        return self.opened

    def read(self):
        self.read_calls += 1
        return True, np.zeros((4, 4, 3), dtype=np.uint8)

    def release(self) -> None:
        self.released = True


class VisionModuleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self) -> None:
        sys.modules.pop("cv2", None)

    async def test_capture_single_frame_writes_file_and_releases_camera(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "frame.jpg"
            fake_cv2 = types.SimpleNamespace(
                VideoCapture=lambda index: FakeCapture(index),
                imwrite=lambda path, frame: Path(path).write_bytes(b"jpg") > 0,
            )
            sys.modules["cv2"] = fake_cv2

            module = VisionModule(camera_index=2, output_path=output_path, warmup_frames=0)
            frame = await module.capture_single_frame()

            self.assertEqual(frame.image_path, str(output_path))
            self.assertTrue(output_path.exists())
            self.assertEqual(FakeCapture.last_instance.camera_index, 2)
            self.assertEqual(FakeCapture.last_instance.read_calls, 1)
            self.assertTrue(FakeCapture.last_instance.released)

    async def test_capture_single_frame_raises_vision_error_when_camera_unavailable(self) -> None:
        fake_cv2 = types.SimpleNamespace(
            VideoCapture=lambda index: FakeCapture(index, opened=False),
            imwrite=lambda path, frame: True,
        )
        sys.modules["cv2"] = fake_cv2

        module = VisionModule()

        with self.assertRaises(VisionError):
            await module.capture_single_frame()

        self.assertTrue(FakeCapture.last_instance.released)


if __name__ == "__main__":
    unittest.main()
