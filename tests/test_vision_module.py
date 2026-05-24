from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

from src.perception.vision_module import VisionError, VisionModule, capture_and_downsample


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
        return True, np.zeros((4, 6, 3), dtype=np.uint8)

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
                resize=lambda frame, target_size, interpolation=None: np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8),
                INTER_AREA=3,
            )
            sys.modules["cv2"] = fake_cv2

            module = VisionModule(
                camera_index=2,
                output_path=output_path,
                debug_dir=Path(tmpdir) / "frames",
                warmup_frames=0,
            )
            frame = await module.capture_single_frame()

            self.assertEqual(frame.image_path, str(output_path))
            self.assertTrue(output_path.exists())
            debug_frames = list((Path(tmpdir) / "frames").glob("frame_*.jpg"))
            self.assertEqual(len(debug_frames), 1)
            self.assertEqual(FakeCapture.last_instance.camera_index, 2)
            self.assertEqual(FakeCapture.last_instance.read_calls, 1)
            self.assertTrue(FakeCapture.last_instance.released)

    async def test_capture_single_frame_can_disable_debug_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "frame.jpg"
            fake_cv2 = types.SimpleNamespace(
                VideoCapture=lambda index: FakeCapture(index),
                imwrite=lambda path, frame: Path(path).write_bytes(b"jpg") > 0,
                resize=lambda frame, target_size, interpolation=None: np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8),
                INTER_AREA=3,
            )
            sys.modules["cv2"] = fake_cv2

            module = VisionModule(
                camera_index=2,
                output_path=output_path,
                debug_dir=None,
                warmup_frames=0,
            )
            await module.capture_single_frame()

            self.assertFalse((Path(tmpdir) / "frames").exists())

    async def test_capture_single_frame_raises_vision_error_when_camera_unavailable(self) -> None:
        fake_cv2 = types.SimpleNamespace(
            VideoCapture=lambda index: FakeCapture(index, opened=False),
            imwrite=lambda path, frame: True,
            resize=lambda frame, target_size, interpolation=None: frame,
            INTER_AREA=3,
        )
        sys.modules["cv2"] = fake_cv2

        module = VisionModule()

        with self.assertRaises(VisionError):
            await module.capture_single_frame()

        self.assertTrue(FakeCapture.last_instance.released)

    async def test_capture_and_downsample_center_crops_before_resize(self) -> None:
        observed: dict[str, np.ndarray] = {}

        def resize(frame, target_size, interpolation=None):
            del interpolation
            observed["crop"] = frame.copy()
            return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

        fake_cv2 = types.SimpleNamespace(resize=resize, INTER_AREA=3)
        sys.modules["cv2"] = fake_cv2
        frame = np.zeros((4, 6, 3), dtype=np.uint8)
        for column in range(6):
            frame[:, column, :] = column

        resized = capture_and_downsample(frame, target_size=(2, 2))

        self.assertEqual(resized.shape, (2, 2, 3))
        self.assertEqual(observed["crop"].shape, (4, 4, 3))
        self.assertTrue(np.all(observed["crop"][:, 0, :] == 1))
        self.assertTrue(np.all(observed["crop"][:, -1, :] == 4))


if __name__ == "__main__":
    unittest.main()
