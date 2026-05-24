from __future__ import annotations

import argparse
import asyncio
import logging

from vision_module import VisionModule


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture one OpenCV camera frame for VLA diagnostics.")
    parser.add_argument("--camera-index", default=0, type=int, help="OpenCV camera index.")
    parser.add_argument("--frame-path", default="tmp/probe_frame.jpg", help="Output path for latest probe frame.")
    parser.add_argument("--debug-dir", default="tmp/probe_frames", help="Directory for timestamped probe frames.")
    parser.add_argument("--warmup-frames", default=5, type=int, help="Frames to discard before saving.")
    return parser


async def run_probe(args: argparse.Namespace) -> None:
    module = VisionModule(
        camera_index=args.camera_index,
        output_path=args.frame_path,
        debug_dir=args.debug_dir,
        warmup_frames=args.warmup_frames,
    )
    frame = await module.capture_single_frame()
    print(f"Captured frame: {frame.image_path} at {frame.captured_at.isoformat()}")
    print(f"Debug snapshots: {args.debug_dir}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    args = build_parser().parse_args()
    asyncio.run(run_probe(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
