from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from src.perception.vision_module import VisionError, VisionModule


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture one OpenCV camera frame for VLA diagnostics.")
    parser.add_argument("--camera-index", default=0, type=int, help="OpenCV camera index.")
    parser.add_argument("--frame-path", default="tmp/probe_frame.jpg", help="Output path for latest probe frame.")
    parser.add_argument("--debug-dir", default="tmp/probe_frames", help="Directory for timestamped probe frames.")
    parser.add_argument("--warmup-frames", default=5, type=int, help="Frames to discard before saving.")
    parser.add_argument("--scan", action="store_true", help="Probe camera indices from 0 to --max-index.")
    parser.add_argument("--max-index", default=5, type=int, help="Maximum camera index to scan.")
    return parser


async def run_probe(args: argparse.Namespace) -> None:
    if args.scan:
        await scan_cameras(args)
        return

    module = VisionModule(
        camera_index=args.camera_index,
        output_path=args.frame_path,
        debug_dir=args.debug_dir,
        warmup_frames=args.warmup_frames,
    )
    frame = await module.capture_single_frame()
    print(f"Captured frame: {frame.image_path} at {frame.captured_at.isoformat()}")
    print(f"Debug snapshots: {args.debug_dir}")


async def scan_cameras(args: argparse.Namespace) -> None:
    output_dir = Path(args.debug_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Scanning camera indices 0..{args.max_index}")
    for index in range(args.max_index + 1):
        output_path = output_dir / f"camera_index_{index}.jpg"
        module = VisionModule(
            camera_index=index,
            output_path=output_path,
            debug_dir=None,
            warmup_frames=args.warmup_frames,
        )
        try:
            frame = await module.capture_single_frame()
        except VisionError as exc:
            print(f"[{index}] unavailable: {exc}")
            continue
        print(f"[{index}] captured: {frame.image_path}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    args = build_parser().parse_args()
    asyncio.run(run_probe(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
