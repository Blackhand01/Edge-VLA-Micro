from __future__ import annotations

import argparse
import asyncio
import logging
import os

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

from src.safety.command_validator import CommandValidator, DroneOperationalState, DroneStateSnapshot
from src.action import DEFAULT_CONNECTION, DroneController
from src.audio import AudioModule, DEFAULT_WHISPER_LANGUAGE
from src.core.agent_loop import AgentLoop
from src.monitoring import BlackboxLogger, PerformanceLogger
from src.perception import CognitionEngine, DEFAULT_MODEL_ID
from src.perception.cognition_service import AsyncCognitionService, ProcessCognitionService, ThreadedCognitionService
from src.perception.vision_module import VisionModule


logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Main async agent loop: audio -> cognition -> safety -> drone action.")
    parser.add_argument("--connection", default=DEFAULT_CONNECTION, help=f"MAVSDK connection string. Default: {DEFAULT_CONNECTION}")
    parser.add_argument("--whisper-model", default="tiny", help="faster-whisper model size.")
    parser.add_argument("--whisper-language", default=DEFAULT_WHISPER_LANGUAGE, help="Whisper language code.")
    parser.add_argument("--audio-device", default=None, help="Optional sounddevice input device.")
    parser.add_argument("--camera-index", default=0, type=int, help="OpenCV camera index.")
    parser.add_argument("--frame-path", default="tmp/frame.jpg", help="Path for the one-shot camera frame.")
    parser.add_argument("--vision-debug-dir", default="tmp/frames", help="Directory for timestamped camera snapshots. Use 'none' to disable.")
    parser.add_argument("--sample-rate", default=16_000, type=int, help="Microphone sample rate.")
    parser.add_argument("--silence-threshold", default=0.006, type=float, help="RMS threshold used to start voice capture.")
    parser.add_argument("--trailing-silence", default=0.80, type=float, help="Seconds of silence used to end a voice command.")
    parser.add_argument("--max-record", default=5.0, type=float, help="Maximum seconds to record a single voice command.")
    parser.add_argument("--cognition-model", default=DEFAULT_MODEL_ID, help="MLX model id for CognitionEngine.")
    parser.add_argument("--vlm-backend", choices=("auto", "mlx", "dummy", "tensorrt"), default="auto", help="VLM inference runtime.")
    parser.add_argument("--cognition-backend", choices=("process", "thread"), default="process", help="VLM isolation backend.")
    parser.add_argument("--cognition-timeout", default=75.0, type=float, help="Seconds before cognition is failed closed.")
    parser.add_argument("--max-cognition-failures", default=3, type=int, help="Consecutive cognition failures before emergency hold.")
    parser.add_argument("--blackbox-dir", default="logs/sessions", help="Directory for blackbox telemetry bundles.")
    parser.add_argument("--performance-log", default="logs/performance.csv", help="CSV path for per-inference latency metrics.")
    return parser


async def run_agent(args: argparse.Namespace) -> None:
    cognition_engine = build_cognition_engine(args)
    cognition_service = build_cognition_service(args, cognition_engine)
    agent = AgentLoop(
        drone_controller=DroneController(connection=args.connection),
        cognition_engine=cognition_engine,
        audio_module=AudioModule(
            model_size=args.whisper_model,
            language=args.whisper_language,
            sample_rate=args.sample_rate,
            silence_threshold=args.silence_threshold,
            trailing_silence_s=args.trailing_silence,
            max_record_s=args.max_record,
            device=args.audio_device,
        ),
        vision_module=VisionModule(
            camera_index=args.camera_index,
            output_path=args.frame_path,
            debug_dir=None if str(args.vision_debug_dir).lower() == "none" else args.vision_debug_dir,
        ),
        blackbox_logger=BlackboxLogger(sessions_dir=args.blackbox_dir),
        performance_logger=PerformanceLogger(path=args.performance_log),
        cognition_service=cognition_service,
        max_cognition_failures=args.max_cognition_failures,
    )
    try:
        await agent.warmup_cognition()
        await agent.run()
    except Exception:
        await agent.shutdown()
        raise


def build_cognition_engine(args: argparse.Namespace) -> CognitionEngine:
    validator = CommandValidator(
        DroneStateSnapshot(
            state=DroneOperationalState.GROUNDED,
            connected=False,
            battery_remaining=None,
        )
    )
    return CognitionEngine(
        validator=validator,
        model_id=args.cognition_model,
        temperature=0.0,
        vlm_backend=args.vlm_backend,
    )


def build_cognition_service(args: argparse.Namespace, cognition_engine: CognitionEngine) -> AsyncCognitionService:
    if args.cognition_backend == "process":
        return ProcessCognitionService(
            model_id=args.cognition_model,
            temperature=0.0,
            timeout_s=args.cognition_timeout,
            vlm_backend=args.vlm_backend,
        )
    return ThreadedCognitionService(cognition_engine)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    try:
        asyncio.run(run_agent(build_parser().parse_args()))
    except KeyboardInterrupt:
        logger.info("Manual interruption received")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
