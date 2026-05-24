from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import queue as queue_module
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.perception.prompts import DEFAULT_BASE_MODEL_ID, DEFAULT_QUANTIZED_MODEL_ID
from src.perception.vision_module import capture_and_downsample
from src.perception.vlm_runtime import VLMRuntime


logger = logging.getLogger("profiling")
BENCHMARK_PROMPT = (
    "Inspect the image and answer with compact JSON only: "
    '{"command":"hold","target_found":true,"reasoning":"benchmark"}'
)


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    image_size: str
    model_id: str
    ttft_ms: float = 0.0
    decode_time_ms: float = 0.0
    generated_tokens: int = 0
    tps: float = 0.0
    elapsed_ms: float = 0.0
    error: Optional[str] = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A/B benchmark for Edge-VLA image downsampling and MLX-VLM 4-bit loading.")
    parser.add_argument("--baseline-model", default=DEFAULT_BASE_MODEL_ID, help="Non-quantized MLX-VLM model id.")
    parser.add_argument("--optimized-model", default=DEFAULT_QUANTIZED_MODEL_ID, help="4-bit MLX-VLM model id.")
    parser.add_argument("--max-tokens", default=48, type=int, help="Maximum generated tokens per benchmark run.")
    parser.add_argument("--target-size", default=224, type=int, help="Optimized square image size.")
    parser.add_argument("--case-timeout", default=300.0, type=float, help="Timeout per benchmark case in seconds.")
    parser.add_argument("--minimum-free-gb", default=8.0, type=float, help="Free disk required before downloading baseline models.")
    parser.add_argument("--no-baseline-fallback", action="store_true", help="Do not retry baseline with the 4-bit model.")
    return parser


def dummy_1080p_frame() -> np.ndarray:
    height, width = 1080, 1920
    x_axis = np.linspace(0, 255, width, dtype=np.uint8)
    y_axis = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = x_axis
    frame[:, :, 1] = y_axis
    frame[height // 3 : 2 * height // 3, width // 3 : 2 * width // 3, 2] = 220
    return frame


def write_image(path: Path, frame: np.ndarray) -> None:
    import cv2  # pylint: disable=import-outside-toplevel

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"failed to write benchmark image: {path}")


def run_case(
    name: str,
    model_id: str,
    image_path: Path,
    max_tokens: int,
    *,
    prefer_quantized: bool,
    timeout_s: float,
    minimum_free_gb: float,
) -> BenchmarkResult:
    logger.info("[PROFILING] Starting %s | model=%s | image=%s", name, model_id, image_path)
    disk_error = model_download_disk_error(model_id, prefer_quantized, minimum_free_gb)
    if disk_error is not None:
        return BenchmarkResult(name=name, image_size=image_dimensions(image_path), model_id=model_id, error=disk_error)
    context = mp.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=run_case_worker,
        args=(result_queue, name, model_id, str(image_path), max_tokens, prefer_quantized),
    )
    process.start()
    process.join(timeout_s)
    if process.is_alive():
        process.terminate()
        process.join(10.0)
        return BenchmarkResult(
            name=name,
            image_size=image_dimensions(image_path),
            model_id=model_id,
            error=f"timeout after {timeout_s:.0f}s during MLX-VLM import/model load/inference",
        )
    try:
        result = result_queue.get_nowait()
    except queue_module.Empty:
        return BenchmarkResult(
            name=name,
            image_size=image_dimensions(image_path),
            model_id=model_id,
            error=f"worker exited without result; exitcode={process.exitcode}",
        )
    if result.error:
        logger.error("[PROFILING] %s failed | %s", name, result.error)
    return result


def run_case_worker(result_queue, name: str, model_id: str, image_path_value: str, max_tokens: int, prefer_quantized: bool) -> None:
    image_path = Path(image_path_value)
    runtime = VLMRuntime(
        model_id=model_id,
        max_tokens=max_tokens,
        temperature=0.0,
        prefer_quantized=prefer_quantized,
        quantized_model_id=model_id,
    )
    started = time.perf_counter()
    try:
        _, profile = runtime.generate_profiled(BENCHMARK_PROMPT, image_path=str(image_path))
    except Exception as exc:
        result_queue.put(BenchmarkResult(name=name, image_size=image_dimensions(image_path), model_id=model_id, error=str(exc)))
        return
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    logger.info(
        "[PROFILING] %s complete | ttft=%.1fms decode=%.1fms tokens=%s tps=%.2f elapsed=%.1fms",
        name,
        profile.ttft_ms,
        profile.decode_time_ms,
        profile.generated_tokens,
        profile.tps,
        elapsed_ms,
    )
    result_queue.put(
        BenchmarkResult(
            name=name,
            image_size=image_dimensions(image_path),
            model_id=runtime.model_id,
            ttft_ms=profile.ttft_ms,
            decode_time_ms=profile.decode_time_ms,
            generated_tokens=profile.generated_tokens,
            tps=profile.tps,
            elapsed_ms=elapsed_ms,
        )
    )


def image_dimensions(path: Path) -> str:
    from PIL import Image

    with Image.open(path) as image:
        return f"{image.width}x{image.height}"


def model_download_disk_error(model_id: str, prefer_quantized: bool, minimum_free_gb: float) -> Optional[str]:
    if prefer_quantized or huggingface_model_cache_exists(model_id):
        return None
    free_gb = shutil.disk_usage(Path.home()).free / (1024**3)
    if free_gb >= minimum_free_gb:
        return None
    return f"insufficient disk for uncached baseline download: {free_gb:.1f}GB free, need {minimum_free_gb:.1f}GB"


def huggingface_model_cache_exists(model_id: str) -> bool:
    cache_name = "models--" + model_id.replace("/", "--")
    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / cache_name
    if any(cache_root.rglob("*.incomplete")):
        return False
    snapshots_root = cache_root / "snapshots"
    if not snapshots_root.exists():
        return False
    for snapshot in snapshots_root.iterdir():
        has_config = (snapshot / "config.json").is_file()
        has_weights = any(path.is_file() and path.suffix in {".safetensors", ".npz"} for path in snapshot.rglob("*"))
        if has_config and has_weights:
            return True
    return False


def improvement_pct(baseline: float, optimized: float, *, higher_is_better: bool = False) -> float:
    if baseline <= 0.0:
        return 0.0
    if higher_is_better:
        return (optimized - baseline) / baseline * 100.0
    return (baseline - optimized) / baseline * 100.0


def print_summary(baseline: BenchmarkResult, optimized: BenchmarkResult) -> None:
    headers = ("Case", "Image", "Model", "TTFT ms", "Decode ms", "Tokens", "TPS", "Delta TTFT", "Delta TPS")
    rows = [
        result_row(baseline, "", ""),
        result_row(
            optimized,
            f"{improvement_pct(baseline.ttft_ms, optimized.ttft_ms):+.1f}%",
            f"{improvement_pct(baseline.tps, optimized.tps, higher_is_better=True):+.1f}%",
        ),
    ]
    widths = [max(len(str(item)) for item in column) for column in zip(headers, *rows)]
    print("\n[PROFILING] Comparative VLM Benchmark\n")
    print(format_row(headers, widths))
    print(format_row(tuple("-" * width for width in widths), widths))
    for row in rows:
        print(format_row(row, widths))


def result_row(result: BenchmarkResult, ttft_delta: str, tps_delta: str) -> tuple[str, ...]:
    if result.error:
        return (result.name, result.image_size, result.model_id, "ERROR", "-", "-", "-", result.error[:36], "-")
    return (
        result.name,
        result.image_size,
        result.model_id,
        f"{result.ttft_ms:.1f}",
        f"{result.decode_time_ms:.1f}",
        str(result.generated_tokens),
        f"{result.tps:.2f}",
        ttft_delta,
        tps_delta,
    )


def format_row(row: tuple[str, ...], widths: list[int]) -> str:
    return " | ".join(str(value).ljust(width) for value, width in zip(row, widths))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args()
    with tempfile.TemporaryDirectory(prefix="edge-vla-benchmark-") as tmpdir:
        raw_path = Path(tmpdir) / "baseline_1080p.jpg"
        optimized_path = Path(tmpdir) / "optimized_224.jpg"
        raw_frame = dummy_1080p_frame()
        write_image(raw_path, raw_frame)
        optimized_frame = capture_and_downsample(raw_frame, target_size=(args.target_size, args.target_size))
        write_image(optimized_path, optimized_frame)
        baseline = run_case(
            "A Baseline",
            args.baseline_model,
            raw_path,
            args.max_tokens,
            prefer_quantized=False,
            timeout_s=args.case_timeout,
            minimum_free_gb=args.minimum_free_gb,
        )
        if baseline.error and not args.no_baseline_fallback:
            logger.warning("[PROFILING] Baseline unavailable; retrying full-resolution baseline with the 4-bit model.")
            baseline = run_case(
                "A Baseline 4-bit Full-Res",
                args.optimized_model,
                raw_path,
                args.max_tokens,
                prefer_quantized=True,
                timeout_s=args.case_timeout,
                minimum_free_gb=args.minimum_free_gb,
            )
        optimized = run_case(
            "B Optimized",
            args.optimized_model,
            optimized_path,
            args.max_tokens,
            prefer_quantized=True,
            timeout_s=args.case_timeout,
            minimum_free_gb=args.minimum_free_gb,
        )
    print_summary(baseline, optimized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
