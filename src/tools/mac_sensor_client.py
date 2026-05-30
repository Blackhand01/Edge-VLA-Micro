from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import platform
import re
import tempfile
import urllib.error
import urllib.request
import uuid
import wave
from pathlib import Path
from typing import Optional

import numpy as np

from src.audio import AudioModule, DEFAULT_WHISPER_LANGUAGE
from src.perception.vision_module import VisionModule


DEFAULT_SERVER_URL = "http://192.168.55.1:8000/process_intent"
logger = logging.getLogger(__name__)
MLX_WHISPER_MODEL_ALIASES = {
    "tiny": "mlx-community/whisper-tiny",
    "base": "mlx-community/whisper-base",
    "small": "mlx-community/whisper-small",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Edge-VLA drone voice command console.")
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--text", default=None, help="Bypass ASR and send this text directly.")
    parser.add_argument("--interactive", action="store_true", help="Run the free-form voice command console.")
    parser.add_argument("--asr-backend", choices=("auto", "faster-whisper", "mlx-whisper"), default="auto")
    parser.add_argument("--whisper-model", default="tiny", help="faster-whisper model size, or mlx-whisper model id.")
    parser.add_argument("--whisper-language", default=DEFAULT_WHISPER_LANGUAGE)
    parser.add_argument("--audio-device", default=None)
    parser.add_argument("--list-audio-devices", action="store_true", help="Print sounddevice input devices and exit.")
    parser.add_argument("--audio-mode", choices=("fixed", "vad"), default="fixed", help="Audio capture mode for faster-whisper.")
    parser.add_argument("--save-audio-path", default=None, help="Optional WAV path for debugging captured microphone audio.")
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--record-seconds", type=float, default=4.0, help="Fixed recording duration for mlx-whisper backend.")
    parser.add_argument("--max-record", type=float, default=5.0, help="Maximum speech segment for faster-whisper backend.")
    parser.add_argument("--silence-threshold", type=float, default=0.006)
    parser.add_argument("--trailing-silence", type=float, default=0.80)
    parser.add_argument("--camera-index", default="auto", help="OpenCV camera index, or 'auto' to use the first available non-OBS camera.")
    parser.add_argument("--frame-path", default="tmp/mac_sensor_frame.jpg")
    parser.add_argument("--detection-debug-path", default="tmp/last_detection_debug.jpg")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--image-mode", choices=("auto", "always", "never"), default="auto")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--no-image", action="store_true", help="Send text only.")
    parser.add_argument("--show-json", action="store_true", help="Print full server JSON in interactive mode.")
    return parser.parse_args()


async def run(args: argparse.Namespace) -> int:
    if args.list_audio_devices:
        list_audio_devices()
        return 0
    if args.interactive:
        return await run_interactive_console(args)

    text = args.text or await transcribe_from_microphone(args)
    if not text:
        print("[ERROR] No speech text captured.")
        return 1

    response = await process_command(args, text)
    print(json.dumps(response, indent=2, sort_keys=True))
    return 0 if response.get("ok") else 2


async def run_interactive_console(args: argparse.Namespace) -> int:
    print("Edge-VLA Drone Voice Command Console")
    print(f"server={args.server_url}")
    print(f"camera_index={args.camera_index} image_mode={effective_image_mode(args)}")
    print("Press Enter to record a voice command. Type a command and press Enter to bypass ASR. Type q to quit.")
    print()

    while True:
        typed = input("[READY] command> ").strip()
        if typed.lower() in {"q", "quit", "exit"}:
            return 0

        text = typed or await transcribe_from_microphone(args)
        if not text:
            print("[RETRY] No speech text captured.")
            continue

        try:
            response = await process_command(args, text)
        except Exception as exc:  # noqa: BLE001 - console should stay alive for the next operator command.
            print(f"[ERROR] {type(exc).__name__}: {exc}")
            continue
        print_response_summary(response)
        if args.show_json:
            print(json.dumps(response, indent=2, sort_keys=True))


async def process_command(args: argparse.Namespace, text: str) -> dict:
    image_path = await maybe_capture_image(args, text)
    payload = build_payload(text, image_path)
    print(f"[INFO] text={text!r}")
    if image_path is not None:
        print(f"[INFO] frame={image_path}")
    response = post_json(args.server_url, payload, timeout=args.timeout)
    debug_path = save_detection_debug_image(args, response)
    if debug_path is not None:
        response = dict(response)
        response["detection_debug_image_base64"] = f"<saved to {debug_path}>"
    return response


def save_detection_debug_image(args: argparse.Namespace, response: dict) -> Optional[Path]:
    encoded = response.get("detection_debug_image_base64")
    if not encoded:
        return None
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        print(f"[WARN] Could not decode detection debug image: {exc}")
        return None

    path = Path(args.detection_debug_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(image_bytes)
    print(f"[INFO] detection_debug={path}")
    return path


async def maybe_capture_image(args: argparse.Namespace, text: str) -> Optional[Path]:
    mode = effective_image_mode(args)
    if mode == "never":
        return None
    if mode == "auto" and not is_visual_command(text):
        return None
    return await capture_webcam_frame(args)


def effective_image_mode(args: argparse.Namespace) -> str:
    return "never" if args.no_image else args.image_mode


def print_response_summary(response: dict) -> None:
    state = response.get("drone_state") or {}
    timings = response.get("timings_ms") or {}
    state_label = state.get("operational_state") or state.get("state") or "<unknown>"
    total_ms = timings.get("total")
    total_fragment = f" total_ms={total_ms:.1f}" if isinstance(total_ms, int | float) else ""
    if response.get("ok"):
        print(
            "[ACCEPTED] "
            f"action={response.get('action')} "
            f"command={response.get('command')} "
            f"state={state_label}"
            f"{total_fragment}"
        )
        return

    print(
        "[REJECTED] "
        f"reason={response.get('reason')} "
        f"details={response.get('details')} "
        f"state={state_label}"
        f"{total_fragment}"
    )


def is_visual_command(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ("object", "red", "target", "toward", "towards", "follow", "approach"))


async def transcribe_from_microphone(args: argparse.Namespace) -> Optional[str]:
    backend = select_asr_backend(args.asr_backend)
    print(f"[INFO] Recording speech with {backend}...")
    if backend == "mlx-whisper":
        return await transcribe_with_mlx_whisper(args)
    return await transcribe_with_faster_whisper(args)


def select_asr_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    if platform.system() == "Darwin":
        try:
            import mlx_whisper  # noqa: F401
        except ImportError:
            return "faster-whisper"
        return "mlx-whisper"
    return "faster-whisper"


async def transcribe_with_faster_whisper(args: argparse.Namespace) -> Optional[str]:
    if args.audio_mode == "fixed":
        return await transcribe_fixed_with_faster_whisper(args)

    audio = AudioModule(
        model_size=args.whisper_model,
        language=args.whisper_language,
        sample_rate=args.sample_rate,
        silence_threshold=args.silence_threshold,
        trailing_silence_s=args.trailing_silence,
        max_record_s=args.max_record,
        device=args.audio_device,
    )
    return clean_transcript(await audio.transcribe_audio())


async def transcribe_fixed_with_faster_whisper(args: argparse.Namespace) -> Optional[str]:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError("Install Mac audio deps: python -m pip install sounddevice") from exc

    print(f"[INFO] Listening for {args.record_seconds:.1f}s on audio device {args.audio_device or '<default>'}...", flush=True)
    sample_count = max(1, int(args.sample_rate * args.record_seconds))
    samples = await asyncio.to_thread(
        sd.rec,
        sample_count,
        samplerate=args.sample_rate,
        channels=1,
        dtype="float32",
        device=args.audio_device,
    )
    await asyncio.to_thread(sd.wait)
    mono = np.asarray(samples, dtype=np.float32).reshape(-1)
    rms = float(np.sqrt(np.mean(np.square(mono)) + 1e-12))
    if args.save_audio_path:
        write_wav(Path(args.save_audio_path), mono, args.sample_rate)
        print(f"[INFO] Saved captured audio to {args.save_audio_path}", flush=True)
    print(f"[INFO] Audio captured rms={rms:.5f}; loading faster-whisper...", flush=True)

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("Install Mac ASR deps: python -m pip install faster-whisper") from exc
    model = await asyncio.to_thread(WhisperModel, args.whisper_model, device="cpu", compute_type="int8")
    print("[INFO] Transcribing audio...", flush=True)
    segments, _ = await asyncio.to_thread(
        model.transcribe,
        mono,
        language=args.whisper_language,
        beam_size=1,
        best_of=1,
        temperature=0.0,
        vad_filter=True,
    )
    text = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
    return clean_transcript(text)


async def transcribe_with_mlx_whisper(args: argparse.Namespace) -> Optional[str]:
    try:
        import sounddevice as sd
        import mlx_whisper
    except ImportError as exc:
        raise RuntimeError("Install Mac ASR deps: python -m pip install sounddevice mlx-whisper") from exc

    sample_count = max(1, int(args.sample_rate * args.record_seconds))
    samples = await asyncio.to_thread(
        sd.rec,
        sample_count,
        samplerate=args.sample_rate,
        channels=1,
        dtype="float32",
        device=args.audio_device,
    )
    await asyncio.to_thread(sd.wait)
    mono = np.asarray(samples, dtype=np.float32).reshape(-1)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        wav_path = Path(handle.name)
    try:
        write_wav(wav_path, mono, args.sample_rate)
        result = await asyncio.to_thread(
            mlx_whisper.transcribe,
            str(wav_path),
            path_or_hf_repo=mlx_whisper_model_id(args.whisper_model),
            language=args.whisper_language,
        )
    finally:
        wav_path.unlink(missing_ok=True)
    text = str(result.get("text", "")).strip() if isinstance(result, dict) else ""
    return clean_transcript(text)


def clean_transcript(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return None
    if is_repetitive_hallucination(compact):
        logger.warning("Rejected likely ASR hallucination: %r", compact[:120])
        return None
    return compact


def list_audio_devices() -> None:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError("Install Mac audio deps: python -m pip install sounddevice") from exc
    print(sd.query_devices())


def is_repetitive_hallucination(text: str) -> bool:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if len(words) < 8:
        return False
    counts: dict[str, int] = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    most_common = max(counts.values())
    return most_common / len(words) >= 0.75


def mlx_whisper_model_id(model: str) -> str:
    return MLX_WHISPER_MODEL_ALIASES.get(model, model)


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


async def capture_webcam_frame(args: argparse.Namespace) -> Path:
    camera_index = resolve_camera_index(args.camera_index)
    vision = VisionModule(
        camera_index=camera_index,
        output_path=args.frame_path,
        debug_dir=None,
        target_size=(args.image_size, args.image_size),
    )
    frame = await vision.capture_single_frame()
    return Path(frame.image_path)


def resolve_camera_index(camera_index: str) -> int:
    if str(camera_index).lower() != "auto":
        return int(camera_index)
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for camera auto-detection.") from exc
    probe_order = (1, 2, 3, 4, 5, 0) if platform.system() == "Darwin" else tuple(range(6))
    for index in probe_order:
        capture = open_capture_for_probe(cv2, index)
        try:
            if capture.isOpened():
                ok, frame = capture.read()
                if ok and frame is not None:
                    print(f"[INFO] Auto-selected camera index {index}", flush=True)
                    return index
        finally:
            capture.release()
    raise RuntimeError("No usable camera found in indices 0..5.")


def open_capture_for_probe(cv2, index: int):  # noqa: ANN001
    if hasattr(cv2, "CAP_AVFOUNDATION"):
        capture = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if capture.isOpened():
            return capture
        capture.release()
    return cv2.VideoCapture(index)


def build_payload(text: str, image_path: Optional[Path]) -> dict:
    payload = {
        "request_id": str(uuid.uuid4()),
        "source": "mac_sensor_client",
        "text": text,
        "metadata": {"sensor_node": "mac"},
    }
    if image_path is not None:
        payload["image_mime_type"] = "image/jpeg"
        payload["image_base64"] = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return payload


def post_json(url: str, payload: dict, *, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Jetson server returned HTTP {exc.code}: {detail}") from exc
    return json.loads(raw)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    return asyncio.run(run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
