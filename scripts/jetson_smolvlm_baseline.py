from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForVision2Seq, AutoProcessor


DEFAULT_MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
DEFAULT_IMAGE_URL = "https://raw.githubusercontent.com/PX4/PX4-Autopilot/main/boards/px4/fmu-v6x/fmu-v6x.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jetson SmolVLM single-device CUDA baseline.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--image-path", default="tmp/jetson_smolvlm_baseline.jpg")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--prompt",
        default="Describe this image briefly and say whether it shows a drone flight controller.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("Jetson SmolVLM baseline")
    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required on Jetson for this benchmark.")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"cuda: {torch.version.cuda}")

    image = load_image(args.image_path, args.image_size)
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        size={"longest_edge": args.image_size},
    )

    try:
        model = AutoModelForVision2Seq.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16,
            _attn_implementation="eager",
            low_cpu_mem_usage=True,
        ).to("cuda")
    except Exception as exc:  # noqa: BLE001
        print("\nMODEL_LOAD_FAILED")
        print(type(exc).__name__)
        print(str(exc))
        print("\nThis is not an offload path. Keep the model single-device and inspect the error above.")
        return 2

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = inputs.to("cuda")

    print("\nStarting inference")
    disable_distributed_generation_checks()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    except Exception as exc:  # noqa: BLE001
        print("\nINFERENCE_FAILED")
        print(type(exc).__name__)
        print(str(exc))
        return 3
    torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - started

    input_len = inputs["input_ids"].shape[1]
    generated_ids = generated_ids[:, input_len:]
    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    token_count = generated_ids.shape[1]
    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)

    print("\n" + "=" * 40)
    print("RESULT")
    print(output_text)
    print("=" * 40)
    print(f"elapsed_s: {elapsed_s:.2f}")
    print(f"tokens: {token_count}")
    print(f"tps: {token_count / elapsed_s if elapsed_s > 0 else 0.0:.2f}")
    print(f"peak_cuda_allocated_gb: {peak_gb:.2f}")
    print("=" * 40)
    return 0


def load_image(image_path: str, image_size: int) -> Image.Image:
    path = Path(image_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        try:
            urllib.request.urlretrieve(DEFAULT_IMAGE_URL, path)
        except Exception as exc:  # noqa: BLE001
            print(f"Image download failed ({type(exc).__name__}: {exc}); using synthetic test image.")
            create_synthetic_image(path, image_size)
    image = Image.open(path).convert("RGB")
    image.thumbnail((image_size, image_size))
    canvas = Image.new("RGB", (image_size, image_size), (0, 0, 0))
    x = (image_size - image.width) // 2
    y = (image_size - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def create_synthetic_image(path: Path, image_size: int) -> None:
    image = Image.new("RGB", (image_size, image_size), (22, 28, 36))
    draw = ImageDraw.Draw(image)
    margin = max(16, image_size // 8)
    draw.rectangle(
        (margin, image_size // 2 - 24, image_size - margin, image_size // 2 + 24),
        outline=(80, 180, 255),
        width=3,
    )
    draw.rectangle(
        (image_size // 2 - 36, margin, image_size // 2 + 36, image_size - margin),
        outline=(255, 180, 70),
        width=3,
    )
    draw.text((margin, image_size - margin - 20), "flight controller", fill=(230, 230, 230))
    image.save(path)


def disable_distributed_generation_checks() -> None:
    try:
        import transformers.generation.utils as generation_utils

        generation_utils.is_deepspeed_zero3_enabled = lambda: False
        generation_utils.is_fsdp_managed_module = lambda _module: False
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: could not patch distributed generation checks: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
