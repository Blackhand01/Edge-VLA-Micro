from __future__ import annotations

import argparse
import importlib.util
import time
import urllib.request
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import (
    AutoConfig,
    AutoProcessor,
    AwqConfig,
    Qwen2VLForConditionalGeneration,
)


DEFAULT_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct-AWQ"
DEFAULT_IMAGE_URL = "https://raw.githubusercontent.com/PX4/PX4-Autopilot/main/boards/px4/fmu-v6x/fmu-v6x.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jetson Qwen2-VL AWQ single-device CUDA baseline.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--image-path", default="tmp/jetson_baseline.jpg")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--prompt",
        default="Describe this image briefly and say whether it shows a drone flight controller.",
    )
    parser.add_argument(
        "--awq-config",
        choices=("model", "explicit"),
        default="explicit",
        help=(
            "AWQ quantization source. 'explicit' rebuilds an AwqConfig from the repository "
            "config and excludes Jetson-incompatible heads; 'model' uses the repository config unchanged."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("Jetson VLM baseline")
    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required on Jetson for this benchmark.")
    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)}")
        print(f"cuda: {torch.version.cuda}")

    print_model_quantization(args.model_id)
    image = load_image(args.image_path, args.image_size)

    try:
        processor = load_processor(args)
        model = load_model(args)
    except Exception as exc:  # noqa: BLE001 - benchmark should report model-load failures clearly.
        report_model_load_failure(exc)
        return 2

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to("cuda")

    print("\nStarting inference")
    disable_distributed_generation_checks()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    started = time.perf_counter()
    try:
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    except Exception as exc:  # noqa: BLE001 - benchmark should report inference failures clearly.
        report_inference_failure(exc)
        return 3
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - started

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    token_count = len(generated_ids_trimmed[0])
    peak_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0

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


def load_processor(args: argparse.Namespace):
    pixel_budget = args.image_size * args.image_size
    print(f"processor_pixel_budget: {pixel_budget}")
    return AutoProcessor.from_pretrained(
        args.model_id,
        use_fast=False,
        size={"shortest_edge": pixel_budget, "longest_edge": pixel_budget},
        min_pixels=pixel_budget,
        max_pixels=pixel_budget,
    )


def load_model(args: argparse.Namespace):
    torch.cuda.empty_cache()
    warn_if_awq_backend_missing(args.model_id)
    model_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": "cuda",
        "low_cpu_mem_usage": True,
    }
    if args.awq_config == "explicit":
        model_kwargs["quantization_config"] = build_awq_config(args.model_id)

    return Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_id,
        **model_kwargs,
    )


def report_model_load_failure(exc: Exception) -> None:
    print("\nMODEL_LOAD_FAILED")
    print(type(exc).__name__)
    print(str(exc))
    message = str(exc).lower()
    if "numpy is not available" in message or "_array_api" in message:
        print("\nThis is a Python environment issue, not a Jetson memory result.")
        print("Next: reinstall numpy<2 inside .venv-jetson, then rerun the same AWQ baseline.")
        return
    if "size must contain" in message and "shortest_edge" in message:
        print("\nThis is a processor configuration issue, not a Jetson memory result.")
        print("Next: pass Qwen2-VL size, min_pixels, and max_pixels explicitly, then rerun the same AWQ baseline.")
        return
    if isinstance(exc, ModuleNotFoundError) or "no module named" in message:
        print("\nThis is a missing dependency issue, not a Jetson memory result.")
        print("Next: install the missing package inside .venv-jetson without replacing the NVIDIA PyTorch wheel.")
        return
    print("\nThis is useful evidence: this model may not fit the current Jetson physical UMA budget.")
    print(
        "Next: use a smaller quantized model, TensorRT/TensorRT-LLM, or another 4-bit runtime. "
        "Do not use CPU/GPU offload on Jetson UMA."
    )


def print_model_quantization(model_id: str) -> None:
    config = AutoConfig.from_pretrained(model_id)
    quantization_config = getattr(config, "quantization_config", None)
    if quantization_config:
        print(f"quantization_config: {quantization_config}")
    else:
        print("quantization_config: <none>")


def build_awq_config(model_id: str) -> AwqConfig:
    config = AutoConfig.from_pretrained(model_id)
    quantization_config = getattr(config, "quantization_config", None)
    if not isinstance(quantization_config, dict):
        raise ValueError(f"{model_id} does not expose an AWQ quantization_config dict.")
    if quantization_config.get("quant_method") != "awq":
        raise ValueError(f"{model_id} quant_method is not awq: {quantization_config!r}")
    modules_to_not_convert = list(quantization_config.get("modules_to_not_convert") or [])
    for module_name in ("visual", "lm_head"):
        if module_name not in modules_to_not_convert:
            modules_to_not_convert.append(module_name)
    print(f"awq_modules_to_not_convert: {modules_to_not_convert}")
    return AwqConfig(
        bits=quantization_config.get("bits", 4),
        group_size=quantization_config.get("group_size", 128),
        zero_point=quantization_config.get("zero_point", True),
        version=quantization_config.get("version", "gemm"),
        modules_to_not_convert=modules_to_not_convert,
    )


def report_inference_failure(exc: Exception) -> None:
    print("\nINFERENCE_FAILED")
    print(type(exc).__name__)
    print(str(exc))
    message = str(exc).lower()
    if "rshift_cuda" in message and "half" in message:
        print("\nThis is an AWQ kernel/path compatibility issue, not a Jetson memory result.")
        print("Next: rerun with explicit AWQ config that leaves lm_head unquantized, or use a TensorRT/LLM runtime.")
        return
    if "no module named 'awq_ext'" in message or "awq_ext" in message:
        print("\nThis is an AutoAWQ native-extension issue, not a Jetson memory result.")
        print("Next: avoid installing generic CUDA extensions blindly; prefer a Jetson-compatible runtime path.")
        return
    print("\nThis is an inference-time failure. Keep the single-device CUDA setup and inspect the runtime error above.")


def warn_if_awq_backend_missing(model_id: str) -> None:
    config = AutoConfig.from_pretrained(model_id)
    quantization_config = getattr(config, "quantization_config", None)
    if not isinstance(quantization_config, dict):
        return
    if quantization_config.get("quant_method") != "awq":
        return
    if importlib.util.find_spec("awq") is None:
        print(
            "Warning: AWQ backend module 'awq' was not found. "
            "Install autoawq in the Jetson venv before running this model."
        )


def load_image(image_path: str, image_size: int) -> Image.Image:
    path = Path(image_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        try:
            urllib.request.urlretrieve(DEFAULT_IMAGE_URL, path)
        except Exception as exc:  # noqa: BLE001 - benchmark should not depend on a remote image URL.
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
    draw.rectangle((28, 92, 196, 132), outline=(80, 180, 255), width=3)
    draw.rectangle((76, 68, 148, 156), outline=(255, 180, 70), width=3)
    draw.line((32, 32, 192, 192), fill=(160, 220, 120), width=2)
    draw.line((192, 32, 32, 192), fill=(160, 220, 120), width=2)
    draw.text((48, 176), "flight controller", fill=(230, 230, 230))
    image.save(path)


def disable_distributed_generation_checks() -> None:
    # NVIDIA Jetson PyTorch wheels can omit torch.distributed c10d/FSDP pieces.
    # This benchmark is strictly single-device, so distributed generation checks
    # should be false instead of importing unavailable distributed modules.
    try:
        import transformers.generation.utils as generation_utils

        generation_utils.is_deepspeed_zero3_enabled = lambda: False
        generation_utils.is_fsdp_managed_module = lambda _module: False
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: could not patch distributed generation checks: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
