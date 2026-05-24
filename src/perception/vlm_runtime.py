from __future__ import annotations

import logging
import time
from typing import Any, Optional

from src.perception.models import GeneratorFn, VLMProfile, estimate_generated_token_count
from src.perception.prompts import DEFAULT_MODEL_ID, DEFAULT_QUANTIZED_MODEL_ID


logger = logging.getLogger(__name__)


class VLMRuntime:
    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        max_tokens: int = 128,
        temperature: float = 0.0,
        external_generator: Optional[GeneratorFn] = None,
        prefer_quantized: bool = True,
        quantized_model_id: str = DEFAULT_QUANTIZED_MODEL_ID,
        quantize_on_load: bool = False,
    ) -> None:
        self.requested_model_id = model_id
        self.model_id = resolve_model_id(model_id, prefer_quantized, quantized_model_id)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.external_generator = external_generator
        self.prefer_quantized = prefer_quantized
        self.quantized_model_id = quantized_model_id
        self.quantize_on_load = quantize_on_load
        self.model = None
        self.processor = None
        self.batch_generate = None
        self.stream_generate = None

    def warmup(self) -> None:
        logger.info("Pre-warming VLM models and compiling Metal shaders...")
        if self.external_generator is not None:
            self.external_generator("warmup", None)
            logger.info("VLM Warm-up complete.")
            return

        self.load_local_model()
        assert self.batch_generate is not None
        self.batch_generate(
            self.model,
            self.processor,
            prompt="warmup",
            image=None,
            max_tokens=1,
            temperature=self.temperature,
            verbose=False,
        )
        logger.info("VLM Warm-up complete.")

    def generate_profiled(self, raw_prompt: str, *, image_path: Optional[str]) -> tuple[str, VLMProfile]:
        if self.external_generator is not None:
            return self.generate_external_profiled(raw_prompt, image_path=image_path)

        self.load_local_model()
        if self.stream_generate is not None:
            try:
                return self.stream_generate_profiled(raw_prompt, image_path=image_path)
            except Exception as exc:
                logger.warning("VLM streaming unavailable; falling back to batch generate: %s", exc)

        return self.batch_generate_profiled(
            raw_prompt,
            image_path=image_path,
            fallback_reason="stream_generate unavailable or failed",
        )

    def generate_external_profiled(self, raw_prompt: str, *, image_path: Optional[str]) -> tuple[str, VLMProfile]:
        started = time.perf_counter()
        assert self.external_generator is not None
        raw_response = self.external_generator(raw_prompt, image_path)
        elapsed_ms = elapsed_ms_since(started)
        generated_tokens = estimate_generated_token_count(raw_response)
        return raw_response, VLMProfile(
            ttft_ms=elapsed_ms,
            generated_tokens=generated_tokens,
            tps=tokens_per_second(generated_tokens, elapsed_ms),
            fallback_reason="external generator is non-streaming",
        )

    def batch_generate_profiled(
        self,
        raw_prompt: str,
        *,
        image_path: Optional[str],
        fallback_reason: str,
    ) -> tuple[str, VLMProfile]:
        started = time.perf_counter()
        result = self.batch_generate(
            self.model,
            self.processor,
            prompt=raw_prompt,
            image=image_path,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            verbose=False,
        )
        elapsed_ms = elapsed_ms_since(started)
        raw_response = result.text if hasattr(result, "text") else str(result)
        generated_tokens = result_token_count(result, raw_response)
        return raw_response, VLMProfile(
            ttft_ms=elapsed_ms,
            generated_tokens=generated_tokens,
            tps=tokens_per_second(generated_tokens, elapsed_ms),
            fallback_reason=fallback_reason,
        )

    def stream_generate_profiled(self, raw_prompt: str, *, image_path: Optional[str]) -> tuple[str, VLMProfile]:
        started = time.perf_counter()
        first_token_at: Optional[float] = None
        ended_at = started
        parts: list[str] = []
        generated_tokens = 0
        final_token_count: Optional[int] = None
        prompt_eval_ms: Optional[float] = None

        for response in self.stream_generate(
            self.model,
            self.processor,
            prompt=raw_prompt,
            image=image_path,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            verbose=False,
        ):
            now = time.perf_counter()
            first_token_at = first_token_at or now
            ended_at = now
            text = response_text(response)
            parts.append(text)
            generated_tokens += response_token_increment(response, text)
            final_token_count = response_final_token_count(response, final_token_count)
            prompt_eval_ms = response_prompt_eval_ms(response, prompt_eval_ms)

        return build_stream_profile(started, first_token_at, ended_at, "".join(parts), generated_tokens, final_token_count, prompt_eval_ms)

    def load_local_model(self) -> None:
        if self.model is not None and self.processor is not None:
            return
        try:
            from mlx_vlm import generate, load
        except Exception as exc:
            raise RuntimeError(
                "Unable to import the MLX-VLM runtime. Reinstall the pinned dependencies with "
                "`python -m pip install -r requirements.txt`. "
                f"Original import error: {type(exc).__name__}: {exc}"
            ) from exc
        try:
            from mlx_vlm import stream_generate
        except ImportError:
            try:
                from mlx_vlm.generate import stream_generate
            except ImportError:
                stream_generate = None

        logger.info("Loading local MLX-VLM model: %s", self.model_id)
        self.model, self.processor = load(self.model_id)
        if self.quantize_on_load:
            self.apply_on_the_fly_quantization()
        self.force_float16_for_supported_layers()
        self.batch_generate = generate
        self.stream_generate = stream_generate

    def apply_on_the_fly_quantization(self) -> None:
        try:
            import mlx.nn as nn  # pylint: disable=import-outside-toplevel
        except ImportError:
            logger.warning("MLX nn module unavailable; skipping on-the-fly 4-bit quantization.")
            return
        quantize = getattr(nn, "quantize", None)
        if quantize is None:
            logger.warning("MLX nn.quantize unavailable; skipping on-the-fly 4-bit quantization.")
            return
        try:
            self.model = quantize(self.model, bits=4)
            logger.info("Applied on-the-fly MLX 4-bit quantization.")
        except TypeError:
            quantize(self.model, bits=4)
            logger.info("Applied in-place MLX 4-bit quantization.")

    def force_float16_for_supported_layers(self) -> None:
        try:
            import mlx.core as mx  # pylint: disable=import-outside-toplevel
        except ImportError:
            return
        dtype = getattr(mx, "float16", None)
        if dtype is None:
            return
        for method_name in ("to", "set_dtype", "astype"):
            method = getattr(self.model, method_name, None)
            if method is None:
                continue
            try:
                converted = method(dtype)
                if converted is not None:
                    self.model = converted
                logger.info("Configured supported non-quantized VLM layers for float16.")
                return
            except (TypeError, AttributeError, ValueError):
                continue


def resolve_model_id(model_id: str, prefer_quantized: bool, quantized_model_id: str) -> str:
    if not prefer_quantized:
        return model_id
    lowered = model_id.lower()
    if "4bit" in lowered or "4-bit" in lowered or "int4" in lowered:
        return model_id
    logger.info("Using 4-bit MLX-VLM model variant: %s -> %s", model_id, quantized_model_id)
    return quantized_model_id


def build_stream_profile(started, first_token_at, ended_at, raw_response, generated_tokens, final_count, prompt_eval_ms):
    if final_count is not None:
        generated_tokens = max(generated_tokens, final_count)
    if first_token_at is None:
        return raw_response, VLMProfile(
            prompt_eval_ms=prompt_eval_ms,
            ttft_ms=elapsed_ms_since(started),
            generated_tokens=generated_tokens,
            used_streaming=True,
        )
    ttft_ms = (first_token_at - started) * 1000.0
    decode_time_ms = max(0.0, (ended_at - first_token_at) * 1000.0)
    return raw_response, VLMProfile(
        prompt_eval_ms=prompt_eval_ms,
        ttft_ms=ttft_ms,
        decode_time_ms=decode_time_ms,
        generated_tokens=generated_tokens,
        tps=tokens_per_second(generated_tokens, decode_time_ms),
        used_streaming=True,
    )


def response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text is not None:
        return str(text)
    if isinstance(response, tuple) and response:
        return str(response[0])
    return str(response)


def response_token_increment(response: Any, text: str) -> int:
    if getattr(response, "token", None) is not None:
        return 1
    tokens = getattr(response, "tokens", None)
    if tokens is None:
        return 1 if text else 0
    try:
        return len(tokens)
    except TypeError:
        return 1


def response_final_token_count(response: Any, current: Optional[int]) -> Optional[int]:
    return first_numeric_attr(response, ("generation_tokens", "generated_tokens", "total_generated_tokens"), current)


def response_prompt_eval_ms(response: Any, current: Optional[float]) -> Optional[float]:
    return first_numeric_attr(response, ("prompt_eval_ms", "prompt_time_ms", "prompt_processing_ms"), current)


def first_numeric_attr(response: Any, attributes: tuple[str, ...], current):
    for attr in attributes:
        value = getattr(response, attr, None)
        if value is not None:
            try:
                return type(current if current is not None else value)(value)
            except (TypeError, ValueError):
                continue
    return current


def result_token_count(result: Any, raw_response: str) -> int:
    value = first_numeric_attr(result, ("generation_tokens", "generated_tokens", "total_generated_tokens"), None)
    return int(value) if value is not None else estimate_generated_token_count(raw_response)


def tokens_per_second(token_count: int, elapsed_ms: float) -> float:
    return (token_count / (elapsed_ms / 1000.0)) if elapsed_ms > 0.0 else 0.0


def elapsed_ms_since(started_at: float) -> float:
    return (time.perf_counter() - started_at) * 1000.0
