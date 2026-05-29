from __future__ import annotations

import unittest

from src.perception.prompts import DEFAULT_QUANTIZED_MODEL_ID
from src.perception.dummy_vlm_runtime import DummyVLMRuntime
from src.perception.vlm_runtime import VLMRuntime, resolve_model_id


class VLMRuntimeOptimizationTests(unittest.TestCase):
    def test_resolve_model_id_prefers_quantized_variant(self) -> None:
        self.assertEqual(
            resolve_model_id("mlx-community/Qwen2-VL-2B-Instruct", True, DEFAULT_QUANTIZED_MODEL_ID),
            DEFAULT_QUANTIZED_MODEL_ID,
        )

    def test_resolve_model_id_preserves_explicit_quantized_model(self) -> None:
        explicit = "mlx-community/Qwen2-VL-2B-Instruct-4bit"

        self.assertEqual(resolve_model_id(explicit, True, DEFAULT_QUANTIZED_MODEL_ID), explicit)

    def test_runtime_can_disable_quantized_routing_for_baseline_benchmark(self) -> None:
        runtime = VLMRuntime(model_id="mlx-community/Qwen2-VL-2B-Instruct", prefer_quantized=False)

        self.assertEqual(runtime.model_id, "mlx-community/Qwen2-VL-2B-Instruct")

    def test_dummy_runtime_returns_valid_profiled_json(self) -> None:
        runtime = DummyVLMRuntime(latency_s=0.0)

        raw_response, profile = runtime.generate_profiled("Arm the drone.", image_path=None)

        self.assertIn('"command":"arm"', raw_response)
        self.assertGreaterEqual(profile.ttft_ms, 0.0)
        self.assertEqual(profile.fallback_reason, "dummy runtime")


if __name__ == "__main__":
    unittest.main()
