from __future__ import annotations

import unittest

from src.perception.prompts import DEFAULT_QUANTIZED_MODEL_ID
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


if __name__ == "__main__":
    unittest.main()
