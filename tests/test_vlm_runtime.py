from __future__ import annotations

import unittest

from src.perception.prompts import DEFAULT_QUANTIZED_MODEL_ID
from src.perception.dummy_vlm_runtime import DummyVLMRuntime
from src.perception.smolvlm_runtime import (
    DEFAULT_SMOLVLM_MODEL_ID,
    build_smolvlm_prompt,
    normalize_smolvlm_response,
    resolve_smolvlm_model_id,
)
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

    def test_smolvlm_runtime_maps_mlx_default_model_to_jetson_model(self) -> None:
        self.assertEqual(
            resolve_smolvlm_model_id("mlx-community/Qwen2-VL-2B-Instruct-4bit"),
            DEFAULT_SMOLVLM_MODEL_ID,
        )

    def test_smolvlm_prompt_extracts_intent_and_state(self) -> None:
        raw_prompt = (
            "CURRENT_DRONE_STATE:\n"
            "{'state': 'GROUNDED'}\n\n"
            "IMAGE_PATH:\n"
            "<NO_IMAGE_AVAILABLE>\n\n"
            "USER_INTENT:\n"
            "Take off.\n\n"
            "JSON:"
        )

        prompt = build_smolvlm_prompt(raw_prompt)

        self.assertIn("USER_INTENT:\nTake off.", prompt)
        self.assertIn("{'state': 'GROUNDED'}", prompt)
        self.assertIn('"command":"arm"', prompt)

    def test_smolvlm_response_falls_back_to_intent_for_truncated_json(self) -> None:
        raw_prompt = "CURRENT_DRONE_STATE:\n{'state': 'AIRBORNE'}\n\nIMAGE_PATH:\nx\n\nUSER_INTENT:\nHold position.\n\nJSON:"

        normalized = normalize_smolvlm_response('{"command":"hold",', raw_prompt)

        self.assertIn('"command":"hold"', normalized)
        self.assertIn('"target_found":true', normalized)

    def test_smolvlm_response_clamps_move_fields(self) -> None:
        raw_prompt = "USER_INTENT:\nMove forward.\n\nJSON:"

        normalized = normalize_smolvlm_response(
            '{"command":"move_velocity","target_found":true,"reasoning":"x","velocity_x":50}',
            raw_prompt,
        )

        self.assertIn('"velocity_x":2.0', normalized)
        self.assertIn('"yaw_deg":0.0', normalized)

    def test_smolvlm_explicit_hold_overrides_wrong_parsed_command(self) -> None:
        raw_prompt = (
            "CURRENT_DRONE_STATE:\n{'state': 'GROUNDED'}\n\n"
            "IMAGE_PATH:\nx\n\n"
            "USER_INTENT:\nhand position Hold position.\n\n"
            "JSON:"
        )

        normalized = normalize_smolvlm_response(
            '{"command":"arm","target_found":true,"reasoning":"..."}',
            raw_prompt,
        )

        self.assertIn('"command":"hold"', normalized)

    def test_smolvlm_vague_text_does_not_allow_critical_model_command(self) -> None:
        raw_prompt = (
            "CURRENT_DRONE_STATE:\n{'state': 'GROUNDED'}\n\n"
            "IMAGE_PATH:\nx\n\n"
            "USER_INTENT:\nWe need to rush.\n\n"
            "JSON:"
        )

        normalized = normalize_smolvlm_response(
            '{"command":"arm","target_found":true,"reasoning":"..."}',
            raw_prompt,
        )

        self.assertIn('"command":"hold"', normalized)
        self.assertIn("without an explicit operator command", normalized)

    def test_smolvlm_moving_toward_red_object_maps_to_velocity(self) -> None:
        raw_prompt = (
            "CURRENT_DRONE_STATE:\n{'state': 'AIRBORNE'}\n\n"
            "IMAGE_PATH:\nx\n\n"
            "USER_INTENT:\nMoving to our red object.\n\n"
            "JSON:"
        )

        normalized = normalize_smolvlm_response(
            '{"command":"hold","target_found":true,"reasoning":"..."}',
            raw_prompt,
        )

        self.assertIn('"command":"move_velocity"', normalized)
        self.assertIn('"velocity_x":0.5', normalized)

    def test_smolvlm_one_meter_per_second_maps_to_one_meter_velocity(self) -> None:
        raw_prompt = (
            "CURRENT_DRONE_STATE:\n{'state': 'AIRBORNE'}\n\n"
            "IMAGE_PATH:\nx\n\n"
            "USER_INTENT:\nMove forward one meter per second.\n\n"
            "JSON:"
        )

        normalized = normalize_smolvlm_response(
            '{"command":"hold","target_found":true,"reasoning":"..."}',
            raw_prompt,
        )

        self.assertIn('"command":"move_velocity"', normalized)
        self.assertIn('"velocity_x":1.0', normalized)


if __name__ == "__main__":
    unittest.main()
