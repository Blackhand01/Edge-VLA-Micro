from __future__ import annotations

import json
import re
from typing import Optional

from src.safety.command_validator import ActionCommand, DroneStateSnapshot


DEFAULT_MODEL_ID = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
SAFE_COMMAND_PATTERNS = (
    ("disarm", re.compile(r"\bdisarm\b", re.IGNORECASE)),
    ("takeoff", re.compile(r"\b(take\s*off|takeoff|launch)\b", re.IGNORECASE)),
    ("arm", re.compile(r"\barm\b", re.IGNORECASE)),
    ("land", re.compile(r"\bland\b", re.IGNORECASE)),
    ("hold", re.compile(r"\b(hold|hover)\b", re.IGNORECASE)),
)

BASE_COMMAND_FIELDS = frozenset({"command", "target_found", "reasoning"})
COMMAND_ALLOWED_FIELDS = {
    "arm": BASE_COMMAND_FIELDS,
    "disarm": BASE_COMMAND_FIELDS,
    "takeoff": BASE_COMMAND_FIELDS,
    "land": BASE_COMMAND_FIELDS,
    "hold": BASE_COMMAND_FIELDS,
    "move_velocity": BASE_COMMAND_FIELDS
    | frozenset({"velocity_x", "velocity_y", "velocity_z", "yaw_deg"}),
}

SYSTEM_PROMPT = (
    "You are the onboard computer of a drone. Inspect the FPV camera image and "
    "read the operator's voice command. Extract spatial intent as valid JSON. "
    "Every JSON object must include target_found and reasoning. If the requested "
    "object is not visible, set target_found=false and explain why. Return exactly "
    "one JSON command. command must be one of: arm, disarm, takeoff, land, hold, "
    "move_velocity. If the voice command contains multiple actions, choose only "
    "the next valid action based on CURRENT_DRONE_STATE. If the state is GROUNDED "
    "and arm/takeoff/move is requested, respond with arm first. Valid move example: "
    '{"command":"move_velocity","target_found":true,"reasoning":"red object visible",'
    '"velocity_x":0.5,"velocity_y":0.0,"velocity_z":0.0,"yaw_deg":0.0}. '
    'Missing target example: {"command":"hold","target_found":false,'
    '"reasoning":"requested target not visible"}. Do not include telemetry fields '
    "such as battery_remaining, state, connected, altitude, or GPS data. "
    "Do not add extra text."
)


def build_cognition_prompt(
    intent_text: str,
    *,
    image_path: Optional[str],
    drone_state: Optional[DroneStateSnapshot],
) -> str:
    schema = json.dumps(ActionCommand.model_json_schema(), indent=2, ensure_ascii=False)
    image_reference = image_path if image_path is not None else "<NO_IMAGE_AVAILABLE>"
    state_reference = drone_state.model_dump(mode="json") if drone_state is not None else "<UNKNOWN>"
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "JSON_SCHEMA:\n"
        f"{schema}\n\n"
        "CURRENT_DRONE_STATE:\n"
        f"{state_reference}\n\n"
        "IMAGE_PATH:\n"
        f"{image_reference}\n\n"
        "USER_INTENT:\n"
        f"{intent_text}\n\n"
        "JSON:"
    )
