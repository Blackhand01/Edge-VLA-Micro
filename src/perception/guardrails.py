from __future__ import annotations

import logging
import re
from typing import Callable, Optional

from src.safety.command_validator import DroneOperationalState, DroneStateSnapshot
from src.perception.prompts import COMMAND_ALLOWED_FIELDS, SAFE_COMMAND_PATTERNS


logger = logging.getLogger(__name__)

COLOR_TARGET_RE = re.compile(r"\b(red|blue)\b|\bhead\s+object\b", re.IGNORECASE)
HSV_MIN_PIXELS = 50
HSV_MIN_RATIO = 0.0002


def apply_hsv_guardrail(
    text_input: str,
    image_path: Optional[str],
    parsed_json: dict,
    color_detector: Callable[[str, str], bool],
) -> None:
    requested_color = requested_target_color(text_input)
    if requested_color is None:
        return

    if not image_path:
        force_target_not_found(
            parsed_json,
            f"OpenCV HSV guardrail: {requested_color} target requested but no frame is available.",
        )
        return

    if not color_detector(image_path, requested_color):
        force_target_not_found(
            parsed_json,
            f"OpenCV HSV guardrail: requested {requested_color} target is absent from the frame.",
        )
        return

    force_target_found(
        parsed_json,
        f"OpenCV HSV guardrail: requested {requested_color} target is present in the frame.",
    )


def requested_target_color(text_input: str) -> Optional[str]:
    match = COLOR_TARGET_RE.search(text_input)
    if match is None:
        return None

    color = (match.group(1) or match.group(0)).lower()
    if color == "head object":
        logger.warning("ASR_COLOR_ALIAS: treating 'head object' as likely 'red object'")
        return "red"
    return color


def force_target_not_found(parsed_json: dict, reasoning: str) -> None:
    logger.warning("SAFETY_OVERRIDE: TARGET_NOT_FOUND | %s", reasoning)
    parsed_json["target_found"] = False
    parsed_json["reasoning"] = reasoning


def force_target_found(parsed_json: dict, reasoning: str) -> None:
    parsed_json["target_found"] = True
    parsed_json["reasoning"] = reasoning


def repair_missing_command_from_text(
    text_input: str,
    parsed_json: dict,
    drone_state: Optional[DroneStateSnapshot],
) -> None:
    if parsed_json.get("command"):
        return

    command = infer_safe_non_motion_command(text_input, drone_state)
    if command is None:
        return

    logger.warning("SCHEMA_REPAIR: INFERRED_COMMAND | command=%s", command)
    parsed_json["command"] = command
    parsed_json.setdefault("target_found", True)
    parsed_json.setdefault(
        "reasoning",
        f"Deterministic non-motion transcript fallback inferred command={command}.",
    )


def repair_missing_target_fields(parsed_json: dict) -> None:
    missing_fields = [field for field in ("target_found", "reasoning") if field not in parsed_json]
    if not missing_fields:
        return

    logger.warning("SCHEMA_REPAIR: MISSING_TARGET_FIELDS | fields=%s", ",".join(missing_fields))
    command = parsed_json.get("command")
    if command in {"arm", "disarm", "takeoff", "land", "hold"}:
        parsed_json.setdefault("target_found", True)
        parsed_json.setdefault(
            "reasoning",
            f"Deterministic schema repair: VLM omitted target metadata for non-visual command={command}.",
        )
        return

    parsed_json["target_found"] = False
    parsed_json["reasoning"] = (
        "SAFETY_OVERRIDE: TARGET_NOT_FOUND. VLM omitted required target_found/reasoning fields."
    )


def drop_extra_command_fields(parsed_json: dict) -> None:
    command = parsed_json.get("command")
    if not isinstance(command, str):
        return

    allowed_fields = COMMAND_ALLOWED_FIELDS.get(command)
    if allowed_fields is None:
        return

    extra_fields = sorted(field for field in parsed_json if field not in allowed_fields)
    if not extra_fields:
        return

    logger.warning("SCHEMA_REPAIR: DROPPED_EXTRA_FIELDS | fields=%s", ",".join(extra_fields))
    for field in extra_fields:
        parsed_json.pop(field, None)


def infer_safe_non_motion_command(
    text_input: str,
    drone_state: Optional[DroneStateSnapshot],
) -> Optional[str]:
    matched = {command for command, pattern in SAFE_COMMAND_PATTERNS if pattern.search(text_input)}
    if not matched:
        return None
    if drone_state is None:
        return first_matched_command(text_input, matched)

    if drone_state.state == DroneOperationalState.GROUNDED:
        return first_allowed_command(matched, ("arm", "disarm", "hold", "takeoff", "land"))
    if drone_state.state == DroneOperationalState.ARMED:
        return first_allowed_command(matched, ("takeoff", "disarm", "hold", "arm", "land"))
    if drone_state.state in {
        DroneOperationalState.AIRBORNE,
        DroneOperationalState.OFFBOARD,
        DroneOperationalState.LANDING,
    }:
        return first_allowed_command(matched, ("land", "hold", "takeoff", "disarm", "arm"))

    return first_matched_command(text_input, matched)


def first_allowed_command(matched: set[str], ordered_commands: tuple[str, ...]) -> Optional[str]:
    return next((command for command in ordered_commands if command in matched), None)


def first_matched_command(text_input: str, matched: set[str]) -> Optional[str]:
    candidates: list[tuple[int, str]] = []
    for command, pattern in SAFE_COMMAND_PATTERNS:
        if command not in matched:
            continue
        match = pattern.search(text_input)
        if match is not None:
            candidates.append((match.start(), command))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def frame_contains_hsv_color(image_path: str, color: str) -> bool:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        logger.warning("SAFETY_OVERRIDE: TARGET_NOT_FOUND | OpenCV unavailable: %s", exc)
        return False

    frame = cv2.imread(image_path)
    if frame is None:
        logger.warning("SAFETY_OVERRIDE: TARGET_NOT_FOUND | OpenCV could not read frame: %s", image_path)
        return False

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if color == "red":
        lower_1 = np.array([0, 80, 50], dtype=np.uint8)
        upper_1 = np.array([10, 255, 255], dtype=np.uint8)
        lower_2 = np.array([170, 80, 50], dtype=np.uint8)
        upper_2 = np.array([180, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_1, upper_1) | cv2.inRange(hsv, lower_2, upper_2)
    elif color == "blue":
        lower = np.array([100, 80, 50], dtype=np.uint8)
        upper = np.array([130, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
    else:
        return False

    matching_pixels = int(cv2.countNonZero(mask))
    total_pixels = int(mask.shape[0] * mask.shape[1])
    ratio = matching_pixels / total_pixels if total_pixels > 0 else 0.0
    logger.info(
        "HSV target check: color=%s pixels=%d total=%d ratio=%.6f threshold_pixels=%d threshold_ratio=%.6f",
        color,
        matching_pixels,
        total_pixels,
        ratio,
        HSV_MIN_PIXELS,
        HSV_MIN_RATIO,
    )
    return matching_pixels >= HSV_MIN_PIXELS and ratio >= HSV_MIN_RATIO
