from __future__ import annotations

import json
import re


JSON_OBJECT_START_RE = re.compile(r"\{")


def extract_json_object(raw_response: str) -> dict:
    cleaned = raw_response.strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = JSON_OBJECT_START_RE.search(cleaned)
    if match is None:
        raise ValueError("LLM response does not contain a JSON object")

    parsed = json.loads(balanced_json_slice(cleaned, match.start()))
    if not isinstance(parsed, dict):
        raise ValueError(f"LLM JSON must be an object, got {type(parsed).__name__}")
    return parsed


def balanced_json_slice(text: str, start: int) -> str:
    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    raise ValueError("LLM response contains an unterminated JSON object")
