from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.perception.models import CognitionError, CognitionResult


def write_cognition_transaction(
    *,
    log_path: Path,
    raw_prompt: str,
    raw_response: str,
    parsed_json: Optional[dict],
    validation_result: CognitionResult | CognitionError,
) -> None:
    block = format_cognition_transaction(
        raw_prompt=raw_prompt,
        raw_response=raw_response,
        parsed_json=parsed_json,
        validation_result=validation_result,
    )
    print(block)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(block)


def format_cognition_transaction(
    *,
    raw_prompt: str,
    raw_response: str,
    parsed_json: Optional[dict],
    validation_result: CognitionResult | CognitionError,
) -> str:
    parsed_text = json.dumps(parsed_json, indent=2, ensure_ascii=False) if parsed_json is not None else "<NONE>"
    return (
        "\n"
        "================ COGNITION TRANSACTION ================\n"
        f"[UTC] {datetime.now(timezone.utc).isoformat()}\n"
        "[PROMPT]\n"
        f"{raw_prompt}\n"
        "[RAW_RESPONSE]\n"
        f"{raw_response or '<EMPTY>'}\n"
        "[PARSED_JSON]\n"
        f"{parsed_text}\n"
        "[VALIDATION_STATUS]\n"
        f"{validation_status(validation_result)}\n"
        "=======================================================\n"
    )


def validation_status(validation_result: CognitionResult | CognitionError) -> str:
    if isinstance(validation_result, CognitionResult):
        command = validation_result.validated_command
        return (
            "ACCEPTED | "
            f"command={command.name} | "
            f"controller_method={command.controller_method} | "
            f"kwargs={command.controller_kwargs()}"
        )

    detail = f" | details={validation_result.details}" if validation_result.details else ""
    return f"REJECTED | reason={validation_result.reason}{detail}"
