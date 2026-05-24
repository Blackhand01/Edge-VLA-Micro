from __future__ import annotations

import contextlib
import io
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from src.perception import CognitionEngine, CognitionError, CognitionResult
from src.safety.command_validator import CommandValidator, DroneOperationalState, DroneStateSnapshot, HoldModel


def _airborne_validator() -> CommandValidator:
    return CommandValidator(
        DroneStateSnapshot(
            state=DroneOperationalState.AIRBORNE,
            connected=True,
            battery_remaining=0.90,
        )
    )


def _process_quiet(engine: CognitionEngine, text: str, **kwargs) -> CognitionResult | CognitionError:
    with contextlib.redirect_stdout(io.StringIO()):
        return engine.process_intent(text, **kwargs)


def _command_json(command: str) -> str:
    return f'{{"command":"{command}","target_found":true,"reasoning":"test command"}}'
