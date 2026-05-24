from __future__ import annotations

import re


EMERGENCY_KEYWORDS = ("stop", "emergency", "emergenza")
EMERGENCY_PATTERN = re.compile(r"\b(stop|emergency|emergenza)\b", re.IGNORECASE)
ACTIONABLE_PATTERN = re.compile(
    r"\b("
    r"arm|armed|disarm|take\s*off|takeoff|launch|land|hold|hover|"
    r"move|forward|backward|left|right|red|object|target|drone"
    r")\b",
    re.IGNORECASE,
)
VISION_REQUIRED_PATTERN = re.compile(
    r"\b(move|forward|backward|left|right|red|blue|object|target)\b",
    re.IGNORECASE,
)
