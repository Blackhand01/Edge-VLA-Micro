from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from src.action import DEFAULT_CONNECTION, DroneController
from src.core.drone_snapshot import DroneSnapshotBuilder, kinematic_state_dict
from src.monitoring import TelemetryCsvLogger
from src.perception.cognition_engine import CognitionEngine
from src.perception.guardrails import hsv_debug_overlay_path, requested_target_color
from src.perception.models import VLMProfile
from src.perception.smolvlm_runtime import explicit_command_from_intent
from src.safety.command_validator import (
    CommandValidator,
    DroneOperationalState,
    DroneStateSnapshot,
    SafetyViolationError,
)


logger = logging.getLogger(__name__)


class IntentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    image_base64: Optional[str] = None
    image_mime_type: str = "image/jpeg"
    source: str = "mac_sensor_client"
    request_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class IntentResponse(BaseModel):
    ok: bool
    action: str
    command: Optional[str] = None
    dry_run: bool
    reason: Optional[str] = None
    details: Optional[str] = None
    parsed_json: Optional[dict[str, Any]] = None
    drone_state: dict[str, Any]
    timings_ms: dict[str, float]
    request_id: Optional[str] = None
    detection_debug_image_base64: Optional[str] = None
    detection_debug_image_mime_type: Optional[str] = None
    detection_debug_image_path: Optional[str] = None
    vlm_profile: Optional[dict[str, Any]] = None


class ServerContext:
    def __init__(
        self,
        *,
        connection: str,
        image_dir: Path,
        max_image_bytes: int,
        dry_run: bool,
        telemetry_log: str | Path,
    ) -> None:
        self.connection = connection
        self.image_dir = image_dir
        self.max_image_bytes = max_image_bytes
        self.dry_run = dry_run
        self.telemetry_logger = TelemetryCsvLogger(path=telemetry_log)
        self.controller = DroneController(connection=connection)
        self.snapshot_builder = DroneSnapshotBuilder()
        self.engine = CognitionEngine(
            validator=CommandValidator(
                DroneStateSnapshot(
                    state=DroneOperationalState.GROUNDED,
                    connected=False,
                    battery_remaining=None,
                )
            ),
            vlm_backend="smolvlm",
            log_path="logs/cognition_server.log",
        )
        self.command_lock = asyncio.Lock()

    async def start(self) -> None:
        self.image_dir.mkdir(parents=True, exist_ok=True)
        await self.controller.connect()
        await self.controller.check_health(
            require_armable=False,
            require_global_position=False,
            require_battery=False,
        )
        logger.info("Cognition server ready on MAVSDK connection %s", self.connection)

    async def close(self) -> None:
        await self.controller.close()


def create_app(
    *,
    connection: str = DEFAULT_CONNECTION,
    image_dir: str | Path = "tmp/bridge_frames",
    max_image_mb: float = 4.0,
    dry_run: bool = False,
    telemetry_log: str | Path = "logs/telemetry.csv",
) -> FastAPI:
    context = ServerContext(
        connection=connection,
        image_dir=Path(image_dir),
        max_image_bytes=int(max_image_mb * 1024 * 1024),
        dry_run=dry_run,
        telemetry_log=telemetry_log,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.context = context
        await context.start()
        try:
            yield
        finally:
            await context.close()

    app = FastAPI(
        title="Edge-VLA Jetson Cognition Server",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        ctx: ServerContext = request.app.state.context
        drone_state = ctx.snapshot_builder.build(ctx.controller.state)
        return {
            "ok": True,
            "connection": ctx.connection,
            "dry_run": ctx.dry_run,
            "drone_state": kinematic_state_dict(ctx.controller.state, drone_state),
        }

    @app.post("/process_intent", response_model=IntentResponse)
    async def process_intent(payload: IntentRequest, request: Request) -> IntentResponse:
        ctx: ServerContext = request.app.state.context
        async with ctx.command_lock:
            response = await process_payload(ctx, payload)
            log_server_telemetry(ctx, payload, response)
            return response

    return app


async def process_payload(ctx: ServerContext, payload: IntentRequest) -> IntentResponse:
    started = time.perf_counter()
    drone_state = ctx.snapshot_builder.build(ctx.controller.state)
    cognition_started = time.perf_counter()
    fast_path_command = explicit_command_from_intent(
        payload.text,
        str(drone_state.model_dump(mode="json")),
    )
    if fast_path_command is not None and should_use_fast_path(fast_path_command, payload):
        cognition_ms = elapsed_ms(cognition_started)
        return await process_validated_payload(
            ctx,
            payload,
            fast_path_command,
            drone_state,
            cognition_ms,
            started,
            source="text_fast_path",
        )

    image_path = decode_image_payload(ctx, payload)
    result = ctx.engine.process_intent(
        payload.text,
        image_path=str(image_path) if image_path is not None else None,
        drone_state=drone_state,
    )
    cognition_ms = elapsed_ms(cognition_started)
    debug_payload = detection_debug_payload(payload.text, image_path)

    current_state = kinematic_state_dict(ctx.controller.state, drone_state)
    if not result.ok:
        return IntentResponse(
            ok=False,
            action="rejected",
            dry_run=ctx.dry_run,
            reason=result.reason,
            details=getattr(result, "details", None),
            parsed_json=result.parsed_json,
            drone_state=current_state,
            timings_ms={"cognition": cognition_ms, "total": elapsed_ms(started)},
            request_id=payload.request_id,
            vlm_profile=result.vlm_profile.model_dump(),
            **debug_payload,
        )

    return await execute_command_response(
        ctx,
        payload,
        result.validated_command,
        result.parsed_json,
        drone_state,
        cognition_ms,
        started,
        debug_payload=debug_payload,
        vlm_profile=result.vlm_profile,
    )


async def process_validated_payload(
    ctx: ServerContext,
    payload: IntentRequest,
    raw_command: dict[str, Any],
    drone_state: DroneStateSnapshot,
    cognition_ms: float,
    started: float,
    *,
    source: str,
) -> IntentResponse:
    try:
        validated = CommandValidator(drone_state).validate(
            json.dumps(raw_command, separators=(",", ":")),
            drone_state=drone_state,
        )
    except SafetyViolationError as exc:
        return IntentResponse(
            ok=False,
            action="rejected",
            dry_run=ctx.dry_run,
            reason="VALIDATION_REJECTED",
            details=str(exc),
            parsed_json=raw_command,
            drone_state=kinematic_state_dict(ctx.controller.state, drone_state),
            timings_ms={
                "cognition": cognition_ms,
                "total": elapsed_ms(started),
            },
            request_id=payload.request_id,
            vlm_profile=VLMProfile().model_dump(),
        )

    raw_command = dict(validated.raw)
    raw_command["reasoning"] = f"{raw_command.get('reasoning', '')}; {source}".strip("; ")
    return await execute_command_response(
        ctx,
        payload,
        validated,
        raw_command,
        drone_state,
        cognition_ms,
        started,
    )


async def execute_command_response(
    ctx: ServerContext,
    payload: IntentRequest,
    command,
    parsed_json: Optional[dict[str, Any]],
    drone_state: DroneStateSnapshot,
    cognition_ms: float,
    started: float,
    debug_payload: Optional[dict[str, Any]] = None,
    vlm_profile: Optional[VLMProfile] = None,
) -> IntentResponse:
    debug_payload = debug_payload or {}
    profile_payload = (vlm_profile or VLMProfile()).model_dump()
    action_started = time.perf_counter()
    current_state = kinematic_state_dict(ctx.controller.state, drone_state)
    try:
        action = "dry_run" if ctx.dry_run else await dispatch_command(ctx.controller, command)
    except Exception as exc:  # noqa: BLE001 - action failures must be returned to the sensor node.
        logger.exception("Action dispatch failed")
        return IntentResponse(
            ok=False,
            action="action_failed",
            command=command.name,
            dry_run=ctx.dry_run,
            reason=type(exc).__name__,
            details=str(exc),
            parsed_json=parsed_json,
            drone_state=current_state,
            timings_ms={
                "cognition": cognition_ms,
                "action": elapsed_ms(action_started),
                "total": elapsed_ms(started),
            },
            request_id=payload.request_id,
            vlm_profile=profile_payload,
            **debug_payload,
        )
    action_ms = elapsed_ms(action_started)
    post_state = ctx.snapshot_builder.build(ctx.controller.state)
    return IntentResponse(
        ok=True,
        action=action,
        command=command.name,
        dry_run=ctx.dry_run,
        parsed_json=parsed_json,
        drone_state=kinematic_state_dict(ctx.controller.state, post_state),
        timings_ms={
            "cognition": cognition_ms,
            "action": action_ms,
            "total": elapsed_ms(started),
        },
        request_id=payload.request_id,
        vlm_profile=profile_payload,
        **debug_payload,
    )


def log_server_telemetry(ctx: ServerContext, payload: IntentRequest, response: IntentResponse) -> None:
    timings = response.timings_ms or {}
    profile = response.vlm_profile or {}
    metadata = payload.metadata or {}
    ctx.telemetry_logger.log_event(
        profile="edge_distributed",
        component="jetson_cognition_server",
        request_id=response.request_id or payload.request_id,
        source=payload.source,
        ok=response.ok,
        action=response.action,
        command=response.command,
        state=response.drone_state.get("operational_state") or response.drone_state.get("state"),
        input_text=payload.text,
        image_present=bool(payload.image_base64),
        audio_ms=metadata.get("audio_ms"),
        vision_ms=metadata.get("vision_ms"),
        cognition_ms=timings.get("cognition"),
        action_ms=timings.get("action"),
        total_ms=timings.get("total"),
        ttft_ms=profile.get("ttft_ms"),
        decode_time_ms=profile.get("decode_time_ms"),
        generated_tokens=profile.get("generated_tokens"),
        tps=profile.get("tps"),
        safety_ms=profile.get("safety_ms"),
        reason=response.reason,
        details=response.details,
    )


def should_use_fast_path(command: dict[str, Any], payload: IntentRequest) -> bool:
    command_name = command.get("command")
    if command_name in {"arm", "disarm", "takeoff", "land", "hold"}:
        return True
    return command_name == "move_velocity" and not payload.image_base64


def decode_image_payload(ctx: ServerContext, payload: IntentRequest) -> Optional[Path]:
    if not payload.image_base64:
        return None
    try:
        raw = base64.b64decode(payload.image_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid image_base64: {exc}") from exc
    if len(raw) > ctx.max_image_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"image payload too large: {len(raw)} bytes > {ctx.max_image_bytes} bytes",
        )

    suffix = ".jpg" if payload.image_mime_type in {"image/jpeg", "image/jpg"} else ".png"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    image_path = ctx.image_dir / f"{payload.request_id or timestamp}{suffix}"
    image_path.write_bytes(raw)
    return image_path


def detection_debug_payload(text: str, image_path: Optional[Path]) -> dict[str, Any]:
    color = requested_target_color(text)
    if image_path is None or color is None:
        return {}

    debug_path = hsv_debug_overlay_path(str(image_path), color)
    if not debug_path.exists():
        return {}

    return {
        "detection_debug_image_base64": base64.b64encode(debug_path.read_bytes()).decode("ascii"),
        "detection_debug_image_mime_type": "image/jpeg",
        "detection_debug_image_path": str(debug_path),
    }


async def dispatch_command(controller: DroneController, command) -> str:
    if command.name == "hold" and not controller.state.in_air:
        logger.info("Skipping hold because vehicle is not airborne")
        return "skipped_hold_not_airborne"
    method = getattr(controller, command.controller_method)
    await method(**command.controller_kwargs())
    return command.name


def elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jetson VLA HTTP cognition/action server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--connection", default=DEFAULT_CONNECTION)
    parser.add_argument("--image-dir", default="tmp/bridge_frames")
    parser.add_argument("--max-image-mb", type=float, default=4.0)
    parser.add_argument("--dry-run", action="store_true", help="Run cognition/safety but do not send MAVSDK actions.")
    parser.add_argument("--telemetry-log", default="logs/telemetry.csv", help="Unified CSV path for Jetson server telemetry.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    import uvicorn

    app = create_app(
        connection=args.connection,
        image_dir=args.image_dir,
        max_image_mb=args.max_image_mb,
        dry_run=args.dry_run,
        telemetry_log=args.telemetry_log,
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
