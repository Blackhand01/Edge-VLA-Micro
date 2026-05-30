from __future__ import annotations

import argparse
import asyncio
import logging
import time

from src.action import DEFAULT_CONNECTION, DroneController
from src.core.drone_snapshot import DroneSnapshotBuilder
from src.perception.cognition_engine import CognitionEngine
from src.safety.command_validator import CommandValidator, DroneOperationalState, DroneStateSnapshot


DEFAULT_COMMANDS = (
    "Arm the drone.",
    "Take off.",
    "Hold position.",
    "Move forward one meter per second.",
    "Hold position.",
    "Land.",
)


logger = logging.getLogger(__name__)


async def run_scripted_smoke(args: argparse.Namespace) -> None:
    controller = DroneController(connection=args.connection)
    snapshot_builder = DroneSnapshotBuilder()
    engine = CognitionEngine(
        validator=CommandValidator(
            DroneStateSnapshot(
                state=DroneOperationalState.GROUNDED,
                connected=False,
                battery_remaining=None,
            )
        ),
        vlm_backend=args.vlm_backend,
        log_path=args.cognition_log,
    )

    try:
        await controller.connect()
        await controller.check_health(
            require_armable=False,
            require_global_position=False,
            require_battery=False,
        )
        for text in commands_from_args(args):
            drone_state = snapshot_builder.build(controller.state)
            started = time.perf_counter()
            result = engine.process_intent(text, drone_state=drone_state)
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            if not result.ok:
                print(f"[REJECTED] state={drone_state.state.value} input={text!r} reason={result.reason}")
                continue

            command = result.validated_command
            print(
                f"[ACCEPTED] state={drone_state.state.value} input={text!r} "
                f"command={command.name} cognition_ms={elapsed_ms:.1f}",
                flush=True,
            )
            if not args.dry_run:
                await dispatch(controller, command)
                await asyncio.sleep(args.post_action_sleep)
    finally:
        await controller.close()


async def dispatch(controller: DroneController, command) -> None:
    if command.name == "hold" and not controller.state.in_air:
        logger.info("Skipping hold because vehicle is not airborne")
        return
    method = getattr(controller, command.controller_method)
    await method(**command.controller_kwargs())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scripted Jetson smoke test: text intent -> cognition -> MAVSDK action.")
    parser.add_argument("--connection", default=DEFAULT_CONNECTION, help=f"MAVSDK connection string. Default: {DEFAULT_CONNECTION}")
    parser.add_argument("--vlm-backend", choices=("dummy", "tensorrt", "smolvlm"), default="dummy")
    parser.add_argument("--cognition-log", default="/tmp/edge-vla-scripted-cognition.log")
    parser.add_argument("--post-action-sleep", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true", help="Validate cognition and safety but do not send MAVSDK actions.")
    parser.add_argument("--command", action="append", default=None, help="Command text. Can be passed multiple times.")
    return parser.parse_args()


def commands_from_args(args: argparse.Namespace) -> tuple[str, ...] | list[str]:
    return args.command if args.command is not None else DEFAULT_COMMANDS


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    asyncio.run(run_scripted_smoke(parse_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
