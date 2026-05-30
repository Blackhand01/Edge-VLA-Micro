from __future__ import annotations

import argparse
import logging

from src.perception.cognition_engine import CognitionEngine
from src.safety.command_validator import CommandValidator, DroneOperationalState, DroneStateSnapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one cognition request without MAVSDK, audio, or camera.")
    parser.add_argument("--vlm-backend", choices=("dummy", "tensorrt", "smolvlm"), default="dummy")
    parser.add_argument("--state", choices=[state.value for state in DroneOperationalState], default=DroneOperationalState.AIRBORNE.value)
    parser.add_argument("--battery", type=float, default=1.0)
    parser.add_argument("--connected", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--image-path", default=None)
    parser.add_argument("--text", default="Hold position.")
    parser.add_argument("--log-path", default="/tmp/edge-vla-cognition-probe.log")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    drone_state = DroneStateSnapshot(
        state=DroneOperationalState(args.state),
        connected=args.connected,
        battery_remaining=args.battery,
    )
    engine = CognitionEngine(
        validator=CommandValidator(drone_state),
        vlm_backend=args.vlm_backend,
        max_tokens=args.max_tokens,
        log_path=args.log_path,
    )
    result = engine.process_intent(args.text, image_path=args.image_path, drone_state=drone_state)

    print(f"ok: {result.ok}")
    print(f"raw: {result.raw_response}")
    print(f"parsed: {result.parsed_json}")
    print(f"details: {getattr(result, 'details', None)}")
    if result.ok:
        print(f"command: {result.validated_command.name}")
        print(f"profile: {result.vlm_profile.model_dump()}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
