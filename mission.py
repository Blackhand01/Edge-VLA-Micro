from __future__ import annotations

import argparse
import asyncio
import logging

from drone_controller import DEFAULT_CONNECTION, DroneController


logger = logging.getLogger(__name__)


async def run_mission(connection: str) -> None:
    async with DroneController(connection=connection) as controller:
        await controller.check_health()
        await controller.arm()
        await controller.takeoff()
        await controller.hold()
        await asyncio.sleep(5)
        await controller.move_velocity(vx=1.0, vy=0.0, vz=0.0, yaw_deg=0.0)
        await asyncio.sleep(3)
        await controller.land()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic Phase 1 MAVSDK control sequence.")
    parser.add_argument("--connection", default=DEFAULT_CONNECTION, help=f"MAVSDK connection string. Default: {DEFAULT_CONNECTION}")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    args = parse_args()
    asyncio.run(run_mission(args.connection))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
