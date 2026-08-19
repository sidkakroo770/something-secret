#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import time

from native.common.scan_adapter import ScanAdapter
from native.controllers.obstacle_avoidance import (
    ObstacleAvoidanceController,
)
from native.hardware.d500_driver import D500Driver


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--port",
        default="/dev/ttyUSB0",
    )

    parser.add_argument(
        "--yaw-offset",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--invert-angle",
        action="store_true",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=8.0,
    )

    args = parser.parse_args()

    adapter = ScanAdapter(
        invert_angle=args.invert_angle,
        yaw_offset_deg=args.yaw_offset,
    )

    controller = (
        ObstacleAvoidanceController()
    )

    controller.enter()

    print()
    print("================================================")
    print(" NATIVE OBSTACLE AVOIDANCE DRY RUN")
    print(" *** NOTHING IS SENT TO THE FLIGHT CONTROLLER ***")
    print("================================================")
    print()

    started = time.monotonic()

    with D500Driver(
        port=args.port
    ) as lidar:

        while (
            time.monotonic()
            - started
            < args.duration
        ):

            raw = lidar.get_scan(
                timeout_s=2.0
            )

            if raw is None:

                print("D500 TIMEOUT")
                continue

            scan = adapter.convert(raw)

            output = controller.step(
                scan,
                attitude=None,
            )

            d = controller.diagnostics()

            obs = d.get(
                "observation",
                {},
            )

            cmd = output.command

            print(
                f"{output.status:<30} "
                f"geom={str(obs.get('geometry_valid', False)):<5} "
                f"front={obs.get('front_clearance_m')} "
                f"Lwall={obs.get('left_wall_valid')} "
                f"Rwall={obs.get('right_wall_valid')} "
                f"face={obs.get('front_face_valid')} "
                f"candidate={obs.get('candidate_state')} "
                f"| vx={cmd.vx_m_s:+.3f} "
                f"vy={cmd.vy_m_s:+.3f} "
                f"yaw="
                f"{math.degrees(cmd.yaw_rate_rad_s):+.2f}deg/s"
            )

            reason = obs.get(
                "candidate_reason"
            )

            if reason:
                print(
                    "    observation:",
                    reason,
                )

            if output.next_state is not None:

                print()
                print(
                    "FSM REQUEST:",
                    output.next_state.value,
                )

                print(
                    "REASON:",
                    output.reason,
                )

                break


if __name__ == "__main__":
    main()
