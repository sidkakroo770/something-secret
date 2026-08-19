#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time

from native.common.scan_adapter import ScanAdapter
from native.controllers.exit_detection import ExitDetectionController
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
        default=10.0,
    )

    args = parser.parse_args()

    adapter = ScanAdapter(
        invert_angle=args.invert_angle,
        yaw_offset_deg=args.yaw_offset,
    )

    controller = ExitDetectionController()
    controller.enter()

    print()
    print("===============================================")
    print(" NATIVE EXIT_DETECTION DRY RUN")
    print(" *** NOTHING IS SENT TO THE FLIGHT CONTROLLER ***")
    print("===============================================")
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
                timeout_s=1.0
            )

            scan = None

            if raw is not None:
                scan = adapter.convert(raw)

            output = controller.step(scan)

            d = controller.diagnostics()
            cmd = output.command

            print(
                f"{output.status:<15} "
                f"t={d['elapsed_s']:5.2f}s "
                f"travel={d['estimated_commanded_travel_m']:5.2f}m "
                f"front={d['front_clearance_m']} "
                f"stop_streak={d['front_stop_streak']} | "
                f"vx={cmd.vx_m_s:+.3f} "
                f"vy={cmd.vy_m_s:+.3f}"
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
