#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import time

from native.common.scan_adapter import ScanAdapter
from native.controllers.corridor_cruise import CorridorCruiseController
from native.hardware.d500_driver import D500Driver


def fmt(value, width=6, decimals=2):
    if value is None:
        return " " * (width - 4) + "----"

    return f"{value:{width}.{decimals}f}"


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

    controller = CorridorCruiseController()
    controller.enter()

    print()
    print("================================================")
    print(" NATIVE CORRIDOR_CRUISE DRY RUN")
    print(" *** NOTHING IS SENT TO THE FLIGHT CONTROLLER ***")
    print("================================================")
    print()

    started = time.monotonic()

    with D500Driver(port=args.port) as lidar:

        while time.monotonic() - started < args.duration:

            raw = lidar.get_scan(timeout_s=2.0)

            if raw is None:
                print("D500 TIMEOUT")
                continue

            scan = adapter.convert(raw)

            result = controller.step(
                scan,
                attitude=None,
            )

            d = controller.diagnostics()
            cmd = result.command

            print(
                f"{result.status:<30} "
                f"strict={str(d.get('strict_valid', False)):<5} "
                f"conf={fmt(d.get('confidence'))} "
                f"yaw={fmt(d.get('yaw_error_deg'))}deg "
                f"lat={fmt(d.get('lateral_error_m'))}m "
                f"width={fmt(d.get('width_m'))}m "
                f"front={fmt(d.get('front_clearance_m'))}m "
                f"Lopen={d.get('side_open_left', False)} "
                f"Ropen={d.get('side_open_right', False)} | "
                f"vx={cmd.vx_m_s:+.3f} "
                f"vy={cmd.vy_m_s:+.3f} "
                f"yawcmd={math.degrees(cmd.yaw_rate_rad_s):+.2f}deg/s"
            )

            if result.next_state is not None:

                print()
                print("FSM REQUEST:", result.next_state.value)
                print("REASON:", result.reason)
                break


if __name__ == "__main__":
    main()
