#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import time

from native.common.scan_adapter import ScanAdapter
from native.controllers.pre_entry import PreEntryController
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
        default=30.0,
    )

    args = parser.parse_args()

    adapter = ScanAdapter(
        invert_angle=args.invert_angle,
        yaw_offset_deg=args.yaw_offset,
    )

    controller = PreEntryController()

    controller.enter()

    print()
    print("==============================================")
    print(" NATIVE PRE_ENTRY DRY RUN")
    print(" *** NO COMMANDS ARE SENT TO THE DRONE ***")
    print("==============================================")
    print()

    print(
        " STATE              "
        "VALID CONF  "
        "YAW(deg) LAT(m) WIDTH "
        "LEFT RIGHT FRONT | "
        "COMMAND vy / yaw"
    )

    print("-" * 120)

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

                print(
                    "LiDAR timeout"
                )

                continue

            scan = adapter.convert(raw)

            result = controller.step(
                scan,
                attitude=None,
            )

            d = controller.diagnostics()

            yaw = d.get(
                "yaw_error_deg"
            )

            lateral = d.get(
                "lateral_error_m"
            )

            width = d.get(
                "width_m"
            )

            left = d.get(
                "left_distance_m"
            )

            right = d.get(
                "right_distance_m"
            )

            front = d.get(
                "front_clearance_m"
            )

            confidence = d.get(
                "confidence",
                0.0,
            )

            strict = d.get(
                "strict_valid",
                False,
            )

            cmd = result.command

            print(
                f"{result.status:<19} "
                f"{str(strict):<5} "
                f"{confidence:4.2f}  "
                f"{fmt(yaw)} "
                f"{fmt(lateral)} "
                f"{fmt(width)} "
                f"{fmt(left)} "
                f"{fmt(right)} "
                f"{fmt(front)} | "
                f"vy={cmd.vy_m_s:+.3f} "
                f"yaw={math.degrees(cmd.yaw_rate_rad_s):+.2f}deg/s"
            )

            if result.next_state is not None:

                print()
                print(
                    "=============================================="
                )

                print(
                    "FSM REQUEST:",
                    result.next_state.value,
                )

                print(
                    "Reason:",
                    result.reason,
                )

                print(
                    "=============================================="
                )

                break


if __name__ == "__main__":
    main()
