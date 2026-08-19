#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time

from native.common.scan_adapter import ScanAdapter
from native.common.types import MissionState

from native.controllers.hover_and_reassess import (
    HoverAndReassessController,
)

from native.hardware.d500_driver import D500Driver


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--port",
        default="/dev/ttyUSB0",
    )

    parser.add_argument(
        "--source",
        default="CORRIDOR_CRUISE",
    )

    parser.add_argument(
        "--reason",
        default="TEST_RECOVERY",
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

    try:

        source = MissionState(
            args.source.strip().upper()
        )

    except ValueError:

        raise SystemExit(
            f"Unknown MissionState: {args.source}"
        )

    adapter = ScanAdapter(
        invert_angle=args.invert_angle,
        yaw_offset_deg=args.yaw_offset,
    )

    controller = (
        HoverAndReassessController()
    )

    controller.enter(
        source,
        args.reason,
    )

    print()
    print("==============================================")
    print(" NATIVE HOVER_AND_REASSESS DRY RUN")
    print(" *** ZERO MOTION AT ALL TIMES ***")
    print("==============================================")
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

            scan = (
                adapter.convert(raw)
                if raw is not None
                else None
            )

            output = controller.step(
                scan
            )

            d = controller.diagnostics()

            cmd = output.command

            print(
                f"{output.status:<23} "
                f"front={d.get('front_clearance_m')} "
                f"corridor={d.get('corridor_stable')} "
                f"width={d.get('corridor_width_m')} "
                f"blocked={d.get('front_blocked')} "
                f"obstacle={d.get('obstacle_candidate')} "
                f"exit={d.get('exit_candidate')} "
                f"history={d.get('candidate_history')} | "
                f"vx={cmd.vx_m_s:+.3f} "
                f"vy={cmd.vy_m_s:+.3f} "
                f"vz={cmd.vz_m_s:+.3f} "
                f"yaw={cmd.yaw_rate_rad_s:+.3f}"
            )

            # HOVER_AND_REASSESS is NEVER allowed
            # to command movement.
            assert cmd.vx_m_s == 0.0
            assert cmd.vy_m_s == 0.0
            assert cmd.vz_m_s == 0.0
            assert cmd.yaw_rate_rad_s == 0.0

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
