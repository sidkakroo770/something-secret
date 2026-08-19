#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import time

from pymavlink import mavutil


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--port",
        required=True,
        help="Pixhawk serial device, preferably /dev/serial/by-id/...",
    )

    parser.add_argument(
        "--baud",
        type=int,
        default=115200,
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=15.0,
    )

    args = parser.parse_args()

    print()
    print("===============================================")
    print(" PIXHAWK MAVLINK READ-ONLY TEST")
    print(" *** NO ARM / MODE / MOVEMENT COMMANDS SENT ***")
    print("===============================================")
    print()
    print("Port :", args.port)
    print("Baud :", args.baud)
    print()
    print("Waiting for Pixhawk HEARTBEAT...")

    connection = mavutil.mavlink_connection(
        args.port,
        baud=args.baud,
        autoreconnect=True,
        source_system=255,
    )

    heartbeat = connection.wait_heartbeat(
        timeout=10,
    )

    if heartbeat is None:
        raise SystemExit(
            "No HEARTBEAT received within 10 seconds."
        )

    print()
    print("HEARTBEAT RECEIVED")
    print("------------------")
    print(
        "Target system   :",
        connection.target_system,
    )
    print(
        "Target component:",
        connection.target_component,
    )
    print(
        "Vehicle type    :",
        heartbeat.type,
    )
    print(
        "Autopilot type  :",
        heartbeat.autopilot,
    )
    print(
        "Base mode       :",
        heartbeat.base_mode,
    )
    print(
        "Custom mode     :",
        heartbeat.custom_mode,
    )

    armed = bool(
        heartbeat.base_mode
        & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
    )

    print(
        "Armed           :",
        armed,
    )

    try:
        mode = mavutil.mode_string_v10(
            heartbeat
        )
    except Exception:
        mode = "UNKNOWN"

    print(
        "Flight mode     :",
        mode,
    )

    print()
    print(
        f"Listening for telemetry for {args.duration:.1f} s..."
    )
    print()

    started = time.monotonic()

    counts = {}

    last_print = {
        "ATTITUDE": 0.0,
        "LOCAL_POSITION_NED": 0.0,
        "GLOBAL_POSITION_INT": 0.0,
        "SYS_STATUS": 0.0,
        "GPS_RAW_INT": 0.0,
    }

    wanted = set(last_print.keys())

    while (
        time.monotonic()
        - started
        < args.duration
    ):

        msg = connection.recv_match(
            blocking=True,
            timeout=1.0,
        )

        if msg is None:
            continue

        msg_type = msg.get_type()

        if msg_type == "BAD_DATA":
            continue

        counts[msg_type] = (
            counts.get(msg_type, 0) + 1
        )

        if msg_type not in wanted:
            continue

        now = time.monotonic()

        # Print each telemetry type at most twice per second.
        if (
            now - last_print[msg_type]
            < 0.5
        ):
            continue

        last_print[msg_type] = now

        if msg_type == "ATTITUDE":

            print(
                "ATTITUDE           "
                f"roll={math.degrees(msg.roll):+7.2f} deg "
                f"pitch={math.degrees(msg.pitch):+7.2f} deg "
                f"yaw={math.degrees(msg.yaw):+7.2f} deg"
            )

        elif msg_type == "LOCAL_POSITION_NED":

            print(
                "LOCAL_POSITION_NED "
                f"N={msg.x:+7.2f} m "
                f"E={msg.y:+7.2f} m "
                f"D={msg.z:+7.2f} m | "
                f"vN={msg.vx:+6.2f} "
                f"vE={msg.vy:+6.2f} "
                f"vD={msg.vz:+6.2f} m/s"
            )

        elif msg_type == "GLOBAL_POSITION_INT":

            print(
                "GLOBAL_POSITION    "
                f"relative_alt="
                f"{msg.relative_alt / 1000.0:+7.2f} m "
                f"vx={msg.vx / 100.0:+6.2f} "
                f"vy={msg.vy / 100.0:+6.2f} "
                f"vz={msg.vz / 100.0:+6.2f} m/s"
            )

        elif msg_type == "GPS_RAW_INT":

            satellites = getattr(
                msg,
                "satellites_visible",
                255,
            )

            print(
                "GPS_RAW_INT         "
                f"fix={msg.fix_type} "
                f"sats={satellites}"
            )

        elif msg_type == "SYS_STATUS":

            voltage = (
                msg.voltage_battery / 1000.0
                if msg.voltage_battery != 65535
                else float("nan")
            )

            print(
                "SYS_STATUS          "
                f"battery={voltage:.2f} V"
            )

    print()
    print("Message summary")
    print("---------------")

    for name in sorted(counts):

        print(
            f"{name:<30} {counts[name]}"
        )

    print()
    print("READ-ONLY TEST COMPLETE")
    print("No vehicle-control commands were sent.")


if __name__ == "__main__":
    main()
