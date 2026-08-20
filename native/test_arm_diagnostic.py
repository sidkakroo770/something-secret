#!/usr/bin/env python3

from __future__ import annotations

import time

from pymavlink import mavutil


PORT = (
    "/dev/serial/by-id/"
    "usb-ArduPilot_fmuv3_170034000351333531383033-if00"
)


def result_name(value: int) -> str:
    try:
        return mavutil.mavlink.enums[
            "MAV_RESULT"
        ][value].name
    except Exception:
        return f"UNKNOWN({value})"


def is_armed(heartbeat) -> bool:
    return bool(
        heartbeat.base_mode
        & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
    )


def main():

    print()
    print("==============================================")
    print(" ARDUPILOT NORMAL-ARM DIAGNOSTIC")
    print("==============================================")
    print()
    print("No force-arm will be used.")
    print("If arming succeeds, automatic disarm follows.")
    print()

    master = mavutil.mavlink_connection(
        PORT,
        baud=115200,
        source_system=255,
        autoreconnect=True,
    )

    print("Waiting for heartbeat...")

    hb = master.wait_heartbeat(
        timeout=10
    )

    if hb is None:
        raise SystemExit(
            "No Pixhawk heartbeat."
        )

    print(
        f"Connected: sys={master.target_system} "
        f"comp={master.target_component}"
    )

    print(
        "Mode:",
        mavutil.mode_string_v10(hb),
    )

    print(
        "Initially armed:",
        is_armed(hb),
    )

    # Drain older messages first so we mainly see messages caused
    # by this arm attempt.
    drain_until = time.monotonic() + 1.0

    while time.monotonic() < drain_until:
        master.recv_match(
            blocking=False
        )
        time.sleep(0.01)

    print()
    print("Sending NORMAL ARM request...")

    master.mav.command_long_send(
        master.target_system,
        0,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,

        # ARM
        1.0,

        # NORMAL ARM -- do not bypass checks
        0.0,

        0,
        0,
        0,
        0,
        0,
    )

    deadline = time.monotonic() + 8.0

    armed = False

    print()
    print("Responses:")
    print("----------------------------------------------")

    try:

        while time.monotonic() < deadline:

            msg = master.recv_match(
                type=[
                    "STATUSTEXT",
                    "COMMAND_ACK",
                    "HEARTBEAT",
                ],
                blocking=True,
                timeout=0.5,
            )

            if msg is None:
                continue

            msg_type = msg.get_type()

            if msg_type == "STATUSTEXT":

                text = str(msg.text).rstrip("\x00")

                print(
                    f"STATUSTEXT severity={msg.severity}: "
                    f"{text}"
                )

            elif msg_type == "COMMAND_ACK":

                if (
                    msg.command
                    == mavutil.mavlink.
                    MAV_CMD_COMPONENT_ARM_DISARM
                ):

                    print(
                        "COMMAND_ACK ARM: "
                        f"{result_name(msg.result)} "
                        f"({msg.result})"
                    )

            elif msg_type == "HEARTBEAT":

                armed = is_armed(msg)

                if armed:

                    print()
                    print("*** ARMED CONFIRMED ***")
                    print(
                        "Will automatically disarm "
                        "after 2 seconds."
                    )

                    time.sleep(2.0)
                    break

    finally:

        if armed:

            print()
            print("Sending normal DISARM...")

            master.mav.command_long_send(
                master.target_system,
                0,
                mavutil.mavlink.
                MAV_CMD_COMPONENT_ARM_DISARM,
                0,

                0.0,   # DISARM
                0.0,   # no force
                0,
                0,
                0,
                0,
                0,
            )

            end = time.monotonic() + 5.0

            while time.monotonic() < end:

                msg = master.recv_match(
                    type="HEARTBEAT",
                    blocking=True,
                    timeout=0.5,
                )

                if (
                    msg is not None
                    and not is_armed(msg)
                ):

                    print(
                        "DISARM CONFIRMED"
                    )
                    break

        master.close()

    print()
    print("----------------------------------------------")

    if not armed:

        print(
            "Arming was rejected."
        )

        print(
            "The STATUSTEXT lines above contain "
            "the reason we need to fix."
        )


if __name__ == "__main__":
    main()
