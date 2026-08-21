#!/usr/bin/env python3

from __future__ import annotations

import time

from pymavlink import mavutil


class FlightControllerManager:
    """
    High-level Pixhawk lifecycle manager.

    Responsibilities:
        - change flight mode
        - arm normally
        - disarm normally
        - request LAND

    Deliberately does NOT:
        - force arm
        - force disarm
        - send motor PWM
        - send throttle
        - send velocity targets
    """

    def __init__(self, mavlink_io) -> None:

        self.fc = mavlink_io

    # ========================================================
    # Basic health
    # ========================================================

    def _require_connection(self) -> None:

        if self.fc is None:
            raise RuntimeError(
                "MAVLink interface is None"
            )

        if self.fc.master is None:
            raise RuntimeError(
                "MAVLink connection is not open"
            )

        if not self.fc.heartbeat_ok():
            raise RuntimeError(
                "Pixhawk heartbeat is stale"
            )

    # ========================================================
    # Mode
    # ========================================================

    def request_mode(
        self,
        mode_name: str,
        timeout_s: float = 5.0,
    ) -> bool:

        self._require_connection()

        mode_name = (
            str(mode_name)
            .strip()
            .upper()
        )

        master = self.fc.master

        mapping = master.mode_mapping()

        if not mapping:
            raise RuntimeError(
                "Could not obtain ArduCopter mode mapping"
            )

        if mode_name not in mapping:
            raise ValueError(
                f"Unsupported mode {mode_name!r}. "
                f"Available modes: "
                f"{', '.join(sorted(mapping))}"
            )

        custom_mode = int(
            mapping[mode_name]
        )

        print(
            f"[FC] requesting mode {mode_name}"
        )

        master.mav.command_long_send(
            master.target_system,
            0,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            0,

            # Enable ArduPilot custom flight-mode number.
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,

            custom_mode,

            0,
            0,
            0,
            0,
            0,
        )

        deadline = (
            time.monotonic()
            + timeout_s
        )

        while (
            time.monotonic()
            < deadline
        ):

            status = self.fc.status()

            if (
                status.mode.upper()
                == mode_name
            ):

                print(
                    f"[FC] mode confirmed: "
                    f"{status.mode}"
                )

                return True

            time.sleep(0.1)

        print(
            f"[FC] mode change to "
            f"{mode_name} NOT confirmed"
        )

        return False

    # ========================================================
    # ARM
    # ========================================================

    def arm(
        self,
        timeout_s: float = 8.0,
    ) -> bool:

        self._require_connection()

        if self.fc.status().armed:

            print(
                "[FC] already armed"
            )

            return True

        master = self.fc.master

        print(
            "[FC] requesting NORMAL arm"
        )

        # Normal arm:
        #
        # param1 = 1
        # param2 = 0
        #
        # We deliberately DO NOT use 21196,
        # which would attempt to force-arm.
        master.mav.command_long_send(
            master.target_system,
            0,
            mavutil.mavlink.
            MAV_CMD_COMPONENT_ARM_DISARM,
            0,

            1.0,
            0.0,

            0,
            0,
            0,
            0,
            0,
        )

        deadline = (
            time.monotonic()
            + timeout_s
        )

        while (
            time.monotonic()
            < deadline
        ):

            if self.fc.status().armed:

                print(
                    "[FC] ARMED confirmed"
                )

                return True

            time.sleep(0.1)

        print(
            "[FC] arm NOT confirmed"
        )

        return False

    # ========================================================
    # DISARM
    # ========================================================

    def disarm(
        self,
        timeout_s: float = 5.0,
    ) -> bool:

        self._require_connection()

        if not self.fc.status().armed:

            print(
                "[FC] already disarmed"
            )

            return True

        master = self.fc.master

        print(
            "[FC] requesting NORMAL disarm"
        )

        # Normal on-ground disarm.
        # No force-disarm parameter.
        master.mav.command_long_send(
            master.target_system,
            0,
            mavutil.mavlink.
            MAV_CMD_COMPONENT_ARM_DISARM,
            0,

            0.0,
            0.0,

            0,
            0,
            0,
            0,
            0,
        )

        deadline = (
            time.monotonic()
            + timeout_s
        )

        while (
            time.monotonic()
            < deadline
        ):

            if not self.fc.status().armed:

                print(
                    "[FC] DISARMED confirmed"
                )

                return True

            time.sleep(0.1)

        print(
            "[FC] disarm NOT confirmed"
        )

        return False

    # ========================================================
    # LAND
    # ========================================================

    def land(self) -> None:

        self._require_connection()

        master = self.fc.master

        print(
            "[FC] requesting LAND"
        )

        master.mav.command_long_send(
            master.target_system,
            0,
            mavutil.mavlink.MAV_CMD_NAV_LAND,
            0,

            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )
