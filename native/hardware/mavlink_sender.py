#!/usr/bin/env python3

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

from native.common.types import (
    BodyVelocity,
    VehicleAction,
)

from native.hardware.mavlink_commands import (
    body_velocity_to_mavlink,
    vehicle_action_to_mavlink,
)


@dataclass(frozen=True)
class DispatchResult:
    """
    Result of attempting to dispatch one flight-controller command.
    """

    kind: str

    transmitted: bool
    blocked: bool

    reason: str

    plan: Optional[Any] = None


class MavlinkCommandSender:
    """
    Flight-control output boundary.

    IMPORTANT:

    Control transmission starts DISABLED.

    Velocity output additionally requires:
        - fresh MAVLink heartbeat
        - GUIDED mode
        - valid horizontal position estimate

    LAND does not require horizontal position because it is the
    terminal safety action.

    This class contains no arm command and no takeoff command.
    """

    ENABLE_CONFIRMATION = (
        "ENABLE_REAL_FLIGHT_CONTROL"
    )

    def __init__(
        self,
        mavlink_io,
    ) -> None:

        self.fc = mavlink_io

        self._control_enabled = False

        self._tx_lock = threading.Lock()

    # ========================================================
    # Control gate
    # ========================================================

    @property
    def control_enabled(self) -> bool:
        return self._control_enabled

    def enable_control(
        self,
        confirmation: str,
    ) -> None:

        if (
            confirmation
            != self.ENABLE_CONFIRMATION
        ):
            raise RuntimeError(
                "Real flight control NOT enabled. "
                "Explicit confirmation token required."
            )

        self._control_enabled = True

        print(
            "[MAVLINK TX] REAL FLIGHT CONTROL ENABLED"
        )

    def disable_control(self) -> None:

        self._control_enabled = False

        print(
            "[MAVLINK TX] flight control disabled"
        )

    # ========================================================
    # Helpers
    # ========================================================

    def _connection_ready(self) -> bool:

        if self.fc is None:
            return False

        if self.fc.master is None:
            return False

        return True

    # ========================================================
    # Body velocity
    # ========================================================

    def send_velocity(
        self,
        command: BodyVelocity,
    ) -> DispatchResult:

        plan = body_velocity_to_mavlink(
            command
        )

        # ----------------------------------------------------
        # HARD GATE
        # ----------------------------------------------------

        if not self._control_enabled:

            return DispatchResult(
                kind="BODY_VELOCITY",
                transmitted=False,
                blocked=False,
                reason="DRY_RUN_CONTROL_DISABLED",
                plan=plan,
            )

        # ----------------------------------------------------
        # Real-transmission safety checks
        # ----------------------------------------------------

        if not self._connection_ready():

            return DispatchResult(
                kind="BODY_VELOCITY",
                transmitted=False,
                blocked=True,
                reason="NO_MAVLINK_CONNECTION",
                plan=plan,
            )

        if not self.fc.heartbeat_ok():

            return DispatchResult(
                kind="BODY_VELOCITY",
                transmitted=False,
                blocked=True,
                reason="HEARTBEAT_STALE",
                plan=plan,
            )

        status = self.fc.status()

        if status.mode != "GUIDED":

            return DispatchResult(
                kind="BODY_VELOCITY",
                transmitted=False,
                blocked=True,
                reason=(
                    "VELOCITY_REQUIRES_GUIDED_MODE"
                ),
                plan=plan,
            )

        if not self.fc.horizontal_position_ok():

            return DispatchResult(
                kind="BODY_VELOCITY",
                transmitted=False,
                blocked=True,
                reason=(
                    "NO_VALID_HORIZONTAL_POSITION"
                ),
                plan=plan,
            )

        master = self.fc.master

        # Sender time in milliseconds.
        time_boot_ms = (
            int(time.monotonic() * 1000)
            & 0xFFFFFFFF
        )

        with self._tx_lock:

            master.mav.set_position_target_local_ned_send(
                time_boot_ms,

                master.target_system,

                # Flight controller or all components.
                0,

                plan.coordinate_frame,
                plan.type_mask,

                # Position fields ignored by mask.
                0.0,
                0.0,
                0.0,

                # Velocity.
                plan.vx_m_s,
                plan.vy_m_s,
                plan.vz_m_s,

                # Acceleration fields ignored.
                0.0,
                0.0,
                0.0,

                # Yaw ignored by mask.
                0.0,

                # Yaw rate used.
                plan.yaw_rate_rad_s,
            )

        return DispatchResult(
            kind="BODY_VELOCITY",
            transmitted=True,
            blocked=False,
            reason="TRANSMITTED",
            plan=plan,
        )

    # ========================================================
    # High-level actions
    # ========================================================

    def send_action(
        self,
        action: VehicleAction,
    ) -> DispatchResult:

        plan = vehicle_action_to_mavlink(
            action
        )

        # ----------------------------------------------------
        # HARD GATE
        # ----------------------------------------------------

        if not self._control_enabled:

            return DispatchResult(
                kind=action.value,
                transmitted=False,
                blocked=False,
                reason="DRY_RUN_CONTROL_DISABLED",
                plan=plan,
            )

        if not self._connection_ready():

            return DispatchResult(
                kind=action.value,
                transmitted=False,
                blocked=True,
                reason="NO_MAVLINK_CONNECTION",
                plan=plan,
            )

        if not self.fc.heartbeat_ok():

            return DispatchResult(
                kind=action.value,
                transmitted=False,
                blocked=True,
                reason="HEARTBEAT_STALE",
                plan=plan,
            )

        master = self.fc.master

        with self._tx_lock:

            master.mav.command_long_send(
                master.target_system,
                0,
                plan.command,
                plan.confirmation,
                *plan.params,
            )

        return DispatchResult(
            kind=action.value,
            transmitted=True,
            blocked=False,
            reason="TRANSMITTED",
            plan=plan,
        )
