#!/usr/bin/env python3

from __future__ import annotations

from native.common.types import (
    BodyVelocity,
    ControllerOutput,
    VehicleAction,
)


class AbortCorridorController:
    """
    Terminal mission controller.

    ABORT_CORRIDOR never attempts to navigate or descend using
    companion-computer velocity commands.

    Instead:

        velocity = exactly zero
        action   = VehicleAction.LAND

    The future MAVLink command layer will translate LAND into the
    appropriate ArduPilot command.

    CURRENTLY THIS CLASS DOES NOT TALK TO THE PIXHAWK.
    """

    def __init__(self) -> None:
        self.active = False
        self.reason = ""

    def enter(
        self,
        reason: str = "",
    ) -> None:

        self.active = True
        self.reason = str(reason)

        print(
            "[ABORT] ABORT_CORRIDOR active: "
            f"{self.reason or 'unspecified reason'}"
        )

        print(
            "[ABORT] navigation commands cancelled"
        )

        print(
            "[ABORT] LAND action requested"
        )

    def reset(self) -> None:
        self.active = False
        self.reason = ""

    def step(
        self,
        landed: bool = False,
    ) -> ControllerOutput:

        # Absolute abort invariant:
        command = BodyVelocity.stop()

        if not self.active:

            return ControllerOutput(
                command=command,
                status="IDLE",
            )

        # Eventually this flag will come from
        # EXTENDED_SYS_STATE.landed_state.
        if landed:

            return ControllerOutput(
                command=command,
                status="LANDED",
                reason=self.reason,
                action=None,
            )

        return ControllerOutput(
            command=command,
            status="REQUEST_LAND",
            reason=self.reason,
            action=VehicleAction.LAND,
        )
