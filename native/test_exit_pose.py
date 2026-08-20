#!/usr/bin/env python3

import math
import time
from dataclasses import dataclass

from native.common.types import (
    MissionState,
)

from native.controllers.exit_detection import (
    ExitDetectionController,
)


@dataclass
class FakePose:
    x_m: float
    y_m: float
    yaw_rad: float
    timestamp: float


def pose_along_heading(
    start_x,
    start_y,
    yaw,
    forward,
    sideways=0.0,
):

    # Forward vector in N/E.
    fx = math.cos(yaw)
    fy = math.sin(yaw)

    # Right-hand perpendicular vector.
    sx = -math.sin(yaw)
    sy = math.cos(yaw)

    return FakePose(
        x_m=(
            start_x
            + forward * fx
            + sideways * sx
        ),
        y_m=(
            start_y
            + forward * fy
            + sideways * sy
        ),
        yaw_rad=yaw,
        timestamp=time.monotonic(),
    )


def main():

    print()
    print("========================================")
    print(" EXIT REAL-POSE DISPLACEMENT TEST")
    print("========================================")
    print()

    yaw = math.radians(37.0)

    x0 = 10.0
    y0 = 20.0

    controller = (
        ExitDetectionController()
    )

    controller.enter()

    # --------------------------------------------------------
    # Start pose
    # --------------------------------------------------------

    p0 = FakePose(
        x_m=x0,
        y_m=y0,
        yaw_rad=yaw,
        timestamp=time.monotonic(),
    )

    out = controller.step(
        scan=None,
        pose=p0,
    )

    assert (
        out.next_state is None
    )

    assert (
        out.command.vx_m_s > 0.0
    )

    print(
        "Start pose latched: PASS"
    )

    # --------------------------------------------------------
    # Move 0.60 m forward
    # --------------------------------------------------------

    p1 = pose_along_heading(
        x0,
        y0,
        yaw,
        forward=0.60,
    )

    out = controller.step(
        scan=None,
        pose=p1,
    )

    assert (
        out.next_state is None
    )

    assert math.isclose(
        controller.measured_travel_m,
        0.60,
        abs_tol=1e-6,
    )

    print(
        "0.60 m measured forward: PASS"
    )

    # --------------------------------------------------------
    # Pure sideways movement should NOT count as forward exit.
    # --------------------------------------------------------

    p_side = pose_along_heading(
        x0,
        y0,
        yaw,
        forward=0.60,
        sideways=2.0,
    )

    out = controller.step(
        scan=None,
        pose=p_side,
    )

    assert (
        out.next_state is None
    )

    assert math.isclose(
        controller.measured_travel_m,
        0.60,
        abs_tol=1e-6,
    )

    print(
        "Sideways displacement ignored: PASS"
    )

    # --------------------------------------------------------
    # Reach real 1.21 m forward displacement
    # --------------------------------------------------------

    p2 = pose_along_heading(
        x0,
        y0,
        yaw,
        forward=1.21,
    )

    out = controller.step(
        scan=None,
        pose=p2,
    )

    assert (
        out.next_state
        == MissionState.CORRIDOR_EXITED
    )

    assert (
        out.command.vx_m_s == 0.0
    )

    print(
        "1.21 m completion: PASS"
    )

    print()
    print(
        "EXIT pose-distance test PASSED"
    )


if __name__ == "__main__":
    main()
