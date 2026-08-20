#!/usr/bin/env python3

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Protocol

import numpy as np

from native.common.types import (
    BodyVelocity,
    ControllerOutput,
    MissionState,
    NativeScan,
)


# ============================================================
# Minimal pose interface
#
# mission_runner.VehiclePose already satisfies this.
# Using a Protocol avoids circular imports.
# ============================================================

class PoseLike(Protocol):
    x_m: float
    y_m: float
    yaw_rad: float
    timestamp: float


@dataclass
class ExitConfig:

    # Forward commit
    exit_forward_speed_m_s: float = 0.15
    exit_commit_distance_m: float = 1.20

    # Real position supervision
    pose_fresh_s: float = 0.50
    pose_loss_timeout_s: float = 1.00

    # If the aircraft cannot actually make the requested progress,
    # never drive forward forever.
    exit_hard_timeout_s: float = 15.0

    # Emergency front safety
    use_front_safety: bool = True
    front_cone_deg: float = 18.0
    front_stop_m: float = 0.60
    front_stop_confirm_scans: int = 2

    # Missing LiDAR alone does not invalidate the exit commit.
    # Position feedback IS required.
    scan_fresh_for_safety_s: float = 0.50


class ExitState(Enum):
    IDLE = auto()
    COMMIT_FORWARD = auto()
    COMPLETE = auto()
    HOLD = auto()


class ExitDetectionController:

    def __init__(
        self,
        config: Optional[ExitConfig] = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else ExitConfig()
        )

        self.state = ExitState.IDLE

        self.commit_start_monotonic: Optional[float] = None

        # Pose latched when real exit movement begins.
        self.start_x_m: Optional[float] = None
        self.start_y_m: Optional[float] = None
        self.start_yaw_rad: Optional[float] = None

        self.measured_travel_m: Optional[float] = None

        self.pose_missing_since: Optional[float] = None

        self.transition_target: Optional[MissionState] = None
        self.transition_reason = ""

        self.front_clearance_m: Optional[float] = None
        self.last_scan_monotonic: Optional[float] = None
        self.front_stop_streak = 0

        self.latest_command = BodyVelocity.stop()

    # ========================================================
    # Lifecycle
    # ========================================================

    def enter(self) -> None:

        self.state = ExitState.COMMIT_FORWARD

        self.commit_start_monotonic = time.monotonic()

        self.start_x_m = None
        self.start_y_m = None
        self.start_yaw_rad = None

        self.measured_travel_m = None
        self.pose_missing_since = None

        self.transition_target = None
        self.transition_reason = ""

        self.front_stop_streak = 0
        self.front_clearance_m = None
        self.last_scan_monotonic = None

        self.latest_command = BodyVelocity.stop()

        print(
            "[EXIT] entered COMMIT_FORWARD: "
            f"vx={self.forward_speed():.2f} m/s, "
            f"measured target={self.commit_distance():.2f} m"
        )

    def reset(self) -> None:

        self.state = ExitState.IDLE

        self.commit_start_monotonic = None

        self.start_x_m = None
        self.start_y_m = None
        self.start_yaw_rad = None

        self.measured_travel_m = None
        self.pose_missing_since = None

        self.transition_target = None
        self.transition_reason = ""

        self.front_clearance_m = None
        self.last_scan_monotonic = None
        self.front_stop_streak = 0

        self.latest_command = BodyVelocity.stop()

    # ========================================================
    # Timing
    # ========================================================

    def forward_speed(self) -> float:
        return max(
            0.05,
            self.config.exit_forward_speed_m_s,
        )

    def commit_distance(self) -> float:
        return max(
            0.20,
            self.config.exit_commit_distance_m,
        )

    def elapsed_s(self) -> float:

        if self.commit_start_monotonic is None:
            return 0.0

        return max(
            0.0,
            time.monotonic()
            - self.commit_start_monotonic,
        )

    # ========================================================
    # Pose / real displacement
    # ========================================================

    @staticmethod
    def pose_age_s(
        pose: PoseLike,
    ) -> float:

        return max(
            0.0,
            time.monotonic()
            - float(pose.timestamp),
        )

    def pose_is_fresh(
        self,
        pose: Optional[PoseLike],
    ) -> bool:

        if pose is None:
            return False

        return (
            self.pose_age_s(pose)
            <= self.config.pose_fresh_s
        )

    def latch_start_pose(
        self,
        pose: PoseLike,
    ) -> None:

        self.start_x_m = float(pose.x_m)
        self.start_y_m = float(pose.y_m)
        self.start_yaw_rad = float(
            pose.yaw_rad
        )

        self.measured_travel_m = 0.0

        print(
            "[EXIT] local start pose latched: "
            f"x={self.start_x_m:.2f}, "
            f"y={self.start_y_m:.2f}, "
            f"yaw="
            f"{math.degrees(self.start_yaw_rad):.1f}°"
        )

    def projected_travel(
        self,
        pose: PoseLike,
    ) -> float:

        if (
            self.start_x_m is None
            or self.start_y_m is None
            or self.start_yaw_rad is None
        ):
            return 0.0

        dx = (
            float(pose.x_m)
            - self.start_x_m
        )

        dy = (
            float(pose.y_m)
            - self.start_y_m
        )

        # LOCAL_POSITION_NED:
        #
        # x = North
        # y = East
        #
        # yaw=0 points North,
        # positive yaw rotates toward East.
        #
        # Project actual horizontal displacement onto the heading
        # present when EXIT_DETECTION started moving.
        return (
            math.cos(self.start_yaw_rad) * dx
            + math.sin(self.start_yaw_rad) * dy
        )

    # ========================================================
    # LiDAR front safety
    # ========================================================

    def extract_front_clearance(
        self,
        scan: NativeScan,
    ) -> Optional[float]:

        ranges = np.asarray(
            scan.ranges_m,
            dtype=np.float64,
        )

        angles = np.asarray(
            scan.angles_rad,
            dtype=np.float64,
        )

        if ranges.size == 0:
            return None

        safe = ranges.copy()

        # No return means clear up to LiDAR maximum range.
        safe[np.isposinf(safe)] = (
            scan.range_max_m
        )

        valid = np.isfinite(safe)

        valid &= safe >= max(
            scan.range_min_m,
            0.05,
        )

        valid &= (
            safe <= scan.range_max_m
        )

        half = math.radians(
            self.config.front_cone_deg
        )

        values = safe[
            valid
            & (np.abs(angles) <= half)
        ]

        if values.size == 0:
            return None

        return float(
            np.percentile(
                values,
                10.0,
            )
        )

    def process_scan(
        self,
        scan: Optional[NativeScan],
    ) -> None:

        c = self.config

        if scan is None:
            return

        self.last_scan_monotonic = (
            time.monotonic()
        )

        self.front_clearance_m = (
            self.extract_front_clearance(scan)
        )

        if (
            self.state
            != ExitState.COMMIT_FORWARD
        ):
            self.front_stop_streak = 0
            return

        if not c.use_front_safety:
            self.front_stop_streak = 0
            return

        if self.front_clearance_m is None:
            self.front_stop_streak = 0
            return

        if (
            0.0
            < self.front_clearance_m
            <= c.front_stop_m
        ):

            self.front_stop_streak += 1

            if (
                self.front_stop_streak
                >= c.front_stop_confirm_scans
            ):

                self.request_transition(
                    MissionState.HOVER_AND_REASSESS,
                    (
                        "unexpected front obstruction "
                        "during exit commit: "
                        f"{self.front_clearance_m:.2f} m"
                    ),
                )

        else:

            self.front_stop_streak = 0

    # ========================================================
    # Main update
    # ========================================================

    def step(
        self,
        scan: Optional[NativeScan] = None,
        pose: Optional[PoseLike] = None,
    ) -> ControllerOutput:

        if self.state == ExitState.IDLE:

            self.latest_command = (
                BodyVelocity.stop()
            )

            return self.output()

        if self.transition_target is not None:

            self.latest_command = (
                BodyVelocity.stop()
            )

            return self.output()

        # Emergency LiDAR check remains active.
        self.process_scan(scan)

        if self.transition_target is not None:

            self.latest_command = (
                BodyVelocity.stop()
            )

            return self.output()

        if (
            self.state
            != ExitState.COMMIT_FORWARD
        ):

            return self.output()

        # ----------------------------------------------------
        # Hard runtime guard.
        # ----------------------------------------------------

        if (
            self.elapsed_s()
            >= self.config.exit_hard_timeout_s
        ):

            self.request_transition(
                MissionState.HOVER_AND_REASSESS,
                (
                    "EXIT_DETECTION progress timeout "
                    f"after {self.elapsed_s():.1f} s"
                ),
            )

            return self.output()

        # ----------------------------------------------------
        # Position is mandatory for real exit completion.
        # ----------------------------------------------------

        if not self.pose_is_fresh(pose):

            self.latest_command = (
                BodyVelocity.stop()
            )

            now = time.monotonic()

            if self.pose_missing_since is None:
                self.pose_missing_since = now

            missing_age = (
                now
                - self.pose_missing_since
            )

            if (
                missing_age
                >= self.config.pose_loss_timeout_s
            ):

                self.request_transition(
                    MissionState.HOVER_AND_REASSESS,
                    (
                        "EXIT_DETECTION has no fresh "
                        "local-position feedback for "
                        f"{missing_age:.1f} s"
                    ),
                )

            return self.output()

        # Fresh position recovered.
        self.pose_missing_since = None

        assert pose is not None

        # First valid pose becomes the reference position.
        if self.start_x_m is None:

            self.latch_start_pose(pose)

        travelled = self.projected_travel(
            pose
        )

        self.measured_travel_m = (
            travelled
        )

        # ----------------------------------------------------
        # Actual displacement decides completion.
        # ----------------------------------------------------

        if (
            travelled
            >= self.commit_distance()
        ):

            self.request_transition(
                MissionState.CORRIDOR_EXITED,
                (
                    "measured forward displacement "
                    f"{travelled:.2f} m "
                    f"(target "
                    f"{self.commit_distance():.2f} m)"
                ),
            )

            self.latest_command = (
                BodyVelocity.stop()
            )

            return self.output()

        # ----------------------------------------------------
        # Continue the committed real movement.
        # ----------------------------------------------------

        self.latest_command = (
            BodyVelocity(
                vx_m_s=self.forward_speed(),
                vy_m_s=0.0,
                vz_m_s=0.0,
                yaw_rate_rad_s=0.0,
            )
        )

        return self.output()

    # ========================================================
    # Transition
    # ========================================================

    def request_transition(
        self,
        target: MissionState,
        reason: str,
    ) -> None:

        if self.transition_target is not None:
            return

        self.transition_target = target
        self.transition_reason = reason

        self.latest_command = (
            BodyVelocity.stop()
        )

        if (
            target
            == MissionState.CORRIDOR_EXITED
        ):

            self.state = ExitState.COMPLETE

        else:

            self.state = ExitState.HOLD

        print(
            f"[EXIT] -> {target.value}: "
            f"{reason}"
        )

    # ========================================================
    # Diagnostics
    # ========================================================

    def scan_age_s(
        self,
    ) -> Optional[float]:

        if self.last_scan_monotonic is None:
            return None

        return max(
            0.0,
            time.monotonic()
            - self.last_scan_monotonic,
        )

    def fresh_front_safety_available(
        self,
    ) -> bool:

        age = self.scan_age_s()

        if age is None:
            return False

        return (
            age
            <= self.config.scan_fresh_for_safety_s
        )

    def output(self) -> ControllerOutput:

        return ControllerOutput(
            command=self.latest_command,
            next_state=self.transition_target,
            status=self.state.name,
            reason=self.transition_reason,
            confidence=None,
        )

    def diagnostics(self) -> dict:

        return {
            "state":
                self.state.name,

            "exit_forward_speed_m_s":
                round(
                    self.forward_speed(),
                    3,
                ),

            "exit_commit_distance_m":
                round(
                    self.commit_distance(),
                    3,
                ),

            "elapsed_s":
                round(
                    self.elapsed_s(),
                    3,
                ),

            "measured_travel_m":
                (
                    round(
                        self.measured_travel_m,
                        3,
                    )
                    if self.measured_travel_m
                    is not None
                    else None
                ),

            "start_pose_latched":
                (
                    self.start_x_m
                    is not None
                ),

            "front_clearance_m":
                (
                    round(
                        self.front_clearance_m,
                        3,
                    )
                    if self.front_clearance_m
                    is not None
                    else None
                ),

            "front_stop_streak":
                self.front_stop_streak,

            "transition_target":
                (
                    self.transition_target.value
                    if self.transition_target
                    is not None
                    else None
                ),

            "transition_reason":
                self.transition_reason
                or None,
        }
