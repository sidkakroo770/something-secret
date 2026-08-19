#!/usr/bin/env python3

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np

from native.common.types import (
    BodyVelocity,
    ControllerOutput,
    MissionState,
    NativeScan,
)


@dataclass
class ExitConfig:

    # Commit behaviour
    exit_forward_speed_m_s: float = 0.15
    exit_commit_distance_m: float = 1.20

    # Emergency front safety
    use_front_safety: bool = True

    front_cone_deg: float = 18.0
    front_stop_m: float = 0.60
    front_stop_confirm_scans: int = 2

    # Missing/stale LiDAR does NOT stop the committed exit.
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

        self.commit_start_monotonic: Optional[
            float
        ] = None

        self.transition_target: Optional[
            MissionState
        ] = None

        self.transition_reason = ""

        self.front_clearance_m: Optional[
            float
        ] = None

        self.last_scan_monotonic: Optional[
            float
        ] = None

        self.front_stop_streak = 0

        self.latest_command = (
            BodyVelocity.stop()
        )

    # ========================================================
    # Lifecycle
    # ========================================================

    def enter(self) -> None:

        self.state = ExitState.COMMIT_FORWARD

        self.commit_start_monotonic = (
            time.monotonic()
        )

        self.transition_target = None
        self.transition_reason = ""

        self.front_stop_streak = 0
        self.front_clearance_m = None
        self.last_scan_monotonic = None

        self.latest_command = (
            BodyVelocity.stop()
        )

        print(
            "[EXIT] entered COMMIT_FORWARD: "
            f"vx={self.forward_speed():.2f} m/s, "
            f"distance={self.commit_distance():.2f} m, "
            f"nominal time="
            f"{self.nominal_duration():.2f} s"
        )

    def reset(self) -> None:

        self.state = ExitState.IDLE

        self.commit_start_monotonic = None

        self.transition_target = None
        self.transition_reason = ""

        self.front_clearance_m = None
        self.last_scan_monotonic = None

        self.front_stop_streak = 0

        self.latest_command = (
            BodyVelocity.stop()
        )

    # ========================================================
    # Basic timing
    # ========================================================

    def forward_speed(self) -> float:

        # Prevent accidental zero/negative speed from
        # creating an infinite state.
        return max(
            0.05,
            self.config.exit_forward_speed_m_s,
        )

    def commit_distance(self) -> float:

        return max(
            0.20,
            self.config.exit_commit_distance_m,
        )

    def nominal_duration(self) -> float:

        return (
            self.commit_distance()
            / self.forward_speed()
        )

    def elapsed_s(self) -> float:

        if self.commit_start_monotonic is None:
            return 0.0

        return max(
            0.0,
            time.monotonic()
            - self.commit_start_monotonic,
        )

    def estimated_travel_m(self) -> float:

        return (
            self.forward_speed()
            * self.elapsed_s()
        )

    # ========================================================
    # LiDAR emergency safety
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

        # No-return / +inf means clear up to sensor max range.
        safe = ranges.copy()

        safe[np.isposinf(safe)] = (
            scan.range_max_m
        )

        valid = np.isfinite(safe)

        valid &= safe >= max(
            scan.range_min_m,
            0.05,
        )

        valid &= safe <= scan.range_max_m

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

        self.process_scan(scan)

        # process_scan may have triggered emergency transition.
        if self.transition_target is not None:

            self.latest_command = (
                BodyVelocity.stop()
            )

            return self.output()

        if self.state == ExitState.COMMIT_FORWARD:

            # CRITICAL:
            # no wall-confidence or side-wall requirement here.
            #
            # Walls disappearing is EXPECTED during corridor exit.
            self.latest_command = BodyVelocity(
                vx_m_s=self.forward_speed(),
                vy_m_s=0.0,
                vz_m_s=0.0,
                yaw_rate_rad_s=0.0,
            )

            if (
                self.estimated_travel_m()
                >= self.commit_distance()
            ):

                travelled = (
                    self.estimated_travel_m()
                )

                self.request_transition(
                    MissionState.CORRIDOR_EXITED,
                    (
                        f"committed forward "
                        f"{travelled:.2f} m "
                        f"(target "
                        f"{self.commit_distance():.2f} m)"
                    ),
                )

                # Stop immediately on completion.
                self.latest_command = (
                    BodyVelocity.stop()
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

        if target == MissionState.CORRIDOR_EXITED:

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

    def scan_age_s(self) -> Optional[float]:

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

            "estimated_commanded_travel_m":
                round(
                    self.estimated_travel_m(),
                    3,
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

            "front_safety_scan_fresh":
                self.fresh_front_safety_available(),

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
