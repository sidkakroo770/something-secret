#!/usr/bin/env python3

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np

from native.common.types import (
    Attitude,
    BodyVelocity,
    ControllerOutput,
    MissionState,
    NativeScan,
)

from native.controllers.pre_entry import (
    CorridorGeometry,
    PreEntryConfig,
    PreEntryController,
    clamp,
    linear_score,
)


# ============================================================
# Configuration
# ============================================================

@dataclass
class CruiseConfig(PreEntryConfig):

    # Cruise looks slightly farther ahead than PRE_ENTRY.
    fit_x_max_m: float = 5.5

    # Verified Gazebo setting.
    # Wider than the original 18-degree default.
    front_cone_deg: float = 35.0

    # Confidence
    nominal_confidence: float = 0.78
    control_confidence_min: float = 0.60

    low_confidence_trigger: float = 0.60
    low_confidence_confirm_scans: int = 3

    strict_geometry_loss_confirm_scans: int = 3

    # Front obstacle
    front_slowdown_start_m: float = 2.50
    front_obstacle_trigger_m: float = 1.35
    front_emergency_stop_m: float = 0.75
    obstacle_confirm_scans: int = 2

    # Normal cruise
    nominal_forward_speed_m_s: float = 0.35
    minimum_forward_speed_m_s: float = 0.10

    k_yaw: float = 0.55
    max_yaw_rate_deg_s: float = 6.0
    yaw_deadband_deg: float = 0.8

    k_lateral: float = 0.30
    max_lateral_speed_m_s: float = 0.12
    lateral_deadband_m: float = 0.04

    # Integrated correction
    correction_enter_yaw_deg: float = 5.0
    correction_exit_yaw_deg: float = 2.5

    correction_enter_lateral_m: float = 0.22
    correction_exit_lateral_m: float = 0.10

    correction_enter_confirm_scans: int = 2
    correction_release_confirm_scans: int = 4

    correction_forward_speed_m_s: float = 0.10

    correction_k_yaw: float = 0.85
    correction_max_yaw_rate_deg_s: float = 8.0

    correction_k_lateral: float = 0.48
    correction_max_lateral_speed_m_s: float = 0.18

    correction_yaw_priority_deg: float = 7.0

    correction_timeout_s: float = 8.0

    correction_extreme_yaw_deg: float = 18.0

    # Important:
    # Your final Gazebo params increased this from 0.70 to 1.10 m
    # because a successful obstacle bypass can intentionally leave the
    # aircraft ~1 m from corridor centre.
    correction_extreme_lateral_m: float = 1.10

    correction_worsening_ratio: float = 1.35
    correction_worsening_confirm_scans: int = 3

    # Exit detection
    exit_guard_time_s: float = 1.5
    exit_side_open_m: float = 2.40
    exit_front_open_m: float = 3.00

    exit_probe_speed_m_s: float = 0.12

    exit_candidate_confirm_scans: int = 3
    exit_precursor_max_scans: int = 20

    # Cruise permits slightly more vehicle tilt.
    max_tilt_deg: float = 10.0


# ============================================================
# Controller state
# ============================================================

class CruiseState(Enum):
    IDLE = auto()
    ACTIVE = auto()
    TRANSITION_REQUESTED = auto()


class CruiseMode(Enum):
    NOMINAL = auto()
    CORRECTING = auto()


# ============================================================
# Native CORRIDOR_CRUISE
# ============================================================

class CorridorCruiseController(PreEntryController):

    def __init__(
        self,
        config: Optional[CruiseConfig] = None,
    ) -> None:

        if config is None:
            config = CruiseConfig()

        super().__init__(config)

        self.config: CruiseConfig = config

        # Cruise ROS implementation used deterministic seed 7.
        self.rng = np.random.default_rng(7)

        self.cruise_state = CruiseState.IDLE
        self.mode = CruiseMode.NOMINAL

        self.transition_target: Optional[MissionState] = None
        self.transition_reason: str = ""
        self.pause_reason: str = ""

        self.session_start_time = time.monotonic()

        self.obstacle_streak = 0
        self.low_conf_streak = 0
        self.strict_geometry_loss_streak = 0

        self.exit_streak = 0
        self.exit_precursor_streak = 0

        self.correction_enter_streak = 0
        self.correction_release_streak = 0

        self.correction_start_time: Optional[float] = None
        self.correction_best_metric = math.inf
        self.correction_worsening_streak = 0

    # ========================================================
    # Lifecycle
    # ========================================================

    def enter(self) -> None:

        self.reset_cruise()

        self.cruise_state = CruiseState.ACTIVE

        print(
            "[CRUISE] enabled -> ACTIVE/NOMINAL"
        )

    def reset_cruise(self) -> None:

        # Reset inherited geometry filters.
        super().reset()

        self.cruise_state = CruiseState.IDLE
        self.mode = CruiseMode.NOMINAL

        self.transition_target = None
        self.transition_reason = ""
        self.pause_reason = ""

        self.session_start_time = time.monotonic()

        self.obstacle_streak = 0
        self.low_conf_streak = 0
        self.strict_geometry_loss_streak = 0

        self.exit_streak = 0
        self.exit_precursor_streak = 0

        self.correction_enter_streak = 0
        self.correction_release_streak = 0

        self.correction_start_time = None
        self.correction_best_metric = math.inf
        self.correction_worsening_streak = 0

        self.latest_command = BodyVelocity.stop()

    def session_age(self) -> float:

        return (
            time.monotonic()
            - self.session_start_time
        )

    # ========================================================
    # Transition handling
    # ========================================================

    def request_transition(
        self,
        target: MissionState,
        reason: str,
        pause_reason: str = "",
    ) -> None:

        if (
            self.cruise_state
            == CruiseState.TRANSITION_REQUESTED
        ):
            return

        self.transition_target = target
        self.transition_reason = reason
        self.pause_reason = pause_reason

        self.cruise_state = (
            CruiseState.TRANSITION_REQUESTED
        )

        self.latest_command = BodyVelocity.stop()

        print(
            f"[CRUISE] -> {target.value}: "
            f"{reason}"
        )

    # ========================================================
    # Main public interface
    # ========================================================

    def step(
        self,
        scan: NativeScan,
        attitude: Optional[Attitude] = None,
    ) -> ControllerOutput:

        if (
            self.cruise_state
            == CruiseState.TRANSITION_REQUESTED
        ):
            return self.output()

        if self.cruise_state != CruiseState.ACTIVE:
            self.latest_command = BodyVelocity.stop()
            return self.output()

        # Native equivalent of ROS LaserScan watchdog.
        if scan.age_s > self.config.scan_stale_s:

            self.request_transition(
                MissionState.HOVER_AND_REASSESS,
                "D500 scan stale",
                "LOW_CONFIDENCE_GEOMETRY",
            )

            return self.output()

        g = self.extract_cruise_geometry(scan)

        self.last_geometry = g

        self.step_cruise(
            g,
            attitude,
        )

        return self.output()

    # ========================================================
    # Geometry + exit classification
    # ========================================================

    def extract_cruise_geometry(
        self,
        scan: NativeScan,
    ) -> CorridorGeometry:

        # Reuse the same verified:
        #
        #   XY conversion
        #   RANSAC
        #   wall pairing
        #   width estimate
        #   lateral estimate
        #   yaw estimate
        #   sector validation
        #   confidence
        #
        # from native PRE_ENTRY.
        g = super().extract_corridor_geometry(scan)

        ranges = np.asarray(
            scan.ranges_m,
            dtype=np.float64,
        )

        angles = np.asarray(
            scan.angles_rad,
            dtype=np.float64,
        )

        (
            side_open_left,
            side_open_right,
        ) = self.classify_side_opening(
            ranges,
            angles,
            scan.range_min_m,
            scan.range_max_m,
        )

        # Dataclasses are intentionally not slotted, so the
        # cruise-specific event fields can be attached here
        # without duplicating the geometry estimator.
        g.side_open_left = side_open_left
        g.side_open_right = side_open_right

        front_open = (
            g.front_clearance
            >= self.config.exit_front_open_m
        )

        exit_guard_passed = (
            self.session_age()
            >= self.config.exit_guard_time_s
        )

        g.exit_candidate = bool(
            exit_guard_passed
            and front_open
            and side_open_left
            and side_open_right
        )

        return g

    def classify_side_opening(
        self,
        ranges: np.ndarray,
        angles: np.ndarray,
        range_min: float,
        range_max: float,
    ) -> tuple[bool, bool]:

        c = self.config

        threshold = c.exit_side_open_m

        half_cone = math.radians(
            c.sector_cone_deg
        )

        max_sector = min(
            range_max,
            c.sector_max_range_m,
        )

        valid = np.isfinite(ranges)

        valid &= ranges >= max(
            range_min,
            0.05,
        )

        valid &= ranges <= max_sector

        def side_open(
            target_deg: float,
        ) -> bool:

            target = math.radians(
                target_deg
            )

            delta = np.abs(
                np.arctan2(
                    np.sin(
                        angles - target
                    ),
                    np.cos(
                        angles - target
                    ),
                )
            )

            values = ranges[
                valid
                & (delta <= half_cone)
            ]

            # No finite wall return in the side cone = open.
            if values.size == 0:
                return True

            return (
                float(
                    np.percentile(
                        values,
                        50.0,
                    )
                )
                >= threshold
            )

        return (
            side_open(+90.0),
            side_open(-90.0),
        )

    # ========================================================
    # Main cruise FSM
    # ========================================================

    def step_cruise(
        self,
        g: CorridorGeometry,
        attitude: Optional[Attitude],
    ) -> None:

        c = self.config

        # ----------------------------------------------------
        # 0 — attitude safety
        # ----------------------------------------------------

        if not self.attitude_is_safe(attitude):

            self.request_transition(
                MissionState.HOVER_AND_REASSESS,
                "roll/pitch exceeds cruise geometry limit",
                "LOW_CONFIDENCE_GEOMETRY",
            )

            return

        # ----------------------------------------------------
        # 1 — obstacle detection
        # ----------------------------------------------------

        if (
            0.0
            < g.front_clearance
            <= c.front_emergency_stop_m
        ):

            self.request_transition(
                MissionState.OBSTACLE_DECISION,
                (
                    "front emergency clearance "
                    f"{g.front_clearance:.2f} m"
                ),
            )

            return

        if (
            0.0
            < g.front_clearance
            <= c.front_obstacle_trigger_m
        ):

            self.obstacle_streak += 1

            self.latest_command = BodyVelocity.stop()

            if (
                self.obstacle_streak
                >= c.obstacle_confirm_scans
            ):

                self.request_transition(
                    MissionState.OBSTACLE_DECISION,
                    (
                        "front obstacle confirmed at "
                        f"{g.front_clearance:.2f} m"
                    ),
                )

            return

        self.obstacle_streak = 0

        # ----------------------------------------------------
        # 2 — strong exit candidate
        # ----------------------------------------------------

        if getattr(
            g,
            "exit_candidate",
            False,
        ):

            self.exit_streak += 1

            if (
                self.exit_streak
                >= c.exit_candidate_confirm_scans
            ):

                self.request_transition(
                    MissionState.EXIT_DETECTION,
                    "persistent corridor-opening candidate",
                )

                return

            self.latest_command = BodyVelocity(
                vx_m_s=max(
                    0.0,
                    c.exit_probe_speed_m_s,
                )
            )

            return

        self.exit_streak = 0

        # ----------------------------------------------------
        # 2B — weak exit precursor
        # ----------------------------------------------------

        exit_guard_passed = (
            self.session_age()
            >= c.exit_guard_time_s
        )

        front_open = (
            g.front_clearance
            >= c.exit_front_open_m
        )

        side_opening_started = (
            getattr(
                g,
                "side_open_left",
                False,
            )
            or getattr(
                g,
                "side_open_right",
                False,
            )
        )

        exit_precursor = (
            exit_guard_passed
            and front_open
            and side_opening_started
        )

        if exit_precursor:

            self.exit_precursor_streak += 1

            # Don't treat deliberate exit probing as
            # ordinary geometry failure.
            self.low_conf_streak = 0
            self.strict_geometry_loss_streak = 0

            if (
                self.exit_precursor_streak
                > c.exit_precursor_max_scans
            ):

                self.request_transition(
                    MissionState.HOVER_AND_REASSESS,
                    (
                        "exit precursor failed to become "
                        "full corridor opening"
                    ),
                    "LOW_CONFIDENCE_GEOMETRY",
                )

                return

            self.latest_command = BodyVelocity(
                vx_m_s=max(
                    0.0,
                    c.exit_probe_speed_m_s,
                )
            )

            return

        self.exit_precursor_streak = 0

        # ----------------------------------------------------
        # 3 — geometry confidence
        # ----------------------------------------------------

        geometry_bad = (
            not g.loose_valid
            or g.confidence
            < c.low_confidence_trigger
            or g.front_clearance <= 0.0
        )

        if geometry_bad:

            self.low_conf_streak += 1

            self.latest_command = BodyVelocity.stop()

            if (
                self.low_conf_streak
                >= c.low_confidence_confirm_scans
            ):

                self.request_transition(
                    MissionState.HOVER_AND_REASSESS,
                    (
                        "wall estimates / geometry confidence "
                        "became unreliable"
                    ),
                    "LOW_CONFIDENCE_GEOMETRY",
                )

            return

        self.low_conf_streak = 0

        # ----------------------------------------------------
        # 4 — strict geometry loss
        # ----------------------------------------------------

        if not g.strict_valid:

            self.strict_geometry_loss_streak += 1

            self.latest_command = BodyVelocity.stop()

            if (
                self.strict_geometry_loss_streak
                >= c.strict_geometry_loss_confirm_scans
            ):

                front_open = (
                    g.front_clearance
                    >= c.exit_front_open_m
                )

                if (
                    g.loose_valid
                    and front_open
                ):

                    self.request_transition(
                        MissionState.EXIT_DETECTION,
                        (
                            "strict wall geometry weakening "
                            "near possible corridor end"
                        ),
                    )

                else:

                    self.request_transition(
                        MissionState.HOVER_AND_REASSESS,
                        (
                            "strict wall geometry unavailable "
                            "for control"
                        ),
                        "LOW_CONFIDENCE_GEOMETRY",
                    )

            return

        self.strict_geometry_loss_streak = 0

        # ----------------------------------------------------
        # 5 — decide whether correction is needed
        # ----------------------------------------------------

        enter_yaw = math.radians(
            c.correction_enter_yaw_deg
        )

        correction_needed = (
            abs(g.yaw_error) >= enter_yaw
            or abs(g.lateral_error)
            >= c.correction_enter_lateral_m
        )

        if self.mode == CruiseMode.NOMINAL:

            if correction_needed:

                self.correction_enter_streak += 1

                if (
                    self.correction_enter_streak
                    >= c.correction_enter_confirm_scans
                ):

                    self.enter_correction_mode(
                        g,
                        (
                            f"lat={g.lateral_error:+.2f} m, "
                            f"yaw="
                            f"{math.degrees(g.yaw_error):+.1f} deg"
                        ),
                    )

            else:
                self.correction_enter_streak = 0

        # ----------------------------------------------------
        # 6 — integrated correction
        # ----------------------------------------------------

        if self.mode == CruiseMode.CORRECTING:

            extreme_yaw = math.radians(
                c.correction_extreme_yaw_deg
            )

            if (
                abs(g.yaw_error)
                >= extreme_yaw
                or abs(g.lateral_error)
                >= c.correction_extreme_lateral_m
            ):

                self.request_transition(
                    MissionState.HOVER_AND_REASSESS,
                    (
                        "correction exceeded safe envelope: "
                        f"lat={g.lateral_error:+.2f} m, "
                        f"yaw="
                        f"{math.degrees(g.yaw_error):+.1f} deg"
                    ),
                    "RECENTER_FAILED",
                )

                return

            metric = self.correction_metric(g)

            if metric < self.correction_best_metric:

                self.correction_best_metric = metric
                self.correction_worsening_streak = 0

            else:

                if (
                    math.isfinite(
                        self.correction_best_metric
                    )
                    and metric
                    > (
                        self.correction_best_metric
                        * c.correction_worsening_ratio
                    )
                ):

                    self.correction_worsening_streak += 1

                else:

                    self.correction_worsening_streak = 0

            if (
                self.correction_worsening_streak
                >= c.correction_worsening_confirm_scans
            ):

                self.request_transition(
                    MissionState.HOVER_AND_REASSESS,
                    (
                        "integrated centering/yaw correction "
                        "is worsening"
                    ),
                    "RECENTER_FAILED",
                )

                return

            if (
                self.correction_age()
                > c.correction_timeout_s
            ):

                self.request_transition(
                    MissionState.HOVER_AND_REASSESS,
                    (
                        "integrated centering/yaw correction "
                        "timed out"
                    ),
                    "RECENTER_FAILED",
                )

                return

            restored = (
                abs(g.yaw_error)
                <= math.radians(
                    c.correction_exit_yaw_deg
                )
                and abs(g.lateral_error)
                <= c.correction_exit_lateral_m
            )

            if restored:

                self.correction_release_streak += 1

                if (
                    self.correction_release_streak
                    >= c.correction_release_confirm_scans
                ):
                    self.leave_correction_mode()

            else:
                self.correction_release_streak = 0

            if self.mode == CruiseMode.CORRECTING:

                self.latest_command = (
                    self.correction_command(g)
                )

                return

        # ----------------------------------------------------
        # 7 — nominal cruise
        # ----------------------------------------------------

        if (
            g.confidence
            < c.control_confidence_min
        ):

            self.latest_command = BodyVelocity.stop()
            return

        self.latest_command = BodyVelocity(
            vx_m_s=self.adaptive_forward_speed(g),

            vy_m_s=self.cruise_lateral_control(
                g.lateral_error
            ),

            yaw_rate_rad_s=self.cruise_yaw_control(
                g.yaw_error
            ),
        )

    # ========================================================
    # Nominal controls
    # ========================================================

    def cruise_yaw_control(
        self,
        error_rad: float,
    ) -> float:

        c = self.config

        deadband = math.radians(
            c.yaw_deadband_deg
        )

        if abs(error_rad) <= deadband:
            return 0.0

        maximum = math.radians(
            c.max_yaw_rate_deg_s
        )

        return clamp(
            c.k_yaw * error_rad,
            -maximum,
            maximum,
        )

    def cruise_lateral_control(
        self,
        error_m: float,
    ) -> float:

        c = self.config

        if (
            abs(error_m)
            <= c.lateral_deadband_m
        ):
            return 0.0

        return clamp(
            c.k_lateral * error_m,
            -c.max_lateral_speed_m_s,
            c.max_lateral_speed_m_s,
        )

    # ========================================================
    # Adaptive forward velocity
    # ========================================================

    def adaptive_forward_speed(
        self,
        g: CorridorGeometry,
    ) -> float:

        c = self.config

        nominal = c.nominal_forward_speed_m_s

        minimum = clamp(
            c.minimum_forward_speed_m_s,
            0.0,
            nominal,
        )

        # Front-clearance factor.
        if (
            g.front_clearance
            >= c.front_slowdown_start_m
        ):

            front_factor = 1.0

        elif (
            g.front_clearance
            <= c.front_obstacle_trigger_m
        ):

            front_factor = 0.0

        else:

            front_factor = (
                (
                    g.front_clearance
                    - c.front_obstacle_trigger_m
                )
                /
                max(
                    c.front_slowdown_start_m
                    - c.front_obstacle_trigger_m,
                    1e-6,
                )
            )

        # Confidence factor.
        if (
            g.confidence
            >= c.nominal_confidence
        ):

            confidence_factor = 1.0

        elif (
            g.confidence
            <= c.control_confidence_min
        ):

            confidence_factor = 0.25

        else:

            confidence_factor = (
                0.25
                + 0.75
                * (
                    (
                        g.confidence
                        - c.control_confidence_min
                    )
                    /
                    max(
                        c.nominal_confidence
                        - c.control_confidence_min,
                        1e-6,
                    )
                )
            )

        # Alignment factor.
        yaw_trigger = math.radians(
            c.correction_enter_yaw_deg
        )

        yaw_factor = linear_score(
            abs(g.yaw_error),
            math.radians(1.5),
            yaw_trigger,
        )

        lateral_factor = linear_score(
            abs(g.lateral_error),
            0.06,
            c.correction_enter_lateral_m,
        )

        alignment_factor = min(
            yaw_factor,
            lateral_factor,
        )

        factor = clamp(
            min(
                front_factor,
                confidence_factor,
                alignment_factor,
            ),
            0.0,
            1.0,
        )

        if factor <= 0.0:
            return 0.0

        return clamp(
            max(
                minimum,
                nominal * factor,
            ),
            0.0,
            nominal,
        )

    # ========================================================
    # Integrated correction
    # ========================================================

    def correction_metric(
        self,
        g: CorridorGeometry,
    ) -> float:

        c = self.config

        yaw_scale = max(
            math.radians(
                c.correction_enter_yaw_deg
            ),
            math.radians(0.5),
        )

        lateral_scale = max(
            c.correction_enter_lateral_m,
            0.03,
        )

        return max(
            abs(g.yaw_error) / yaw_scale,
            abs(g.lateral_error) / lateral_scale,
        )

    def correction_age(self) -> float:

        if self.correction_start_time is None:
            return 0.0

        return (
            time.monotonic()
            - self.correction_start_time
        )

    def enter_correction_mode(
        self,
        g: CorridorGeometry,
        reason: str,
    ) -> None:

        if self.mode == CruiseMode.CORRECTING:
            return

        self.mode = CruiseMode.CORRECTING

        self.correction_start_time = (
            time.monotonic()
        )

        self.correction_release_streak = 0
        self.correction_worsening_streak = 0

        self.correction_best_metric = (
            self.correction_metric(g)
        )

        print(
            "[CRUISE] NOMINAL -> CORRECTING: "
            + reason
        )

    def leave_correction_mode(self) -> None:

        if self.mode != CruiseMode.CORRECTING:
            return

        self.mode = CruiseMode.NOMINAL

        self.correction_start_time = None
        self.correction_enter_streak = 0
        self.correction_release_streak = 0
        self.correction_worsening_streak = 0

        self.correction_best_metric = math.inf

        print(
            "[CRUISE] CORRECTING -> NOMINAL: "
            "alignment restored"
        )

    def correction_command(
        self,
        g: CorridorGeometry,
    ) -> BodyVelocity:

        c = self.config

        yaw_error = (
            0.0
            if abs(g.yaw_error)
            <= math.radians(
                c.yaw_deadband_deg
            )
            else g.yaw_error
        )

        lateral_error = (
            0.0
            if abs(g.lateral_error)
            <= c.lateral_deadband_m
            else g.lateral_error
        )

        yaw_rate = clamp(
            c.correction_k_yaw
            * yaw_error,

            -math.radians(
                c.correction_max_yaw_rate_deg_s
            ),

            math.radians(
                c.correction_max_yaw_rate_deg_s
            ),
        )

        yaw_priority = math.radians(
            c.correction_yaw_priority_deg
        )

        # Major yaw error: rotate in place first.
        if abs(g.yaw_error) >= yaw_priority:

            return BodyVelocity(
                vx_m_s=0.0,
                vy_m_s=0.0,
                yaw_rate_rad_s=yaw_rate,
            )

        vy = clamp(
            c.correction_k_lateral
            * lateral_error,

            -c.correction_max_lateral_speed_m_s,
            c.correction_max_lateral_speed_m_s,
        )

        requested_vx = max(
            0.0,
            c.correction_forward_speed_m_s,
        )

        vx = min(
            requested_vx,
            self.adaptive_forward_speed(g),
        )

        return BodyVelocity(
            vx_m_s=vx,
            vy_m_s=vy,
            yaw_rate_rad_s=yaw_rate,
        )

    # ========================================================
    # Native output
    # ========================================================

    def output(self) -> ControllerOutput:

        return ControllerOutput(
            command=self.latest_command,

            next_state=self.transition_target,

            status=(
                f"{self.cruise_state.name}/"
                f"{self.mode.name}"
            ),

            reason=self.transition_reason,

            confidence=(
                self.last_geometry.confidence
                if self.last_geometry is not None
                else None
            ),
        )

    # ========================================================
    # Diagnostics
    # ========================================================

    def diagnostics(self) -> dict:

        g = self.last_geometry

        result = {
            "state": self.cruise_state.name,
            "mode": self.mode.name,

            "transition_target": (
                self.transition_target.value
                if self.transition_target is not None
                else None
            ),

            "transition_reason":
                self.transition_reason,

            "pause_reason":
                self.pause_reason,

            "session_age_s":
                round(self.session_age(), 2),

            "obstacle_streak":
                self.obstacle_streak,

            "low_conf_streak":
                self.low_conf_streak,

            "strict_loss_streak":
                self.strict_geometry_loss_streak,

            "exit_streak":
                self.exit_streak,

            "exit_precursor_streak":
                self.exit_precursor_streak,
        }

        if g is None:
            return result

        result.update(
            {
                "strict_valid":
                    g.strict_valid,

                "loose_valid":
                    g.loose_valid,

                "confidence":
                    round(g.confidence, 3),

                "yaw_error_deg":
                    round(
                        math.degrees(
                            g.yaw_error
                        ),
                        2,
                    ),

                "lateral_error_m":
                    round(
                        g.lateral_error,
                        3,
                    ),

                "width_m":
                    (
                        round(g.width, 3)
                        if math.isfinite(g.width)
                        else None
                    ),

                "front_clearance_m":
                    round(
                        g.front_clearance,
                        3,
                    ),

                "side_open_left":
                    getattr(
                        g,
                        "side_open_left",
                        False,
                    ),

                "side_open_right":
                    getattr(
                        g,
                        "side_open_right",
                        False,
                    ),

                "exit_candidate":
                    getattr(
                        g,
                        "exit_candidate",
                        False,
                    ),
            }
        )

        return result
