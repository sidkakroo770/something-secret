#!/usr/bin/env python3

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Deque, Dict, Optional

import numpy as np

from native.common.types import (
    Attitude,
    BodyVelocity,
    ControllerOutput,
    MissionState,
    NativeScan,
)


# ============================================================
# Helpers
# ============================================================

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wrap_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def linear_score(error: float, good: float, bad: float) -> float:
    if bad <= good:
        return 1.0 if error <= good else 0.0

    if error <= good:
        return 1.0

    if error >= bad:
        return 0.0

    return 1.0 - (error - good) / (bad - good)


# ============================================================
# Configuration
# ============================================================

@dataclass
class PreEntryConfig:

    # Geometry
    corridor_width_m: float = 3.5
    width_tolerance_m: float = 0.45
    loose_width_tolerance_m: float = 0.80

    max_parallel_error_deg: float = 7.0
    loose_parallel_error_deg: float = 15.0
    max_corridor_yaw_deg: float = 35.0

    geometry_max_range_m: float = 8.0

    # Wall point selection
    fit_x_min_m: float = -0.20
    fit_x_max_m: float = 5.0

    fit_side_min_m: float = 0.25
    fit_side_max_m: float = 3.5

    # RANSAC
    ransac_iterations: int = 120
    ransac_distance_m: float = 0.08

    min_wall_inliers: int = 14
    min_wall_span_m: float = 0.80
    max_fit_rms_m: float = 0.10
    max_wall_line_angle_deg: float = 45.0

    # Sector validator
    use_sector_validation: bool = True

    sector_cone_deg: float = 6.0
    sector_percentile: float = 35.0
    sector_filter_window: int = 5
    sector_max_range_m: float = 6.0

    sector_lr_tolerance_m: float = 0.35
    sector_diag_tolerance_m: float = 0.55

    # Geometry filtering / confidence
    geometry_filter_window: int = 3

    control_confidence_min: float = 0.62
    verify_confidence_min: float = 0.72

    # Safety
    front_cone_deg: float = 18.0
    front_stop_m: float = 0.80
    scan_stale_s: float = 0.30

    require_imu: bool = False
    max_tilt_deg: float = 8.0

    # Controller
    k_yaw: float = 0.9
    max_yaw_rate_deg_s: float = 10.0
    min_yaw_rate_deg_s: float = 1.5

    k_lateral: float = 0.45
    max_lateral_speed_m_s: float = 0.18
    min_lateral_speed_m_s: float = 0.03

    # Hysteresis
    yaw_realign_deg: float = 4.5
    yaw_aligned_deg: float = 2.5
    yaw_lock_deg: float = 3.0

    lateral_recenter_m: float = 0.18
    lateral_centered_m: float = 0.10
    lateral_lock_m: float = 0.12

    # Timing
    verify_scans: int = 6
    geometry_grace_scans: int = 2

    acquire_timeout_s: float = 8.0
    alignment_timeout_s: float = 15.0

    hold_recovery_scans: int = 5


# ============================================================
# Internal structures
# ============================================================

class LockState(Enum):
    IDLE = auto()
    ACQUIRE_GEOMETRY = auto()
    ALIGN_YAW = auto()
    CENTER_LATERALLY = auto()
    VERIFY_LOCK = auto()
    LOCKED = auto()
    HOLD = auto()


@dataclass
class LineModel:
    valid: bool = False

    point: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float64)
    )

    direction: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float64)
    )

    inlier_points: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float64)
    )

    rms: float = math.inf
    span: float = 0.0
    inlier_ratio: float = 0.0


@dataclass
class SectorSnapshot:
    values: Dict[str, Optional[float]] = field(default_factory=dict)
    front_clearance: float = 0.0


@dataclass
class CorridorGeometry:

    strict_valid: bool = False
    loose_valid: bool = False

    yaw_error_raw: float = 0.0
    lateral_error_raw: float = 0.0
    width_raw: float = math.inf

    yaw_error: float = 0.0
    lateral_error: float = 0.0
    width: float = math.inf

    d_left: float = math.inf
    d_right: float = math.inf

    parallel_error: float = math.inf

    left_rms: float = math.inf
    right_rms: float = math.inf

    left_inliers: int = 0
    right_inliers: int = 0

    left_span: float = 0.0
    right_span: float = 0.0

    left_inlier_ratio: float = 0.0
    right_inlier_ratio: float = 0.0

    front_clearance: float = 0.0

    sector_score: float = 0.5
    temporal_score: float = 0.7

    confidence: float = 0.0

    observed_lr_sum: Optional[float] = None
    expected_lr_sum: Optional[float] = None
    lr_sum_residual: Optional[float] = None

    sectors: Dict[str, Optional[float]] = field(default_factory=dict)


# ============================================================
# Sector validation
# ============================================================

class SectorMonitor:

    BEARINGS_DEG = {
        "L": 90.0,
        "FL": 45.0,
        "F": 0.0,
        "FR": -45.0,
        "R": -90.0,
    }

    def __init__(self, window: int) -> None:

        self.window = max(1, int(window))

        self.histories: Dict[str, Deque[float]] = {
            key: deque(maxlen=self.window)
            for key in self.BEARINGS_DEG
        }

    def reset(self) -> None:
        for history in self.histories.values():
            history.clear()

    @staticmethod
    def angle_difference(
        angles: np.ndarray,
        target: float,
    ) -> np.ndarray:

        return np.abs(
            np.arctan2(
                np.sin(angles - target),
                np.cos(angles - target),
            )
        )

    def ingest(
        self,
        ranges: np.ndarray,
        angles: np.ndarray,
        range_min: float,
        range_max: float,
        config: PreEntryConfig,
    ) -> SectorSnapshot:

        snapshot = SectorSnapshot()

        valid = np.isfinite(ranges)

        valid &= ranges >= max(
            float(range_min),
            0.05,
        )

        valid &= ranges <= min(
            float(range_max),
            config.sector_max_range_m,
        )

        half_cone = math.radians(
            config.sector_cone_deg
        )

        percentile = clamp(
            config.sector_percentile,
            0.0,
            100.0,
        )

        for name, bearing_deg in self.BEARINGS_DEG.items():

            target = math.radians(bearing_deg)

            mask = (
                valid
                & (
                    self.angle_difference(
                        angles,
                        target,
                    )
                    <= half_cone
                )
            )

            values = ranges[mask]

            if values.size:

                sample = float(
                    np.percentile(
                        values,
                        percentile,
                    )
                )

                self.histories[name].append(sample)

            else:
                self.histories[name].append(float("nan"))

            finite_history = [
                value
                for value in self.histories[name]
                if math.isfinite(value)
            ]

            snapshot.values[name] = (
                float(np.median(finite_history))
                if finite_history
                else None
            )

        # Conservative front safety measurement.
        safe_ranges = ranges.copy()

        safe_ranges[np.isposinf(safe_ranges)] = range_max

        front_valid = np.isfinite(safe_ranges)

        front_valid &= safe_ranges >= max(
            range_min,
            0.05,
        )

        front_valid &= safe_ranges <= range_max

        front_half = math.radians(
            config.front_cone_deg
        )

        front_mask = (
            front_valid
            & (np.abs(angles) <= front_half)
        )

        front_values = safe_ranges[front_mask]

        if front_values.size:

            snapshot.front_clearance = float(
                np.percentile(
                    front_values,
                    10.0,
                )
            )

        else:
            snapshot.front_clearance = 0.0

        return snapshot


# ============================================================
# Temporal geometry filtering
# ============================================================

class GeometryHistory:

    def __init__(self, window: int) -> None:

        window = max(1, int(window))

        self.yaw: Deque[float] = deque(maxlen=window)
        self.lateral: Deque[float] = deque(maxlen=window)
        self.width: Deque[float] = deque(maxlen=window)

    def reset(self) -> None:

        self.yaw.clear()
        self.lateral.clear()
        self.width.clear()

    def push(
        self,
        yaw: float,
        lateral: float,
        width: float,
    ) -> None:

        self.yaw.append(float(yaw))
        self.lateral.append(float(lateral))
        self.width.append(float(width))

    def medians(
        self,
    ) -> tuple[float, float, float]:

        return (
            float(np.median(self.yaw)),
            float(np.median(self.lateral)),
            float(np.median(self.width)),
        )

    def stability_score(self) -> float:

        if len(self.yaw) < 3:
            return 0.70

        yaw = np.asarray(
            self.yaw,
            dtype=np.float64,
        )

        lateral = np.asarray(
            self.lateral,
            dtype=np.float64,
        )

        width = np.asarray(
            self.width,
            dtype=np.float64,
        )

        yaw_mad = float(
            np.median(
                np.abs(
                    yaw - np.median(yaw)
                )
            )
        )

        lateral_mad = float(
            np.median(
                np.abs(
                    lateral - np.median(lateral)
                )
            )
        )

        width_mad = float(
            np.median(
                np.abs(
                    width - np.median(width)
                )
            )
        )

        yaw_score = linear_score(
            yaw_mad,
            math.radians(0.5),
            math.radians(4.0),
        )

        lateral_score = linear_score(
            lateral_mad,
            0.02,
            0.20,
        )

        width_score = linear_score(
            width_mad,
            0.03,
            0.25,
        )

        return float(
            (
                yaw_score
                + lateral_score
                + width_score
            )
            / 3.0
        )


# ============================================================
# Native PRE_ENTRY controller
# ============================================================

class PreEntryController:

    def __init__(
        self,
        config: Optional[PreEntryConfig] = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else PreEntryConfig()
        )

        self.state = LockState.IDLE

        now = time.monotonic()

        self.state_enter_time = now
        self.session_start_time = now

        self.alignment_start_time: Optional[float] = None

        self.failure_reason: Optional[str] = None

        self.stable_scan_count = 0
        self.invalid_scan_count = 0
        self.hold_recovery_count = 0

        self.last_geometry: Optional[CorridorGeometry] = None

        self.latest_command = BodyVelocity.stop()

        self.sector_monitor = SectorMonitor(
            self.config.sector_filter_window
        )

        self.geometry_history = GeometryHistory(
            self.config.geometry_filter_window
        )

        # Same deterministic seed as ROS implementation.
        self.rng = np.random.default_rng(2026)

    # ========================================================
    # Lifecycle
    # ========================================================

    def enter(self) -> None:

        self.reset()

        self.transition(
            LockState.ACQUIRE_GEOMETRY,
            "controller enabled",
        )

    def reset(self) -> None:

        now = time.monotonic()

        self.state = LockState.IDLE
        self.state_enter_time = now
        self.session_start_time = now

        self.alignment_start_time = None

        self.failure_reason = None

        self.stable_scan_count = 0
        self.invalid_scan_count = 0
        self.hold_recovery_count = 0

        self.last_geometry = None

        self.latest_command = BodyVelocity.stop()

        self.sector_monitor.reset()
        self.geometry_history.reset()

    def transition(
        self,
        new_state: LockState,
        reason: str,
    ) -> None:

        if new_state == self.state:
            return

        old = self.state

        self.state = new_state
        self.state_enter_time = time.monotonic()

        self.stable_scan_count = 0
        self.invalid_scan_count = 0

        if (
            new_state
            in (
                LockState.ALIGN_YAW,
                LockState.CENTER_LATERALLY,
                LockState.VERIFY_LOCK,
            )
            and self.alignment_start_time is None
        ):
            self.alignment_start_time = time.monotonic()

        if new_state != LockState.HOLD:
            self.hold_recovery_count = 0

        print(
            f"[PRE_ENTRY] "
            f"{old.name} -> {new_state.name}: "
            f"{reason}"
        )

    def fail(self, reason: str) -> None:

        self.failure_reason = reason
        self.latest_command = BodyVelocity.stop()

        self.transition(
            LockState.HOLD,
            f"FAILED: {reason}",
        )

    def state_age(self) -> float:
        return time.monotonic() - self.state_enter_time

    def alignment_age(self) -> float:

        if self.alignment_start_time is None:
            return 0.0

        return (
            time.monotonic()
            - self.alignment_start_time
        )

    # ========================================================
    # Main controller input
    # ========================================================

    def step(
        self,
        scan: NativeScan,
        attitude: Optional[Attitude] = None,
    ) -> ControllerOutput:

        # Scan watchdog.
        if scan.age_s > self.config.scan_stale_s:

            self.latest_command = BodyVelocity.stop()

            if self.state not in (
                LockState.IDLE,
                LockState.HOLD,
            ):
                self.transition(
                    LockState.HOLD,
                    "LiDAR scan stale",
                )

            return self.output()

        geometry = self.extract_corridor_geometry(scan)

        self.last_geometry = geometry

        self.step_fsm(
            geometry,
            attitude,
        )

        return self.output()

    # ========================================================
    # Safety
    # ========================================================

    def attitude_is_safe(
        self,
        attitude: Optional[Attitude],
    ) -> bool:

        if attitude is None:
            return not self.config.require_imu

        max_tilt = math.radians(
            self.config.max_tilt_deg
        )

        return (
            abs(attitude.roll_rad) <= max_tilt
            and abs(attitude.pitch_rad) <= max_tilt
        )

    # ========================================================
    # FSM
    # ========================================================

    def step_fsm(
        self,
        g: CorridorGeometry,
        attitude: Optional[Attitude],
    ) -> None:

        c = self.config

        if not self.attitude_is_safe(attitude):

            self.latest_command = BodyVelocity.stop()

            self.transition(
                LockState.HOLD,
                "roll/pitch exceeds tilt limit",
            )

            return

        if g.front_clearance < c.front_stop_m:

            self.latest_command = BodyVelocity.stop()

            self.transition(
                LockState.HOLD,
                "front clearance below stop distance",
            )

            return

        yaw_realign = math.radians(
            c.yaw_realign_deg
        )

        yaw_aligned = math.radians(
            c.yaw_aligned_deg
        )

        yaw_lock = math.radians(
            c.yaw_lock_deg
        )

        if self.state == LockState.IDLE:

            self.latest_command = BodyVelocity.stop()
            return

        # ----------------------------------------------------
        # ACQUIRE
        # ----------------------------------------------------

        if self.state == LockState.ACQUIRE_GEOMETRY:

            self.latest_command = BodyVelocity.stop()

            if (
                g.strict_valid
                and g.confidence
                >= c.control_confidence_min
            ):

                if abs(g.yaw_error) > yaw_aligned:

                    self.transition(
                        LockState.ALIGN_YAW,
                        "trustworthy walls acquired",
                    )

                elif (
                    abs(g.lateral_error)
                    > c.lateral_centered_m
                ):

                    self.transition(
                        LockState.CENTER_LATERALLY,
                        "yaw acceptable; centre laterally",
                    )

                else:

                    self.transition(
                        LockState.VERIFY_LOCK,
                        "already near alignment",
                    )

                return

            if self.state_age() > c.acquire_timeout_s:

                self.fail(
                    "LOW_CONFIDENCE_GEOMETRY"
                )

            return

        # ----------------------------------------------------
        # YAW
        # ----------------------------------------------------

        if self.state == LockState.ALIGN_YAW:

            if (
                not g.strict_valid
                or g.confidence
                < c.control_confidence_min
            ):

                self.invalid_scan_count += 1

                self.latest_command = BodyVelocity.stop()

                if (
                    self.invalid_scan_count
                    > c.geometry_grace_scans
                ):
                    self.transition(
                        LockState.HOLD,
                        "geometry confidence lost during yaw",
                    )

                return

            self.invalid_scan_count = 0

            if (
                self.alignment_age()
                > c.alignment_timeout_s
            ):

                self.fail("ENTRY_LOCK_FAILED")
                return

            if abs(g.yaw_error) <= yaw_aligned:

                self.latest_command = BodyVelocity.stop()

                if (
                    abs(g.lateral_error)
                    > c.lateral_centered_m
                ):

                    self.transition(
                        LockState.CENTER_LATERALLY,
                        "yaw aligned; correct lateral position",
                    )

                else:

                    self.transition(
                        LockState.VERIFY_LOCK,
                        "yaw and lateral near target",
                    )

                return

            self.latest_command = BodyVelocity(
                vx_m_s=0.0,
                vy_m_s=0.0,
                vz_m_s=0.0,
                yaw_rate_rad_s=self.yaw_control(
                    g.yaw_error
                ),
            )

            return

        # ----------------------------------------------------
        # LATERAL CENTERING
        # ----------------------------------------------------

        if self.state == LockState.CENTER_LATERALLY:

            if (
                not g.strict_valid
                or g.confidence
                < c.control_confidence_min
            ):

                self.invalid_scan_count += 1

                self.latest_command = BodyVelocity.stop()

                if (
                    self.invalid_scan_count
                    > c.geometry_grace_scans
                ):

                    self.transition(
                        LockState.HOLD,
                        "geometry confidence lost while centring",
                    )

                return

            self.invalid_scan_count = 0

            if (
                self.alignment_age()
                > c.alignment_timeout_s
            ):

                self.fail("ENTRY_LOCK_FAILED")
                return

            if abs(g.yaw_error) > yaw_realign:

                self.latest_command = BodyVelocity.stop()

                self.transition(
                    LockState.ALIGN_YAW,
                    "yaw drift exceeded hysteresis",
                )

                return

            if (
                abs(g.lateral_error)
                <= c.lateral_centered_m
            ):

                self.latest_command = BodyVelocity.stop()

                self.transition(
                    LockState.VERIFY_LOCK,
                    "lateral centre reached",
                )

                return

            vy = self.lateral_control(
                g.lateral_error
            )

            yaw_rate = 0.0

            if (
                abs(g.yaw_error)
                > math.radians(0.8)
            ):
                yaw_rate = (
                    0.5
                    * self.yaw_control(
                        g.yaw_error
                    )
                )

            self.latest_command = BodyVelocity(
                vx_m_s=0.0,
                vy_m_s=vy,
                vz_m_s=0.0,
                yaw_rate_rad_s=yaw_rate,
            )

            return

        # ----------------------------------------------------
        # VERIFY
        # ----------------------------------------------------

        if self.state == LockState.VERIFY_LOCK:

            self.latest_command = BodyVelocity.stop()

            good = (
                g.strict_valid
                and g.confidence
                >= c.verify_confidence_min
                and abs(g.yaw_error)
                <= yaw_lock
                and abs(g.lateral_error)
                <= c.lateral_lock_m
                and g.front_clearance
                >= c.front_stop_m
                and self.attitude_is_safe(attitude)
            )

            if good:

                self.stable_scan_count += 1

                if (
                    self.stable_scan_count
                    >= c.verify_scans
                ):

                    self.transition(
                        LockState.LOCKED,
                        "stable geometry lock achieved",
                    )

                return

            self.stable_scan_count = 0

            if (
                not g.strict_valid
                or g.confidence
                < c.control_confidence_min
            ):

                self.invalid_scan_count += 1

                if (
                    self.invalid_scan_count
                    > c.geometry_grace_scans
                ):

                    self.transition(
                        LockState.HOLD,
                        "geometry invalid during verification",
                    )

                return

            self.invalid_scan_count = 0

            if abs(g.yaw_error) > yaw_realign:

                self.transition(
                    LockState.ALIGN_YAW,
                    "yaw left verification band",
                )

            elif (
                abs(g.lateral_error)
                > c.lateral_recenter_m
            ):

                self.transition(
                    LockState.CENTER_LATERALLY,
                    "lateral position left verification band",
                )

            return

        # ----------------------------------------------------
        # LOCKED
        # ----------------------------------------------------

        if self.state == LockState.LOCKED:

            self.latest_command = BodyVelocity.stop()
            return

        # ----------------------------------------------------
        # HOLD
        # ----------------------------------------------------

        if self.state == LockState.HOLD:

            self.latest_command = BodyVelocity.stop()

            if self.failure_reason is not None:
                return

            recoverable = (
                g.strict_valid
                and g.confidence
                >= c.control_confidence_min
                and g.front_clearance
                >= c.front_stop_m
                and self.attitude_is_safe(attitude)
            )

            if recoverable:

                self.hold_recovery_count += 1

                if (
                    self.hold_recovery_count
                    >= c.hold_recovery_scans
                ):

                    self.transition(
                        LockState.ACQUIRE_GEOMETRY,
                        "fresh stable geometry recovered",
                    )

            else:
                self.hold_recovery_count = 0

    # ========================================================
    # Motion controllers
    # ========================================================

    def yaw_control(
        self,
        error_rad: float,
    ) -> float:

        c = self.config

        maximum = math.radians(
            c.max_yaw_rate_deg_s
        )

        minimum = math.radians(
            c.min_yaw_rate_deg_s
        )

        command = clamp(
            c.k_yaw * error_rad,
            -maximum,
            maximum,
        )

        if (
            abs(command) < minimum
            and abs(error_rad) > 1e-6
        ):

            command = math.copysign(
                minimum,
                command,
            )

        return command

    def lateral_control(
        self,
        error_m: float,
    ) -> float:

        c = self.config

        command = clamp(
            c.k_lateral * error_m,
            -c.max_lateral_speed_m_s,
            c.max_lateral_speed_m_s,
        )

        if (
            abs(command)
            < c.min_lateral_speed_m_s
            and abs(error_m) > 1e-6
        ):

            command = math.copysign(
                c.min_lateral_speed_m_s,
                command,
            )

        return command

    # ========================================================
    # Geometry extraction
    # ========================================================

    def extract_corridor_geometry(
        self,
        scan: NativeScan,
    ) -> CorridorGeometry:

        c = self.config

        g = CorridorGeometry()

        ranges = np.asarray(
            scan.ranges_m,
            dtype=np.float64,
        )

        angles = np.asarray(
            scan.angles_rad,
            dtype=np.float64,
        )

        if ranges.size < 20:
            return g

        sectors = self.sector_monitor.ingest(
            ranges=ranges,
            angles=angles,
            range_min=scan.range_min_m,
            range_max=scan.range_max_m,
            config=c,
        )

        g.front_clearance = sectors.front_clearance
        g.sectors = dict(sectors.values)

        valid = np.isfinite(ranges)

        valid &= ranges >= max(
            scan.range_min_m,
            0.05,
        )

        valid &= ranges <= min(
            scan.range_max_m,
            c.geometry_max_range_m,
        )

        if np.count_nonzero(valid) < 20:
            return g

        r = ranges[valid]
        a = angles[valid]

        points = np.column_stack(
            (
                r * np.cos(a),
                r * np.sin(a),
            )
        )

        x = points[:, 0]
        y = points[:, 1]

        longitudinal = (
            (x >= c.fit_x_min_m)
            & (x <= c.fit_x_max_m)
        )

        left_points = points[
            longitudinal
            & (y >= c.fit_side_min_m)
            & (y <= c.fit_side_max_m)
        ]

        right_points = points[
            longitudinal
            & (y <= -c.fit_side_min_m)
            & (y >= -c.fit_side_max_m)
        ]

        left_line = self.fit_line_ransac(
            left_points
        )

        right_line = self.fit_line_ransac(
            right_points
        )

        if (
            not left_line.valid
            or not right_line.valid
        ):
            return g

        g.left_rms = left_line.rms
        g.right_rms = right_line.rms

        g.left_inliers = int(
            left_line.inlier_points.shape[0]
        )

        g.right_inliers = int(
            right_line.inlier_points.shape[0]
        )

        g.left_span = left_line.span
        g.right_span = right_line.span

        g.left_inlier_ratio = (
            left_line.inlier_ratio
        )

        g.right_inlier_ratio = (
            right_line.inlier_ratio
        )

        d1 = left_line.direction.copy()
        d2 = right_line.direction.copy()

        if d1[0] < 0:
            d1 = -d1

        if d2[0] < 0:
            d2 = -d2

        dot = clamp(
            float(np.dot(d1, d2)),
            -1.0,
            1.0,
        )

        g.parallel_error = math.acos(dot)

        corridor_direction = d1 + d2

        norm = float(
            np.linalg.norm(
                corridor_direction
            )
        )

        if norm < 1e-6:
            return g

        corridor_direction /= norm

        if corridor_direction[0] < 0:
            corridor_direction = -corridor_direction

        g.yaw_error_raw = wrap_pi(
            math.atan2(
                float(corridor_direction[1]),
                float(corridor_direction[0]),
            )
        )

        left_normal = np.array(
            [
                -corridor_direction[1],
                corridor_direction[0],
            ],
            dtype=np.float64,
        )

        left_coordinate = float(
            np.median(
                left_line.inlier_points
                @ left_normal
            )
        )

        right_coordinate = float(
            np.median(
                right_line.inlier_points
                @ left_normal
            )
        )

        # Vehicle should be between both wall lines.
        if (
            left_coordinate <= 0.0
            or right_coordinate >= 0.0
        ):
            return g

        g.d_left = left_coordinate
        g.d_right = -right_coordinate

        g.width_raw = (
            left_coordinate
            - right_coordinate
        )

        # Positive = corridor centre is to aircraft LEFT.
        g.lateral_error_raw = (
            0.5
            * (
                left_coordinate
                + right_coordinate
            )
        )

        strict_parallel = math.radians(
            c.max_parallel_error_deg
        )

        loose_parallel = math.radians(
            c.loose_parallel_error_deg
        )

        max_yaw = math.radians(
            c.max_corridor_yaw_deg
        )

        g.loose_valid = (
            abs(
                g.width_raw
                - c.corridor_width_m
            )
            <= c.loose_width_tolerance_m
            and g.parallel_error
            <= loose_parallel
            and abs(g.yaw_error_raw)
            <= max_yaw
        )

        g.strict_valid = (
            abs(
                g.width_raw
                - c.corridor_width_m
            )
            <= c.width_tolerance_m
            and g.parallel_error
            <= strict_parallel
            and abs(g.yaw_error_raw)
            <= max_yaw
            and g.left_span
            >= c.min_wall_span_m
            and g.right_span
            >= c.min_wall_span_m
            and g.left_rms
            <= c.max_fit_rms_m
            and g.right_rms
            <= c.max_fit_rms_m
            and g.left_inliers
            >= c.min_wall_inliers
            and g.right_inliers
            >= c.min_wall_inliers
        )

        if g.loose_valid:

            self.geometry_history.push(
                g.yaw_error_raw,
                g.lateral_error_raw,
                g.width_raw,
            )

        if self.geometry_history.yaw:

            (
                g.yaw_error,
                g.lateral_error,
                g.width,
            ) = self.geometry_history.medians()

            g.temporal_score = (
                self.geometry_history
                .stability_score()
            )

        else:

            g.yaw_error = g.yaw_error_raw
            g.lateral_error = g.lateral_error_raw
            g.width = g.width_raw

        g.sector_score = self.compute_sector_score(
            sectors.values,
            left_line,
            right_line,
        )

        left_sector = sectors.values.get("L")
        right_sector = sectors.values.get("R")

        if (
            left_sector is not None
            and right_sector is not None
        ):

            g.observed_lr_sum = float(
                left_sector
                + right_sector
            )

            cosine = math.cos(
                g.yaw_error
            )

            if (
                abs(cosine) > 0.20
                and math.isfinite(g.width)
            ):

                g.expected_lr_sum = float(
                    g.width
                    / abs(cosine)
                )

                g.lr_sum_residual = abs(
                    g.observed_lr_sum
                    - g.expected_lr_sum
                )

        g.confidence = self.compute_confidence(g)

        return g

    # ========================================================
    # RANSAC
    # ========================================================

    def fit_line_ransac(
        self,
        points: np.ndarray,
    ) -> LineModel:

        c = self.config

        if (
            points.shape[0]
            < c.min_wall_inliers
        ):
            return LineModel()

        threshold = c.ransac_distance_m

        max_angle = math.radians(
            c.max_wall_line_angle_deg
        )

        n = points.shape[0]

        best_mask = None
        best_score = -math.inf

        for _ in range(
            c.ransac_iterations
        ):

            i, j = self.rng.choice(
                n,
                size=2,
                replace=False,
            )

            p1 = points[i]
            p2 = points[j]

            segment = p2 - p1

            length = float(
                np.linalg.norm(segment)
            )

            if length < 0.20:
                continue

            direction = (
                segment
                / length
            )

            if direction[0] < 0:
                direction = -direction

            candidate_angle = abs(
                math.atan2(
                    float(direction[1]),
                    float(direction[0]),
                )
            )

            if candidate_angle > max_angle:
                continue

            normal = np.array(
                [
                    -direction[1],
                    direction[0],
                ],
                dtype=np.float64,
            )

            residuals = np.abs(
                (points - p1)
                @ normal
            )

            mask = (
                residuals
                <= threshold
            )

            count = int(
                np.count_nonzero(mask)
            )

            if count < c.min_wall_inliers:
                continue

            inliers = points[mask]

            projections = (
                (inliers - p1)
                @ direction
            )

            span = float(
                np.percentile(
                    projections,
                    95.0,
                )
                - np.percentile(
                    projections,
                    5.0,
                )
            )

            median_residual = float(
                np.median(
                    residuals[mask]
                )
            )

            span_factor = (
                0.5
                + min(
                    span
                    / max(
                        c.min_wall_span_m,
                        0.1,
                    ),
                    3.0,
                )
            )

            residual_factor = (
                1.0
                / (
                    1.0
                    + 10.0
                    * median_residual
                )
            )

            score = (
                count
                * span_factor
                * residual_factor
            )

            if score > best_score:

                best_score = score
                best_mask = mask

        if best_mask is None:
            return LineModel()

        inliers = points[best_mask]

        centroid = np.mean(
            inliers,
            axis=0,
        )

        centered = (
            inliers
            - centroid
        )

        if inliers.shape[0] < 2:
            return LineModel()

        _, _, vh = np.linalg.svd(
            centered,
            full_matrices=False,
        )

        direction = vh[0]

        norm = float(
            np.linalg.norm(direction)
        )

        if norm < 1e-9:
            return LineModel()

        direction /= norm

        if direction[0] < 0:
            direction = -direction

        refined_angle = abs(
            math.atan2(
                float(direction[1]),
                float(direction[0]),
            )
        )

        if refined_angle > max_angle:
            return LineModel()

        normal = np.array(
            [
                -direction[1],
                direction[0],
            ],
            dtype=np.float64,
        )

        residuals = np.abs(
            centered
            @ normal
        )

        rms = float(
            math.sqrt(
                np.mean(
                    residuals
                    * residuals
                )
            )
        )

        along = (
            centered
            @ direction
        )

        span = float(
            np.percentile(
                along,
                95.0,
            )
            - np.percentile(
                along,
                5.0,
            )
        )

        inlier_ratio = float(
            inliers.shape[0]
            / max(
                points.shape[0],
                1,
            )
        )

        return LineModel(
            valid=True,
            point=centroid,
            direction=direction,
            inlier_points=inliers,
            rms=rms,
            span=span,
            inlier_ratio=inlier_ratio,
        )

    # ========================================================
    # Sector-model validation
    # ========================================================

    @staticmethod
    def ray_line_intersection_range(
        line: LineModel,
        bearing_rad: float,
    ) -> Optional[float]:

        if not line.valid:
            return None

        ray = np.array(
            [
                math.cos(bearing_rad),
                math.sin(bearing_rad),
            ],
            dtype=np.float64,
        )

        denominator = cross2(
            ray,
            line.direction,
        )

        if abs(denominator) < 1e-8:
            return None

        distance = (
            cross2(
                line.point,
                line.direction,
            )
            / denominator
        )

        if (
            distance <= 0
            or not math.isfinite(distance)
        ):
            return None

        return float(distance)

    def compute_sector_score(
        self,
        observed: Dict[str, Optional[float]],
        left_line: LineModel,
        right_line: LineModel,
    ) -> float:

        c = self.config

        if not c.use_sector_validation:
            return 0.75

        checks = [
            (
                "L",
                left_line,
                math.radians(90.0),
                c.sector_lr_tolerance_m,
                1.0,
            ),
            (
                "R",
                right_line,
                math.radians(-90.0),
                c.sector_lr_tolerance_m,
                1.0,
            ),
            (
                "FL",
                left_line,
                math.radians(45.0),
                c.sector_diag_tolerance_m,
                0.35,
            ),
            (
                "FR",
                right_line,
                math.radians(-45.0),
                c.sector_diag_tolerance_m,
                0.35,
            ),
        ]

        weighted = 0.0
        total_weight = 0.0

        strong_seen = 0

        for (
            name,
            line,
            bearing,
            tolerance,
            weight,
        ) in checks:

            actual = observed.get(name)

            predicted = (
                self.ray_line_intersection_range(
                    line,
                    bearing,
                )
            )

            if (
                actual is None
                or predicted is None
            ):
                continue

            residual = abs(
                actual
                - predicted
            )

            score = linear_score(
                residual,
                0.08,
                tolerance,
            )

            weighted += (
                weight
                * score
            )

            total_weight += weight

            if name in ("L", "R"):
                strong_seen += 1

        if total_weight <= 1e-9:
            return 0.50

        score = (
            weighted
            / total_weight
        )

        if strong_seen == 0:
            score = min(
                score,
                0.55,
            )

        elif strong_seen == 1:
            score = min(
                score,
                0.75,
            )

        return float(
            clamp(
                score,
                0.0,
                1.0,
            )
        )

    # ========================================================
    # Confidence
    # ========================================================

    def compute_confidence(
        self,
        g: CorridorGeometry,
    ) -> float:

        c = self.config

        loose_parallel = math.radians(
            c.loose_parallel_error_deg
        )

        wall_scores = []

        for rms, span, inliers in (
            (
                g.left_rms,
                g.left_span,
                g.left_inliers,
            ),
            (
                g.right_rms,
                g.right_span,
                g.right_inliers,
            ),
        ):

            rms_score = linear_score(
                rms,
                0.025,
                c.max_fit_rms_m
                * 1.4,
            )

            span_score = clamp(
                span
                / max(
                    1.5
                    * c.min_wall_span_m,
                    0.1,
                ),
                0.0,
                1.0,
            )

            inlier_score = clamp(
                inliers
                / max(
                    2.0
                    * c.min_wall_inliers,
                    1.0,
                ),
                0.0,
                1.0,
            )

            wall_scores.append(
                0.45 * rms_score
                + 0.30 * span_score
                + 0.25 * inlier_score
            )

        fit_score = (
            float(np.mean(wall_scores))
            if wall_scores
            else 0.0
        )

        width_error = abs(
            g.width_raw
            - c.corridor_width_m
        )

        width_score = linear_score(
            width_error,
            0.12,
            c.loose_width_tolerance_m,
        )

        parallel_score = linear_score(
            g.parallel_error,
            math.radians(1.5),
            loose_parallel,
        )

        confidence = (
            0.34 * fit_score
            + 0.22 * width_score
            + 0.16 * parallel_score
            + 0.16 * g.sector_score
            + 0.12 * g.temporal_score
        )

        if not g.loose_valid:
            confidence = min(
                confidence,
                0.35,
            )

        return float(
            clamp(
                confidence,
                0.0,
                1.0,
            )
        )

    # ========================================================
    # Output
    # ========================================================

    def output(self) -> ControllerOutput:

        next_state = None
        reason = ""

        if self.failure_reason is not None:

            next_state = (
                MissionState.HOVER_AND_REASSESS
            )

            reason = self.failure_reason

        elif self.state == LockState.LOCKED:

            next_state = (
                MissionState.ENTER_CORRIDOR
            )

            reason = (
                "stable corridor geometry lock"
            )

        confidence = (
            self.last_geometry.confidence
            if self.last_geometry is not None
            else None
        )

        return ControllerOutput(
            command=self.latest_command,
            next_state=next_state,
            status=self.state.name,
            reason=reason,
            confidence=confidence,
        )

    # ========================================================
    # Human-readable diagnostics
    # ========================================================

    def diagnostics(self) -> dict:

        g = self.last_geometry

        if g is None:

            return {
                "state": self.state.name,
                "failure_reason": self.failure_reason,
            }

        return {
            "state": self.state.name,
            "failure_reason": self.failure_reason,

            "strict_valid": g.strict_valid,
            "loose_valid": g.loose_valid,

            "confidence": round(
                g.confidence,
                3,
            ),

            "yaw_error_deg": round(
                math.degrees(
                    g.yaw_error
                ),
                2,
            ),

            "lateral_error_m": round(
                g.lateral_error,
                3,
            ),

            "width_m": (
                round(
                    g.width,
                    3,
                )
                if math.isfinite(g.width)
                else None
            ),

            "left_distance_m": (
                round(
                    g.d_left,
                    3,
                )
                if math.isfinite(g.d_left)
                else None
            ),

            "right_distance_m": (
                round(
                    g.d_right,
                    3,
                )
                if math.isfinite(g.d_right)
                else None
            ),

            "parallel_error_deg": (
                round(
                    math.degrees(
                        g.parallel_error
                    ),
                    2,
                )
                if math.isfinite(
                    g.parallel_error
                )
                else None
            ),

            "front_clearance_m": round(
                g.front_clearance,
                3,
            ),

            "left_inliers": g.left_inliers,
            "right_inliers": g.right_inliers,

            "left_span_m": round(
                g.left_span,
                2,
            ),

            "right_span_m": round(
                g.right_span,
                2,
            ),

            "sector_score": round(
                g.sector_score,
                3,
            ),

            "temporal_score": round(
                g.temporal_score,
                3,
            ),

            "sectors": {
                key: (
                    round(value, 2)
                    if value is not None
                    else None
                )
                for key, value
                in g.sectors.items()
            },
        }
