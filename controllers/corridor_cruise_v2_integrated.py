#!/usr/bin/env python3
"""Integrated CORRIDOR_CRUISE state for SAEINDIA AeroTHON 2026.

Designed for a D500 2-D LiDAR publishing sensor_msgs/LaserScan on /scan.

This V2 merges the former RECENTER behavior into CORRIDOR_CRUISE.
There is no external RECENTER state.

While CORRIDOR_CRUISE is active it internally switches between:
    NOMINAL
        Normal forward flight with continuous small yaw/lateral corrections.

    CORRECTING
        Reduced/zero forward speed with stronger yaw/lateral correction when
        drift becomes too large for nominal cruise.

Built-in external transitions:
    front obstacle detected
        -> OBSTACLE_DECISION

    corridor exit candidate detected
        -> EXIT_DETECTION

    low-confidence geometry, unsafe attitude, stale scan, or correction
    that fails to converge
        -> HOVER_AND_REASSESS

Otherwise the vehicle remains in CORRIDOR_CRUISE. When correction succeeds,
the node returns from CORRECTING to NOMINAL without a global FSM transition.

Estimator:
    full D500 scan -> XY -> RANSAC left/right walls -> yaw/centre/width
    plus L/FL/F/FR/R sector validation and short temporal filtering.

ROS command convention is FLU:
    +x forward, +y left, +z up, +yaw counter-clockwise.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Deque, Dict, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Bool, String


# ============================================================================
# Helpers
# ============================================================================


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wrap_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def linear_score(error: float, good: float, bad: float) -> float:
    """Return 1 at/below good, 0 at/above bad, linear in between."""
    if bad <= good:
        return 1.0 if error <= good else 0.0
    if error <= good:
        return 1.0
    if error >= bad:
        return 0.0
    return 1.0 - (error - good) / (bad - good)


def quaternion_to_roll_pitch(msg: Imu) -> tuple[float, float]:
    q = msg.orientation

    sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
    cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (q.w * q.y - q.z * q.x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    return roll, pitch


# ============================================================================
# Data structures
# ============================================================================


class CruiseState(Enum):
    IDLE = auto()
    ACTIVE = auto()
    TRANSITION_REQUESTED = auto()


class CruiseMode(Enum):
    """Internal controller mode; not a global mission FSM state."""
    NOMINAL = auto()
    CORRECTING = auto()


@dataclass
class LineModel:
    valid: bool = False
    point: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    direction: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
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
    # Individual wall detection is kept even if a complete pair cannot be fit.
    left_wall_valid: bool = False
    right_wall_valid: bool = False

    # Pair geometry validity.
    strict_valid: bool = False
    loose_valid: bool = False

    # Raw single-scan estimates.
    yaw_error_raw: float = 0.0
    lateral_error_raw: float = 0.0
    width_raw: float = math.inf

    # Filtered controller estimates.
    yaw_error: float = 0.0
    lateral_error: float = 0.0
    width: float = math.inf

    # Wall geometry / quality.
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

    # Validation / safety.
    front_clearance: float = 0.0
    sector_score: float = 0.5
    temporal_score: float = 0.7
    confidence: float = 0.0
    observed_lr_sum: Optional[float] = None
    expected_lr_sum: Optional[float] = None
    lr_sum_residual: Optional[float] = None
    sectors: Dict[str, Optional[float]] = field(default_factory=dict)

    # Cruise event classification.
    exit_candidate: bool = False
    side_open_left: bool = False
    side_open_right: bool = False


# ============================================================================
# Sector monitor -- validator / safety layer, not primary wall controller
# ============================================================================


class SectorMonitor:
    """Robust L/FL/F/FR/R ranges using ROS FLU bearings.

    After scan-angle correction:
        F=0, FL=+45, L=+90, FR=-45, R=-90 degrees.
    """

    BEARINGS_DEG = {
        "L": 90.0,
        "FL": 45.0,
        "F": 0.0,
        "FR": -45.0,
        "R": -90.0,
    }

    def __init__(self, window: int) -> None:
        n = max(1, int(window))
        self.histories: Dict[str, Deque[float]] = {
            key: deque(maxlen=n) for key in self.BEARINGS_DEG
        }

    def reset(self) -> None:
        for hist in self.histories.values():
            hist.clear()

    @staticmethod
    def _angle_difference_array(angles: np.ndarray, target: float) -> np.ndarray:
        return np.abs(np.arctan2(np.sin(angles - target), np.cos(angles - target)))

    def ingest(
        self,
        ranges: np.ndarray,
        angles: np.ndarray,
        range_min: float,
        range_max: float,
        cone_deg: float,
        percentile: float,
        max_sector_range: float,
        front_cone_deg: float,
    ) -> SectorSnapshot:
        snapshot = SectorSnapshot()

        valid = np.isfinite(ranges)
        valid &= ranges >= max(float(range_min), 0.05)
        valid &= ranges <= min(float(range_max), float(max_sector_range))

        half_cone = math.radians(float(cone_deg))
        q = clamp(float(percentile), 0.0, 100.0)

        for name, bearing_deg in self.BEARINGS_DEG.items():
            target = math.radians(bearing_deg)
            mask = valid & (self._angle_difference_array(angles, target) <= half_cone)
            vals = ranges[mask]

            if vals.size:
                sample = float(np.percentile(vals, q))
                self.histories[name].append(sample)
            else:
                # NaN ages stale sector readings out of the temporal window.
                self.histories[name].append(float("nan"))

            finite_hist = [v for v in self.histories[name] if math.isfinite(v)]
            snapshot.values[name] = (
                float(np.median(finite_hist)) if finite_hist else None
            )

        # Front safety is deliberately more conservative than the individual
        # sector value. +inf means no return and therefore open up to range_max.
        safe_ranges = ranges.copy()
        safe_ranges[np.isposinf(safe_ranges)] = float(range_max)
        front_valid = np.isfinite(safe_ranges)
        front_valid &= safe_ranges >= max(float(range_min), 0.05)
        front_valid &= safe_ranges <= float(range_max)

        front_half = math.radians(float(front_cone_deg))
        front_mask = front_valid & (np.abs(angles) <= front_half)
        front_vals = safe_ranges[front_mask]
        if front_vals.size:
            snapshot.front_clearance = float(np.percentile(front_vals, 10.0))
        else:
            # Unknown front is unsafe.
            snapshot.front_clearance = 0.0

        return snapshot


# ============================================================================
# Temporal geometry filter
# ============================================================================


class GeometryHistory:
    def __init__(self, window: int) -> None:
        n = max(1, int(window))
        self.yaw: Deque[float] = deque(maxlen=n)
        self.lateral: Deque[float] = deque(maxlen=n)
        self.width: Deque[float] = deque(maxlen=n)

    def reset(self) -> None:
        self.yaw.clear()
        self.lateral.clear()
        self.width.clear()

    def push(self, yaw: float, lateral: float, width: float) -> None:
        self.yaw.append(float(yaw))
        self.lateral.append(float(lateral))
        self.width.append(float(width))

    def medians(self) -> tuple[float, float, float]:
        return (
            float(np.median(self.yaw)),
            float(np.median(self.lateral)),
            float(np.median(self.width)),
        )

    def stability_score(self) -> float:
        if len(self.yaw) < 3:
            return 0.70

        yaw_arr = np.asarray(self.yaw, dtype=np.float64)
        lat_arr = np.asarray(self.lateral, dtype=np.float64)
        width_arr = np.asarray(self.width, dtype=np.float64)

        yaw_mad = float(np.median(np.abs(yaw_arr - np.median(yaw_arr))))
        lat_mad = float(np.median(np.abs(lat_arr - np.median(lat_arr))))
        width_mad = float(np.median(np.abs(width_arr - np.median(width_arr))))

        yaw_score = linear_score(yaw_mad, math.radians(0.5), math.radians(4.0))
        lat_score = linear_score(lat_mad, 0.02, 0.20)
        width_score = linear_score(width_mad, 0.03, 0.25)
        return float((yaw_score + lat_score + width_score) / 3.0)


# ============================================================================
# CORRIDOR_CRUISE ROS node
# ============================================================================


class CorridorCruiseV2(Node):
    def __init__(self) -> None:
        super().__init__("corridor_cruise_v2")

        # ------------------------------------------------------------------
        # Topics
        # ------------------------------------------------------------------
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("imu_topic", "/imu/data")
        self.declare_parameter("enable_topic", "/corridor/cruise/enable")
        self.declare_parameter("cmd_topic", "/corridor/cruise/cmd_vel")
        self.declare_parameter("state_topic", "/corridor/cruise/state")
        self.declare_parameter("mode_topic", "/corridor/cruise/mode")
        self.declare_parameter("result_topic", "/corridor/cruise/result")
        self.declare_parameter("next_state_topic", "/corridor/cruise/next_state")
        self.declare_parameter("diagnostics_topic", "/corridor/cruise/diagnostics")

        # ------------------------------------------------------------------
        # D500 / scan orientation
        # ------------------------------------------------------------------
        self.declare_parameter("invert_scan_angle", False)
        self.declare_parameter("lidar_yaw_offset_deg", 0.0)
        self.declare_parameter("geometry_max_range_m", 8.0)

        # ------------------------------------------------------------------
        # Known corridor geometry
        # ------------------------------------------------------------------
        self.declare_parameter("corridor_width_m", 3.5)
        self.declare_parameter("width_tolerance_m", 0.45)
        self.declare_parameter("loose_width_tolerance_m", 0.80)
        self.declare_parameter("max_parallel_error_deg", 7.0)
        self.declare_parameter("loose_parallel_error_deg", 15.0)
        self.declare_parameter("max_corridor_yaw_deg", 35.0)

        # ------------------------------------------------------------------
        # RANSAC wall fitting
        # ------------------------------------------------------------------
        # Cruise looks slightly farther ahead than the static lock state.
        self.declare_parameter("fit_x_min_m", -0.20)
        self.declare_parameter("fit_x_max_m", 5.5)
        self.declare_parameter("fit_side_min_m", 0.25)
        self.declare_parameter("fit_side_max_m", 3.5)
        self.declare_parameter("ransac_iterations", 120)
        self.declare_parameter("ransac_distance_m", 0.08)
        self.declare_parameter("min_wall_inliers", 14)
        self.declare_parameter("min_wall_span_m", 0.80)
        self.declare_parameter("max_fit_rms_m", 0.10)
        self.declare_parameter("max_wall_line_angle_deg", 45.0)

        # ------------------------------------------------------------------
        # Sector validation
        # ------------------------------------------------------------------
        self.declare_parameter("use_sector_validation", True)
        self.declare_parameter("sector_cone_deg", 6.0)
        self.declare_parameter("sector_percentile", 35.0)
        self.declare_parameter("sector_filter_window", 5)
        self.declare_parameter("sector_max_range_m", 6.0)
        self.declare_parameter("sector_lr_tolerance_m", 0.35)
        self.declare_parameter("sector_diag_tolerance_m", 0.55)

        # ------------------------------------------------------------------
        # Temporal filtering / confidence
        # ------------------------------------------------------------------
        self.declare_parameter("geometry_filter_window", 3)
        self.declare_parameter("nominal_confidence", 0.78)
        self.declare_parameter("control_confidence_min", 0.60)
        self.declare_parameter("low_confidence_trigger", 0.60)
        self.declare_parameter("low_confidence_confirm_scans", 3)
        # loose_valid can occasionally survive while strict wall geometry is
        # temporarily unusable.  Do not sit stopped forever in that condition.
        self.declare_parameter("strict_geometry_loss_confirm_scans", 3)

        # ------------------------------------------------------------------
        # Front obstacle handling
        # ------------------------------------------------------------------
        self.declare_parameter("front_cone_deg", 18.0)
        self.declare_parameter("front_slowdown_start_m", 2.50)
        self.declare_parameter("front_obstacle_trigger_m", 1.35)
        self.declare_parameter("front_emergency_stop_m", 0.75)
        self.declare_parameter("obstacle_confirm_scans", 2)

        # ------------------------------------------------------------------
        # Normal cruise controller
        # ------------------------------------------------------------------
        self.declare_parameter("nominal_forward_speed_m_s", 0.35)
        self.declare_parameter("minimum_forward_speed_m_s", 0.10)
        self.declare_parameter("k_yaw", 0.55)
        self.declare_parameter("max_yaw_rate_deg_s", 6.0)
        self.declare_parameter("yaw_deadband_deg", 0.8)
        self.declare_parameter("k_lateral", 0.30)
        self.declare_parameter("max_lateral_speed_m_s", 0.12)
        self.declare_parameter("lateral_deadband_m", 0.04)

        # ------------------------------------------------------------------
        # Integrated re-centering / alignment correction
        # Enter on larger thresholds and leave only after tighter thresholds
        # remain satisfied. This hysteresis prevents chatter.
        self.declare_parameter("correction_enter_yaw_deg", 5.0)
        self.declare_parameter("correction_exit_yaw_deg", 2.5)
        self.declare_parameter("correction_enter_lateral_m", 0.22)
        self.declare_parameter("correction_exit_lateral_m", 0.10)
        self.declare_parameter("correction_enter_confirm_scans", 2)
        self.declare_parameter("correction_release_confirm_scans", 4)

        self.declare_parameter("correction_forward_speed_m_s", 0.10)
        self.declare_parameter("correction_k_yaw", 0.85)
        self.declare_parameter("correction_max_yaw_rate_deg_s", 8.0)
        self.declare_parameter("correction_k_lateral", 0.48)
        self.declare_parameter("correction_max_lateral_speed_m_s", 0.18)

        # Large yaw: rotate first, with no forward/lateral translation.
        self.declare_parameter("correction_yaw_priority_deg", 7.0)

        # Correction failure supervision.
        self.declare_parameter("correction_timeout_s", 8.0)
        self.declare_parameter("correction_extreme_yaw_deg", 18.0)
        self.declare_parameter("correction_extreme_lateral_m", 0.70)
        self.declare_parameter("correction_worsening_ratio", 1.35)
        self.declare_parameter("correction_worsening_confirm_scans", 3)

        # ------------------------------------------------------------------
        # EXIT_DETECTION candidate generation
        # ------------------------------------------------------------------
        # Cruise does not *confirm* the exit; it only notices an opening-like
        # pattern and hands it to EXIT_DETECTION.
        self.declare_parameter("exit_guard_time_s", 1.5)
        self.declare_parameter("exit_side_open_m", 2.40)
        self.declare_parameter("exit_front_open_m", 3.00)
        self.declare_parameter("exit_probe_speed_m_s", 0.12)
        self.declare_parameter("exit_candidate_confirm_scans", 3)
        self.declare_parameter("exit_precursor_max_scans", 20)

        # ------------------------------------------------------------------
        # General safety
        # ------------------------------------------------------------------
        self.declare_parameter("scan_stale_s", 0.30)
        self.declare_parameter("require_imu", False)
        self.declare_parameter("max_tilt_deg", 10.0)

        # ------------------------------------------------------------------
        # ROS wiring
        # ------------------------------------------------------------------
        scan_topic = str(self.get_parameter("scan_topic").value)
        imu_topic = str(self.get_parameter("imu_topic").value)
        enable_topic = str(self.get_parameter("enable_topic").value)
        cmd_topic = str(self.get_parameter("cmd_topic").value)

        self.scan_sub = self.create_subscription(LaserScan, scan_topic, self.scan_callback, 10)
        self.imu_sub = self.create_subscription(Imu, imu_topic, self.imu_callback, 10)
        self.enable_sub = self.create_subscription(Bool, enable_topic, self.enable_callback, 10)

        self.cmd_pub = self.create_publisher(TwistStamped, cmd_topic, 10)
        self.state_pub = self.create_publisher(
            String, str(self.get_parameter("state_topic").value), 10
        )
        self.mode_pub = self.create_publisher(
            String, str(self.get_parameter("mode_topic").value), 10
        )
        self.result_pub = self.create_publisher(
            String, str(self.get_parameter("result_topic").value), 10
        )
        self.next_state_pub = self.create_publisher(
            String, str(self.get_parameter("next_state_topic").value), 10
        )
        self.diagnostics_pub = self.create_publisher(
            String, str(self.get_parameter("diagnostics_topic").value), 10
        )

        # Publish commands/status at 20 Hz even though D500 scans are ~10 Hz.
        self.timer = self.create_timer(0.05, self.publish_outputs)

        # ------------------------------------------------------------------
        # Runtime state
        # ------------------------------------------------------------------
        self.state = CruiseState.IDLE
        self.enabled = False
        self.transition_target: Optional[str] = None
        self.transition_reason: Optional[str] = None
        self.pause_reason: Optional[str] = None

        self.last_scan_time = None
        self.session_start_time = None
        self.latest_command = self.zero_command()
        self.last_geometry: Optional[CorridorGeometry] = None

        self.roll: Optional[float] = None
        self.pitch: Optional[float] = None

        self.obstacle_streak = 0
        self.low_conf_streak = 0
        self.strict_geometry_loss_streak = 0
        self.exit_streak = 0
        self.exit_precursor_streak = 0
        

        self.mode = CruiseMode.NOMINAL
        self.correction_enter_streak = 0
        self.correction_release_streak = 0
        self.correction_start_time = None
        self.correction_best_metric = math.inf
        self.correction_worsening_streak = 0

        self.rng = np.random.default_rng(7)
        self.sector_monitor = SectorMonitor(
            int(self.get_parameter("sector_filter_window").value)
        )
        self.geometry_history = GeometryHistory(
            int(self.get_parameter("geometry_filter_window").value)
        )

        self.get_logger().info("CORRIDOR_CRUISE V2 integrated correction started")

    # ======================================================================
    # Lifecycle / callbacks
    # ======================================================================

    def reset_session(self) -> None:
        self.transition_target = None
        self.transition_reason = None
        self.pause_reason = None
        self.last_geometry = None
        self.latest_command = self.zero_command()
        self.last_scan_time = None
        self.session_start_time = self.get_clock().now()

        self.obstacle_streak = 0
        self.low_conf_streak = 0
        self.strict_geometry_loss_streak = 0
        self.exit_streak = 0

        self.mode = CruiseMode.NOMINAL
        self.correction_enter_streak = 0
        self.correction_release_streak = 0
        self.correction_start_time = None
        self.correction_best_metric = math.inf
        self.correction_worsening_streak = 0

        self.sector_monitor.reset()
        self.geometry_history.reset()

    def enable_callback(self, msg: Bool) -> None:
        if msg.data and not self.enabled:
            self.reset_session()
            self.enabled = True
            self.state = CruiseState.ACTIVE
            self.get_logger().info("CORRIDOR_CRUISE enabled")
        elif not msg.data and self.enabled:
            self.enabled = False
            self.state = CruiseState.IDLE
            self.transition_target = None
            self.transition_reason = None
            self.pause_reason = None
            self.mode = CruiseMode.NOMINAL
            self.latest_command = self.zero_command()
            self.get_logger().info("CORRIDOR_CRUISE disabled")

    def imu_callback(self, msg: Imu) -> None:
        self.roll, self.pitch = quaternion_to_roll_pitch(msg)

    def scan_callback(self, scan: LaserScan) -> None:
        self.last_scan_time = self.get_clock().now()

        if not self.enabled or self.state != CruiseState.ACTIVE:
            return

        g = self.extract_corridor_geometry(scan)
        self.last_geometry = g
        self.step_cruise(g)

    def session_age(self) -> float:
        if self.session_start_time is None:
            return 0.0
        return (
            self.get_clock().now() - self.session_start_time
        ).nanoseconds * 1e-9

    def attitude_is_safe(self) -> bool:
        require_imu = bool(self.get_parameter("require_imu").value)
        if self.roll is None or self.pitch is None:
            return not require_imu

        max_tilt = math.radians(float(self.get_parameter("max_tilt_deg").value))
        return abs(self.roll) <= max_tilt and abs(self.pitch) <= max_tilt

    def request_transition(
        self,
        target: str,
        reason: str,
        pause_reason: Optional[str] = None,
    ) -> None:
        if self.state == CruiseState.TRANSITION_REQUESTED:
            return

        self.transition_target = target
        self.transition_reason = reason
        self.pause_reason = pause_reason
        self.state = CruiseState.TRANSITION_REQUESTED
        self.latest_command = self.zero_command()
        self.get_logger().warning(
            f"CORRIDOR_CRUISE -> {target}: {reason}"
        )

    # ======================================================================
    # Integrated correction-mode helpers
    # ======================================================================

    def correction_age(self) -> float:
        if self.correction_start_time is None:
            return 0.0
        return (
            self.get_clock().now() - self.correction_start_time
        ).nanoseconds * 1e-9

    def correction_metric(self, g: CorridorGeometry) -> float:
        """Dimensionless alignment error, used only to supervise progress."""
        yaw_scale = max(
            math.radians(float(self.get_parameter("correction_enter_yaw_deg").value)),
            math.radians(0.5),
        )
        lat_scale = max(
            float(self.get_parameter("correction_enter_lateral_m").value),
            0.03,
        )
        return max(
            abs(g.yaw_error) / yaw_scale,
            abs(g.lateral_error) / lat_scale,
        )

    def enter_correction_mode(self, g: CorridorGeometry, reason: str) -> None:
        if self.mode == CruiseMode.CORRECTING:
            return
        self.mode = CruiseMode.CORRECTING
        self.correction_start_time = self.get_clock().now()
        self.correction_release_streak = 0
        self.correction_worsening_streak = 0
        self.correction_best_metric = self.correction_metric(g)
        self.get_logger().warning(
            "CORRIDOR_CRUISE internal NOMINAL -> CORRECTING: " + reason
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
        self.get_logger().info(
            "CORRIDOR_CRUISE internal CORRECTING -> NOMINAL: alignment restored"
        )

    def correction_command(self, g: CorridorGeometry) -> TwistStamped:
        """Strong bounded correction while remaining in CORRIDOR_CRUISE."""
        yaw_deadband = math.radians(
            float(self.get_parameter("yaw_deadband_deg").value)
        )
        lat_deadband = float(self.get_parameter("lateral_deadband_m").value)

        yaw_error = 0.0 if abs(g.yaw_error) <= yaw_deadband else g.yaw_error
        lat_error = (
            0.0 if abs(g.lateral_error) <= lat_deadband else g.lateral_error
        )

        yaw_rate = clamp(
            float(self.get_parameter("correction_k_yaw").value) * yaw_error,
            -math.radians(
                float(self.get_parameter("correction_max_yaw_rate_deg_s").value)
            ),
            math.radians(
                float(self.get_parameter("correction_max_yaw_rate_deg_s").value)
            ),
        )

        yaw_priority = math.radians(
            float(self.get_parameter("correction_yaw_priority_deg").value)
        )
        if abs(g.yaw_error) >= yaw_priority:
            # Correct major heading error first.
            return self.make_command(vx=0.0, vy_left=0.0, yaw_rate=yaw_rate)

        vy_left = clamp(
            float(self.get_parameter("correction_k_lateral").value) * lat_error,
            -float(
                self.get_parameter("correction_max_lateral_speed_m_s").value
            ),
            float(
                self.get_parameter("correction_max_lateral_speed_m_s").value
            ),
        )

        requested_vx = max(
            0.0, float(self.get_parameter("correction_forward_speed_m_s").value)
        )
        vx = min(requested_vx, self.adaptive_forward_speed(g))
        return self.make_command(vx=vx, vy_left=vy_left, yaw_rate=yaw_rate)

    # ======================================================================
    # Global FSM state logic
    # ======================================================================

    def step_cruise(self, g: CorridorGeometry) -> None:
        """Process one D500 scan while CORRIDOR_CRUISE is active.

        External transition priority:
          1. obstacle -> OBSTACLE_DECISION
          2. exit candidate -> EXIT_DETECTION
          3. unreliable geometry / attitude -> HOVER_AND_REASSESS
          4. failed integrated correction -> HOVER_AND_REASSESS
          5. otherwise remain in CORRIDOR_CRUISE

        Large drift/yaw changes only the internal mode to CORRECTING.
        """

        # 0) IMU / scan-plane safety.
        if not self.attitude_is_safe():
            self.request_transition(
                "HOVER_AND_REASSESS",
                "roll/pitch exceeds cruise geometry limit",
                "LOW_CONFIDENCE_GEOMETRY",
            )
            return

        # 1) Front obstacle.
        emergency = float(self.get_parameter("front_emergency_stop_m").value)
        obstacle_trigger = float(
            self.get_parameter("front_obstacle_trigger_m").value
        )
        obstacle_confirm = int(
            self.get_parameter("obstacle_confirm_scans").value
        )

        if 0.0 < g.front_clearance <= emergency:
            self.request_transition(
                "OBSTACLE_DECISION",
                f"front emergency clearance {g.front_clearance:.2f} m",
            )
            return

        if 0.0 < g.front_clearance <= obstacle_trigger:
            self.obstacle_streak += 1
            self.latest_command = self.zero_command()
            if self.obstacle_streak >= obstacle_confirm:
                self.request_transition(
                    "OBSTACLE_DECISION",
                    f"front obstacle confirmed at {g.front_clearance:.2f} m",
                )
            return
        self.obstacle_streak = 0

        # 2) Exit candidate before low-confidence handling, because a real
        # exit naturally causes side-wall geometry to open/disappear.
        if g.exit_candidate:
            self.exit_streak += 1
            if self.exit_streak >= int(
                self.get_parameter("exit_candidate_confirm_scans").value
            ):
                self.request_transition(
                    "EXIT_DETECTION",
                    "persistent corridor-opening candidate",
                )
                return

            probe = float(self.get_parameter("exit_probe_speed_m_s").value)
            self.latest_command = self.make_command(vx=max(0.0, probe))
            return
        self.exit_streak = 0

        # --------------------------------------------------------------
        # 2B) Weak exit precursor
        #
        # Near the corridor exit, strict wall geometry can disappear
        # before both side sectors have completely cleared the walls.
        #
        # If we stop immediately, the aircraft may never move far enough
        # to observe the complete opening. Therefore, when the front is
        # open and at least one side has opened, allow a short bounded
        # forward probe.
        # --------------------------------------------------------------

        exit_guard_passed = (
            self.session_age()
            >= float(self.get_parameter("exit_guard_time_s").value)
        )

        front_open = (
            g.front_clearance
            >= float(self.get_parameter("exit_front_open_m").value)
        )

        side_opening_started = (
            g.side_open_left or g.side_open_right
        )

        exit_precursor = (
            exit_guard_passed
            and front_open
            and side_opening_started
        )

        if exit_precursor:
            self.exit_precursor_streak += 1

            # Do not accumulate ordinary wall-loss failure counters
            # while deliberately probing a plausible corridor exit.
            self.low_conf_streak = 0
            self.strict_geometry_loss_streak = 0

            max_precursor_scans = int(
                self.get_parameter("exit_precursor_max_scans").value
            )

            if self.exit_precursor_streak > max_precursor_scans:
                self.latest_command = self.zero_command()

                self.request_transition(
                    "HOVER_AND_REASSESS",
                    "exit precursor failed to become full corridor opening",
                    "LOW_CONFIDENCE_GEOMETRY",
                )
                return

            probe = float(
                self.get_parameter("exit_probe_speed_m_s").value
            )

            self.latest_command = self.make_command(
                vx=max(0.0, probe)
            )
            return

        self.exit_precursor_streak = 0

        # 3) Geometry confidence.
        low_conf_trigger = float(
            self.get_parameter("low_confidence_trigger").value
        )
        low_conf_confirm = int(
            self.get_parameter("low_confidence_confirm_scans").value
        )

        geometry_bad = (
            not g.loose_valid
            or g.confidence < low_conf_trigger
            or g.front_clearance <= 0.0
        )

        if geometry_bad:
            self.low_conf_streak += 1
            self.latest_command = self.zero_command()
            if self.low_conf_streak >= low_conf_confirm:
                self.request_transition(
                    "HOVER_AND_REASSESS",
                    "wall estimates / geometry confidence became unreliable",
                    "LOW_CONFIDENCE_GEOMETRY",
                )
            return
        self.low_conf_streak = 0

        # 4) Control-quality geometry gate.
        # The Excel FSM says a geometry drop during cruise / re-centering must
        # go to HOVER_AND_REASSESS.  A single imperfect D500 scan is allowed,
        # but persistent loss of strict wall geometry is not allowed to leave
        # the vehicle silently stopped inside CORRIDOR_CRUISE forever.
        if not g.strict_valid:
            self.strict_geometry_loss_streak += 1
            self.latest_command = self.zero_command()

            if self.strict_geometry_loss_streak >= int(
                self.get_parameter("strict_geometry_loss_confirm_scans").value
            ):

                front_open = (
                    g.front_clearance
                    >= float(self.get_parameter("exit_front_open_m").value)
                )

        # If loose corridor geometry still exists and the forward path is
        # clearly open, treat persistent loss of strict wall quality as a
        # possible corridor end and hand control to EXIT_DETECTION.
                if g.loose_valid and front_open:

                    self.request_transition(
                        "EXIT_DETECTION",
                        "strict wall geometry weakening near possible corridor end",
                    )

                else:

                    self.request_transition(
                        "HOVER_AND_REASSESS",
                        "strict wall geometry unavailable for control",
                        "LOW_CONFIDENCE_GEOMETRY",
                    )   

            return

        self.strict_geometry_loss_streak = 0

        # 5) Decide whether stronger integrated correction is needed.
        enter_yaw = math.radians(
            float(self.get_parameter("correction_enter_yaw_deg").value)
        )
        exit_yaw = math.radians(
            float(self.get_parameter("correction_exit_yaw_deg").value)
        )
        enter_lat = float(
            self.get_parameter("correction_enter_lateral_m").value
        )
        exit_lat = float(
            self.get_parameter("correction_exit_lateral_m").value
        )

        correction_needed = (
            abs(g.yaw_error) >= enter_yaw
            or abs(g.lateral_error) >= enter_lat
        )

        if self.mode == CruiseMode.NOMINAL:
            if correction_needed:
                self.correction_enter_streak += 1
                if self.correction_enter_streak >= int(
                    self.get_parameter(
                        "correction_enter_confirm_scans"
                    ).value
                ):
                    self.enter_correction_mode(
                        g,
                        (
                            f"lat={g.lateral_error:+.2f} m, "
                            f"yaw={math.degrees(g.yaw_error):+.1f} deg"
                        ),
                    )
            else:
                self.correction_enter_streak = 0

        # 6) Strong correction, but still globally CORRIDOR_CRUISE.
        if self.mode == CruiseMode.CORRECTING:
            extreme_yaw = math.radians(
                float(self.get_parameter("correction_extreme_yaw_deg").value)
            )
            extreme_lat = float(
                self.get_parameter("correction_extreme_lateral_m").value
            )

            if (
                abs(g.yaw_error) >= extreme_yaw
                or abs(g.lateral_error) >= extreme_lat
            ):
                self.request_transition(
                    "HOVER_AND_REASSESS",
                    (
                        "correction exceeded safe envelope: "
                        f"lat={g.lateral_error:+.2f} m, "
                        f"yaw={math.degrees(g.yaw_error):+.1f} deg"
                    ),
                    "RECENTER_FAILED",
                )
                return

            metric = self.correction_metric(g)
            if metric < self.correction_best_metric:
                self.correction_best_metric = metric
                self.correction_worsening_streak = 0
            else:
                worsening_ratio = float(
                    self.get_parameter("correction_worsening_ratio").value
                )
                if (
                    math.isfinite(self.correction_best_metric)
                    and metric
                    > self.correction_best_metric * worsening_ratio
                ):
                    self.correction_worsening_streak += 1
                else:
                    self.correction_worsening_streak = 0

            if self.correction_worsening_streak >= int(
                self.get_parameter(
                    "correction_worsening_confirm_scans"
                ).value
            ):
                self.request_transition(
                    "HOVER_AND_REASSESS",
                    "integrated centering/yaw correction is worsening",
                    "RECENTER_FAILED",
                )
                return

            if self.correction_age() > float(
                self.get_parameter("correction_timeout_s").value
            ):
                self.request_transition(
                    "HOVER_AND_REASSESS",
                    "integrated centering/yaw correction timed out",
                    "RECENTER_FAILED",
                )
                return

            restored = (
                abs(g.yaw_error) <= exit_yaw
                and abs(g.lateral_error) <= exit_lat
            )
            if restored:
                self.correction_release_streak += 1
                if self.correction_release_streak >= int(
                    self.get_parameter(
                        "correction_release_confirm_scans"
                    ).value
                ):
                    self.leave_correction_mode()
            else:
                self.correction_release_streak = 0

            if self.mode == CruiseMode.CORRECTING:
                self.latest_command = self.correction_command(g)
                return

        # 7) Nominal corridor cruise.
        control_conf = float(
            self.get_parameter("control_confidence_min").value
        )
        if g.confidence < control_conf:
            self.latest_command = self.zero_command()
            return

        yaw_rate = self.cruise_yaw_control(g.yaw_error)
        vy_left = self.cruise_lateral_control(g.lateral_error)
        vx = self.adaptive_forward_speed(g)

        self.latest_command = self.make_command(
            vx=vx,
            vy_left=vy_left,
            yaw_rate=yaw_rate,
        )

    # ======================================================================
    # Controllers
    # ======================================================================

    def cruise_yaw_control(self, error_rad: float) -> float:
        deadband = math.radians(float(self.get_parameter("yaw_deadband_deg").value))
        if abs(error_rad) <= deadband:
            return 0.0

        max_rate = math.radians(float(self.get_parameter("max_yaw_rate_deg_s").value))
        k = float(self.get_parameter("k_yaw").value)
        return clamp(k * error_rad, -max_rate, max_rate)

    def cruise_lateral_control(self, error_m: float) -> float:
        deadband = float(self.get_parameter("lateral_deadband_m").value)
        if abs(error_m) <= deadband:
            return 0.0

        max_speed = float(self.get_parameter("max_lateral_speed_m_s").value)
        k = float(self.get_parameter("k_lateral").value)
        return clamp(k * error_m, -max_speed, max_speed)

    def adaptive_forward_speed(self, g: CorridorGeometry) -> float:
        nominal = float(self.get_parameter("nominal_forward_speed_m_s").value)
        minimum = float(self.get_parameter("minimum_forward_speed_m_s").value)
        minimum = clamp(minimum, 0.0, nominal)

        # Front-clearance factor: full speed when far away, approaches zero at
        # the obstacle trigger.  The event logic above will stop/transition at
        # or below the trigger itself.
        slow_start = float(self.get_parameter("front_slowdown_start_m").value)
        obstacle = float(self.get_parameter("front_obstacle_trigger_m").value)
        if g.front_clearance >= slow_start:
            front_factor = 1.0
        elif g.front_clearance <= obstacle:
            front_factor = 0.0
        else:
            front_factor = (g.front_clearance - obstacle) / max(slow_start - obstacle, 1e-6)

        # Confidence factor.
        nominal_conf = float(self.get_parameter("nominal_confidence").value)
        control_conf = float(self.get_parameter("control_confidence_min").value)
        if g.confidence >= nominal_conf:
            confidence_factor = 1.0
        elif g.confidence <= control_conf:
            confidence_factor = 0.25
        else:
            confidence_factor = 0.25 + 0.75 * (
                (g.confidence - control_conf) / max(nominal_conf - control_conf, 1e-6)
            )

        # Slow as we approach the threshold that engages stronger
        # integrated correction.
        yaw_trigger = math.radians(
            float(self.get_parameter("correction_enter_yaw_deg").value)
        )
        lat_trigger = float(
            self.get_parameter("correction_enter_lateral_m").value
        )

        yaw_factor = linear_score(abs(g.yaw_error), math.radians(1.5), yaw_trigger)
        lat_factor = linear_score(abs(g.lateral_error), 0.06, lat_trigger)
        alignment_factor = min(yaw_factor, lat_factor)

        factor = clamp(min(front_factor, confidence_factor, alignment_factor), 0.0, 1.0)
        if factor <= 0.0:
            return 0.0

        # Keep a small positive crawl only while all safety gates remain valid.
        return clamp(max(minimum, nominal * factor), 0.0, nominal)

    # ======================================================================
    # Geometry extraction
    # ======================================================================

    def corrected_scan_arrays(self, scan: LaserScan) -> tuple[np.ndarray, np.ndarray]:
        ranges = np.asarray(scan.ranges, dtype=np.float64)
        count = ranges.size
        angles = scan.angle_min + np.arange(count, dtype=np.float64) * scan.angle_increment
        angles = np.arctan2(np.sin(angles), np.cos(angles))

        if bool(self.get_parameter("invert_scan_angle").value):
            angles = -angles

        angles += math.radians(float(self.get_parameter("lidar_yaw_offset_deg").value))
        angles = np.arctan2(np.sin(angles), np.cos(angles))
        return ranges, angles

    def extract_corridor_geometry(self, scan: LaserScan) -> CorridorGeometry:
        g = CorridorGeometry()

        ranges, angles = self.corrected_scan_arrays(scan)
        if ranges.size < 20:
            return g

        sectors = self.sector_monitor.ingest(
            ranges=ranges,
            angles=angles,
            range_min=scan.range_min,
            range_max=scan.range_max,
            cone_deg=float(self.get_parameter("sector_cone_deg").value),
            percentile=float(self.get_parameter("sector_percentile").value),
            max_sector_range=float(self.get_parameter("sector_max_range_m").value),
            front_cone_deg=float(self.get_parameter("front_cone_deg").value),
        )
        g.front_clearance = sectors.front_clearance
        g.sectors = dict(sectors.values)

        # Exit-candidate evidence is available even when wall fitting fails.
        # Use the CURRENT scan for side-opening classification, not the
        # temporally filtered L/R sectors: at a real exit the old 1.75 m wall
        # readings would otherwise linger for several scans and could let the
        # low-confidence route fire before EXIT_DETECTION.
        self.classify_side_opening(g, ranges, angles, scan.range_min, scan.range_max)

        valid = np.isfinite(ranges)
        valid &= ranges >= max(float(scan.range_min), 0.05)
        valid &= ranges <= min(
            float(scan.range_max),
            float(self.get_parameter("geometry_max_range_m").value),
        )

        if np.count_nonzero(valid) < 20:
            self.finish_exit_candidate(g)
            return g

        r = ranges[valid]
        a = angles[valid]
        points = np.column_stack((r * np.cos(a), r * np.sin(a)))

        x_min = float(self.get_parameter("fit_x_min_m").value)
        x_max = float(self.get_parameter("fit_x_max_m").value)
        side_min = float(self.get_parameter("fit_side_min_m").value)
        side_max = float(self.get_parameter("fit_side_max_m").value)

        x = points[:, 0]
        y = points[:, 1]
        longitudinal = (x >= x_min) & (x <= x_max)

        left_points = points[
            longitudinal & (y >= side_min) & (y <= side_max)
        ]
        right_points = points[
            longitudinal & (y <= -side_min) & (y >= -side_max)
        ]

        left_line = self.fit_line_ransac(left_points)
        right_line = self.fit_line_ransac(right_points)
        g.left_wall_valid = left_line.valid
        g.right_wall_valid = right_line.valid

        if not left_line.valid or not right_line.valid:
            self.finish_exit_candidate(g)
            return g

        g.left_rms = left_line.rms
        g.right_rms = right_line.rms
        g.left_inliers = int(left_line.inlier_points.shape[0])
        g.right_inliers = int(right_line.inlier_points.shape[0])
        g.left_span = left_line.span
        g.right_span = right_line.span
        g.left_inlier_ratio = left_line.inlier_ratio
        g.right_inlier_ratio = right_line.inlier_ratio

        d1 = left_line.direction.copy()
        d2 = right_line.direction.copy()
        if d1[0] < 0.0:
            d1 = -d1
        if d2[0] < 0.0:
            d2 = -d2

        dot = clamp(float(np.dot(d1, d2)), -1.0, 1.0)
        g.parallel_error = math.acos(dot)

        corridor_direction = d1 + d2
        norm = float(np.linalg.norm(corridor_direction))
        if norm < 1e-6:
            self.finish_exit_candidate(g)
            return g
        corridor_direction /= norm
        if corridor_direction[0] < 0.0:
            corridor_direction = -corridor_direction

        g.yaw_error_raw = wrap_pi(
            math.atan2(float(corridor_direction[1]), float(corridor_direction[0]))
        )

        left_normal = np.array(
            [-corridor_direction[1], corridor_direction[0]], dtype=np.float64
        )

        left_coordinate = float(np.median(left_line.inlier_points @ left_normal))
        right_coordinate = float(np.median(right_line.inlier_points @ left_normal))

        # In CORRIDOR_CRUISE the aircraft should remain between the walls.
        if left_coordinate <= 0.0 or right_coordinate >= 0.0:
            self.finish_exit_candidate(g)
            return g

        g.d_left = left_coordinate
        g.d_right = -right_coordinate
        g.width_raw = left_coordinate - right_coordinate
        # Positive = corridor centre lies left of vehicle (+y FLU).
        g.lateral_error_raw = 0.5 * (left_coordinate + right_coordinate)

        corridor_width = float(self.get_parameter("corridor_width_m").value)
        strict_width_tol = float(self.get_parameter("width_tolerance_m").value)
        loose_width_tol = float(self.get_parameter("loose_width_tolerance_m").value)
        strict_parallel = math.radians(
            float(self.get_parameter("max_parallel_error_deg").value)
        )
        loose_parallel = math.radians(
            float(self.get_parameter("loose_parallel_error_deg").value)
        )
        max_yaw = math.radians(float(self.get_parameter("max_corridor_yaw_deg").value))
        min_span = float(self.get_parameter("min_wall_span_m").value)
        max_rms = float(self.get_parameter("max_fit_rms_m").value)
        min_inliers = int(self.get_parameter("min_wall_inliers").value)

        g.loose_valid = (
            abs(g.width_raw - corridor_width) <= loose_width_tol
            and g.parallel_error <= loose_parallel
            and abs(g.yaw_error_raw) <= max_yaw
        )

        g.strict_valid = (
            abs(g.width_raw - corridor_width) <= strict_width_tol
            and g.parallel_error <= strict_parallel
            and abs(g.yaw_error_raw) <= max_yaw
            and g.left_span >= min_span
            and g.right_span >= min_span
            and g.left_rms <= max_rms
            and g.right_rms <= max_rms
            and g.left_inliers >= min_inliers
            and g.right_inliers >= min_inliers
        )

        if g.loose_valid:
            self.geometry_history.push(
                g.yaw_error_raw,
                g.lateral_error_raw,
                g.width_raw,
            )

        if self.geometry_history.yaw:
            g.yaw_error, g.lateral_error, g.width = self.geometry_history.medians()
            g.temporal_score = self.geometry_history.stability_score()
        else:
            g.yaw_error = g.yaw_error_raw
            g.lateral_error = g.lateral_error_raw
            g.width = g.width_raw

        g.sector_score = self.compute_sector_score(
            sectors.values,
            left_line,
            right_line,
        )

        L = sectors.values.get("L")
        R = sectors.values.get("R")
        if L is not None and R is not None:
            g.observed_lr_sum = float(L + R)
            c = math.cos(g.yaw_error)
            if abs(c) > 0.20 and math.isfinite(g.width):
                g.expected_lr_sum = float(g.width / abs(c))
                g.lr_sum_residual = abs(g.observed_lr_sum - g.expected_lr_sum)

        g.confidence = self.compute_confidence(g)
        self.finish_exit_candidate(g)
        return g

    # ------------------------------------------------------------------
    # Exit-candidate classification
    # ------------------------------------------------------------------

    def classify_side_opening(
        self,
        g: CorridorGeometry,
        ranges: np.ndarray,
        angles: np.ndarray,
        range_min: float,
        range_max: float,
    ) -> None:
        """Classify left/right opening from the current scan only.

        This deliberately does NOT use the sector history because stale wall
        ranges are useful for control smoothing but harmful for exit onset
        detection.  No finite return inside the side cone is treated as open.
        """
        threshold = float(self.get_parameter("exit_side_open_m").value)
        half_cone = math.radians(float(self.get_parameter("sector_cone_deg").value))
        max_sector = min(
            float(range_max),
            float(self.get_parameter("sector_max_range_m").value),
        )

        valid = np.isfinite(ranges)
        valid &= ranges >= max(float(range_min), 0.05)
        valid &= ranges <= max_sector

        def side_open(target_deg: float) -> bool:
            target = math.radians(target_deg)
            delta = np.abs(
                np.arctan2(np.sin(angles - target), np.cos(angles - target))
            )
            vals = ranges[valid & (delta <= half_cone)]
            if vals.size == 0:
                return True
            # A median/central percentile avoids one spurious long beam
            # declaring an opening while a wall is still present.
            return float(np.percentile(vals, 50.0)) >= threshold

        g.side_open_left = side_open(90.0)
        g.side_open_right = side_open(-90.0)

    def finish_exit_candidate(self, g: CorridorGeometry) -> None:
        if self.session_age() < float(self.get_parameter("exit_guard_time_s").value):
            g.exit_candidate = False
            return

        front_open = g.front_clearance >= float(
            self.get_parameter("exit_front_open_m").value
        )

        # Strong candidate: both side walls look open and the forward path is
        # open.  The dedicated EXIT_DETECTION state must still confirm it.
        g.exit_candidate = bool(
            front_open and g.side_open_left and g.side_open_right
        )

    # ------------------------------------------------------------------
    # RANSAC line fitting
    # ------------------------------------------------------------------

    def fit_line_ransac(self, points: np.ndarray) -> LineModel:
        min_inliers = int(self.get_parameter("min_wall_inliers").value)
        if points.shape[0] < min_inliers:
            return LineModel()

        iterations = int(self.get_parameter("ransac_iterations").value)
        threshold = float(self.get_parameter("ransac_distance_m").value)
        max_angle = math.radians(
            float(self.get_parameter("max_wall_line_angle_deg").value)
        )
        min_span_ref = float(self.get_parameter("min_wall_span_m").value)

        n = points.shape[0]
        best_mask = None
        best_score = -math.inf

        for _ in range(iterations):
            i, j = self.rng.choice(n, size=2, replace=False)
            p1 = points[i]
            p2 = points[j]
            segment = p2 - p1
            segment_length = float(np.linalg.norm(segment))
            if segment_length < 0.20:
                continue

            direction = segment / segment_length
            if direction[0] < 0.0:
                direction = -direction

            candidate_angle = abs(
                math.atan2(float(direction[1]), float(direction[0]))
            )
            if candidate_angle > max_angle:
                continue

            normal = np.array([-direction[1], direction[0]], dtype=np.float64)
            residuals = np.abs((points - p1) @ normal)
            mask = residuals <= threshold
            count = int(np.count_nonzero(mask))
            if count < min_inliers:
                continue

            inliers = points[mask]
            projections = (inliers - p1) @ direction
            span = float(
                np.percentile(projections, 95.0)
                - np.percentile(projections, 5.0)
            )
            median_residual = float(np.median(residuals[mask]))

            # Prefer long, well-supported wall models over short obstacle faces.
            span_factor = 0.5 + min(span / max(min_span_ref, 0.1), 3.0)
            residual_factor = 1.0 / (1.0 + 10.0 * median_residual)
            score = count * span_factor * residual_factor

            if score > best_score:
                best_score = score
                best_mask = mask

        if best_mask is None:
            return LineModel()

        inliers = points[best_mask]
        if inliers.shape[0] < 2:
            return LineModel()

        centroid = np.mean(inliers, axis=0)
        centered = inliers - centroid
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        direction = vh[0]
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            return LineModel()
        direction /= norm
        if direction[0] < 0.0:
            direction = -direction

        refined_angle = abs(
            math.atan2(float(direction[1]), float(direction[0]))
        )
        if refined_angle > max_angle:
            return LineModel()

        normal = np.array([-direction[1], direction[0]], dtype=np.float64)
        residuals = np.abs(centered @ normal)
        rms = float(math.sqrt(np.mean(residuals * residuals)))

        along = centered @ direction
        span = float(np.percentile(along, 95.0) - np.percentile(along, 5.0))
        inlier_ratio = float(inliers.shape[0] / max(points.shape[0], 1))

        return LineModel(
            valid=True,
            point=centroid,
            direction=direction,
            inlier_points=inliers,
            rms=rms,
            span=span,
            inlier_ratio=inlier_ratio,
        )

    # ------------------------------------------------------------------
    # Sector-model validation
    # ------------------------------------------------------------------

    @staticmethod
    def ray_line_intersection_range(
        line: LineModel, bearing_rad: float
    ) -> Optional[float]:
        if not line.valid:
            return None

        ray = np.array(
            [math.cos(bearing_rad), math.sin(bearing_rad)],
            dtype=np.float64,
        )
        denom = cross2(ray, line.direction)
        if abs(denom) < 1e-8:
            return None

        distance = cross2(line.point, line.direction) / denom
        if distance <= 0.0 or not math.isfinite(distance):
            return None
        return float(distance)

    def compute_sector_score(
        self,
        observed: Dict[str, Optional[float]],
        left_line: LineModel,
        right_line: LineModel,
    ) -> float:
        if not bool(self.get_parameter("use_sector_validation").value):
            return 0.75

        lr_tol = float(self.get_parameter("sector_lr_tolerance_m").value)
        diag_tol = float(self.get_parameter("sector_diag_tolerance_m").value)

        checks = [
            ("L", left_line, math.radians(90.0), lr_tol, 1.0),
            ("R", right_line, math.radians(-90.0), lr_tol, 1.0),
            # Diagonals are weak evidence because an obstacle can legitimately
            # appear before the wall.
            ("FL", left_line, math.radians(45.0), diag_tol, 0.35),
            ("FR", right_line, math.radians(-45.0), diag_tol, 0.35),
        ]

        weighted_score = 0.0
        total_weight = 0.0
        strong_seen = 0

        for name, line, bearing, tol, weight in checks:
            obs = observed.get(name)
            pred = self.ray_line_intersection_range(line, bearing)
            if obs is None or pred is None:
                continue

            residual = abs(obs - pred)
            score = linear_score(residual, 0.08, tol)
            weighted_score += weight * score
            total_weight += weight
            if name in ("L", "R"):
                strong_seen += 1

        if total_weight <= 1e-9:
            return 0.50

        score = weighted_score / total_weight
        if strong_seen == 0:
            score = min(score, 0.55)
        elif strong_seen == 1:
            score = min(score, 0.75)

        return float(clamp(score, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Geometry confidence
    # ------------------------------------------------------------------

    def compute_confidence(self, g: CorridorGeometry) -> float:
        corridor_width = float(self.get_parameter("corridor_width_m").value)
        loose_width_tol = float(self.get_parameter("loose_width_tolerance_m").value)
        loose_parallel = math.radians(
            float(self.get_parameter("loose_parallel_error_deg").value)
        )
        max_rms = float(self.get_parameter("max_fit_rms_m").value)
        min_span = float(self.get_parameter("min_wall_span_m").value)
        min_inliers = int(self.get_parameter("min_wall_inliers").value)

        wall_scores = []
        for rms, span, inliers in (
            (g.left_rms, g.left_span, g.left_inliers),
            (g.right_rms, g.right_span, g.right_inliers),
        ):
            rms_score = linear_score(rms, 0.025, max_rms * 1.4)
            span_score = clamp(span / max(1.5 * min_span, 0.1), 0.0, 1.0)
            inlier_score = clamp(inliers / max(2.0 * min_inliers, 1.0), 0.0, 1.0)
            wall_scores.append(
                0.45 * rms_score + 0.30 * span_score + 0.25 * inlier_score
            )
        fit_score = float(np.mean(wall_scores)) if wall_scores else 0.0

        width_score = linear_score(
            abs(g.width_raw - corridor_width),
            0.12,
            loose_width_tol,
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
            confidence = min(confidence, 0.35)

        return float(clamp(confidence, 0.0, 1.0))

    # ======================================================================
    # Publishing / watchdog
    # ======================================================================

    def make_command(
        self,
        vx: float = 0.0,
        vy_left: float = 0.0,
        vz_up: float = 0.0,
        yaw_rate: float = 0.0,
    ) -> TwistStamped:
        msg = TwistStamped()
        msg.header.frame_id = "base_link"
        msg.twist.linear.x = float(vx)
        msg.twist.linear.y = float(vy_left)
        msg.twist.linear.z = float(vz_up)
        msg.twist.angular.z = float(yaw_rate)
        return msg

    def zero_command(self) -> TwistStamped:
        return self.make_command()

    def diagnostics_dict(self) -> Dict[str, object]:
        g = self.last_geometry
        base = {
            "state": self.state.name,
            "control_mode": self.mode.name,
            "enabled": self.enabled,
            "transition_target": self.transition_target,
            "transition_reason": self.transition_reason,
            "pause_reason": self.pause_reason,
            "session_age_s": round(self.session_age(), 3),
            "streaks": {
                "obstacle": self.obstacle_streak,
                "low_confidence": self.low_conf_streak,
                "strict_geometry_loss": self.strict_geometry_loss_streak,
                "exit": self.exit_streak,
                "correction_enter": self.correction_enter_streak,
                "correction_release": self.correction_release_streak,
                "correction_worsening": self.correction_worsening_streak,
            },
            "correction": {
                "age_s": round(self.correction_age(), 3),
                "best_metric": (
                    round(self.correction_best_metric, 3)
                    if math.isfinite(self.correction_best_metric)
                    else None
                ),
            },
            "command": {
                "vx": round(float(self.latest_command.twist.linear.x), 4),
                "vy_left": round(float(self.latest_command.twist.linear.y), 4),
                "yaw_rate_deg_s": round(
                    math.degrees(float(self.latest_command.twist.angular.z)), 3
                ),
            },
        }

        if g is None:
            return base

        base.update(
            {
                "geometry": {
                    "strict_valid": g.strict_valid,
                    "loose_valid": g.loose_valid,
                    "left_wall_valid": g.left_wall_valid,
                    "right_wall_valid": g.right_wall_valid,
                    "confidence": round(g.confidence, 3),
                    "yaw_error_deg": round(math.degrees(g.yaw_error), 3),
                    "lateral_error_m": round(g.lateral_error, 3),
                    "width_m": round(g.width, 3) if math.isfinite(g.width) else None,
                    "d_left_m": round(g.d_left, 3) if math.isfinite(g.d_left) else None,
                    "d_right_m": round(g.d_right, 3) if math.isfinite(g.d_right) else None,
                    "parallel_error_deg": (
                        round(math.degrees(g.parallel_error), 3)
                        if math.isfinite(g.parallel_error)
                        else None
                    ),
                    "sector_score": round(g.sector_score, 3),
                    "temporal_score": round(g.temporal_score, 3),
                    "left_rms_m": round(g.left_rms, 4) if math.isfinite(g.left_rms) else None,
                    "right_rms_m": round(g.right_rms, 4) if math.isfinite(g.right_rms) else None,
                    "left_span_m": round(g.left_span, 3),
                    "right_span_m": round(g.right_span, 3),
                },
                "safety": {
                    "front_clearance_m": round(g.front_clearance, 3),
                    "exit_candidate": g.exit_candidate,
                    "side_open_left": g.side_open_left,
                    "side_open_right": g.side_open_right,
                },
                "sectors": {
                    k: (round(v, 3) if v is not None else None)
                    for k, v in g.sectors.items()
                },
                "lr_sanity": {
                    "observed_sum_m": (
                        round(g.observed_lr_sum, 3)
                        if g.observed_lr_sum is not None
                        else None
                    ),
                    "expected_sum_m": (
                        round(g.expected_lr_sum, 3)
                        if g.expected_lr_sum is not None
                        else None
                    ),
                    "residual_m": (
                        round(g.lr_sum_residual, 3)
                        if g.lr_sum_residual is not None
                        else None
                    ),
                },
            }
        )
        return base

    def publish_outputs(self) -> None:
        now = self.get_clock().now()

        # D500 stale-scan watchdog.
        
        # D500 stale-scan watchdog.
        if self.enabled and self.state == CruiseState.ACTIVE:
            stale_limit = float(self.get_parameter("scan_stale_s").value)

            if self.last_scan_time is None:
        # Cruise may be enabled between two LiDAR scans.
        # Allow one stale-limit interval for the first fresh scan to arrive.
                scan_stale = self.session_age() > stale_limit
            else:
                scan_age = (now - self.last_scan_time).nanoseconds * 1e-9
                scan_stale = scan_age > stale_limit

            if scan_stale:
                self.request_transition(
                    "HOVER_AND_REASSESS",
                    "LaserScan stale",
                    "LOW_CONFIDENCE_GEOMETRY",
                )
                
        
            
        self.latest_command.header.stamp = now.to_msg()
        self.cmd_pub.publish(self.latest_command)

        state_msg = String()
        if self.state == CruiseState.ACTIVE:
            state_msg.data = "CORRIDOR_CRUISE"
        elif self.state == CruiseState.TRANSITION_REQUESTED:
            state_msg.data = self.transition_target or "TRANSITION_REQUESTED"
        else:
            state_msg.data = "IDLE"
        self.state_pub.publish(state_msg)

        mode_msg = String()
        mode_msg.data = self.mode.name
        self.mode_pub.publish(mode_msg)

        result_msg = String()
        next_msg = String()

        if self.state == CruiseState.TRANSITION_REQUESTED:
            suffix = self.transition_reason or ""
            result_msg.data = f"TRANSITION:{self.transition_target}:{suffix}"
            next_msg.data = self.transition_target or ""
        elif self.state == CruiseState.ACTIVE:
            result_msg.data = f"ACTIVE:{self.mode.name}"
            next_msg.data = ""
        else:
            result_msg.data = "IDLE"
            next_msg.data = ""

        self.result_pub.publish(result_msg)
        self.next_state_pub.publish(next_msg)

        diag_msg = String()
        diag_msg.data = json.dumps(self.diagnostics_dict(), separators=(",", ":"))
        self.diagnostics_pub.publish(diag_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CorridorCruiseV2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
