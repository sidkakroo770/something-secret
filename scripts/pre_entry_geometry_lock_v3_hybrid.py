#!/usr/bin/env python3
"""Hybrid PRE_ENTRY_GEOMETRY_LOCK for SAEINDIA AeroTHON 2026.

Designed for a D500 2-D LiDAR publishing sensor_msgs/LaserScan on /scan.

Purpose
-------
This node implements ONLY the pre-entry geometry-lock stage:

    PRE_ENTRY_HOVER -> PRE_ENTRY_GEOMETRY_LOCK -> ENTER_CORRIDOR
                                      |
                                      +-> HOVER_AND_REASSESS (failure)

During this state the vehicle is assumed to already be at / just inside the
corridor mouth.  Therefore this node NEVER commands forward motion.
It commands only:

    * body-frame yaw rate, to become parallel to the corridor walls
    * body-frame lateral velocity, to move onto the corridor centreline

Primary estimator
-----------------
The full D500 scan is converted to XY points and robust straight lines are fit
to the left and right walls with RANSAC.  Wall direction gives yaw error;
wall position gives lateral error.

Secondary validator (best part of the sector-based prototype)
--------------------------------------------------------------
Narrow L / FL / F / FR / R sectors are also extracted and temporally median
filtered.  They do NOT drive the vehicle directly.  Instead they validate the
wall model, provide front safety, and make diagnostics much easier to inspect.
If the sector measurements disagree with the RANSAC model, geometry confidence
falls and the vehicle holds instead of averaging conflicting controllers.

Robustness additions
--------------------
* temporal median filtering of yaw / lateral / width estimates
* confidence score built from fit quality, corridor width, parallelism,
  sector-model agreement, and temporal stability
* hysteresis between "start correcting" and "correction complete" thresholds
* scan-stale watchdog
* optional IMU roll/pitch safety gate
* strict no-forward-motion contract for this FSM state
* latched failure result for mission-manager handoff

ROS convention
--------------
Published command is ROS FLU body convention:
    +x forward, +y left, +z up, +yaw counter-clockwise.

The MAVLink bridge must convert ROS FLU to the autopilot's expected frame.
Do not wire this topic directly to motors / attitude outputs.
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
# Small helpers
# ============================================================================


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wrap_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def linear_score(error: float, good: float, bad: float) -> float:
    """1 at/below good error, 0 at/above bad error, linear in between."""
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
    # Geometry validity
    strict_valid: bool = False
    loose_valid: bool = False

    # Raw (single scan) estimates
    yaw_error_raw: float = 0.0
    lateral_error_raw: float = 0.0
    width_raw: float = math.inf

    # Filtered estimates used by the controller
    yaw_error: float = 0.0
    lateral_error: float = 0.0
    width: float = math.inf

    # Wall geometry / quality
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

    # Safety / validation
    front_clearance: float = 0.0
    sector_score: float = 0.5
    temporal_score: float = 0.7
    confidence: float = 0.0
    observed_lr_sum: Optional[float] = None
    expected_lr_sum: Optional[float] = None
    lr_sum_residual: Optional[float] = None

    sectors: Dict[str, Optional[float]] = field(default_factory=dict)


# ============================================================================
# Sector monitor -- validator, not primary controller
# ============================================================================


class SectorMonitor:
    """Extract robust L/FL/F/FR/R ranges from a corrected LaserScan angle array.

    Standard ROS FLU bearings are used after optional scan inversion / mounting
    offset has already been applied:
        F=0, FL=+45, L=+90, FR=-45, R=-90 degrees.

    Each scan uses a percentile inside a small cone rather than a raw minimum.
    A median across recent scans removes spikes/dropouts.
    """

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
            key: deque(maxlen=self.window) for key in self.BEARINGS_DEG
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
                # Age old measurements out instead of keeping a stale sector
                # value forever when that direction stops returning data.
                self.histories[name].append(float("nan"))

            hist = self.histories[name]
            finite_hist = [v for v in hist if math.isfinite(v)]
            snapshot.values[name] = (
                float(np.median(finite_hist)) if finite_hist else None
            )

        # Front safety is deliberately separate and more conservative.
        # +inf means no return and is treated as sensor max-range for clearance.
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
            # No trustworthy data in the front cone is unsafe.
            snapshot.front_clearance = 0.0

        return snapshot


# ============================================================================
# Temporal filtering for full-wall geometry
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
# Main node
# ============================================================================


class PreEntryGeometryLockV3(Node):
    def __init__(self) -> None:
        super().__init__("pre_entry_geometry_lock_v3")

        # ------------------------------------------------------------------
        # Topics
        # ------------------------------------------------------------------
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("imu_topic", "/imu/data")
        self.declare_parameter("enable_topic", "/corridor/pre_entry/enable")
        self.declare_parameter("cmd_topic", "/corridor/pre_entry/cmd_vel")
        self.declare_parameter("state_topic", "/corridor/pre_entry/state")
        self.declare_parameter("locked_topic", "/corridor/pre_entry/locked")
        self.declare_parameter("result_topic", "/corridor/pre_entry/result")
        self.declare_parameter("next_state_topic", "/corridor/pre_entry/next_state")
        self.declare_parameter("diagnostics_topic", "/corridor/pre_entry/diagnostics")

        # ------------------------------------------------------------------
        # D500 / scan orientation
        # ------------------------------------------------------------------
        # After correction this node assumes ROS FLU angles:
        # 0 front, +90 left, -90 right.
        self.declare_parameter("invert_scan_angle", False)
        self.declare_parameter("lidar_yaw_offset_deg", 0.0)
        self.declare_parameter("geometry_max_range_m", 8.0)

        # ------------------------------------------------------------------
        # Known corridor geometry (AeroTHON mission: 3.5 m)
        # ------------------------------------------------------------------
        self.declare_parameter("corridor_width_m", 3.5)
        self.declare_parameter("width_tolerance_m", 0.45)
        self.declare_parameter("loose_width_tolerance_m", 0.80)
        self.declare_parameter("max_parallel_error_deg", 7.0)
        self.declare_parameter("loose_parallel_error_deg", 15.0)
        self.declare_parameter("max_corridor_yaw_deg", 35.0)

        # ------------------------------------------------------------------
        # Point selection / wall RANSAC
        # ------------------------------------------------------------------
        self.declare_parameter("fit_x_min_m", -0.20)
        self.declare_parameter("fit_x_max_m", 5.0)
        self.declare_parameter("fit_side_min_m", 0.25)
        self.declare_parameter("fit_side_max_m", 3.5)
        self.declare_parameter("ransac_iterations", 120)
        self.declare_parameter("ransac_distance_m", 0.08)
        self.declare_parameter("min_wall_inliers", 14)
        self.declare_parameter("min_wall_span_m", 0.80)
        self.declare_parameter("max_fit_rms_m", 0.10)
        self.declare_parameter("max_wall_line_angle_deg", 45.0)

        # ------------------------------------------------------------------
        # Sector validator
        # ------------------------------------------------------------------
        self.declare_parameter("use_sector_validation", True)
        self.declare_parameter("sector_cone_deg", 6.0)
        self.declare_parameter("sector_percentile", 35.0)
        self.declare_parameter("sector_filter_window", 5)
        self.declare_parameter("sector_max_range_m", 6.0)
        self.declare_parameter("sector_lr_tolerance_m", 0.35)
        self.declare_parameter("sector_diag_tolerance_m", 0.55)

        # ------------------------------------------------------------------
        # Temporal geometry filtering / confidence
        # ------------------------------------------------------------------
        self.declare_parameter("geometry_filter_window", 3)
        self.declare_parameter("control_confidence_min", 0.62)
        self.declare_parameter("verify_confidence_min", 0.72)

        # ------------------------------------------------------------------
        # Safety
        # ------------------------------------------------------------------
        self.declare_parameter("front_cone_deg", 18.0)
        self.declare_parameter("front_stop_m", 0.80)
        self.declare_parameter("scan_stale_s", 0.30)
        self.declare_parameter("require_imu", False)
        self.declare_parameter("max_tilt_deg", 8.0)

        # ------------------------------------------------------------------
        # Controller: NO FORWARD MOTION in this state
        # ------------------------------------------------------------------
        self.declare_parameter("k_yaw", 0.9)
        self.declare_parameter("max_yaw_rate_deg_s", 10.0)
        self.declare_parameter("min_yaw_rate_deg_s", 1.5)
        self.declare_parameter("k_lateral", 0.45)
        self.declare_parameter("max_lateral_speed_m_s", 0.18)
        self.declare_parameter("min_lateral_speed_m_s", 0.03)

        # Hysteresis: start correcting at larger error, stop at smaller error.
        self.declare_parameter("yaw_realign_deg", 4.5)
        self.declare_parameter("yaw_aligned_deg", 2.5)
        self.declare_parameter("yaw_lock_deg", 3.0)
        self.declare_parameter("lateral_recenter_m", 0.18)
        self.declare_parameter("lateral_centered_m", 0.10)
        self.declare_parameter("lateral_lock_m", 0.12)

        # ------------------------------------------------------------------
        # FSM timing
        # ------------------------------------------------------------------
        self.declare_parameter("verify_scans", 6)
        self.declare_parameter("geometry_grace_scans", 2)
        self.declare_parameter("acquire_timeout_s", 8.0)
        self.declare_parameter("alignment_timeout_s", 15.0)
        self.declare_parameter("hold_recovery_scans", 5)

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
        self.locked_pub = self.create_publisher(
            Bool, str(self.get_parameter("locked_topic").value), 10
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

        # 20 Hz command/status publishing. D500 scans are normally ~10 Hz.
        self.timer = self.create_timer(0.05, self.publish_outputs)

        # Runtime state
        self.enabled = False
        self.state = LockState.IDLE
        self.state_enter_time = self.get_clock().now()
        self.session_start_time = self.get_clock().now()
        self.alignment_start_time = None
        self.last_scan_time = None
        self.last_geometry: Optional[CorridorGeometry] = None
        self.latest_command = self.zero_command()

        self.roll: Optional[float] = None
        self.pitch: Optional[float] = None

        self.stable_scan_count = 0
        self.invalid_scan_count = 0
        self.hold_recovery_count = 0
        self.failure_reason: Optional[str] = None

        self.sector_monitor = SectorMonitor(
            int(self.get_parameter("sector_filter_window").value)
        )
        self.geometry_history = GeometryHistory(
            int(self.get_parameter("geometry_filter_window").value)
        )

        self.rng = np.random.default_rng(2026)
        self.get_logger().info(
            "Hybrid D500 pre-entry geometry lock V3 started (vx is hard-fixed to 0)."
        )

    # ======================================================================
    # Lifecycle / callbacks
    # ======================================================================

    def reset_session(self) -> None:
        self.failure_reason = None
        self.stable_scan_count = 0
        self.invalid_scan_count = 0
        self.hold_recovery_count = 0
        self.alignment_start_time = None
        self.last_geometry = None
        self.latest_command = self.zero_command()
        self.sector_monitor.reset()
        self.geometry_history.reset()
        self.session_start_time = self.get_clock().now()

    def enable_callback(self, msg: Bool) -> None:
        if msg.data and not self.enabled:
            self.reset_session()
            self.enabled = True
            self.transition(LockState.ACQUIRE_GEOMETRY, "enabled")
        elif not msg.data and self.enabled:
            self.enabled = False
            self.failure_reason = None
            self.transition(LockState.IDLE, "disabled")
            self.latest_command = self.zero_command()

    def imu_callback(self, msg: Imu) -> None:
        self.roll, self.pitch = quaternion_to_roll_pitch(msg)

    def scan_callback(self, scan: LaserScan) -> None:
        self.last_scan_time = self.get_clock().now()

        if not self.enabled or self.failure_reason is not None:
            return

        g = self.extract_corridor_geometry(scan)
        self.last_geometry = g
        self.step_fsm(g)

    def transition(self, new_state: LockState, reason: str) -> None:
        if new_state == self.state:
            return

        old_state = self.state
        self.state = new_state
        self.state_enter_time = self.get_clock().now()
        self.stable_scan_count = 0
        self.invalid_scan_count = 0

        if new_state in (
            LockState.ALIGN_YAW,
            LockState.CENTER_LATERALLY,
            LockState.VERIFY_LOCK,
        ) and self.alignment_start_time is None:
            self.alignment_start_time = self.get_clock().now()

        if new_state != LockState.HOLD:
            self.hold_recovery_count = 0

        self.get_logger().info(f"FSM {old_state.name} -> {new_state.name}: {reason}")

    def fail(self, reason: str) -> None:
        self.failure_reason = reason
        self.latest_command = self.zero_command()
        self.transition(LockState.HOLD, f"FAILED: {reason}")
        self.get_logger().error(f"PRE_ENTRY_GEOMETRY_LOCK failed: {reason}")

    def state_age(self) -> float:
        return (self.get_clock().now() - self.state_enter_time).nanoseconds * 1e-9

    def alignment_age(self) -> float:
        if self.alignment_start_time is None:
            return 0.0
        return (
            self.get_clock().now() - self.alignment_start_time
        ).nanoseconds * 1e-9

    def attitude_is_safe(self) -> bool:
        require_imu = bool(self.get_parameter("require_imu").value)
        if self.roll is None or self.pitch is None:
            return not require_imu

        max_tilt = math.radians(float(self.get_parameter("max_tilt_deg").value))
        return abs(self.roll) <= max_tilt and abs(self.pitch) <= max_tilt

    # ======================================================================
    # FSM
    # ======================================================================

    def step_fsm(self, g: CorridorGeometry) -> None:
        # Hard safety gates first.
        if not self.attitude_is_safe():
            self.latest_command = self.zero_command()
            self.transition(LockState.HOLD, "roll/pitch exceeds tilt limit")
            return

        front_stop = float(self.get_parameter("front_stop_m").value)
        if g.front_clearance < front_stop:
            self.latest_command = self.zero_command()
            self.transition(LockState.HOLD, "front clearance below stop distance")
            return

        control_conf = float(self.get_parameter("control_confidence_min").value)
        verify_conf = float(self.get_parameter("verify_confidence_min").value)

        yaw_realign = math.radians(float(self.get_parameter("yaw_realign_deg").value))
        yaw_aligned = math.radians(float(self.get_parameter("yaw_aligned_deg").value))
        yaw_lock = math.radians(float(self.get_parameter("yaw_lock_deg").value))
        lat_recenter = float(self.get_parameter("lateral_recenter_m").value)
        lat_centered = float(self.get_parameter("lateral_centered_m").value)
        lat_lock = float(self.get_parameter("lateral_lock_m").value)

        verify_scans = int(self.get_parameter("verify_scans").value)
        grace_scans = int(self.get_parameter("geometry_grace_scans").value)

        if self.state == LockState.IDLE:
            self.latest_command = self.zero_command()
            return

        # ------------------------------------------------------------------
        # ACQUIRE_GEOMETRY: hover and wait for trustworthy full-wall geometry.
        # No forward creep. That belongs to another FSM state if ever needed.
        # ------------------------------------------------------------------
        if self.state == LockState.ACQUIRE_GEOMETRY:
            self.latest_command = self.zero_command()

            if g.strict_valid and g.confidence >= control_conf:
                if abs(g.yaw_error) > yaw_aligned:
                    self.transition(LockState.ALIGN_YAW, "trustworthy walls acquired")
                elif abs(g.lateral_error) > lat_centered:
                    self.transition(
                        LockState.CENTER_LATERALLY,
                        "yaw already acceptable; centre laterally",
                    )
                else:
                    self.transition(LockState.VERIFY_LOCK, "already near alignment")
                return

            if self.state_age() > float(self.get_parameter("acquire_timeout_s").value):
                self.fail("LOW_CONFIDENCE_GEOMETRY")
            return

        # ------------------------------------------------------------------
        # ALIGN_YAW: rotate in place. vx=0, vy=0.
        # ------------------------------------------------------------------
        if self.state == LockState.ALIGN_YAW:
            if not g.strict_valid or g.confidence < control_conf:
                self.invalid_scan_count += 1
                self.latest_command = self.zero_command()
                if self.invalid_scan_count > grace_scans:
                    self.transition(LockState.HOLD, "geometry confidence lost during yaw")
                return

            self.invalid_scan_count = 0

            if self.alignment_age() > float(
                self.get_parameter("alignment_timeout_s").value
            ):
                self.fail("ENTRY_LOCK_FAILED")
                return

            if abs(g.yaw_error) <= yaw_aligned:
                self.latest_command = self.zero_command()
                if abs(g.lateral_error) > lat_centered:
                    self.transition(
                        LockState.CENTER_LATERALLY,
                        "yaw aligned; correct lateral position",
                    )
                else:
                    self.transition(LockState.VERIFY_LOCK, "yaw and lateral near target")
                return

            yaw_rate = self.yaw_control(g.yaw_error)
            self.latest_command = self.make_command(yaw_rate=yaw_rate)
            return

        # ------------------------------------------------------------------
        # CENTER_LATERALLY: translate left/right while keeping yaw trimmed.
        # ------------------------------------------------------------------
        if self.state == LockState.CENTER_LATERALLY:
            if not g.strict_valid or g.confidence < control_conf:
                self.invalid_scan_count += 1
                self.latest_command = self.zero_command()
                if self.invalid_scan_count > grace_scans:
                    self.transition(
                        LockState.HOLD, "geometry confidence lost while centring"
                    )
                return

            self.invalid_scan_count = 0

            if self.alignment_age() > float(
                self.get_parameter("alignment_timeout_s").value
            ):
                self.fail("ENTRY_LOCK_FAILED")
                return

            # Hysteresis: if yaw grows beyond the larger threshold, stop
            # translating and re-enter yaw alignment.
            if abs(g.yaw_error) > yaw_realign:
                self.latest_command = self.zero_command()
                self.transition(LockState.ALIGN_YAW, "yaw drift exceeded hysteresis")
                return

            if abs(g.lateral_error) <= lat_centered:
                self.latest_command = self.zero_command()
                self.transition(LockState.VERIFY_LOCK, "lateral centre reached")
                return

            vy_left = self.lateral_control(g.lateral_error)

            # Small yaw trim is allowed during lateral translation, but at
            # half gain and only while yaw remains inside the realign band.
            yaw_rate = 0.0
            if abs(g.yaw_error) > math.radians(0.8):
                yaw_rate = 0.5 * self.yaw_control(g.yaw_error)

            self.latest_command = self.make_command(
                vy_left=vy_left,
                yaw_rate=yaw_rate,
            )
            return

        # ------------------------------------------------------------------
        # VERIFY_LOCK: no motion. Require several consecutive good D500 scans.
        # ------------------------------------------------------------------
        if self.state == LockState.VERIFY_LOCK:
            self.latest_command = self.zero_command()

            good = (
                g.strict_valid
                and g.confidence >= verify_conf
                and abs(g.yaw_error) <= yaw_lock
                and abs(g.lateral_error) <= lat_lock
                and g.front_clearance >= front_stop
                and self.attitude_is_safe()
            )

            if good:
                self.stable_scan_count += 1
                if self.stable_scan_count >= verify_scans:
                    self.transition(LockState.LOCKED, "stable geometry lock achieved")
                return

            self.stable_scan_count = 0

            if not g.strict_valid or g.confidence < control_conf:
                self.invalid_scan_count += 1
                if self.invalid_scan_count > grace_scans:
                    self.transition(LockState.HOLD, "geometry invalid during verification")
                return

            self.invalid_scan_count = 0

            if abs(g.yaw_error) > yaw_realign:
                self.transition(LockState.ALIGN_YAW, "yaw left verification band")
            elif abs(g.lateral_error) > lat_recenter:
                self.transition(
                    LockState.CENTER_LATERALLY,
                    "lateral position left verification band",
                )
            return

        # ------------------------------------------------------------------
        # LOCKED: mission manager should transition to ENTER_CORRIDOR.
        # ------------------------------------------------------------------
        if self.state == LockState.LOCKED:
            self.latest_command = self.zero_command()
            return

        # ------------------------------------------------------------------
        # HOLD: transient safety hold can self-recover; latched failure cannot.
        # ------------------------------------------------------------------
        if self.state == LockState.HOLD:
            self.latest_command = self.zero_command()

            if self.failure_reason is not None:
                return

            recoverable = (
                g.strict_valid
                and g.confidence >= control_conf
                and g.front_clearance >= front_stop
                and self.attitude_is_safe()
            )
            if recoverable:
                self.hold_recovery_count += 1
                if self.hold_recovery_count >= int(
                    self.get_parameter("hold_recovery_scans").value
                ):
                    self.transition(
                        LockState.ACQUIRE_GEOMETRY,
                        "fresh stable geometry recovered",
                    )
            else:
                self.hold_recovery_count = 0
            return

    # ======================================================================
    # Controllers
    # ======================================================================

    def yaw_control(self, error_rad: float) -> float:
        max_rate = math.radians(float(self.get_parameter("max_yaw_rate_deg_s").value))
        min_rate = math.radians(float(self.get_parameter("min_yaw_rate_deg_s").value))
        k = float(self.get_parameter("k_yaw").value)

        cmd = clamp(k * error_rad, -max_rate, max_rate)
        if abs(cmd) < min_rate and abs(error_rad) > 1e-6:
            cmd = math.copysign(min_rate, cmd)
        return cmd

    def lateral_control(self, error_m: float) -> float:
        max_speed = float(self.get_parameter("max_lateral_speed_m_s").value)
        min_speed = float(self.get_parameter("min_lateral_speed_m_s").value)
        k = float(self.get_parameter("k_lateral").value)

        cmd = clamp(k * error_m, -max_speed, max_speed)
        if abs(cmd) < min_speed and abs(error_m) > 1e-6:
            cmd = math.copysign(min_speed, cmd)
        return cmd

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

        valid = np.isfinite(ranges)
        valid &= ranges >= max(float(scan.range_min), 0.05)
        valid &= ranges <= min(
            float(scan.range_max),
            float(self.get_parameter("geometry_max_range_m").value),
        )

        if np.count_nonzero(valid) < 20:
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

        if not left_line.valid or not right_line.valid:
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
            return g
        corridor_direction /= norm
        if corridor_direction[0] < 0.0:
            corridor_direction = -corridor_direction

        g.yaw_error_raw = wrap_pi(
            math.atan2(float(corridor_direction[1]), float(corridor_direction[0]))
        )

        # Normal points to the left of the estimated corridor direction.
        left_normal = np.array(
            [-corridor_direction[1], corridor_direction[0]], dtype=np.float64
        )

        left_coordinate = float(np.median(left_line.inlier_points @ left_normal))
        right_coordinate = float(np.median(right_line.inlier_points @ left_normal))

        # Since PRE_ENTRY_GEOMETRY_LOCK starts at the mouth / within the
        # corridor, the vehicle origin should lie between the wall lines.
        if left_coordinate <= 0.0 or right_coordinate >= 0.0:
            return g

        g.d_left = left_coordinate
        g.d_right = -right_coordinate
        g.width_raw = left_coordinate - right_coordinate

        # Positive means corridor centre lies to the vehicle's left (+y FLU).
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

        # Add plausible geometry to the short temporal filter.  We use loose
        # validity here so a single scan just outside strict thresholds does
        # not destroy the history.
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

        # Sector-model agreement uses actual measured sectors as an independent
        # check on the RANSAC wall model.  It never directly commands motion.
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
        return g

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

            # Walls should run approximately along the corridor, not across it.
            candidate_angle = abs(math.atan2(float(direction[1]), float(direction[0])))
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
            span = float(np.percentile(projections, 95.0) - np.percentile(projections, 5.0))
            median_residual = float(np.median(residuals[mask]))

            # Prefer models supported by many points AND long wall span.
            # This makes short obstacle faces less attractive than a long wall.
            span_factor = 0.5 + min(span / max(min_span_ref, 0.1), 3.0)
            residual_factor = 1.0 / (1.0 + 10.0 * median_residual)
            score = count * span_factor * residual_factor

            if score > best_score:
                best_score = score
                best_mask = mask

        if best_mask is None:
            return LineModel()

        inliers = points[best_mask]
        centroid = np.mean(inliers, axis=0)
        centered = inliers - centroid

        if inliers.shape[0] < 2:
            return LineModel()

        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        direction = vh[0]
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            return LineModel()
        direction /= norm
        if direction[0] < 0.0:
            direction = -direction

        refined_angle = abs(math.atan2(float(direction[1]), float(direction[0])))
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

        ray = np.array([math.cos(bearing_rad), math.sin(bearing_rad)], dtype=np.float64)
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
            # Diagonal beams can legitimately hit an obstacle before a wall,
            # so they are only weak supporting evidence.
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

        # If one/both orthogonal side checks are unavailable, don't let weak
        # diagonal checks alone create a falsely high validation score.
        if strong_seen == 0:
            score = min(score, 0.55)
        elif strong_seen == 1:
            score = min(score, 0.75)

        return float(clamp(score, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Confidence score
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

        # Wall fit quality: RMS + span + inlier count, averaged across walls.
        wall_scores = []
        for rms, span, inliers in (
            (g.left_rms, g.left_span, g.left_inliers),
            (g.right_rms, g.right_span, g.right_inliers),
        ):
            rms_score = linear_score(rms, 0.025, max_rms * 1.4)
            span_score = clamp(span / max(1.5 * min_span, 0.1), 0.0, 1.0)
            inlier_score = clamp(inliers / max(2.0 * min_inliers, 1.0), 0.0, 1.0)
            wall_scores.append(0.45 * rms_score + 0.30 * span_score + 0.25 * inlier_score)
        fit_score = float(np.mean(wall_scores)) if wall_scores else 0.0

        width_error = abs(g.width_raw - corridor_width)
        width_score = linear_score(width_error, 0.12, loose_width_tol)

        parallel_score = linear_score(
            g.parallel_error,
            math.radians(1.5),
            loose_parallel,
        )

        sector_score = g.sector_score
        temporal_score = g.temporal_score

        # Primary full-wall geometry dominates. Sector evidence is a validator,
        # not an equal estimator.
        confidence = (
            0.34 * fit_score
            + 0.22 * width_score
            + 0.16 * parallel_score
            + 0.16 * sector_score
            + 0.12 * temporal_score
        )

        # A non-loose geometry hypothesis should never be treated as confident.
        if not g.loose_valid:
            confidence = min(confidence, 0.35)

        return float(clamp(confidence, 0.0, 1.0))

    # ======================================================================
    # Command / status publishing
    # ======================================================================

    def zero_command(self) -> TwistStamped:
        return self.make_command()

    def make_command(
        self,
        vy_left: float = 0.0,
        vz_up: float = 0.0,
        yaw_rate: float = 0.0,
    ) -> TwistStamped:
        msg = TwistStamped()
        msg.header.frame_id = "base_link"

        # CRITICAL FSM CONTRACT: PRE_ENTRY_GEOMETRY_LOCK never moves forward.
        msg.twist.linear.x = 0.0
        msg.twist.linear.y = float(vy_left)
        msg.twist.linear.z = float(vz_up)
        msg.twist.angular.z = float(yaw_rate)
        return msg

    def diagnostics_dict(self) -> Dict[str, object]:
        g = self.last_geometry
        if g is None:
            return {
                "state": self.state.name,
                "enabled": self.enabled,
                "failure_reason": self.failure_reason,
            }

        return {
            "state": self.state.name,
            "enabled": self.enabled,
            "failure_reason": self.failure_reason,
            "strict_valid": g.strict_valid,
            "loose_valid": g.loose_valid,
            "confidence": round(g.confidence, 3),
            "yaw_error_deg": round(math.degrees(g.yaw_error), 3),
            "yaw_error_raw_deg": round(math.degrees(g.yaw_error_raw), 3),
            "lateral_error_m": round(g.lateral_error, 4),
            "lateral_error_raw_m": round(g.lateral_error_raw, 4),
            "width_m": round(g.width, 4) if math.isfinite(g.width) else None,
            "width_raw_m": round(g.width_raw, 4) if math.isfinite(g.width_raw) else None,
            "left_distance_m": round(g.d_left, 4) if math.isfinite(g.d_left) else None,
            "right_distance_m": round(g.d_right, 4) if math.isfinite(g.d_right) else None,
            "parallel_error_deg": round(math.degrees(g.parallel_error), 3)
            if math.isfinite(g.parallel_error)
            else None,
            "front_clearance_m": round(g.front_clearance, 3),
            "left_rms_m": round(g.left_rms, 4) if math.isfinite(g.left_rms) else None,
            "right_rms_m": round(g.right_rms, 4) if math.isfinite(g.right_rms) else None,
            "left_inliers": g.left_inliers,
            "right_inliers": g.right_inliers,
            "left_span_m": round(g.left_span, 3),
            "right_span_m": round(g.right_span, 3),
            "sector_score": round(g.sector_score, 3),
            "temporal_score": round(g.temporal_score, 3),
            "sectors_m": {
                k: (round(v, 3) if v is not None else None) for k, v in g.sectors.items()
            },
            "observed_lr_sum_m": round(g.observed_lr_sum, 3)
            if g.observed_lr_sum is not None
            else None,
            "expected_lr_sum_m": round(g.expected_lr_sum, 3)
            if g.expected_lr_sum is not None
            else None,
            "lr_sum_residual_m": round(g.lr_sum_residual, 3)
            if g.lr_sum_residual is not None
            else None,
            "roll_deg": round(math.degrees(self.roll), 2) if self.roll is not None else None,
            "pitch_deg": round(math.degrees(self.pitch), 2) if self.pitch is not None else None,
        }

    def publish_outputs(self) -> None:
        now = self.get_clock().now()

        # Stale scan watchdog. It is a transient hold unless a terminal failure
        # has already been latched.
        if self.enabled and self.failure_reason is None:
            stale = True
            if self.last_scan_time is not None:
                age = (now - self.last_scan_time).nanoseconds * 1e-9
                stale = age > float(self.get_parameter("scan_stale_s").value)

            if stale:
                self.latest_command = self.zero_command()
                if self.state not in (LockState.IDLE, LockState.HOLD):
                    self.transition(LockState.HOLD, "LaserScan stale")

        # Always stamp immediately before publishing.
        self.latest_command.header.stamp = now.to_msg()
        self.cmd_pub.publish(self.latest_command)

        state_msg = String()
        state_msg.data = self.state.name
        self.state_pub.publish(state_msg)

        locked_msg = Bool()
        locked_msg.data = self.state == LockState.LOCKED
        self.locked_pub.publish(locked_msg)

        result_msg = String()
        next_state_msg = String()
        if self.failure_reason is not None:
            result_msg.data = f"FAILED:{self.failure_reason}"
            next_state_msg.data = "HOVER_AND_REASSESS"
        elif self.state == LockState.LOCKED:
            result_msg.data = "LOCKED"
            next_state_msg.data = "ENTER_CORRIDOR"
        elif self.enabled:
            result_msg.data = "IN_PROGRESS"
            next_state_msg.data = ""
        else:
            result_msg.data = "IDLE"
            next_state_msg.data = ""

        self.result_pub.publish(result_msg)
        self.next_state_pub.publish(next_state_msg)

        diag_msg = String()
        diag_msg.data = json.dumps(self.diagnostics_dict(), separators=(",", ":"))
        self.diagnostics_pub.publish(diag_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PreEntryGeometryLockV3()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
