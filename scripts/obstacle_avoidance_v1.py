#!/usr/bin/env python3
"""OBSTACLE_DECISION + AVOID_LEFT + AVOID_RIGHT for SAE AeroTHON corridor.

Designed around the existing D500 /scan convention and the same ROS FLU body
command convention used by PRE_ENTRY_GEOMETRY_LOCK and CORRIDOR_CRUISE:
    +x forward, +y left, +z up, +yaw counter-clockwise.

Global-state behaviour
----------------------
When enabled this node starts in OBSTACLE_DECISION and commands zero motion.
It uses constrained RANSAC to detect:
  * the longitudinal corridor wall(s), approximately parallel to the flight path
  * a transverse obstacle face, approximately perpendicular to the flight path

The obstacle is assumed to be a static wall protruding from exactly one side of
an approximately 3.5 m corridor.

If the transverse face touches the RIGHT corridor wall, the free passage is on
LEFT  -> AVOID_LEFT.
If the transverse face touches the LEFT corridor wall, the free passage is on
RIGHT -> AVOID_RIGHT.

Avoidance is deliberately split internally into two phases while keeping the
published global state as AVOID_LEFT / AVOID_RIGHT:

  SHIFT:
      vx = 0.  Translate laterally into the centre of the free passage while
      trimming yaw from the open-side corridor wall.

  PASS:
      Move forward slowly while holding the same open-side wall clearance.
      The obstacle direction is latched; it is NOT re-decided every scan.
      Completion is confirmed when the same transverse obstacle face is behind
      the aircraft by a configurable margin for several scans.

After a successful bypass the node requests CORRIDOR_CRUISE.  The existing
CORRIDOR_CRUISE integrated correction can then smoothly restore the normal
corridor centreline.

Important control note
----------------------
This node publishes BODY-FRAME LATERAL VELOCITY, not raw roll angle.  On the
real vehicle the flight controller should convert that velocity request into
roll.  That matches the existing Gazebo bridge and avoids bypassing Pixhawk's
attitude controller.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Deque, Dict, Optional, Tuple

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


def axis_angle_from_x(direction: np.ndarray) -> float:
    """Unsigned line-axis angle in [0, pi/2] relative to corridor +x."""
    a = abs(math.atan2(float(direction[1]), float(direction[0])))
    if a > math.pi / 2.0:
        a = math.pi - a
    return abs(a)


# ============================================================================
# Data structures
# ============================================================================


class ObstacleState(Enum):
    IDLE = auto()
    OBSTACLE_DECISION = auto()
    AVOID_LEFT = auto()
    AVOID_RIGHT = auto()
    TRANSITION_REQUESTED = auto()


class AvoidPhase(Enum):
    NONE = auto()
    SHIFT = auto()
    PASS = auto()


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
class FaceObservation:
    valid: bool = False
    x_m: float = math.inf
    y_min_m: float = math.inf
    y_max_m: float = -math.inf
    span_m: float = 0.0
    rms_m: float = math.inf
    inliers: int = 0
    touches_left: bool = False
    touches_right: bool = False


@dataclass
class ObstacleObservation:
    geometry_valid: bool = False

    front_clearance_m: float = 0.0

    left_wall_valid: bool = False
    right_wall_valid: bool = False
    left_wall_yaw_rad: Optional[float] = None
    right_wall_yaw_rad: Optional[float] = None

    left_wall_y_m: Optional[float] = None
    right_wall_y_m: Optional[float] = None
    d_left_m: Optional[float] = None
    d_right_m: Optional[float] = None
    corridor_width_m: Optional[float] = None

    front_face: FaceObservation = field(default_factory=FaceObservation)
    rear_face: FaceObservation = field(default_factory=FaceObservation)

    candidate_state: Optional[str] = None
    obstacle_side: Optional[str] = None
    open_gap_m: Optional[float] = None
    target_outer_clearance_m: Optional[float] = None
    candidate_reason: str = ""


# ============================================================================
# Main node
# ============================================================================


class ObstacleAvoidanceV1(Node):
    def __init__(self) -> None:
        super().__init__("obstacle_avoidance_v1")

        # ------------------------------------------------------------------
        # Topics
        # ------------------------------------------------------------------
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("imu_topic", "/imu/data")
        self.declare_parameter("enable_topic", "/corridor/obstacle/enable")
        self.declare_parameter("cmd_topic", "/corridor/obstacle/cmd_vel")
        self.declare_parameter("state_topic", "/corridor/obstacle/state")
        self.declare_parameter("phase_topic", "/corridor/obstacle/phase")
        self.declare_parameter("result_topic", "/corridor/obstacle/result")
        self.declare_parameter("next_state_topic", "/corridor/obstacle/next_state")
        self.declare_parameter("diagnostics_topic", "/corridor/obstacle/diagnostics")

        # ------------------------------------------------------------------
        # D500 orientation / usable range
        # ------------------------------------------------------------------
        self.declare_parameter("invert_scan_angle", False)
        self.declare_parameter("lidar_yaw_offset_deg", 0.0)
        self.declare_parameter("geometry_max_range_m", 8.0)

        # ------------------------------------------------------------------
        # Corridor / airframe geometry
        # ------------------------------------------------------------------
        self.declare_parameter("corridor_width_m", 3.5)
        self.declare_parameter("corridor_width_tolerance_m", 0.65)

        # IMPORTANT: set this to actual maximum horizontal airframe width.
        self.declare_parameter("vehicle_width_m", 0.65)
        self.declare_parameter("passage_side_margin_m", 0.25)

        # ------------------------------------------------------------------
        # Longitudinal corridor-wall RANSAC
        # ------------------------------------------------------------------
        self.declare_parameter("wall_fit_x_min_m", -0.50)
        self.declare_parameter("wall_fit_x_max_m", 4.50)
        self.declare_parameter("wall_fit_side_min_m", 0.30)
        self.declare_parameter("wall_fit_side_max_m", 3.20)
        self.declare_parameter("wall_max_axis_angle_deg", 22.0)
        self.declare_parameter("wall_ransac_iterations", 140)
        self.declare_parameter("wall_ransac_distance_m", 0.08)
        self.declare_parameter("wall_min_inliers", 12)
        self.declare_parameter("wall_min_span_m", 0.75)
        self.declare_parameter("wall_max_rms_m", 0.11)

        # ------------------------------------------------------------------
        # Transverse obstacle-face RANSAC
        # ------------------------------------------------------------------
        self.declare_parameter("face_x_min_m", 0.20)
        self.declare_parameter("face_x_max_m", 2.20)
        self.declare_parameter("rear_face_x_min_m", -2.20)
        self.declare_parameter("rear_face_x_max_m", -0.15)
        self.declare_parameter("face_axis_tolerance_deg", 18.0)
        self.declare_parameter("face_ransac_iterations", 180)
        self.declare_parameter("face_ransac_distance_m", 0.07)
        self.declare_parameter("face_min_inliers", 10)
        self.declare_parameter("face_min_span_m", 0.45)
        self.declare_parameter("face_max_rms_m", 0.09)
        self.declare_parameter("face_wall_touch_tolerance_m", 0.35)
        # Remove points already explained by the long outer corridor walls
        # before transverse-face RANSAC. Without this, two wall samples at
        # similar x can form a fake vertical line spanning the corridor.
        self.declare_parameter("face_exclude_outer_wall_band_m", 0.12)

        # ------------------------------------------------------------------
        # Decision persistence
        # ------------------------------------------------------------------
        self.declare_parameter("decision_confirm_scans", 4)
        self.declare_parameter("decision_timeout_s", 24.0)
        self.declare_parameter("front_no_obstacle_release_m", 1.80)
        self.declare_parameter("no_obstacle_release_scans", 4)

        # ------------------------------------------------------------------
        # Avoidance control
        # ------------------------------------------------------------------
        self.declare_parameter("shift_k_lateral", 0.55)
        self.declare_parameter("shift_max_lateral_speed_m_s", 0.20)
        self.declare_parameter("shift_min_lateral_speed_m_s", 0.035)
        self.declare_parameter("shift_clearance_tolerance_m", 0.10)
        self.declare_parameter("shift_yaw_tolerance_deg", 2.5)
        self.declare_parameter("shift_confirm_scans", 4)
        self.declare_parameter("shift_timeout_s", 17.0)

        self.declare_parameter("avoid_forward_speed_m_s", 0.18)
        self.declare_parameter("pass_k_lateral", 0.45)
        self.declare_parameter("pass_max_lateral_speed_m_s", 0.14)
        self.declare_parameter("pass_clearance_deadband_m", 0.05)

        self.declare_parameter("k_yaw", 0.75)
        self.declare_parameter("yaw_deadband_deg", 0.8)
        self.declare_parameter("max_yaw_rate_deg_s", 7.0)
        self.declare_parameter("yaw_priority_deg", 6.0)

        # Side / front safety while bypassing.
        self.declare_parameter("hard_outer_wall_clearance_m", 0.45)
        self.declare_parameter("front_emergency_stop_m", 0.60)
        self.declare_parameter("avoid_timeout_s", 30.0)

        # Passing is complete when the same obstacle face is behind the drone.
        self.declare_parameter("rear_pass_margin_m", 0.35)
        self.declare_parameter("pass_confirm_scans", 4)

        # ------------------------------------------------------------------
        # General safety
        # ------------------------------------------------------------------
        self.declare_parameter("front_cone_deg", 16.0)
        self.declare_parameter("scan_stale_s", 0.30)
        self.declare_parameter("require_imu", False)
        self.declare_parameter("max_tilt_deg", 12.0)

        # ------------------------------------------------------------------
        # ROS wiring
        # ------------------------------------------------------------------
        scan_topic = str(self.get_parameter("scan_topic").value)
        imu_topic = str(self.get_parameter("imu_topic").value)
        enable_topic = str(self.get_parameter("enable_topic").value)
        cmd_topic = str(self.get_parameter("cmd_topic").value)

        self.create_subscription(LaserScan, scan_topic, self.scan_callback, 10)
        self.create_subscription(Imu, imu_topic, self.imu_callback, 10)
        self.create_subscription(Bool, enable_topic, self.enable_callback, 10)

        self.cmd_pub = self.create_publisher(TwistStamped, cmd_topic, 10)
        self.state_pub = self.create_publisher(
            String, str(self.get_parameter("state_topic").value), 10
        )
        self.phase_pub = self.create_publisher(
            String, str(self.get_parameter("phase_topic").value), 10
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

        self.timer = self.create_timer(0.05, self.publish_outputs)

        # ------------------------------------------------------------------
        # Runtime
        # ------------------------------------------------------------------
        self.enabled = False
        self.state = ObstacleState.IDLE
        self.phase = AvoidPhase.NONE
        self.state_enter_time = self.get_clock().now()
        self.phase_enter_time = self.get_clock().now()
        self.last_scan_time = None
        self.last_observation: Optional[ObstacleObservation] = None
        self.latest_command = self.zero_command()

        self.transition_target: Optional[str] = None
        self.transition_reason: Optional[str] = None

        self.roll: Optional[float] = None
        self.pitch: Optional[float] = None

        n_decide = max(1, int(self.get_parameter("decision_confirm_scans").value))
        self.decision_history: Deque[str] = deque(maxlen=n_decide)
        self.no_obstacle_streak = 0
        self.shift_streak = 0
        self.pass_streak = 0

        # Latched bypass geometry.  Once chosen, the side is never flipped
        # until the bypass finishes or the node fails safe.
        self.latched_state: Optional[ObstacleState] = None
        self.latched_obstacle_side: Optional[str] = None
        self.target_outer_clearance_m: Optional[float] = None

        self.rng = np.random.default_rng(2026)

        self.get_logger().info(
            "OBSTACLE_DECISION / AVOID_LEFT / AVOID_RIGHT V1 started"
        )

    # ======================================================================
    # Lifecycle
    # ======================================================================

    def reset_session(self) -> None:
        self.state = ObstacleState.OBSTACLE_DECISION
        self.phase = AvoidPhase.NONE
        self.state_enter_time = self.get_clock().now()
        self.phase_enter_time = self.get_clock().now()
        self.last_observation = None
        self.latest_command = self.zero_command()
        self.transition_target = None
        self.transition_reason = None
        self.decision_history.clear()
        self.no_obstacle_streak = 0
        self.shift_streak = 0
        self.pass_streak = 0
        self.latched_state = None
        self.latched_obstacle_side = None
        self.target_outer_clearance_m = None

    def enable_callback(self, msg: Bool) -> None:
        if msg.data and not self.enabled:
            self.enabled = True
            self.reset_session()
            self.get_logger().warning("Entered OBSTACLE_DECISION")
        elif not msg.data and self.enabled:
            self.enabled = False
            self.state = ObstacleState.IDLE
            self.phase = AvoidPhase.NONE
            self.latest_command = self.zero_command()
            self.transition_target = None
            self.transition_reason = None
            self.get_logger().info("Obstacle avoidance disabled")

    def imu_callback(self, msg: Imu) -> None:
        self.roll, self.pitch = quaternion_to_roll_pitch(msg)

    def attitude_is_safe(self) -> bool:
        require_imu = bool(self.get_parameter("require_imu").value)
        if self.roll is None or self.pitch is None:
            return not require_imu

        limit = math.radians(float(self.get_parameter("max_tilt_deg").value))
        return abs(self.roll) <= limit and abs(self.pitch) <= limit

    def state_age(self) -> float:
        return (self.get_clock().now() - self.state_enter_time).nanoseconds * 1e-9

    def phase_age(self) -> float:
        return (self.get_clock().now() - self.phase_enter_time).nanoseconds * 1e-9

    def request_transition(self, target: str, reason: str) -> None:
        if self.state == ObstacleState.TRANSITION_REQUESTED:
            return
        self.transition_target = target
        self.transition_reason = reason
        self.latest_command = self.zero_command()
        self.state = ObstacleState.TRANSITION_REQUESTED
        self.phase = AvoidPhase.NONE
        self.get_logger().warning(f"OBSTACLE FSM -> {target}: {reason}")

    # ======================================================================
    # Scan processing
    # ======================================================================

    def corrected_scan_arrays(self, scan: LaserScan) -> tuple[np.ndarray, np.ndarray]:
        ranges = np.asarray(scan.ranges, dtype=np.float64)
        angles = scan.angle_min + np.arange(ranges.size, dtype=np.float64) * scan.angle_increment
        angles = np.arctan2(np.sin(angles), np.cos(angles))

        if bool(self.get_parameter("invert_scan_angle").value):
            angles = -angles

        angles += math.radians(float(self.get_parameter("lidar_yaw_offset_deg").value))
        angles = np.arctan2(np.sin(angles), np.cos(angles))
        return ranges, angles

    def scan_callback(self, scan: LaserScan) -> None:
        self.last_scan_time = self.get_clock().now()

        if not self.enabled or self.state in (
            ObstacleState.IDLE,
            ObstacleState.TRANSITION_REQUESTED,
        ):
            return

        obs = self.extract_observation(scan)
        self.last_observation = obs

        if not self.attitude_is_safe():
            self.request_transition(
                "HOVER_AND_REASSESS",
                "roll/pitch exceeds obstacle-avoidance limit",
            )
            return

        if self.state == ObstacleState.OBSTACLE_DECISION:
            self.step_decision(obs)
        elif self.state in (ObstacleState.AVOID_LEFT, ObstacleState.AVOID_RIGHT):
            self.step_avoid(obs)

    # ======================================================================
    # RANSAC
    # ======================================================================

    def fit_line_ransac(
        self,
        points: np.ndarray,
        iterations: int,
        distance_threshold: float,
        min_inliers: int,
        min_span: float,
        max_rms: float,
        target_axis_angle_rad: float,
        axis_tolerance_rad: float,
    ) -> LineModel:
        if points.shape[0] < min_inliers:
            return LineModel()

        n = points.shape[0]
        best_mask = None
        best_score = -math.inf

        for _ in range(max(1, iterations)):
            i, j = self.rng.choice(n, size=2, replace=False)
            p1 = points[i]
            p2 = points[j]
            segment = p2 - p1
            length = float(np.linalg.norm(segment))
            if length < 0.12:
                continue

            direction = segment / length
            axis_angle = axis_angle_from_x(direction)
            if abs(axis_angle - target_axis_angle_rad) > axis_tolerance_rad:
                continue

            normal = np.array([-direction[1], direction[0]], dtype=np.float64)
            residuals = np.abs((points - p1) @ normal)
            mask = residuals <= distance_threshold
            count = int(np.count_nonzero(mask))
            if count < min_inliers:
                continue

            inliers = points[mask]
            projection = (inliers - p1) @ direction
            span = float(np.percentile(projection, 95.0) - np.percentile(projection, 5.0))
            if span < 0.6 * min_span:
                continue

            median_residual = float(np.median(residuals[mask]))
            span_factor = 0.5 + min(span / max(min_span, 0.1), 3.0)
            residual_factor = 1.0 / (1.0 + 12.0 * median_residual)
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

        # For longitudinal walls, prefer +x direction.  Transverse lines are
        # sign-invariant; the axis-angle folding handles either sign.
        if direction[0] < 0.0:
            direction = -direction

        axis_angle = axis_angle_from_x(direction)
        if abs(axis_angle - target_axis_angle_rad) > axis_tolerance_rad:
            return LineModel()

        normal = np.array([-direction[1], direction[0]], dtype=np.float64)
        residuals = np.abs(centered @ normal)
        rms = float(math.sqrt(np.mean(residuals * residuals)))

        projection = centered @ direction
        span = float(np.percentile(projection, 95.0) - np.percentile(projection, 5.0))
        inlier_ratio = float(inliers.shape[0] / max(points.shape[0], 1))

        if span < min_span or rms > max_rms:
            return LineModel()

        return LineModel(
            valid=True,
            point=centroid,
            direction=direction,
            inlier_points=inliers,
            rms=rms,
            span=span,
            inlier_ratio=inlier_ratio,
        )

    def fit_wall(self, points: np.ndarray) -> LineModel:
        return self.fit_line_ransac(
            points=points,
            iterations=int(self.get_parameter("wall_ransac_iterations").value),
            distance_threshold=float(self.get_parameter("wall_ransac_distance_m").value),
            min_inliers=int(self.get_parameter("wall_min_inliers").value),
            min_span=float(self.get_parameter("wall_min_span_m").value),
            max_rms=float(self.get_parameter("wall_max_rms_m").value),
            target_axis_angle_rad=0.0,
            axis_tolerance_rad=math.radians(
                float(self.get_parameter("wall_max_axis_angle_deg").value)
            ),
        )

    def fit_face(self, points: np.ndarray) -> LineModel:
        return self.fit_line_ransac(
            points=points,
            iterations=int(self.get_parameter("face_ransac_iterations").value),
            distance_threshold=float(self.get_parameter("face_ransac_distance_m").value),
            min_inliers=int(self.get_parameter("face_min_inliers").value),
            min_span=float(self.get_parameter("face_min_span_m").value),
            max_rms=float(self.get_parameter("face_max_rms_m").value),
            target_axis_angle_rad=math.pi / 2.0,
            axis_tolerance_rad=math.radians(
                float(self.get_parameter("face_axis_tolerance_deg").value)
            ),
        )

    # ======================================================================
    # Geometry extraction
    # ======================================================================

    def front_clearance(
        self,
        ranges: np.ndarray,
        angles: np.ndarray,
        range_min: float,
        range_max: float,
    ) -> float:
        safe = ranges.copy()
        safe[np.isposinf(safe)] = float(range_max)
        valid = np.isfinite(safe)
        valid &= safe >= max(float(range_min), 0.05)
        valid &= safe <= float(range_max)

        half = math.radians(float(self.get_parameter("front_cone_deg").value))
        mask = valid & (np.abs(angles) <= half)
        vals = safe[mask]
        if vals.size == 0:
            return 0.0
        return float(np.percentile(vals, 10.0))

    @staticmethod
    def corridor_basis_from_walls(
        left: LineModel, right: LineModel
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        directions = []
        for line in (left, right):
            if line.valid:
                d = line.direction.copy()
                if d[0] < 0.0:
                    d = -d
                directions.append(d)

        if not directions:
            return None

        d = np.sum(np.asarray(directions), axis=0)
        norm = float(np.linalg.norm(d))
        if norm < 1e-9:
            return None
        d /= norm
        if d[0] < 0.0:
            d = -d

        n_left = np.array([-d[1], d[0]], dtype=np.float64)
        return d, n_left

    def make_face_observation(
        self,
        points_corridor: np.ndarray,
        x_min: float,
        x_max: float,
        left_y: float,
        right_y: float,
    ) -> FaceObservation:
        exclude_band = float(
            self.get_parameter("face_exclude_outer_wall_band_m").value
        )
        mask = (
            (points_corridor[:, 0] >= x_min)
            & (points_corridor[:, 0] <= x_max)
            & (points_corridor[:, 1] <= left_y - exclude_band)
            & (points_corridor[:, 1] >= right_y + exclude_band)
        )
        roi = points_corridor[mask]
        line = self.fit_face(roi)
        if not line.valid:
            return FaceObservation()

        ys = line.inlier_points[:, 1]
        xs = line.inlier_points[:, 0]
        y_min = float(np.percentile(ys, 5.0))
        y_max = float(np.percentile(ys, 95.0))
        x_med = float(np.median(xs))

        touch_tol = float(self.get_parameter("face_wall_touch_tolerance_m").value)
        touches_left = abs(left_y - y_max) <= touch_tol
        touches_right = abs(y_min - right_y) <= touch_tol

        return FaceObservation(
            valid=True,
            x_m=x_med,
            y_min_m=y_min,
            y_max_m=y_max,
            span_m=float(y_max - y_min),
            rms_m=line.rms,
            inliers=int(line.inlier_points.shape[0]),
            touches_left=touches_left,
            touches_right=touches_right,
        )

    def extract_observation(self, scan: LaserScan) -> ObstacleObservation:
        obs = ObstacleObservation()
        ranges, angles = self.corrected_scan_arrays(scan)
        if ranges.size < 20:
            return obs

        obs.front_clearance_m = self.front_clearance(
            ranges, angles, scan.range_min, scan.range_max
        )

        valid = np.isfinite(ranges)
        valid &= ranges >= max(float(scan.range_min), 0.05)
        valid &= ranges <= min(
            float(scan.range_max),
            float(self.get_parameter("geometry_max_range_m").value),
        )
        if np.count_nonzero(valid) < 20:
            return obs

        r = ranges[valid]
        a = angles[valid]
        points = np.column_stack((r * np.cos(a), r * np.sin(a)))

        x_min = float(self.get_parameter("wall_fit_x_min_m").value)
        x_max = float(self.get_parameter("wall_fit_x_max_m").value)
        side_min = float(self.get_parameter("wall_fit_side_min_m").value)
        side_max = float(self.get_parameter("wall_fit_side_max_m").value)

        x = points[:, 0]
        y = points[:, 1]
        longitudinal = (x >= x_min) & (x <= x_max)

        left_points = points[
            longitudinal & (y >= side_min) & (y <= side_max)
        ]
        right_points = points[
            longitudinal & (y <= -side_min) & (y >= -side_max)
        ]

        left_line = self.fit_wall(left_points)
        right_line = self.fit_wall(right_points)
        obs.left_wall_valid = left_line.valid
        obs.right_wall_valid = right_line.valid

        if left_line.valid:
            d = left_line.direction.copy()
            if d[0] < 0.0:
                d = -d
            obs.left_wall_yaw_rad = wrap_pi(math.atan2(float(d[1]), float(d[0])))

        if right_line.valid:
            d = right_line.direction.copy()
            if d[0] < 0.0:
                d = -d
            obs.right_wall_yaw_rad = wrap_pi(math.atan2(float(d[1]), float(d[0])))

        basis = self.corridor_basis_from_walls(left_line, right_line)
        if basis is None:
            obs.candidate_reason = "no trustworthy longitudinal wall"
            return obs

        corridor_dir, left_normal = basis
        x_corr = points @ corridor_dir
        y_corr = points @ left_normal
        points_corr = np.column_stack((x_corr, y_corr))

        width_nominal = float(self.get_parameter("corridor_width_m").value)

        left_y: Optional[float] = None
        right_y: Optional[float] = None

        if left_line.valid:
            left_y = float(np.median(left_line.inlier_points @ left_normal))
        if right_line.valid:
            right_y = float(np.median(right_line.inlier_points @ left_normal))

        # The opposite wall is often the cleanest wall during an obstacle
        # encounter. Infer the obscured corridor boundary from known width.
        if left_y is None and right_y is not None:
            left_y = right_y + width_nominal
        elif right_y is None and left_y is not None:
            right_y = left_y - width_nominal

        if left_y is None or right_y is None:
            obs.candidate_reason = "cannot reconstruct corridor boundaries"
            return obs

        # Vehicle should still be between the outer corridor boundaries.
        if left_y <= 0.0 or right_y >= 0.0:
            obs.candidate_reason = "vehicle origin not between reconstructed walls"
            return obs

        width = left_y - right_y
        width_tol = float(self.get_parameter("corridor_width_tolerance_m").value)
        if abs(width - width_nominal) > width_tol:
            obs.candidate_reason = f"corridor width implausible ({width:.2f} m)"
            return obs

        obs.geometry_valid = True
        obs.left_wall_y_m = left_y
        obs.right_wall_y_m = right_y
        obs.d_left_m = left_y
        obs.d_right_m = -right_y
        obs.corridor_width_m = width

        obs.front_face = self.make_face_observation(
            points_corr,
            float(self.get_parameter("face_x_min_m").value),
            float(self.get_parameter("face_x_max_m").value),
            left_y,
            right_y,
        )
        obs.rear_face = self.make_face_observation(
            points_corr,
            float(self.get_parameter("rear_face_x_min_m").value),
            float(self.get_parameter("rear_face_x_max_m").value),
            left_y,
            right_y,
        )

        face = obs.front_face
        if not face.valid:
            obs.candidate_reason = "no transverse obstacle face ahead"
            return obs

        required_gap = (
            float(self.get_parameter("vehicle_width_m").value)
            + 2.0 * float(self.get_parameter("passage_side_margin_m").value)
        )

        # Exactly one wall-touch is expected for a side-protruding obstacle.
        if face.touches_left and not face.touches_right:
            # Obstacle grows from LEFT wall.  Open channel is RIGHT.
            open_gap = face.y_min_m - right_y
            obs.obstacle_side = "LEFT"
            obs.open_gap_m = open_gap

            if not right_line.valid:
                obs.candidate_reason = "left obstacle seen but right/open wall invalid"
                return obs
            if open_gap < required_gap:
                obs.candidate_reason = (
                    f"right passage too narrow ({open_gap:.2f} < {required_gap:.2f} m)"
                )
                return obs

            obs.candidate_state = "AVOID_RIGHT"
            obs.target_outer_clearance_m = 0.5 * open_gap
            obs.candidate_reason = "transverse face touches left wall; right passage open"
            return obs

        if face.touches_right and not face.touches_left:
            # Obstacle grows from RIGHT wall.  Open channel is LEFT.
            open_gap = left_y - face.y_max_m
            obs.obstacle_side = "RIGHT"
            obs.open_gap_m = open_gap

            if not left_line.valid:
                obs.candidate_reason = "right obstacle seen but left/open wall invalid"
                return obs
            if open_gap < required_gap:
                obs.candidate_reason = (
                    f"left passage too narrow ({open_gap:.2f} < {required_gap:.2f} m)"
                )
                return obs

            obs.candidate_state = "AVOID_LEFT"
            obs.target_outer_clearance_m = 0.5 * open_gap
            obs.candidate_reason = "transverse face touches right wall; left passage open"
            return obs

        if face.touches_left and face.touches_right:
            obs.candidate_reason = "transverse face appears to span entire corridor"
        else:
            obs.candidate_reason = "transverse face does not connect to exactly one wall"
        return obs

    # ======================================================================
    # FSM: OBSTACLE_DECISION
    # ======================================================================

    def step_decision(self, obs: ObstacleObservation) -> None:
        self.latest_command = self.zero_command()

        if self.state_age() > float(self.get_parameter("decision_timeout_s").value):
            self.request_transition(
                "HOVER_AND_REASSESS",
                "obstacle side remained ambiguous until decision timeout",
            )
            return

        # If the cruise trigger was just noise and the front is persistently
        # open again, hand control back rather than inventing a bypass.
        if (
            obs.front_clearance_m
            >= float(self.get_parameter("front_no_obstacle_release_m").value)
            and not obs.front_face.valid
        ):
            self.no_obstacle_streak += 1
            if self.no_obstacle_streak >= int(
                self.get_parameter("no_obstacle_release_scans").value
            ):
                self.request_transition(
                    "CORRIDOR_CRUISE",
                    "front obstacle no longer present during decision",
                )
            return
        self.no_obstacle_streak = 0

        candidate = obs.candidate_state or "AMBIGUOUS"
        self.decision_history.append(candidate)

        needed = int(self.get_parameter("decision_confirm_scans").value)
        if len(self.decision_history) < needed:
            return

        if not all(v == self.decision_history[-1] for v in self.decision_history):
            return

        chosen = self.decision_history[-1]
        if chosen not in ("AVOID_LEFT", "AVOID_RIGHT"):
            return

        if obs.target_outer_clearance_m is None or obs.obstacle_side is None:
            return

        self.target_outer_clearance_m = float(obs.target_outer_clearance_m)
        self.latched_obstacle_side = obs.obstacle_side
        self.latched_state = (
            ObstacleState.AVOID_LEFT if chosen == "AVOID_LEFT" else ObstacleState.AVOID_RIGHT
        )

        self.state = self.latched_state
        self.state_enter_time = self.get_clock().now()
        self.phase = AvoidPhase.SHIFT
        self.phase_enter_time = self.get_clock().now()
        self.shift_streak = 0
        self.pass_streak = 0
        self.latest_command = self.zero_command()

        self.get_logger().warning(
            f"OBSTACLE_DECISION -> {chosen}: {obs.candidate_reason}; "
            f"open_gap={obs.open_gap_m:.2f} m, "
            f"target_outer_clearance={self.target_outer_clearance_m:.2f} m"
        )

    # ======================================================================
    # FSM: AVOID_LEFT / AVOID_RIGHT
    # ======================================================================

    def open_wall_measurements(
        self, obs: ObstacleObservation
    ) -> tuple[Optional[float], Optional[float]]:
        """Return (distance_to_open_outer_wall, yaw_error_from_that_wall)."""
        if self.state == ObstacleState.AVOID_LEFT:
            if not obs.left_wall_valid:
                return None, None
            return obs.d_left_m, obs.left_wall_yaw_rad
        if self.state == ObstacleState.AVOID_RIGHT:
            if not obs.right_wall_valid:
                return None, None
            return obs.d_right_m, obs.right_wall_yaw_rad
        return None, None

    def lateral_command_to_outer_clearance(self, actual_clearance: float) -> float:
        assert self.target_outer_clearance_m is not None

        error = actual_clearance - self.target_outer_clearance_m

        # Open-left: moving +y (left) REDUCES distance to left wall.
        # Open-right: moving -y (right) REDUCES distance to right wall.
        sign = +1.0 if self.state == ObstacleState.AVOID_LEFT else -1.0

        if self.phase == AvoidPhase.SHIFT:
            k = float(self.get_parameter("shift_k_lateral").value)
            max_speed = float(self.get_parameter("shift_max_lateral_speed_m_s").value)
            min_speed = float(self.get_parameter("shift_min_lateral_speed_m_s").value)
            cmd = sign * clamp(k * error, -max_speed, max_speed)
            if abs(cmd) < min_speed and abs(error) > 1e-6:
                cmd = math.copysign(min_speed, cmd)
            return cmd

        deadband = float(self.get_parameter("pass_clearance_deadband_m").value)
        if abs(error) <= deadband:
            return 0.0

        k = float(self.get_parameter("pass_k_lateral").value)
        max_speed = float(self.get_parameter("pass_max_lateral_speed_m_s").value)
        return sign * clamp(k * error, -max_speed, max_speed)

    def yaw_command(self, yaw_error: float) -> float:
        deadband = math.radians(float(self.get_parameter("yaw_deadband_deg").value))
        if abs(yaw_error) <= deadband:
            return 0.0

        max_rate = math.radians(float(self.get_parameter("max_yaw_rate_deg_s").value))
        k = float(self.get_parameter("k_yaw").value)
        return clamp(k * yaw_error, -max_rate, max_rate)

    def rear_face_matches_latched_obstacle(self, obs: ObstacleObservation) -> bool:
        face = obs.rear_face
        if not face.valid:
            return False

        margin = float(self.get_parameter("rear_pass_margin_m").value)
        if face.x_m > -margin:
            return False

        if self.latched_obstacle_side == "LEFT":
            return face.touches_left and not face.touches_right
        if self.latched_obstacle_side == "RIGHT":
            return face.touches_right and not face.touches_left
        return False

    def step_avoid(self, obs: ObstacleObservation) -> None:
        if self.target_outer_clearance_m is None or self.latched_obstacle_side is None:
            self.request_transition(
                "HOVER_AND_REASSESS",
                "avoidance entered without latched bypass geometry",
            )
            return

        if self.state_age() > float(self.get_parameter("avoid_timeout_s").value):
            self.request_transition(
                "HOVER_AND_REASSESS",
                "obstacle bypass exceeded total timeout",
            )
            return

        actual_clearance, yaw_error = self.open_wall_measurements(obs)
        if actual_clearance is None or yaw_error is None:
            self.latest_command = self.zero_command()
            self.request_transition(
                "HOVER_AND_REASSESS",
                "open-side corridor wall lost during bypass",
            )
            return

        if actual_clearance <= float(
            self.get_parameter("hard_outer_wall_clearance_m").value
        ):
            self.request_transition(
                "HOVER_AND_REASSESS",
                f"open-side wall clearance critically low ({actual_clearance:.2f} m)",
            )
            return

        yaw_rate = self.yaw_command(yaw_error)
        yaw_priority = math.radians(float(self.get_parameter("yaw_priority_deg").value))

        # --------------------------------------------------------------
        # SHIFT: lateral translation only.  This is the obstacle version
        # of PRE_ENTRY centering: same P-control idea, different target.
        # We centre in the FREE GAP, not on the corridor centreline.
        # --------------------------------------------------------------
        if self.phase == AvoidPhase.SHIFT:
            if self.phase_age() > float(self.get_parameter("shift_timeout_s").value):
                self.request_transition(
                    "HOVER_AND_REASSESS",
                    "failed to reach bypass lane before shift timeout",
                )
                return

            clearance_error = actual_clearance - self.target_outer_clearance_m
            tolerance = float(self.get_parameter("shift_clearance_tolerance_m").value)

            shift_yaw_tol = math.radians(
                float(self.get_parameter("shift_yaw_tolerance_deg").value)
            )
            if abs(clearance_error) <= tolerance and abs(yaw_error) <= shift_yaw_tol:
                self.shift_streak += 1
                self.latest_command = self.zero_command()
                if self.shift_streak >= int(
                    self.get_parameter("shift_confirm_scans").value
                ):
                    self.phase = AvoidPhase.PASS
                    self.phase_enter_time = self.get_clock().now()
                    self.shift_streak = 0
                    self.pass_streak = 0
                    self.get_logger().info(
                        f"{self.state.name}: SHIFT complete -> PASS"
                    )
                return

            self.shift_streak = 0

            if abs(yaw_error) > yaw_priority:
                # Do not roll/translate sideways while badly yawed.
                self.latest_command = self.make_command(yaw_rate=yaw_rate)
                return

            vy = self.lateral_command_to_outer_clearance(actual_clearance)
            self.latest_command = self.make_command(vy_left=vy, yaw_rate=yaw_rate)
            return

        # --------------------------------------------------------------
        # PASS: fixed slow forward motion while holding the free-gap lane.
        # --------------------------------------------------------------
        if self.phase == AvoidPhase.PASS:
            emergency = float(self.get_parameter("front_emergency_stop_m").value)
            if 0.0 < obs.front_clearance_m <= emergency:
                self.request_transition(
                    "HOVER_AND_REASSESS",
                    f"unexpected object in bypass lane ({obs.front_clearance_m:.2f} m)",
                )
                return

            if self.rear_face_matches_latched_obstacle(obs):
                self.pass_streak += 1
                if self.pass_streak >= int(
                    self.get_parameter("pass_confirm_scans").value
                ):
                    self.request_transition(
                        "CORRIDOR_CRUISE",
                        f"{self.state.name} complete; obstacle face confirmed behind",
                    )
                    return
            else:
                self.pass_streak = 0

            if abs(yaw_error) > yaw_priority:
                self.latest_command = self.make_command(yaw_rate=yaw_rate)
                return

            vy = self.lateral_command_to_outer_clearance(actual_clearance)
            vx = float(self.get_parameter("avoid_forward_speed_m_s").value)
            self.latest_command = self.make_command(
                vx=max(0.0, vx),
                vy_left=vy,
                yaw_rate=yaw_rate,
            )
            return

        self.request_transition(
            "HOVER_AND_REASSESS",
            "invalid internal avoidance phase",
        )

    # ======================================================================
    # Commands / diagnostics
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
        base: Dict[str, object] = {
            "enabled": self.enabled,
            "state": self.state.name,
            "phase": self.phase.name,
            "transition_target": self.transition_target,
            "transition_reason": self.transition_reason,
            "latched_obstacle_side": self.latched_obstacle_side,
            "target_outer_clearance_m": (
                round(self.target_outer_clearance_m, 3)
                if self.target_outer_clearance_m is not None
                else None
            ),
            "decision_history": list(self.decision_history),
        }

        obs = self.last_observation
        if obs is None:
            return base

        base["observation"] = {
            "geometry_valid": obs.geometry_valid,
            "front_clearance_m": round(obs.front_clearance_m, 3),
            "left_wall_valid": obs.left_wall_valid,
            "right_wall_valid": obs.right_wall_valid,
            "d_left_m": round(obs.d_left_m, 3) if obs.d_left_m is not None else None,
            "d_right_m": round(obs.d_right_m, 3) if obs.d_right_m is not None else None,
            "corridor_width_m": (
                round(obs.corridor_width_m, 3)
                if obs.corridor_width_m is not None
                else None
            ),
            "candidate_state": obs.candidate_state,
            "obstacle_side": obs.obstacle_side,
            "open_gap_m": round(obs.open_gap_m, 3) if obs.open_gap_m is not None else None,
            "candidate_reason": obs.candidate_reason,
            "front_face": {
                "valid": obs.front_face.valid,
                "x_m": round(obs.front_face.x_m, 3) if obs.front_face.valid else None,
                "y_min_m": round(obs.front_face.y_min_m, 3) if obs.front_face.valid else None,
                "y_max_m": round(obs.front_face.y_max_m, 3) if obs.front_face.valid else None,
                "span_m": round(obs.front_face.span_m, 3),
                "touches_left": obs.front_face.touches_left,
                "touches_right": obs.front_face.touches_right,
            },
            "rear_face": {
                "valid": obs.rear_face.valid,
                "x_m": round(obs.rear_face.x_m, 3) if obs.rear_face.valid else None,
                "touches_left": obs.rear_face.touches_left,
                "touches_right": obs.rear_face.touches_right,
            },
        }
        return base

    def publish_outputs(self) -> None:
        now = self.get_clock().now()

        if self.enabled and self.state not in (
            ObstacleState.IDLE,
            ObstacleState.TRANSITION_REQUESTED,
        ):
            stale = True
            if self.last_scan_time is not None:
                age = (now - self.last_scan_time).nanoseconds * 1e-9
                stale = age > float(self.get_parameter("scan_stale_s").value)

            if stale:
                self.request_transition(
                    "HOVER_AND_REASSESS",
                    "LaserScan stale during obstacle avoidance",
                )

        self.latest_command.header.stamp = now.to_msg()
        self.cmd_pub.publish(self.latest_command)

        state_msg = String()
        if self.state == ObstacleState.TRANSITION_REQUESTED:
            state_msg.data = self.transition_target or "TRANSITION_REQUESTED"
        else:
            state_msg.data = self.state.name
        self.state_pub.publish(state_msg)

        phase_msg = String()
        phase_msg.data = self.phase.name
        self.phase_pub.publish(phase_msg)

        result_msg = String()
        next_msg = String()
        if self.state == ObstacleState.TRANSITION_REQUESTED:
            result_msg.data = (
                f"TRANSITION:{self.transition_target}:{self.transition_reason or ''}"
            )
            next_msg.data = self.transition_target or ""
        elif self.enabled:
            result_msg.data = f"ACTIVE:{self.state.name}:{self.phase.name}"
            next_msg.data = ""
        else:
            result_msg.data = "IDLE"
            next_msg.data = ""

        self.result_pub.publish(result_msg)
        self.next_state_pub.publish(next_msg)

        diag = String()
        diag.data = json.dumps(self.diagnostics_dict(), separators=(",", ":"))
        self.diagnostics_pub.publish(diag)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ObstacleAvoidanceV1()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
