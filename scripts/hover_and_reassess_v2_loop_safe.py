#!/usr/bin/env python3
"""HOVER_AND_REASSESS V2 loop-safe recovery supervisor for the AeroTHON corridor FSM.

This node implements the recovery semantics described in the user's corridor
FSM spreadsheet.  It NEVER commands forward/lateral/yaw motion: while active,
the aircraft is held stationary by publishing a zero TwistStamped command.
Fresh D500 LaserScan data is then used to decide which normal FSM state can be
safely retried.

Context is supplied by the mission bridge as JSON on /corridor/reassess/context:
    {"source_state":"AVOID_RIGHT", "pause_reason":"..."}

Recovery policy used for the currently tested LiDAR corridor states:
  PRE_ENTRY_GEOMETRY_LOCK:
      stable corridor geometry + clear front -> PRE_ENTRY_GEOMETRY_LOCK
  CORRIDOR_CRUISE:
      clear stable corridor -> CORRIDOR_CRUISE
      trustworthy obstacle face -> OBSTACLE_DECISION
      trustworthy opening -> EXIT_DETECTION
  OBSTACLE_DECISION / AVOID_LEFT / AVOID_RIGHT:
      trustworthy one-sided obstacle -> AVOID_LEFT / AVOID_RIGHT
      obstacle disappeared + corridor stable -> CORRIDOR_CRUISE
  EXIT_DETECTION:
      opening persists -> EXIT_DETECTION
      corridor structure returns -> CORRIDOR_CRUISE
  ENTER_CORRIDOR:
      stable corridor + clear front -> ENTER_CORRIDOR

If no trustworthy recovery action is available before recovery_timeout_s, the
node requests ABORT_CORRIDOR. Missing context also has its own timeout, and
recursive/unsupported recovery targets are forced to ABORT_CORRIDOR.

Important implementation detail
-------------------------------
The existing obstacle_avoidance_v1 node always re-enters its own
OBSTACLE_DECISION state when enabled.  Therefore if this supervisor recommends
AVOID_LEFT / AVOID_RIGHT, the mission bridge re-enables the obstacle controller
and lets it re-validate the side before motion.  The recovery decision is never
used to blindly resume a stale PASS manoeuvre.

ROS FLU convention:
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
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def angle_diff(a: float, b: float) -> float:
    return math.atan2(math.sin(a - b), math.cos(a - b))


@dataclass
class LineModel:
    valid: bool = False
    point: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    direction: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0], dtype=np.float64))
    inlier_points: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float64))
    rms: float = math.inf
    span: float = 0.0


@dataclass
class FaceObservation:
    valid: bool = False
    x_m: float = math.inf
    y_min_m: float = math.inf
    y_max_m: float = -math.inf
    span_m: float = 0.0
    touches_left: bool = False
    touches_right: bool = False


@dataclass
class ReassessObservation:
    front_clearance_m: float = 0.0
    sectors: Dict[str, float] = field(default_factory=dict)

    left_wall_valid: bool = False
    right_wall_valid: bool = False
    left_wall_y_m: Optional[float] = None
    right_wall_y_m: Optional[float] = None
    corridor_width_m: Optional[float] = None
    parallel_error_rad: float = math.inf
    corridor_stable: bool = False

    front_face: FaceObservation = field(default_factory=FaceObservation)
    obstacle_candidate: Optional[str] = None
    obstacle_side: Optional[str] = None
    open_gap_m: Optional[float] = None

    front_blocked: bool = False
    exit_candidate: bool = False


class ReassessState(Enum):
    IDLE = auto()
    WAIT_CONTEXT = auto()
    OBSERVE = auto()
    TRANSITION_REQUESTED = auto()


class HoverAndReassessV2(Node):
    def __init__(self) -> None:
        super().__init__("hover_and_reassess_v2")

        # Topics
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("enable_topic", "/corridor/reassess/enable")
        self.declare_parameter("context_topic", "/corridor/reassess/context")
        self.declare_parameter("cmd_topic", "/corridor/reassess/cmd_vel")
        self.declare_parameter("state_topic", "/corridor/reassess/state")
        self.declare_parameter("result_topic", "/corridor/reassess/result")
        self.declare_parameter("next_state_topic", "/corridor/reassess/next_state")
        self.declare_parameter("diagnostics_topic", "/corridor/reassess/diagnostics")

        # D500 orientation
        self.declare_parameter("invert_scan_angle", False)
        self.declare_parameter("lidar_yaw_offset_deg", 0.0)
        self.declare_parameter("geometry_max_range_m", 8.0)

        # Corridor geometry
        self.declare_parameter("corridor_width_m", 3.5)
        self.declare_parameter("corridor_width_tolerance_m", 0.80)
        self.declare_parameter("max_parallel_error_deg", 15.0)

        # Airframe / bypass safety
        self.declare_parameter("vehicle_width_m", 0.65)
        self.declare_parameter("passage_side_margin_m", 0.25)

        # Longitudinal wall RANSAC
        self.declare_parameter("wall_fit_x_min_m", -0.60)
        self.declare_parameter("wall_fit_x_max_m", 4.50)
        self.declare_parameter("wall_fit_side_min_m", 0.25)
        self.declare_parameter("wall_fit_side_max_m", 3.20)
        self.declare_parameter("wall_axis_tolerance_deg", 25.0)
        self.declare_parameter("wall_ransac_iterations", 120)
        self.declare_parameter("wall_ransac_distance_m", 0.09)
        self.declare_parameter("wall_min_inliers", 10)
        self.declare_parameter("wall_min_span_m", 0.65)
        self.declare_parameter("wall_max_rms_m", 0.13)

        # Transverse face RANSAC
        self.declare_parameter("face_x_min_m", 0.15)
        self.declare_parameter("face_x_max_m", 2.60)
        self.declare_parameter("face_axis_tolerance_deg", 20.0)
        self.declare_parameter("face_ransac_iterations", 150)
        self.declare_parameter("face_ransac_distance_m", 0.08)
        self.declare_parameter("face_min_inliers", 8)
        self.declare_parameter("face_min_span_m", 0.40)
        self.declare_parameter("face_max_rms_m", 0.11)
        self.declare_parameter("face_wall_touch_tolerance_m", 0.38)
        self.declare_parameter("face_exclude_outer_wall_band_m", 0.13)

        # Sector / classification thresholds
        self.declare_parameter("sector_cone_deg", 8.0)
        self.declare_parameter("front_cone_deg", 20.0)
        self.declare_parameter("front_blocked_m", 1.45)
        self.declare_parameter("front_clear_m", 1.80)
        self.declare_parameter("exit_front_clear_m", 2.60)
        self.declare_parameter("exit_side_open_m", 2.25)

        # Recovery persistence / timeout
        self.declare_parameter("recovery_confirm_scans", 5)
        self.declare_parameter("recovery_timeout_s", 8.0)
        self.declare_parameter("context_timeout_s", 2.0)
        self.declare_parameter("scan_stale_s", 0.35)

        self.scan_sub = self.create_subscription(
            LaserScan,
            str(self.get_parameter("scan_topic").value),
            self.scan_callback,
            10,
        )
        self.enable_sub = self.create_subscription(
            Bool,
            str(self.get_parameter("enable_topic").value),
            self.enable_callback,
            10,
        )
        self.context_sub = self.create_subscription(
            String,
            str(self.get_parameter("context_topic").value),
            self.context_callback,
            10,
        )

        self.cmd_pub = self.create_publisher(
            TwistStamped, str(self.get_parameter("cmd_topic").value), 10
        )
        self.state_pub = self.create_publisher(
            String, str(self.get_parameter("state_topic").value), 10
        )
        self.result_pub = self.create_publisher(
            String, str(self.get_parameter("result_topic").value), 10
        )
        self.next_state_pub = self.create_publisher(
            String, str(self.get_parameter("next_state_topic").value), 10
        )
        self.diag_pub = self.create_publisher(
            String, str(self.get_parameter("diagnostics_topic").value), 10
        )

        self.enabled = False
        self.state = ReassessState.IDLE
        self.source_state = ""
        self.pause_reason = ""
        self.context_received = False
        self.state_enter_time = self.get_clock().now()
        self.last_scan_time = None
        self.last_observation: Optional[ReassessObservation] = None
        self.transition_target: Optional[str] = None
        self.transition_reason: Optional[str] = None
        self.candidate_history: Deque[str] = deque(
            maxlen=max(1, int(self.get_parameter("recovery_confirm_scans").value))
        )
        self.rng = np.random.default_rng(2026)

        self.timer = self.create_timer(0.05, self.publish_outputs)
        self.get_logger().info(
            "HOVER_AND_REASSESS V2 started (zero-motion recovery supervisor)"
        )

    # ------------------------------------------------------------------
    # Lifecycle / context
    # ------------------------------------------------------------------

    def reset_session(self) -> None:
        self.state = ReassessState.WAIT_CONTEXT
        self.source_state = ""
        self.pause_reason = ""
        self.context_received = False
        self.state_enter_time = self.get_clock().now()
        self.last_scan_time = None
        self.last_observation = None
        self.transition_target = None
        self.transition_reason = None
        self.candidate_history.clear()

    def enable_callback(self, msg: Bool) -> None:
        if msg.data and not self.enabled:
            self.enabled = True
            self.reset_session()
            self.get_logger().warning("HOVER_AND_REASSESS enabled: immediate zero-motion hold")
        elif not msg.data and self.enabled:
            self.enabled = False
            self.state = ReassessState.IDLE
            self.transition_target = None
            self.transition_reason = None
            self.candidate_history.clear()

    def context_callback(self, msg: String) -> None:
        if not self.enabled:
            return
        try:
            payload = json.loads(msg.data)
            source = str(payload.get("source_state", "")).strip().upper()
            reason = str(payload.get("pause_reason", "")).strip()
        except Exception as exc:
            self.get_logger().warning(f"Invalid reassess context JSON: {exc}")
            return

        if not source:
            return

        changed = source != self.source_state or reason != self.pause_reason
        self.source_state = source
        self.pause_reason = reason
        self.context_received = True

        if self.state == ReassessState.WAIT_CONTEXT:
            self.state = ReassessState.OBSERVE
            self.state_enter_time = self.get_clock().now()
            self.candidate_history.clear()
            self.get_logger().warning(
                f"Reassess context: source={self.source_state}, reason={self.pause_reason or 'unspecified'}"
            )
        elif changed and self.state == ReassessState.OBSERVE:
            self.candidate_history.clear()

    def state_age(self) -> float:
        return (self.get_clock().now() - self.state_enter_time).nanoseconds * 1e-9

    # ------------------------------------------------------------------
    # Scan processing
    # ------------------------------------------------------------------

    def corrected_scan_arrays(self, scan: LaserScan) -> tuple[np.ndarray, np.ndarray]:
        ranges = np.asarray(scan.ranges, dtype=np.float64)
        angles = scan.angle_min + np.arange(ranges.size, dtype=np.float64) * scan.angle_increment
        angles = np.arctan2(np.sin(angles), np.cos(angles))
        if bool(self.get_parameter("invert_scan_angle").value):
            angles = -angles
        angles += math.radians(float(self.get_parameter("lidar_yaw_offset_deg").value))
        angles = np.arctan2(np.sin(angles), np.cos(angles))
        return ranges, angles

    def sector_range(
        self,
        ranges: np.ndarray,
        angles: np.ndarray,
        scan: LaserScan,
        bearing_deg: float,
        cone_deg: float,
        percentile: float = 20.0,
    ) -> float:
        # Treat +inf as max range: for exit/open-space logic, a no-return beam is
        # evidence of free space rather than zero clearance.
        rr = ranges.copy()
        rr[np.isposinf(rr)] = float(scan.range_max)
        valid = np.isfinite(rr)
        valid &= rr >= max(float(scan.range_min), 0.05)
        valid &= rr <= float(scan.range_max)
        target = math.radians(float(bearing_deg))
        delta = np.abs(np.arctan2(np.sin(angles - target), np.cos(angles - target)))
        mask = valid & (delta <= math.radians(float(cone_deg)))
        vals = rr[mask]
        if vals.size == 0:
            return 0.0
        return float(np.percentile(vals, clamp(percentile, 0.0, 100.0)))

    def scan_callback(self, scan: LaserScan) -> None:
        self.last_scan_time = self.get_clock().now()
        if not self.enabled or self.state != ReassessState.OBSERVE:
            return
        if not self.context_received:
            return

        obs = self.extract_observation(scan)
        self.last_observation = obs

        candidate, reason = self.choose_recovery(obs)
        if candidate:
            self.candidate_history.append(candidate)
            needed = int(self.get_parameter("recovery_confirm_scans").value)
            if len(self.candidate_history) >= needed and all(
                x == candidate for x in self.candidate_history
            ):
                self.request_transition(candidate, reason)
                return
        else:
            self.candidate_history.clear()

        if self.state_age() > float(self.get_parameter("recovery_timeout_s").value):
            self.request_transition(
                "ABORT_CORRIDOR",
                f"reassessment timeout from {self.source_state}: no trustworthy recovery action",
            )

    # ------------------------------------------------------------------
    # Geometry extraction
    # ------------------------------------------------------------------

    def fit_line_ransac(
        self,
        points: np.ndarray,
        target_axis_angle_rad: float,
        axis_tolerance_rad: float,
        iterations: int,
        distance_threshold: float,
        min_inliers: int,
        min_span: float,
        max_rms: float,
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
            seg = p2 - p1
            length = float(np.linalg.norm(seg))
            if length < 0.12:
                continue
            d = seg / length
            phi = math.atan2(float(d[1]), float(d[0]))
            # Undirected line: compare modulo pi.
            axis_err = abs(math.atan2(math.sin(phi - target_axis_angle_rad), math.cos(phi - target_axis_angle_rad)))
            axis_err = min(axis_err, abs(math.pi - axis_err))
            if axis_err > axis_tolerance_rad:
                continue

            normal = np.array([-d[1], d[0]], dtype=np.float64)
            residuals = np.abs((points - p1) @ normal)
            mask = residuals <= distance_threshold
            count = int(np.count_nonzero(mask))
            if count < min_inliers:
                continue
            inliers = points[mask]
            along = (inliers - p1) @ d
            span = float(np.percentile(along, 95.0) - np.percentile(along, 5.0))
            if span < min_span:
                continue
            score = count * (1.0 + min(span, 3.0)) / (1.0 + 8.0 * float(np.median(residuals[mask])))
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
        d = vh[0]
        d /= max(float(np.linalg.norm(d)), 1e-9)
        if d[0] < 0.0:
            d = -d

        phi = math.atan2(float(d[1]), float(d[0]))
        axis_err = abs(math.atan2(math.sin(phi - target_axis_angle_rad), math.cos(phi - target_axis_angle_rad)))
        axis_err = min(axis_err, abs(math.pi - axis_err))
        if axis_err > axis_tolerance_rad:
            return LineModel()

        normal = np.array([-d[1], d[0]], dtype=np.float64)
        residuals = np.abs(centered @ normal)
        rms = float(math.sqrt(np.mean(residuals * residuals)))
        along = centered @ d
        span = float(np.percentile(along, 95.0) - np.percentile(along, 5.0))
        if rms > max_rms or span < min_span or inliers.shape[0] < min_inliers:
            return LineModel()

        return LineModel(True, centroid, d, inliers, rms, span)

    @staticmethod
    def line_y_at_x0(line: LineModel) -> Optional[float]:
        if not line.valid:
            return None
        dx = float(line.direction[0])
        if abs(dx) < 1e-6:
            return None
        t = -float(line.point[0]) / dx
        return float(line.point[1] + t * line.direction[1])

    def extract_observation(self, scan: LaserScan) -> ReassessObservation:
        obs = ReassessObservation()
        ranges, angles = self.corrected_scan_arrays(scan)
        if ranges.size < 20:
            return obs

        sector_cone = float(self.get_parameter("sector_cone_deg").value)
        obs.sectors = {
            "L": self.sector_range(ranges, angles, scan, 90.0, sector_cone, 25.0),
            "FL": self.sector_range(ranges, angles, scan, 45.0, sector_cone, 20.0),
            "F": self.sector_range(
                ranges,
                angles,
                scan,
                0.0,
                float(self.get_parameter("front_cone_deg").value),
                10.0,
            ),
            "FR": self.sector_range(ranges, angles, scan, -45.0, sector_cone, 20.0),
            "R": self.sector_range(ranges, angles, scan, -90.0, sector_cone, 25.0),
        }
        obs.front_clearance_m = obs.sectors["F"]

        valid = np.isfinite(ranges)
        valid &= ranges >= max(float(scan.range_min), 0.05)
        valid &= ranges <= min(
            float(scan.range_max), float(self.get_parameter("geometry_max_range_m").value)
        )
        if np.count_nonzero(valid) < 15:
            obs.front_blocked = obs.front_clearance_m <= float(
                self.get_parameter("front_blocked_m").value
            )
            obs.exit_candidate = self.exit_sector_candidate(obs)
            return obs

        r = ranges[valid]
        a = angles[valid]
        points = np.column_stack((r * np.cos(a), r * np.sin(a)))

        x_min = float(self.get_parameter("wall_fit_x_min_m").value)
        x_max = float(self.get_parameter("wall_fit_x_max_m").value)
        side_min = float(self.get_parameter("wall_fit_side_min_m").value)
        side_max = float(self.get_parameter("wall_fit_side_max_m").value)
        long_mask = (points[:, 0] >= x_min) & (points[:, 0] <= x_max)

        left_pts = points[long_mask & (points[:, 1] >= side_min) & (points[:, 1] <= side_max)]
        right_pts = points[long_mask & (points[:, 1] <= -side_min) & (points[:, 1] >= -side_max)]

        wall_args = dict(
            target_axis_angle_rad=0.0,
            axis_tolerance_rad=math.radians(float(self.get_parameter("wall_axis_tolerance_deg").value)),
            iterations=int(self.get_parameter("wall_ransac_iterations").value),
            distance_threshold=float(self.get_parameter("wall_ransac_distance_m").value),
            min_inliers=int(self.get_parameter("wall_min_inliers").value),
            min_span=float(self.get_parameter("wall_min_span_m").value),
            max_rms=float(self.get_parameter("wall_max_rms_m").value),
        )
        left_line = self.fit_line_ransac(left_pts, **wall_args)
        right_line = self.fit_line_ransac(right_pts, **wall_args)
        obs.left_wall_valid = left_line.valid
        obs.right_wall_valid = right_line.valid

        left_y = self.line_y_at_x0(left_line)
        right_y = self.line_y_at_x0(right_line)
        obs.left_wall_y_m = left_y
        obs.right_wall_y_m = right_y

        if left_line.valid and right_line.valid and left_y is not None and right_y is not None:
            d1 = left_line.direction.copy()
            d2 = right_line.direction.copy()
            if d1[0] < 0:
                d1 = -d1
            if d2[0] < 0:
                d2 = -d2
            dot = clamp(float(np.dot(d1, d2)), -1.0, 1.0)
            obs.parallel_error_rad = math.acos(dot)
            obs.corridor_width_m = left_y - right_y
            obs.corridor_stable = (
                left_y > 0.0
                and right_y < 0.0
                and abs(obs.corridor_width_m - float(self.get_parameter("corridor_width_m").value))
                <= float(self.get_parameter("corridor_width_tolerance_m").value)
                and obs.parallel_error_rad
                <= math.radians(float(self.get_parameter("max_parallel_error_deg").value))
            )

        obs.front_face = self.detect_front_face(points, left_line, right_line, left_y, right_y)
        self.classify_obstacle(obs)
        obs.front_blocked = (
            obs.front_clearance_m <= float(self.get_parameter("front_blocked_m").value)
            or obs.front_face.valid
        )
        obs.exit_candidate = self.exit_sector_candidate(obs)
        return obs

    def detect_front_face(
        self,
        points: np.ndarray,
        left_line: LineModel,
        right_line: LineModel,
        left_y: Optional[float],
        right_y: Optional[float],
    ) -> FaceObservation:
        x = points[:, 0]
        y = points[:, 1]
        mask = (
            (x >= float(self.get_parameter("face_x_min_m").value))
            & (x <= float(self.get_parameter("face_x_max_m").value))
            & (np.abs(y) <= 3.0)
        )
        candidates = points[mask]
        if candidates.shape[0] == 0:
            return FaceObservation()

        # Remove points already explained by the two long corridor walls.
        band = float(self.get_parameter("face_exclude_outer_wall_band_m").value)
        keep = np.ones(candidates.shape[0], dtype=bool)
        for line in (left_line, right_line):
            if not line.valid:
                continue
            normal = np.array([-line.direction[1], line.direction[0]], dtype=np.float64)
            dist = np.abs((candidates - line.point) @ normal)
            keep &= dist > band
        candidates = candidates[keep]
        if candidates.shape[0] < int(self.get_parameter("face_min_inliers").value):
            return FaceObservation()

        face_line = self.fit_line_ransac(
            candidates,
            target_axis_angle_rad=math.pi / 2.0,
            axis_tolerance_rad=math.radians(float(self.get_parameter("face_axis_tolerance_deg").value)),
            iterations=int(self.get_parameter("face_ransac_iterations").value),
            distance_threshold=float(self.get_parameter("face_ransac_distance_m").value),
            min_inliers=int(self.get_parameter("face_min_inliers").value),
            min_span=float(self.get_parameter("face_min_span_m").value),
            max_rms=float(self.get_parameter("face_max_rms_m").value),
        )
        if not face_line.valid:
            return FaceObservation()

        pts = face_line.inlier_points
        x_m = float(np.median(pts[:, 0]))
        y_min = float(np.percentile(pts[:, 1], 5.0))
        y_max = float(np.percentile(pts[:, 1], 95.0))
        span = y_max - y_min
        touch_tol = float(self.get_parameter("face_wall_touch_tolerance_m").value)
        touches_left = left_y is not None and abs(y_max - left_y) <= touch_tol
        touches_right = right_y is not None and abs(y_min - right_y) <= touch_tol
        return FaceObservation(True, x_m, y_min, y_max, span, touches_left, touches_right)

    def classify_obstacle(self, obs: ReassessObservation) -> None:
        face = obs.front_face
        if not face.valid or obs.left_wall_y_m is None or obs.right_wall_y_m is None:
            return

        required_gap = float(self.get_parameter("vehicle_width_m").value) + 2.0 * float(
            self.get_parameter("passage_side_margin_m").value
        )

        if face.touches_left and not face.touches_right:
            gap = face.y_min_m - obs.right_wall_y_m
            obs.obstacle_side = "LEFT"
            obs.open_gap_m = gap
            if gap >= required_gap:
                obs.obstacle_candidate = "AVOID_RIGHT"
            return

        if face.touches_right and not face.touches_left:
            gap = obs.left_wall_y_m - face.y_max_m
            obs.obstacle_side = "RIGHT"
            obs.open_gap_m = gap
            if gap >= required_gap:
                obs.obstacle_candidate = "AVOID_LEFT"

    def exit_sector_candidate(self, obs: ReassessObservation) -> bool:
        return (
            obs.front_clearance_m >= float(self.get_parameter("exit_front_clear_m").value)
            and obs.sectors.get("L", 0.0) >= float(self.get_parameter("exit_side_open_m").value)
            and obs.sectors.get("R", 0.0) >= float(self.get_parameter("exit_side_open_m").value)
        )

    # ------------------------------------------------------------------
    # Context-specific recovery policy
    # ------------------------------------------------------------------

    def choose_recovery(self, obs: ReassessObservation) -> tuple[Optional[str], str]:
        src = self.source_state.upper()
        front_clear = obs.front_clearance_m >= float(self.get_parameter("front_clear_m").value)

        if src == "PRE_ENTRY_GEOMETRY_LOCK":
            if obs.corridor_stable and front_clear:
                return "PRE_ENTRY_GEOMETRY_LOCK", "fresh stable entry geometry recovered"
            return None, "entry geometry still unreliable"

        if src == "ENTER_CORRIDOR":
            if obs.corridor_stable and front_clear:
                return "ENTER_CORRIDOR", "corridor geometry recovered during entry"
            if obs.obstacle_candidate:
                return "OBSTACLE_DECISION", "front blockage dominates during entry recovery"
            return None, "entry path still unsafe"

        if src == "CORRIDOR_CRUISE":
            # Spreadsheet ordering: reassess whether blockage, genuine exit, or
            # normal corridor geometry now dominates.
            if obs.obstacle_candidate:
                return "OBSTACLE_DECISION", "fresh scan confirms one-sided front obstacle"
            if obs.exit_candidate:
                return "EXIT_DETECTION", "fresh scan supports genuine corridor opening"
            if obs.corridor_stable and front_clear and not obs.front_blocked:
                return "CORRIDOR_CRUISE", "normal corridor structure stable again"
            return None, "cruise geometry remains ambiguous"

        if src == "OBSTACLE_DECISION":
            if obs.obstacle_candidate:
                return obs.obstacle_candidate, "one bypass side is now consistently safe"
            if obs.corridor_stable and front_clear and not obs.front_blocked:
                return "CORRIDOR_CRUISE", "front blockage cleared; resume centered cruise"
            return None, "no safe reliable bypass side yet"

        if src in {"AVOID_LEFT", "AVOID_RIGHT"}:
            # The spreadsheet sends AVOID failures here but does not define a
            # dedicated context-row recovery.  Safest implementation is to
            # re-evaluate from fresh geometry rather than resume stale PASS
            # state.  The bridge will re-enable obstacle_avoidance_v1, which
            # begins with OBSTACLE_DECISION and independently re-validates.
            if obs.obstacle_candidate:
                return obs.obstacle_candidate, "bypass geometry recovered; revalidate obstacle side"
            if obs.corridor_stable and front_clear and not obs.front_blocked:
                return "CORRIDOR_CRUISE", "obstacle no longer blocks path; corridor stable"
            return None, "bypass geometry still unsafe or ambiguous"

        if src == "EXIT_DETECTION":
            if obs.exit_candidate:
                return "EXIT_DETECTION", "exit opening persists on fresh scans"
            if obs.corridor_stable and front_clear:
                return "CORRIDOR_CRUISE", "exit candidate rejected; corridor structure returned"
            return None, "exit geometry remains ambiguous"

        if src == "SEARCH_ENTRY_MARKER":
            # The spreadsheet correctly makes this a camera-driven recovery.
            # This LiDAR-only recovery node cannot honestly decide marker
            # stability, so it waits for timeout rather than fabricating it.
            return None, "camera marker recovery not available in LiDAR-only node"

        return None, f"unsupported recovery source state {src or 'UNKNOWN'}"

    # ------------------------------------------------------------------
    # Transition / outputs
    # ------------------------------------------------------------------

    def request_transition(self, target: str, reason: str) -> None:
        if self.state == ReassessState.TRANSITION_REQUESTED:
            return

        requested = str(target).upper()
        allowed = {
            "PRE_ENTRY_GEOMETRY_LOCK",
            "ENTER_CORRIDOR",
            "CORRIDOR_CRUISE",
            "OBSTACLE_DECISION",
            "AVOID_LEFT",
            "AVOID_RIGHT",
            "EXIT_DETECTION",
            "ABORT_CORRIDOR",
        }
        # Recursive recovery or an unknown target can never be allowed to
        # create a loop. Fail closed to the terminal mission abort state.
        if requested == "HOVER_AND_REASSESS" or requested not in allowed:
            reason = f"invalid/recursive recovery target {requested or 'EMPTY'}; aborting"
            requested = "ABORT_CORRIDOR"

        self.transition_target = requested
        self.transition_reason = reason
        self.state = ReassessState.TRANSITION_REQUESTED
        self.get_logger().warning(
            f"HOVER_AND_REASSESS -> {self.transition_target}: {reason}"
        )

    @staticmethod
    def zero_command() -> TwistStamped:
        msg = TwistStamped()
        msg.header.frame_id = "base_link"
        return msg

    def diagnostics_dict(self) -> Dict[str, object]:
        data: Dict[str, object] = {
            "enabled": self.enabled,
            "state": self.state.name,
            "source_state": self.source_state or None,
            "pause_reason": self.pause_reason or None,
            "state_age_s": round(self.state_age(), 3) if self.enabled else 0.0,
            "candidate_history": list(self.candidate_history),
            "transition_target": self.transition_target,
            "transition_reason": self.transition_reason,
        }
        obs = self.last_observation
        if obs is not None:
            data.update(
                {
                    "front_clearance_m": round(obs.front_clearance_m, 3),
                    "sectors_m": {k: round(v, 3) for k, v in obs.sectors.items()},
                    "corridor_stable": obs.corridor_stable,
                    "corridor_width_m": round(obs.corridor_width_m, 3)
                    if obs.corridor_width_m is not None
                    else None,
                    "parallel_error_deg": round(math.degrees(obs.parallel_error_rad), 2)
                    if math.isfinite(obs.parallel_error_rad)
                    else None,
                    "front_blocked": obs.front_blocked,
                    "exit_candidate": obs.exit_candidate,
                    "obstacle_candidate": obs.obstacle_candidate,
                    "obstacle_side": obs.obstacle_side,
                    "open_gap_m": round(obs.open_gap_m, 3) if obs.open_gap_m is not None else None,
                    "front_face": {
                        "valid": obs.front_face.valid,
                        "x_m": round(obs.front_face.x_m, 3) if obs.front_face.valid else None,
                        "y_min_m": round(obs.front_face.y_min_m, 3) if obs.front_face.valid else None,
                        "y_max_m": round(obs.front_face.y_max_m, 3) if obs.front_face.valid else None,
                        "touches_left": obs.front_face.touches_left,
                        "touches_right": obs.front_face.touches_right,
                    },
                }
            )
        return data

    def publish_outputs(self) -> None:
        now = self.get_clock().now()

        if self.enabled and self.state == ReassessState.WAIT_CONTEXT:
            # Do not permit a lost/missed context message to leave the vehicle
            # in HOVER_AND_REASSESS forever.
            context_timeout = max(
                0.25, float(self.get_parameter("context_timeout_s").value)
            )
            if self.state_age() > context_timeout:
                self.request_transition(
                    "ABORT_CORRIDOR",
                    f"reassessment context not received within {context_timeout:.1f} s",
                )

        if self.enabled and self.state == ReassessState.OBSERVE:
            if self.last_scan_time is None:
                stale = True
            else:
                age = (now - self.last_scan_time).nanoseconds * 1e-9
                stale = age > float(self.get_parameter("scan_stale_s").value)
            if stale and self.state_age() > float(self.get_parameter("recovery_timeout_s").value):
                self.request_transition(
                    "ABORT_CORRIDOR",
                    "LaserScan remained stale during reassessment",
                )

        cmd = self.zero_command()
        cmd.header.stamp = now.to_msg()
        self.cmd_pub.publish(cmd)

        state_msg = String()
        state_msg.data = "HOVER_AND_REASSESS" if self.enabled else "IDLE"
        self.state_pub.publish(state_msg)

        result_msg = String()
        next_msg = String()
        if self.state == ReassessState.TRANSITION_REQUESTED:
            result_msg.data = f"TRANSITION:{self.transition_target}:{self.transition_reason or ''}"
            next_msg.data = self.transition_target or ""
        elif self.enabled:
            result_msg.data = f"ACTIVE:{self.state.name}:{self.source_state or 'WAITING_CONTEXT'}"
            next_msg.data = ""
        else:
            result_msg.data = "IDLE"
            next_msg.data = ""
        self.result_pub.publish(result_msg)
        self.next_state_pub.publish(next_msg)

        diag = String()
        diag.data = json.dumps(self.diagnostics_dict(), separators=(",", ":"))
        self.diag_pub.publish(diag)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HoverAndReassessV2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
