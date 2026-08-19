#!/usr/bin/env python3

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Deque, Optional, Tuple

import numpy as np

from native.common.types import (
    Attitude,
    BodyVelocity,
    ControllerOutput,
    MissionState,
    NativeScan,
)

from native.controllers.pre_entry import (
    LineModel,
    clamp,
    wrap_pi,
)


# ============================================================
# Helpers
# ============================================================

def axis_angle_from_x(direction: np.ndarray) -> float:
    """
    Unsigned line-axis angle in [0, pi/2] relative to corridor +x.
    """
    angle = abs(
        math.atan2(
            float(direction[1]),
            float(direction[0]),
        )
    )

    if angle > math.pi / 2.0:
        angle = math.pi - angle

    return abs(angle)


# ============================================================
# Configuration
# ============================================================

@dataclass
class ObstacleConfig:

    # D500
    geometry_max_range_m: float = 8.0

    # Corridor / airframe
    corridor_width_m: float = 3.5
    corridor_width_tolerance_m: float = 0.65

    # IMPORTANT:
    # Change this later to measured maximum drone horizontal width.
    vehicle_width_m: float = 0.65

    passage_side_margin_m: float = 0.25

    # Longitudinal corridor-wall RANSAC
    wall_fit_x_min_m: float = -0.50
    wall_fit_x_max_m: float = 4.50

    wall_fit_side_min_m: float = 0.30
    wall_fit_side_max_m: float = 3.20

    wall_max_axis_angle_deg: float = 22.0

    wall_ransac_iterations: int = 140
    wall_ransac_distance_m: float = 0.08

    wall_min_inliers: int = 12
    wall_min_span_m: float = 0.75
    wall_max_rms_m: float = 0.11

    # Transverse obstacle-face RANSAC
    face_x_min_m: float = 0.20
    face_x_max_m: float = 2.20

    rear_face_x_min_m: float = -2.20
    rear_face_x_max_m: float = -0.15

    face_axis_tolerance_deg: float = 18.0

    face_ransac_iterations: int = 180
    face_ransac_distance_m: float = 0.07

    face_min_inliers: int = 10
    face_min_span_m: float = 0.45
    face_max_rms_m: float = 0.09

    face_wall_touch_tolerance_m: float = 0.35

    # Prevent outer corridor-wall samples from forming a fake
    # transverse obstacle.
    face_exclude_outer_wall_band_m: float = 0.12

    # Decision
    decision_confirm_scans: int = 4
    decision_timeout_s: float = 24.0

    front_no_obstacle_release_m: float = 1.80
    no_obstacle_release_scans: int = 4

    # SHIFT
    shift_k_lateral: float = 0.55
    shift_max_lateral_speed_m_s: float = 0.20
    shift_min_lateral_speed_m_s: float = 0.035

    shift_clearance_tolerance_m: float = 0.10
    shift_yaw_tolerance_deg: float = 2.5

    shift_confirm_scans: int = 4
    shift_timeout_s: float = 17.0

    # PASS
    avoid_forward_speed_m_s: float = 0.18

    pass_k_lateral: float = 0.45
    pass_max_lateral_speed_m_s: float = 0.14
    pass_clearance_deadband_m: float = 0.05

    # Yaw
    k_yaw: float = 0.75
    yaw_deadband_deg: float = 0.8
    max_yaw_rate_deg_s: float = 7.0
    yaw_priority_deg: float = 6.0

    # Safety
    hard_outer_wall_clearance_m: float = 0.45
    front_emergency_stop_m: float = 0.60

    avoid_timeout_s: float = 30.0

    # Passing complete when same obstacle face is behind aircraft
    rear_pass_margin_m: float = 0.35
    pass_confirm_scans: int = 4

    front_cone_deg: float = 16.0
    scan_stale_s: float = 0.30

    require_imu: bool = False
    max_tilt_deg: float = 12.0


# ============================================================
# FSM structures
# ============================================================

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

    front_face: FaceObservation = field(
        default_factory=FaceObservation
    )

    rear_face: FaceObservation = field(
        default_factory=FaceObservation
    )

    candidate_state: Optional[str] = None
    obstacle_side: Optional[str] = None

    open_gap_m: Optional[float] = None
    target_outer_clearance_m: Optional[float] = None

    candidate_reason: str = ""


# ============================================================
# Native obstacle controller
# ============================================================

class ObstacleAvoidanceController:

    def __init__(
        self,
        config: Optional[ObstacleConfig] = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else ObstacleConfig()
        )

        self.state = ObstacleState.IDLE
        self.phase = AvoidPhase.NONE

        now = time.monotonic()

        self.state_enter_time = now
        self.phase_enter_time = now

        self.last_observation: Optional[
            ObstacleObservation
        ] = None

        self.latest_command = BodyVelocity.stop()

        self.transition_target: Optional[
            MissionState
        ] = None

        self.transition_reason = ""

        n = max(
            1,
            self.config.decision_confirm_scans,
        )

        self.decision_history: Deque[str] = deque(
            maxlen=n
        )

        self.no_obstacle_streak = 0
        self.shift_streak = 0
        self.pass_streak = 0

        # Latches.
        self.latched_state: Optional[
            ObstacleState
        ] = None

        self.latched_obstacle_side: Optional[
            str
        ] = None

        self.target_outer_clearance_m: Optional[
            float
        ] = None

        self.rng = np.random.default_rng(2026)

    # ========================================================
    # Lifecycle
    # ========================================================

    def enter(self) -> None:

        self.reset()

        self.state = (
            ObstacleState.OBSTACLE_DECISION
        )

        self.state_enter_time = time.monotonic()

        print(
            "[OBSTACLE] entered OBSTACLE_DECISION"
        )

    def reset(self) -> None:

        now = time.monotonic()

        self.state = ObstacleState.IDLE
        self.phase = AvoidPhase.NONE

        self.state_enter_time = now
        self.phase_enter_time = now

        self.last_observation = None

        self.latest_command = BodyVelocity.stop()

        self.transition_target = None
        self.transition_reason = ""

        self.decision_history.clear()

        self.no_obstacle_streak = 0
        self.shift_streak = 0
        self.pass_streak = 0

        self.latched_state = None
        self.latched_obstacle_side = None
        self.target_outer_clearance_m = None

    def state_age(self) -> float:
        return (
            time.monotonic()
            - self.state_enter_time
        )

    def phase_age(self) -> float:
        return (
            time.monotonic()
            - self.phase_enter_time
        )

    # ========================================================
    # Safety
    # ========================================================

    def attitude_is_safe(
        self,
        attitude: Optional[Attitude],
    ) -> bool:

        c = self.config

        if attitude is None:
            return not c.require_imu

        limit = math.radians(
            c.max_tilt_deg
        )

        return (
            abs(attitude.roll_rad) <= limit
            and abs(attitude.pitch_rad) <= limit
        )

    # ========================================================
    # Transition
    # ========================================================

    def request_transition(
        self,
        target: MissionState,
        reason: str,
    ) -> None:

        if (
            self.state
            == ObstacleState.TRANSITION_REQUESTED
        ):
            return

        self.transition_target = target
        self.transition_reason = reason

        self.latest_command = BodyVelocity.stop()

        self.state = (
            ObstacleState.TRANSITION_REQUESTED
        )

        self.phase = AvoidPhase.NONE

        print(
            f"[OBSTACLE] -> {target.value}: "
            f"{reason}"
        )

    # ========================================================
    # Public interface
    # ========================================================

    def step(
        self,
        scan: NativeScan,
        attitude: Optional[Attitude] = None,
    ) -> ControllerOutput:

        if (
            self.state
            == ObstacleState.TRANSITION_REQUESTED
        ):
            return self.output()

        if self.state == ObstacleState.IDLE:
            self.latest_command = BodyVelocity.stop()
            return self.output()

        if scan.age_s > self.config.scan_stale_s:

            self.request_transition(
                MissionState.HOVER_AND_REASSESS,
                "D500 scan stale during obstacle avoidance",
            )

            return self.output()

        obs = self.extract_observation(scan)

        self.last_observation = obs

        if not self.attitude_is_safe(attitude):

            self.request_transition(
                MissionState.HOVER_AND_REASSESS,
                "roll/pitch exceeds obstacle-avoidance limit",
            )

            return self.output()

        if (
            self.state
            == ObstacleState.OBSTACLE_DECISION
        ):

            self.step_decision(obs)

        elif self.state in (
            ObstacleState.AVOID_LEFT,
            ObstacleState.AVOID_RIGHT,
        ):

            self.step_avoid(obs)

        return self.output()

    # ========================================================
    # RANSAC
    # ========================================================

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

            if length < 0.12:
                continue

            direction = segment / length

            axis_angle = axis_angle_from_x(
                direction
            )

            if (
                abs(
                    axis_angle
                    - target_axis_angle_rad
                )
                > axis_tolerance_rad
            ):
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
                <= distance_threshold
            )

            count = int(
                np.count_nonzero(mask)
            )

            if count < min_inliers:
                continue

            inliers = points[mask]

            projection = (
                (inliers - p1)
                @ direction
            )

            span = float(
                np.percentile(
                    projection,
                    95.0,
                )
                - np.percentile(
                    projection,
                    5.0,
                )
            )

            if span < 0.6 * min_span:
                continue

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
                        min_span,
                        0.1,
                    ),
                    3.0,
                )
            )

            residual_factor = (
                1.0
                / (
                    1.0
                    + 12.0
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

        axis_angle = axis_angle_from_x(
            direction
        )

        if (
            abs(
                axis_angle
                - target_axis_angle_rad
            )
            > axis_tolerance_rad
        ):
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

        projection = (
            centered
            @ direction
        )

        span = float(
            np.percentile(
                projection,
                95.0,
            )
            - np.percentile(
                projection,
                5.0,
            )
        )

        ratio = float(
            inliers.shape[0]
            / max(
                points.shape[0],
                1,
            )
        )

        if (
            span < min_span
            or rms > max_rms
        ):
            return LineModel()

        return LineModel(
            valid=True,
            point=centroid,
            direction=direction,
            inlier_points=inliers,
            rms=rms,
            span=span,
            inlier_ratio=ratio,
        )

    def fit_wall(
        self,
        points: np.ndarray,
    ) -> LineModel:

        c = self.config

        return self.fit_line_ransac(
            points=points,
            iterations=c.wall_ransac_iterations,
            distance_threshold=c.wall_ransac_distance_m,
            min_inliers=c.wall_min_inliers,
            min_span=c.wall_min_span_m,
            max_rms=c.wall_max_rms_m,
            target_axis_angle_rad=0.0,
            axis_tolerance_rad=math.radians(
                c.wall_max_axis_angle_deg
            ),
        )

    def fit_face(
        self,
        points: np.ndarray,
    ) -> LineModel:

        c = self.config

        return self.fit_line_ransac(
            points=points,
            iterations=c.face_ransac_iterations,
            distance_threshold=c.face_ransac_distance_m,
            min_inliers=c.face_min_inliers,
            min_span=c.face_min_span_m,
            max_rms=c.face_max_rms_m,
            target_axis_angle_rad=math.pi / 2.0,
            axis_tolerance_rad=math.radians(
                c.face_axis_tolerance_deg
            ),
        )

    # ========================================================
    # Geometry
    # ========================================================

    def front_clearance(
        self,
        ranges: np.ndarray,
        angles: np.ndarray,
        range_min: float,
        range_max: float,
    ) -> float:

        safe = ranges.copy()

        safe[np.isposinf(safe)] = range_max

        valid = np.isfinite(safe)

        valid &= safe >= max(
            range_min,
            0.05,
        )

        valid &= safe <= range_max

        half = math.radians(
            self.config.front_cone_deg
        )

        values = safe[
            valid
            & (np.abs(angles) <= half)
        ]

        if values.size == 0:
            return 0.0

        return float(
            np.percentile(
                values,
                10.0,
            )
        )

    @staticmethod
    def corridor_basis_from_walls(
        left: LineModel,
        right: LineModel,
    ) -> Optional[
        Tuple[np.ndarray, np.ndarray]
    ]:

        directions = []

        for line in (left, right):

            if not line.valid:
                continue

            direction = line.direction.copy()

            if direction[0] < 0:
                direction = -direction

            directions.append(direction)

        if not directions:
            return None

        direction = np.sum(
            np.asarray(directions),
            axis=0,
        )

        norm = float(
            np.linalg.norm(direction)
        )

        if norm < 1e-9:
            return None

        direction /= norm

        if direction[0] < 0:
            direction = -direction

        left_normal = np.array(
            [
                -direction[1],
                direction[0],
            ],
            dtype=np.float64,
        )

        return direction, left_normal

    def make_face_observation(
        self,
        points_corridor: np.ndarray,
        x_min: float,
        x_max: float,
        left_y: float,
        right_y: float,
    ) -> FaceObservation:

        c = self.config

        band = (
            c.face_exclude_outer_wall_band_m
        )

        mask = (
            (points_corridor[:, 0] >= x_min)
            & (points_corridor[:, 0] <= x_max)
            & (
                points_corridor[:, 1]
                <= left_y - band
            )
            & (
                points_corridor[:, 1]
                >= right_y + band
            )
        )

        roi = points_corridor[mask]

        line = self.fit_face(roi)

        if not line.valid:
            return FaceObservation()

        ys = line.inlier_points[:, 1]
        xs = line.inlier_points[:, 0]

        y_min = float(
            np.percentile(ys, 5.0)
        )

        y_max = float(
            np.percentile(ys, 95.0)
        )

        x_med = float(
            np.median(xs)
        )

        touches_left = (
            abs(left_y - y_max)
            <= c.face_wall_touch_tolerance_m
        )

        touches_right = (
            abs(y_min - right_y)
            <= c.face_wall_touch_tolerance_m
        )

        return FaceObservation(
            valid=True,
            x_m=x_med,
            y_min_m=y_min,
            y_max_m=y_max,
            span_m=y_max - y_min,
            rms_m=line.rms,
            inliers=int(
                line.inlier_points.shape[0]
            ),
            touches_left=touches_left,
            touches_right=touches_right,
        )

    def extract_observation(
        self,
        scan: NativeScan,
    ) -> ObstacleObservation:

        c = self.config
        obs = ObstacleObservation()

        ranges = np.asarray(
            scan.ranges_m,
            dtype=np.float64,
        )

        angles = np.asarray(
            scan.angles_rad,
            dtype=np.float64,
        )

        if ranges.size < 20:
            return obs

        obs.front_clearance_m = (
            self.front_clearance(
                ranges,
                angles,
                scan.range_min_m,
                scan.range_max_m,
            )
        )

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
            return obs

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
            (x >= c.wall_fit_x_min_m)
            & (x <= c.wall_fit_x_max_m)
        )

        left_points = points[
            longitudinal
            & (y >= c.wall_fit_side_min_m)
            & (y <= c.wall_fit_side_max_m)
        ]

        right_points = points[
            longitudinal
            & (y <= -c.wall_fit_side_min_m)
            & (y >= -c.wall_fit_side_max_m)
        ]

        left_line = self.fit_wall(
            left_points
        )

        right_line = self.fit_wall(
            right_points
        )

        obs.left_wall_valid = (
            left_line.valid
        )

        obs.right_wall_valid = (
            right_line.valid
        )

        if left_line.valid:

            direction = (
                left_line.direction.copy()
            )

            if direction[0] < 0:
                direction = -direction

            obs.left_wall_yaw_rad = (
                wrap_pi(
                    math.atan2(
                        float(direction[1]),
                        float(direction[0]),
                    )
                )
            )

        if right_line.valid:

            direction = (
                right_line.direction.copy()
            )

            if direction[0] < 0:
                direction = -direction

            obs.right_wall_yaw_rad = (
                wrap_pi(
                    math.atan2(
                        float(direction[1]),
                        float(direction[0]),
                    )
                )
            )

        basis = (
            self.corridor_basis_from_walls(
                left_line,
                right_line,
            )
        )

        if basis is None:

            obs.candidate_reason = (
                "no trustworthy longitudinal wall"
            )

            return obs

        corridor_dir, left_normal = basis

        x_corr = (
            points
            @ corridor_dir
        )

        y_corr = (
            points
            @ left_normal
        )

        points_corridor = np.column_stack(
            (
                x_corr,
                y_corr,
            )
        )

        left_y = None
        right_y = None

        if left_line.valid:

            left_y = float(
                np.median(
                    left_line.inlier_points
                    @ left_normal
                )
            )

        if right_line.valid:

            right_y = float(
                np.median(
                    right_line.inlier_points
                    @ left_normal
                )
            )

        # If obstacle hides one wall, reconstruct it using known
        # 3.5 m corridor width.
        if (
            left_y is None
            and right_y is not None
        ):

            left_y = (
                right_y
                + c.corridor_width_m
            )

        elif (
            right_y is None
            and left_y is not None
        ):

            right_y = (
                left_y
                - c.corridor_width_m
            )

        if (
            left_y is None
            or right_y is None
        ):

            obs.candidate_reason = (
                "cannot reconstruct corridor boundaries"
            )

            return obs

        if (
            left_y <= 0.0
            or right_y >= 0.0
        ):

            obs.candidate_reason = (
                "vehicle origin not between reconstructed walls"
            )

            return obs

        width = left_y - right_y

        if (
            abs(
                width
                - c.corridor_width_m
            )
            > c.corridor_width_tolerance_m
        ):

            obs.candidate_reason = (
                "corridor width implausible "
                f"({width:.2f} m)"
            )

            return obs

        obs.geometry_valid = True

        obs.left_wall_y_m = left_y
        obs.right_wall_y_m = right_y

        obs.d_left_m = left_y
        obs.d_right_m = -right_y

        obs.corridor_width_m = width

        obs.front_face = (
            self.make_face_observation(
                points_corridor,
                c.face_x_min_m,
                c.face_x_max_m,
                left_y,
                right_y,
            )
        )

        obs.rear_face = (
            self.make_face_observation(
                points_corridor,
                c.rear_face_x_min_m,
                c.rear_face_x_max_m,
                left_y,
                right_y,
            )
        )

        face = obs.front_face

        if not face.valid:

            obs.candidate_reason = (
                "no transverse obstacle face ahead"
            )

            return obs

        required_gap = (
            c.vehicle_width_m
            + 2.0
            * c.passage_side_margin_m
        )

        # Obstacle attached to LEFT wall.
        if (
            face.touches_left
            and not face.touches_right
        ):

            open_gap = (
                face.y_min_m
                - right_y
            )

            obs.obstacle_side = "LEFT"
            obs.open_gap_m = open_gap

            if not right_line.valid:

                obs.candidate_reason = (
                    "left obstacle seen but right/open wall invalid"
                )

                return obs

            if open_gap < required_gap:

                obs.candidate_reason = (
                    f"right passage too narrow "
                    f"({open_gap:.2f} < "
                    f"{required_gap:.2f} m)"
                )

                return obs

            obs.candidate_state = (
                "AVOID_RIGHT"
            )

            obs.target_outer_clearance_m = (
                0.5 * open_gap
            )

            obs.candidate_reason = (
                "transverse face touches left wall; "
                "right passage open"
            )

            return obs

        # Obstacle attached to RIGHT wall.
        if (
            face.touches_right
            and not face.touches_left
        ):

            open_gap = (
                left_y
                - face.y_max_m
            )

            obs.obstacle_side = "RIGHT"
            obs.open_gap_m = open_gap

            if not left_line.valid:

                obs.candidate_reason = (
                    "right obstacle seen but left/open wall invalid"
                )

                return obs

            if open_gap < required_gap:

                obs.candidate_reason = (
                    f"left passage too narrow "
                    f"({open_gap:.2f} < "
                    f"{required_gap:.2f} m)"
                )

                return obs

            obs.candidate_state = (
                "AVOID_LEFT"
            )

            obs.target_outer_clearance_m = (
                0.5 * open_gap
            )

            obs.candidate_reason = (
                "transverse face touches right wall; "
                "left passage open"
            )

            return obs

        if (
            face.touches_left
            and face.touches_right
        ):

            obs.candidate_reason = (
                "transverse face appears to span entire corridor"
            )

        else:

            obs.candidate_reason = (
                "transverse face does not connect to exactly one wall"
            )

        return obs

    # ========================================================
    # OBSTACLE_DECISION
    # ========================================================

    def step_decision(
        self,
        obs: ObstacleObservation,
    ) -> None:

        c = self.config

        # Decision state NEVER commands motion.
        self.latest_command = BodyVelocity.stop()

        if (
            self.state_age()
            > c.decision_timeout_s
        ):

            self.request_transition(
                MissionState.HOVER_AND_REASSESS,
                (
                    "obstacle side remained ambiguous "
                    "until decision timeout"
                ),
            )

            return

        # Cruise trigger may have been noise.
        if (
            obs.front_clearance_m
            >= c.front_no_obstacle_release_m
            and not obs.front_face.valid
        ):

            self.no_obstacle_streak += 1

            if (
                self.no_obstacle_streak
                >= c.no_obstacle_release_scans
            ):

                self.request_transition(
                    MissionState.CORRIDOR_CRUISE,
                    (
                        "front obstacle no longer present "
                        "during decision"
                    ),
                )

            return

        self.no_obstacle_streak = 0

        candidate = (
            obs.candidate_state
            or "AMBIGUOUS"
        )

        self.decision_history.append(
            candidate
        )

        needed = (
            c.decision_confirm_scans
        )

        if (
            len(self.decision_history)
            < needed
        ):
            return

        if not all(
            value
            == self.decision_history[-1]
            for value
            in self.decision_history
        ):
            return

        chosen = (
            self.decision_history[-1]
        )

        if chosen not in (
            "AVOID_LEFT",
            "AVOID_RIGHT",
        ):
            return

        if (
            obs.target_outer_clearance_m
            is None
            or obs.obstacle_side
            is None
        ):
            return

        self.target_outer_clearance_m = (
            float(
                obs.target_outer_clearance_m
            )
        )

        self.latched_obstacle_side = (
            obs.obstacle_side
        )

        if chosen == "AVOID_LEFT":

            self.latched_state = (
                ObstacleState.AVOID_LEFT
            )

        else:

            self.latched_state = (
                ObstacleState.AVOID_RIGHT
            )

        self.state = self.latched_state

        self.state_enter_time = (
            time.monotonic()
        )

        self.phase = AvoidPhase.SHIFT

        self.phase_enter_time = (
            time.monotonic()
        )

        self.shift_streak = 0
        self.pass_streak = 0

        self.latest_command = (
            BodyVelocity.stop()
        )

        print(
            f"[OBSTACLE] OBSTACLE_DECISION -> "
            f"{chosen}: "
            f"{obs.candidate_reason}; "
            f"open_gap={obs.open_gap_m:.2f} m; "
            f"target_outer_clearance="
            f"{self.target_outer_clearance_m:.2f} m"
        )

    # ========================================================
    # AVOID helpers
    # ========================================================

    def open_wall_measurements(
        self,
        obs: ObstacleObservation,
    ) -> tuple[
        Optional[float],
        Optional[float],
    ]:

        if (
            self.state
            == ObstacleState.AVOID_LEFT
        ):

            if not obs.left_wall_valid:
                return None, None

            return (
                obs.d_left_m,
                obs.left_wall_yaw_rad,
            )

        if (
            self.state
            == ObstacleState.AVOID_RIGHT
        ):

            if not obs.right_wall_valid:
                return None, None

            return (
                obs.d_right_m,
                obs.right_wall_yaw_rad,
            )

        return None, None

    def lateral_command_to_outer_clearance(
        self,
        actual_clearance: float,
    ) -> float:

        assert (
            self.target_outer_clearance_m
            is not None
        )

        c = self.config

        error = (
            actual_clearance
            - self.target_outer_clearance_m
        )

        # Open-left:
        # +vy moves LEFT and reduces left-wall distance.
        #
        # Open-right:
        # -vy moves RIGHT and reduces right-wall distance.
        sign = (
            +1.0
            if self.state
            == ObstacleState.AVOID_LEFT
            else -1.0
        )

        if self.phase == AvoidPhase.SHIFT:

            command = sign * clamp(
                c.shift_k_lateral
                * error,

                -c.shift_max_lateral_speed_m_s,
                c.shift_max_lateral_speed_m_s,
            )

            if (
                abs(command)
                < c.shift_min_lateral_speed_m_s
                and abs(error) > 1e-6
            ):

                command = math.copysign(
                    c.shift_min_lateral_speed_m_s,
                    command,
                )

            return command

        if (
            abs(error)
            <= c.pass_clearance_deadband_m
        ):
            return 0.0

        return sign * clamp(
            c.pass_k_lateral
            * error,

            -c.pass_max_lateral_speed_m_s,
            c.pass_max_lateral_speed_m_s,
        )

    def yaw_command(
        self,
        yaw_error: float,
    ) -> float:

        c = self.config

        deadband = math.radians(
            c.yaw_deadband_deg
        )

        if abs(yaw_error) <= deadband:
            return 0.0

        maximum = math.radians(
            c.max_yaw_rate_deg_s
        )

        return clamp(
            c.k_yaw * yaw_error,
            -maximum,
            maximum,
        )

    def rear_face_matches_latched_obstacle(
        self,
        obs: ObstacleObservation,
    ) -> bool:

        face = obs.rear_face

        if not face.valid:
            return False

        if (
            face.x_m
            > -self.config.rear_pass_margin_m
        ):
            return False

        if (
            self.latched_obstacle_side
            == "LEFT"
        ):

            return (
                face.touches_left
                and not face.touches_right
            )

        if (
            self.latched_obstacle_side
            == "RIGHT"
        ):

            return (
                face.touches_right
                and not face.touches_left
            )

        return False

    # ========================================================
    # AVOID_LEFT / AVOID_RIGHT
    # ========================================================

    def step_avoid(
        self,
        obs: ObstacleObservation,
    ) -> None:

        c = self.config

        if (
            self.target_outer_clearance_m
            is None
            or self.latched_obstacle_side
            is None
        ):

            self.request_transition(
                MissionState.HOVER_AND_REASSESS,
                (
                    "avoidance entered without "
                    "latched bypass geometry"
                ),
            )

            return

        if (
            self.state_age()
            > c.avoid_timeout_s
        ):

            self.request_transition(
                MissionState.HOVER_AND_REASSESS,
                "obstacle bypass exceeded total timeout",
            )

            return

        (
            actual_clearance,
            yaw_error,
        ) = self.open_wall_measurements(
            obs
        )

        if (
            actual_clearance is None
            or yaw_error is None
        ):

            self.latest_command = (
                BodyVelocity.stop()
            )

            self.request_transition(
                MissionState.HOVER_AND_REASSESS,
                (
                    "open-side corridor wall "
                    "lost during bypass"
                ),
            )

            return

        if (
            actual_clearance
            <= c.hard_outer_wall_clearance_m
        ):

            self.request_transition(
                MissionState.HOVER_AND_REASSESS,
                (
                    "open-side wall clearance "
                    "critically low "
                    f"({actual_clearance:.2f} m)"
                ),
            )

            return

        yaw_rate = self.yaw_command(
            yaw_error
        )

        yaw_priority = math.radians(
            c.yaw_priority_deg
        )

        # ----------------------------------------------------
        # SHIFT
        # ----------------------------------------------------

        if self.phase == AvoidPhase.SHIFT:

            if (
                self.phase_age()
                > c.shift_timeout_s
            ):

                self.request_transition(
                    MissionState.HOVER_AND_REASSESS,
                    (
                        "failed to reach bypass lane "
                        "before shift timeout"
                    ),
                )

                return

            clearance_error = (
                actual_clearance
                - self.target_outer_clearance_m
            )

            yaw_tolerance = math.radians(
                c.shift_yaw_tolerance_deg
            )

            aligned = (
                abs(clearance_error)
                <= c.shift_clearance_tolerance_m
                and abs(yaw_error)
                <= yaw_tolerance
            )

            if aligned:

                self.shift_streak += 1

                self.latest_command = (
                    BodyVelocity.stop()
                )

                if (
                    self.shift_streak
                    >= c.shift_confirm_scans
                ):

                    self.phase = AvoidPhase.PASS

                    self.phase_enter_time = (
                        time.monotonic()
                    )

                    self.shift_streak = 0
                    self.pass_streak = 0

                    print(
                        f"[OBSTACLE] "
                        f"{self.state.name}: "
                        f"SHIFT -> PASS"
                    )

                return

            self.shift_streak = 0

            # Major yaw error: rotate first.
            if abs(yaw_error) > yaw_priority:

                self.latest_command = (
                    BodyVelocity(
                        yaw_rate_rad_s=yaw_rate
                    )
                )

                return

            vy = (
                self.lateral_command_to_outer_clearance(
                    actual_clearance
                )
            )

            self.latest_command = BodyVelocity(
                vx_m_s=0.0,
                vy_m_s=vy,
                yaw_rate_rad_s=yaw_rate,
            )

            return

        # ----------------------------------------------------
        # PASS
        # ----------------------------------------------------

        if self.phase == AvoidPhase.PASS:

            if (
                0.0
                < obs.front_clearance_m
                <= c.front_emergency_stop_m
            ):

                self.request_transition(
                    MissionState.HOVER_AND_REASSESS,
                    (
                        "unexpected object in bypass lane "
                        f"({obs.front_clearance_m:.2f} m)"
                    ),
                )

                return

            if (
                self.rear_face_matches_latched_obstacle(
                    obs
                )
            ):

                self.pass_streak += 1

                if (
                    self.pass_streak
                    >= c.pass_confirm_scans
                ):

                    self.request_transition(
                        MissionState.CORRIDOR_CRUISE,
                        (
                            f"{self.state.name} complete; "
                            "obstacle face confirmed behind"
                        ),
                    )

                    return

            else:

                self.pass_streak = 0

            if abs(yaw_error) > yaw_priority:

                self.latest_command = (
                    BodyVelocity(
                        yaw_rate_rad_s=yaw_rate
                    )
                )

                return

            vy = (
                self.lateral_command_to_outer_clearance(
                    actual_clearance
                )
            )

            self.latest_command = BodyVelocity(
                vx_m_s=max(
                    0.0,
                    c.avoid_forward_speed_m_s,
                ),
                vy_m_s=vy,
                yaw_rate_rad_s=yaw_rate,
            )

            return

        self.request_transition(
            MissionState.HOVER_AND_REASSESS,
            "invalid internal avoidance phase",
        )

    # ========================================================
    # Output / diagnostics
    # ========================================================

    def output(self) -> ControllerOutput:

        return ControllerOutput(
            command=self.latest_command,
            next_state=self.transition_target,
            status=(
                f"{self.state.name}/"
                f"{self.phase.name}"
            ),
            reason=self.transition_reason,
            confidence=None,
        )

    def diagnostics(self) -> dict:

        obs = self.last_observation

        result = {
            "state": self.state.name,
            "phase": self.phase.name,

            "transition_target": (
                self.transition_target.value
                if self.transition_target
                is not None
                else None
            ),

            "transition_reason":
                self.transition_reason,

            "latched_obstacle_side":
                self.latched_obstacle_side,

            "target_outer_clearance_m":
                self.target_outer_clearance_m,

            "decision_history":
                list(self.decision_history),
        }

        if obs is None:
            return result

        result["observation"] = {

            "geometry_valid":
                obs.geometry_valid,

            "front_clearance_m":
                round(
                    obs.front_clearance_m,
                    3,
                ),

            "left_wall_valid":
                obs.left_wall_valid,

            "right_wall_valid":
                obs.right_wall_valid,

            "d_left_m":
                (
                    round(obs.d_left_m, 3)
                    if obs.d_left_m
                    is not None
                    else None
                ),

            "d_right_m":
                (
                    round(obs.d_right_m, 3)
                    if obs.d_right_m
                    is not None
                    else None
                ),

            "corridor_width_m":
                (
                    round(
                        obs.corridor_width_m,
                        3,
                    )
                    if obs.corridor_width_m
                    is not None
                    else None
                ),

            "candidate_state":
                obs.candidate_state,

            "obstacle_side":
                obs.obstacle_side,

            "open_gap_m":
                (
                    round(obs.open_gap_m, 3)
                    if obs.open_gap_m
                    is not None
                    else None
                ),

            "candidate_reason":
                obs.candidate_reason,

            "front_face_valid":
                obs.front_face.valid,

            "front_face_x_m":
                (
                    round(
                        obs.front_face.x_m,
                        3,
                    )
                    if obs.front_face.valid
                    else None
                ),

            "front_touches_left":
                obs.front_face.touches_left,

            "front_touches_right":
                obs.front_face.touches_right,

            "rear_face_valid":
                obs.rear_face.valid,
        }

        return result


