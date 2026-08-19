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
    BodyVelocity,
    ControllerOutput,
    MissionState,
    NativeScan,
)

from native.controllers.pre_entry import (
    LineModel,
    clamp,
)


# ============================================================
# Configuration
# ============================================================

@dataclass
class ReassessConfig:

    # LiDAR
    geometry_max_range_m: float = 8.0

    # Corridor geometry
    corridor_width_m: float = 3.5
    corridor_width_tolerance_m: float = 0.80
    max_parallel_error_deg: float = 15.0

    # Airframe / bypass safety
    vehicle_width_m: float = 0.65
    passage_side_margin_m: float = 0.25

    # Longitudinal wall RANSAC
    wall_fit_x_min_m: float = -0.60
    wall_fit_x_max_m: float = 4.50

    wall_fit_side_min_m: float = 0.25
    wall_fit_side_max_m: float = 3.20

    wall_axis_tolerance_deg: float = 25.0

    wall_ransac_iterations: int = 120
    wall_ransac_distance_m: float = 0.09

    wall_min_inliers: int = 10
    wall_min_span_m: float = 0.65
    wall_max_rms_m: float = 0.13

    # Transverse obstacle face
    face_x_min_m: float = 0.15
    face_x_max_m: float = 2.60

    face_axis_tolerance_deg: float = 20.0

    face_ransac_iterations: int = 150
    face_ransac_distance_m: float = 0.08

    face_min_inliers: int = 8
    face_min_span_m: float = 0.40
    face_max_rms_m: float = 0.11

    face_wall_touch_tolerance_m: float = 0.38
    face_exclude_outer_wall_band_m: float = 0.13

    # Sector classification
    sector_cone_deg: float = 8.0
    front_cone_deg: float = 20.0

    front_blocked_m: float = 1.45
    front_clear_m: float = 1.80

    exit_front_clear_m: float = 2.60
    exit_side_open_m: float = 2.25

    # Recovery supervision
    recovery_confirm_scans: int = 5
    recovery_timeout_s: float = 8.0
    scan_stale_s: float = 0.35


# ============================================================
# Observation structures
# ============================================================

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

    sectors: Dict[str, float] = field(
        default_factory=dict
    )

    left_wall_valid: bool = False
    right_wall_valid: bool = False

    left_wall_y_m: Optional[float] = None
    right_wall_y_m: Optional[float] = None

    corridor_width_m: Optional[float] = None

    parallel_error_rad: float = math.inf

    corridor_stable: bool = False

    front_face: FaceObservation = field(
        default_factory=FaceObservation
    )

    obstacle_candidate: Optional[
        MissionState
    ] = None

    obstacle_side: Optional[str] = None

    open_gap_m: Optional[float] = None

    front_blocked: bool = False

    exit_candidate: bool = False


class ReassessState(Enum):

    IDLE = auto()

    OBSERVE = auto()

    TRANSITION_REQUESTED = auto()


# ============================================================
# Controller
# ============================================================

class HoverAndReassessController:

    """
    Zero-motion recovery supervisor.

    This controller NEVER requests non-zero vehicle motion.
    """

    def __init__(
        self,
        config: Optional[ReassessConfig] = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else ReassessConfig()
        )

        self.state = ReassessState.IDLE

        self.source_state: Optional[
            MissionState
        ] = None

        self.pause_reason = ""

        self.state_enter_time = (
            time.monotonic()
        )

        self.last_scan_time: Optional[
            float
        ] = None

        self.last_observation: Optional[
            ReassessObservation
        ] = None

        self.transition_target: Optional[
            MissionState
        ] = None

        self.transition_reason = ""

        self.candidate_history: Deque[
            MissionState
        ] = deque(
            maxlen=max(
                1,
                self.config.recovery_confirm_scans,
            )
        )

        self.latest_command = (
            BodyVelocity.stop()
        )

        self.rng = np.random.default_rng(
            2026
        )

    # ========================================================
    # Lifecycle
    # ========================================================

    def enter(
        self,
        source_state: MissionState,
        pause_reason: str = "",
    ) -> None:

        self.state = (
            ReassessState.OBSERVE
        )

        self.source_state = source_state

        self.pause_reason = str(
            pause_reason
        )

        self.state_enter_time = (
            time.monotonic()
        )

        self.last_scan_time = None
        self.last_observation = None

        self.transition_target = None
        self.transition_reason = ""

        self.candidate_history.clear()

        self.latest_command = (
            BodyVelocity.stop()
        )

        print(
            "[REASSESS] entered from "
            f"{source_state.value}: "
            f"{self.pause_reason or 'unspecified'}"
        )

    def reset(self) -> None:

        self.state = ReassessState.IDLE

        self.source_state = None
        self.pause_reason = ""

        self.state_enter_time = (
            time.monotonic()
        )

        self.last_scan_time = None
        self.last_observation = None

        self.transition_target = None
        self.transition_reason = ""

        self.candidate_history.clear()

        self.latest_command = (
            BodyVelocity.stop()
        )

    def state_age(self) -> float:

        return max(
            0.0,
            time.monotonic()
            - self.state_enter_time,
        )

    # ========================================================
    # Public step
    # ========================================================

    def step(
        self,
        scan: Optional[NativeScan],
    ) -> ControllerOutput:

        # Absolute invariant:
        #
        # HOVER_AND_REASSESS never commands motion.
        self.latest_command = (
            BodyVelocity.stop()
        )

        if self.state == ReassessState.IDLE:
            return self.output()

        if (
            self.state
            == ReassessState.TRANSITION_REQUESTED
        ):
            return self.output()

        if self.source_state is None:

            self.request_transition(
                MissionState.ABORT_CORRIDOR,
                "reassessment entered without source state",
            )

            return self.output()

        # Missing/stale scan is tolerated until recovery timeout.
        if scan is None:

            if (
                self.state_age()
                >= self.config.recovery_timeout_s
            ):

                self.request_transition(
                    MissionState.ABORT_CORRIDOR,
                    "D500 remained unavailable during reassessment",
                )

            return self.output()

        if (
            scan.age_s
            > self.config.scan_stale_s
        ):

            if (
                self.state_age()
                >= self.config.recovery_timeout_s
            ):

                self.request_transition(
                    MissionState.ABORT_CORRIDOR,
                    "D500 remained stale during reassessment",
                )

            return self.output()

        self.last_scan_time = (
            time.monotonic()
        )

        obs = self.extract_observation(
            scan
        )

        self.last_observation = obs

        candidate, reason = (
            self.choose_recovery(obs)
        )

        if candidate is not None:

            self.candidate_history.append(
                candidate
            )

            needed = (
                self.config.recovery_confirm_scans
            )

            if (
                len(self.candidate_history)
                >= needed
                and all(
                    item == candidate
                    for item
                    in self.candidate_history
                )
            ):

                self.request_transition(
                    candidate,
                    reason,
                )

                return self.output()

        else:

            self.candidate_history.clear()

        if (
            self.state_age()
            >= self.config.recovery_timeout_s
        ):

            self.request_transition(
                MissionState.ABORT_CORRIDOR,
                (
                    "reassessment timeout from "
                    f"{self.source_state.value}: "
                    "no trustworthy recovery action"
                ),
            )

        return self.output()

    # ========================================================
    # Sector measurements
    # ========================================================

    def sector_range(
        self,
        ranges: np.ndarray,
        angles: np.ndarray,
        scan: NativeScan,
        bearing_deg: float,
        cone_deg: float,
        percentile: float,
    ) -> float:

        # +inf / no-return means free space for
        # opening/exit classification.
        rr = ranges.copy()

        rr[np.isposinf(rr)] = (
            scan.range_max_m
        )

        valid = np.isfinite(rr)

        valid &= rr >= max(
            scan.range_min_m,
            0.05,
        )

        valid &= (
            rr <= scan.range_max_m
        )

        target = math.radians(
            bearing_deg
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

        mask = (
            valid
            & (
                delta
                <= math.radians(cone_deg)
            )
        )

        values = rr[mask]

        if values.size == 0:
            return 0.0

        return float(
            np.percentile(
                values,
                clamp(
                    percentile,
                    0.0,
                    100.0,
                ),
            )
        )

    # ========================================================
    # Generic constrained RANSAC
    # ========================================================

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

        if (
            points.shape[0]
            < min_inliers
        ):
            return LineModel()

        n = points.shape[0]

        best_mask = None
        best_score = -math.inf

        for _ in range(
            max(1, iterations)
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

            if length < 0.12:
                continue

            direction = (
                segment / length
            )

            phi = math.atan2(
                float(direction[1]),
                float(direction[0]),
            )

            # Undirected line comparison modulo pi.
            axis_error = abs(
                math.atan2(
                    math.sin(
                        phi
                        - target_axis_angle_rad
                    ),
                    math.cos(
                        phi
                        - target_axis_angle_rad
                    ),
                )
            )

            axis_error = min(
                axis_error,
                abs(
                    math.pi
                    - axis_error
                ),
            )

            if (
                axis_error
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

            along = (
                (inliers - p1)
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

            if span < min_span:
                continue

            median_residual = float(
                np.median(
                    residuals[mask]
                )
            )

            score = (
                count
                * (
                    1.0
                    + min(span, 3.0)
                )
                / (
                    1.0
                    + 8.0
                    * median_residual
                )
            )

            if score > best_score:

                best_score = score
                best_mask = mask

        if best_mask is None:
            return LineModel()

        inliers = points[
            best_mask
        ]

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

        if direction[0] < 0.0:
            direction = -direction

        phi = math.atan2(
            float(direction[1]),
            float(direction[0]),
        )

        axis_error = abs(
            math.atan2(
                math.sin(
                    phi
                    - target_axis_angle_rad
                ),
                math.cos(
                    phi
                    - target_axis_angle_rad
                ),
            )
        )

        axis_error = min(
            axis_error,
            abs(
                math.pi
                - axis_error
            ),
        )

        if (
            axis_error
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

        if (
            rms > max_rms
            or span < min_span
            or inliers.shape[0]
            < min_inliers
        ):
            return LineModel()

        return LineModel(
            valid=True,
            point=centroid,
            direction=direction,
            inlier_points=inliers,
            rms=rms,
            span=span,
        )

    @staticmethod
    def line_y_at_x0(
        line: LineModel,
    ) -> Optional[float]:

        if not line.valid:
            return None

        dx = float(
            line.direction[0]
        )

        if abs(dx) < 1e-6:
            return None

        t = (
            -float(line.point[0])
            / dx
        )

        return float(
            line.point[1]
            + t
            * line.direction[1]
        )

    # ========================================================
    # Observation extraction
    # ========================================================

    def extract_observation(
        self,
        scan: NativeScan,
    ) -> ReassessObservation:

        c = self.config

        obs = ReassessObservation()

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

        # ----------------------------------------------------
        # Sectors
        # ----------------------------------------------------

        obs.sectors = {

            "L":
                self.sector_range(
                    ranges,
                    angles,
                    scan,
                    +90.0,
                    c.sector_cone_deg,
                    25.0,
                ),

            "FL":
                self.sector_range(
                    ranges,
                    angles,
                    scan,
                    +45.0,
                    c.sector_cone_deg,
                    20.0,
                ),

            "F":
                self.sector_range(
                    ranges,
                    angles,
                    scan,
                    0.0,
                    c.front_cone_deg,
                    10.0,
                ),

            "FR":
                self.sector_range(
                    ranges,
                    angles,
                    scan,
                    -45.0,
                    c.sector_cone_deg,
                    20.0,
                ),

            "R":
                self.sector_range(
                    ranges,
                    angles,
                    scan,
                    -90.0,
                    c.sector_cone_deg,
                    25.0,
                ),
        }

        obs.front_clearance_m = (
            obs.sectors["F"]
        )

        # ----------------------------------------------------
        # XY geometry
        # ----------------------------------------------------

        valid = np.isfinite(ranges)

        valid &= ranges >= max(
            scan.range_min_m,
            0.05,
        )

        valid &= ranges <= min(
            scan.range_max_m,
            c.geometry_max_range_m,
        )

        if (
            np.count_nonzero(valid)
            < 15
        ):

            obs.front_blocked = (
                obs.front_clearance_m
                <= c.front_blocked_m
            )

            obs.exit_candidate = (
                self.exit_sector_candidate(
                    obs
                )
            )

            return obs

        r = ranges[valid]
        a = angles[valid]

        points = np.column_stack(
            (
                r * np.cos(a),
                r * np.sin(a),
            )
        )

        longitudinal = (
            (points[:, 0]
             >= c.wall_fit_x_min_m)
            & (points[:, 0]
               <= c.wall_fit_x_max_m)
        )

        left_points = points[
            longitudinal
            & (
                points[:, 1]
                >= c.wall_fit_side_min_m
            )
            & (
                points[:, 1]
                <= c.wall_fit_side_max_m
            )
        ]

        right_points = points[
            longitudinal
            & (
                points[:, 1]
                <= -c.wall_fit_side_min_m
            )
            & (
                points[:, 1]
                >= -c.wall_fit_side_max_m
            )
        ]

        wall_args = dict(

            target_axis_angle_rad=0.0,

            axis_tolerance_rad=math.radians(
                c.wall_axis_tolerance_deg
            ),

            iterations=
                c.wall_ransac_iterations,

            distance_threshold=
                c.wall_ransac_distance_m,

            min_inliers=
                c.wall_min_inliers,

            min_span=
                c.wall_min_span_m,

            max_rms=
                c.wall_max_rms_m,
        )

        left_line = self.fit_line_ransac(
            left_points,
            **wall_args,
        )

        right_line = self.fit_line_ransac(
            right_points,
            **wall_args,
        )

        obs.left_wall_valid = (
            left_line.valid
        )

        obs.right_wall_valid = (
            right_line.valid
        )

        left_y = self.line_y_at_x0(
            left_line
        )

        right_y = self.line_y_at_x0(
            right_line
        )

        obs.left_wall_y_m = left_y
        obs.right_wall_y_m = right_y

        # ----------------------------------------------------
        # Stable corridor test
        # ----------------------------------------------------

        if (
            left_line.valid
            and right_line.valid
            and left_y is not None
            and right_y is not None
        ):

            d1 = (
                left_line.direction.copy()
            )

            d2 = (
                right_line.direction.copy()
            )

            if d1[0] < 0:
                d1 = -d1

            if d2[0] < 0:
                d2 = -d2

            dot = clamp(
                float(
                    np.dot(d1, d2)
                ),
                -1.0,
                1.0,
            )

            obs.parallel_error_rad = (
                math.acos(dot)
            )

            obs.corridor_width_m = (
                left_y
                - right_y
            )

            obs.corridor_stable = (

                left_y > 0.0

                and right_y < 0.0

                and abs(
                    obs.corridor_width_m
                    - c.corridor_width_m
                )
                <= c.corridor_width_tolerance_m

                and obs.parallel_error_rad
                <= math.radians(
                    c.max_parallel_error_deg
                )
            )

        # ----------------------------------------------------
        # Obstacle / exit classification
        # ----------------------------------------------------

        obs.front_face = (
            self.detect_front_face(
                points,
                left_line,
                right_line,
                left_y,
                right_y,
            )
        )

        self.classify_obstacle(obs)

        obs.front_blocked = (
            obs.front_clearance_m
            <= c.front_blocked_m
            or obs.front_face.valid
        )

        obs.exit_candidate = (
            self.exit_sector_candidate(
                obs
            )
        )

        return obs

    # ========================================================
    # Obstacle face
    # ========================================================

    def detect_front_face(
        self,
        points: np.ndarray,
        left_line: LineModel,
        right_line: LineModel,
        left_y: Optional[float],
        right_y: Optional[float],
    ) -> FaceObservation:

        c = self.config

        x = points[:, 0]
        y = points[:, 1]

        mask = (
            (x >= c.face_x_min_m)
            & (x <= c.face_x_max_m)
            & (np.abs(y) <= 3.0)
        )

        candidates = points[mask]

        if candidates.shape[0] == 0:
            return FaceObservation()

        # Remove points explained by long outer walls.
        keep = np.ones(
            candidates.shape[0],
            dtype=bool,
        )

        for line in (
            left_line,
            right_line,
        ):

            if not line.valid:
                continue

            normal = np.array(
                [
                    -line.direction[1],
                    line.direction[0],
                ],
                dtype=np.float64,
            )

            distance = np.abs(
                (candidates - line.point)
                @ normal
            )

            keep &= (
                distance
                > c.face_exclude_outer_wall_band_m
            )

        candidates = (
            candidates[keep]
        )

        if (
            candidates.shape[0]
            < c.face_min_inliers
        ):
            return FaceObservation()

        face_line = (
            self.fit_line_ransac(

                candidates,

                target_axis_angle_rad=
                    math.pi / 2.0,

                axis_tolerance_rad=
                    math.radians(
                        c.face_axis_tolerance_deg
                    ),

                iterations=
                    c.face_ransac_iterations,

                distance_threshold=
                    c.face_ransac_distance_m,

                min_inliers=
                    c.face_min_inliers,

                min_span=
                    c.face_min_span_m,

                max_rms=
                    c.face_max_rms_m,
            )
        )

        if not face_line.valid:
            return FaceObservation()

        pts = face_line.inlier_points

        x_m = float(
            np.median(
                pts[:, 0]
            )
        )

        y_min = float(
            np.percentile(
                pts[:, 1],
                5.0,
            )
        )

        y_max = float(
            np.percentile(
                pts[:, 1],
                95.0,
            )
        )

        span = (
            y_max
            - y_min
        )

        touches_left = (
            left_y is not None
            and abs(
                y_max
                - left_y
            )
            <= c.face_wall_touch_tolerance_m
        )

        touches_right = (
            right_y is not None
            and abs(
                y_min
                - right_y
            )
            <= c.face_wall_touch_tolerance_m
        )

        return FaceObservation(
            valid=True,
            x_m=x_m,
            y_min_m=y_min,
            y_max_m=y_max,
            span_m=span,
            touches_left=touches_left,
            touches_right=touches_right,
        )

    # ========================================================
    # Obstacle / exit classification
    # ========================================================

    def classify_obstacle(
        self,
        obs: ReassessObservation,
    ) -> None:

        c = self.config
        face = obs.front_face

        if (
            not face.valid
            or obs.left_wall_y_m is None
            or obs.right_wall_y_m is None
        ):
            return

        required_gap = (
            c.vehicle_width_m
            + 2.0
            * c.passage_side_margin_m
        )

        # Obstacle attached LEFT -> go RIGHT.
        if (
            face.touches_left
            and not face.touches_right
        ):

            gap = (
                face.y_min_m
                - obs.right_wall_y_m
            )

            obs.obstacle_side = "LEFT"
            obs.open_gap_m = gap

            if gap >= required_gap:

                obs.obstacle_candidate = (
                    MissionState.AVOID_RIGHT
                )

            return

        # Obstacle attached RIGHT -> go LEFT.
        if (
            face.touches_right
            and not face.touches_left
        ):

            gap = (
                obs.left_wall_y_m
                - face.y_max_m
            )

            obs.obstacle_side = "RIGHT"
            obs.open_gap_m = gap

            if gap >= required_gap:

                obs.obstacle_candidate = (
                    MissionState.AVOID_LEFT
                )

    def exit_sector_candidate(
        self,
        obs: ReassessObservation,
    ) -> bool:

        c = self.config

        return (
            obs.front_clearance_m
            >= c.exit_front_clear_m

            and obs.sectors.get(
                "L",
                0.0,
            )
            >= c.exit_side_open_m

            and obs.sectors.get(
                "R",
                0.0,
            )
            >= c.exit_side_open_m
        )

    # ========================================================
    # Context-specific recovery policy
    # ========================================================

    def choose_recovery(
        self,
        obs: ReassessObservation,
    ) -> tuple[
        Optional[MissionState],
        str,
    ]:

        assert (
            self.source_state is not None
        )

        c = self.config

        src = self.source_state

        front_clear = (
            obs.front_clearance_m
            >= c.front_clear_m
        )

        # ----------------------------------------------------
        # PRE_ENTRY
        # ----------------------------------------------------

        if (
            src
            == MissionState.PRE_ENTRY_GEOMETRY_LOCK
        ):

            if (
                obs.corridor_stable
                and front_clear
            ):

                return (
                    MissionState.PRE_ENTRY_GEOMETRY_LOCK,
                    "fresh stable entry geometry recovered",
                )

            return (
                None,
                "entry geometry still unreliable",
            )

        # ----------------------------------------------------
        # ENTER_CORRIDOR
        # ----------------------------------------------------

        if (
            src
            == MissionState.ENTER_CORRIDOR
        ):

            if (
                obs.corridor_stable
                and front_clear
            ):

                return (
                    MissionState.ENTER_CORRIDOR,
                    "corridor geometry recovered during entry",
                )

            if (
                obs.obstacle_candidate
                is not None
            ):

                return (
                    MissionState.OBSTACLE_DECISION,
                    "front blockage dominates during entry recovery",
                )

            return (
                None,
                "entry path still unsafe",
            )

        # ----------------------------------------------------
        # CRUISE
        # ----------------------------------------------------

        if (
            src
            == MissionState.CORRIDOR_CRUISE
        ):

            if (
                obs.obstacle_candidate
                is not None
            ):

                return (
                    MissionState.OBSTACLE_DECISION,
                    "fresh scan confirms one-sided front obstacle",
                )

            if obs.exit_candidate:

                return (
                    MissionState.EXIT_DETECTION,
                    "fresh scan supports genuine corridor opening",
                )

            if (
                obs.corridor_stable
                and front_clear
                and not obs.front_blocked
            ):

                return (
                    MissionState.CORRIDOR_CRUISE,
                    "normal corridor structure stable again",
                )

            return (
                None,
                "cruise geometry remains ambiguous",
            )

        # ----------------------------------------------------
        # OBSTACLE_DECISION
        # ----------------------------------------------------

        if (
            src
            == MissionState.OBSTACLE_DECISION
        ):

            if (
                obs.obstacle_candidate
                is not None
            ):

                return (
                    obs.obstacle_candidate,
                    "one bypass side is now consistently safe",
                )

            if (
                obs.corridor_stable
                and front_clear
                and not obs.front_blocked
            ):

                return (
                    MissionState.CORRIDOR_CRUISE,
                    "front blockage cleared; resume centered cruise",
                )

            return (
                None,
                "no safe reliable bypass side yet",
            )

        # ----------------------------------------------------
        # AVOID_LEFT / AVOID_RIGHT
        # ----------------------------------------------------

        if src in (
            MissionState.AVOID_LEFT,
            MissionState.AVOID_RIGHT,
        ):

            # Never resume a stale PASS manoeuvre.
            #
            # Mission manager will re-enter obstacle controller
            # from OBSTACLE_DECISION and revalidate the side.
            if (
                obs.obstacle_candidate
                is not None
            ):

                return (
                    obs.obstacle_candidate,
                    "bypass geometry recovered; revalidate obstacle side",
                )

            if (
                obs.corridor_stable
                and front_clear
                and not obs.front_blocked
            ):

                return (
                    MissionState.CORRIDOR_CRUISE,
                    "obstacle no longer blocks path; corridor stable",
                )

            return (
                None,
                "bypass geometry still unsafe or ambiguous",
            )

        # ----------------------------------------------------
        # EXIT
        # ----------------------------------------------------

        if (
            src
            == MissionState.EXIT_DETECTION
        ):

            if obs.exit_candidate:

                return (
                    MissionState.EXIT_DETECTION,
                    "exit opening persists on fresh scans",
                )

            if (
                obs.corridor_stable
                and front_clear
            ):

                return (
                    MissionState.CORRIDOR_CRUISE,
                    "exit candidate rejected; corridor structure returned",
                )

            return (
                None,
                "exit geometry remains ambiguous",
            )

        return (
            None,
            (
                "unsupported recovery source "
                f"{src.value}"
            ),
        )

    # ========================================================
    # Transition validation
    # ========================================================

    def request_transition(
        self,
        target: MissionState,
        reason: str,
    ) -> None:

        if (
            self.state
            == ReassessState.TRANSITION_REQUESTED
        ):
            return

        allowed = {

            MissionState.PRE_ENTRY_GEOMETRY_LOCK,

            MissionState.ENTER_CORRIDOR,

            MissionState.CORRIDOR_CRUISE,

            MissionState.OBSTACLE_DECISION,

            MissionState.AVOID_LEFT,

            MissionState.AVOID_RIGHT,

            MissionState.EXIT_DETECTION,

            MissionState.ABORT_CORRIDOR,
        }

        # HOVER_AND_REASSESS is explicitly forbidden from
        # requesting itself.
        if (
            target
            == MissionState.HOVER_AND_REASSESS
            or target not in allowed
        ):

            target = (
                MissionState.ABORT_CORRIDOR
            )

            reason = (
                "invalid or recursive reassessment "
                "target; aborting corridor"
            )

        self.transition_target = target
        self.transition_reason = reason

        self.state = (
            ReassessState.TRANSITION_REQUESTED
        )

        self.latest_command = (
            BodyVelocity.stop()
        )

        print(
            f"[REASSESS] -> "
            f"{target.value}: "
            f"{reason}"
        )

    # ========================================================
    # Output / diagnostics
    # ========================================================

    def output(self) -> ControllerOutput:

        # Safety invariant repeated here deliberately.
        command = BodyVelocity.stop()

        self.latest_command = command

        return ControllerOutput(

            command=command,

            next_state=self.transition_target,

            status=self.state.name,

            reason=self.transition_reason,

            confidence=None,
        )

    def diagnostics(self) -> dict:

        obs = self.last_observation

        data = {

            "state":
                self.state.name,

            "source_state":
                (
                    self.source_state.value
                    if self.source_state
                    is not None
                    else None
                ),

            "pause_reason":
                self.pause_reason
                or None,

            "state_age_s":
                round(
                    self.state_age(),
                    3,
                ),

            "candidate_history":
                [
                    item.value
                    for item
                    in self.candidate_history
                ],

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

        if obs is None:
            return data

        data.update(
            {

                "front_clearance_m":
                    round(
                        obs.front_clearance_m,
                        3,
                    ),

                "sectors_m":
                    {
                        key:
                            round(value, 3)

                        for key, value
                        in obs.sectors.items()
                    },

                "corridor_stable":
                    obs.corridor_stable,

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

                "parallel_error_deg":
                    (
                        round(
                            math.degrees(
                                obs.parallel_error_rad
                            ),
                            2,
                        )
                        if math.isfinite(
                            obs.parallel_error_rad
                        )
                        else None
                    ),

                "front_blocked":
                    obs.front_blocked,

                "exit_candidate":
                    obs.exit_candidate,

                "obstacle_candidate":
                    (
                        obs.obstacle_candidate.value
                        if obs.obstacle_candidate
                        is not None
                        else None
                    ),

                "obstacle_side":
                    obs.obstacle_side,

                "open_gap_m":
                    (
                        round(
                            obs.open_gap_m,
                            3,
                        )
                        if obs.open_gap_m
                        is not None
                        else None
                    ),

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
            }
        )

        return data
