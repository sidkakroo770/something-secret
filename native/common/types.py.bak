#!/usr/bin/env python3

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


# ============================================================
# Sensor data
# ============================================================

@dataclass
class NativeScan:
    """
    One corrected 2-D LiDAR revolution in aircraft body coordinates.

    Coordinate convention:

        +x = forward
        +y = left

    Angles:

          +90° = left
            0° = forward
          -90° = right
         ±180° = rear
    """

    angles_rad: np.ndarray
    ranges_m: np.ndarray
    intensities: np.ndarray

    timestamp: float

    range_min_m: float = 0.02
    range_max_m: float = 12.0

    @property
    def age_s(self) -> float:
        return max(0.0, time.monotonic() - self.timestamp)

    @property
    def size(self) -> int:
        return int(self.ranges_m.size)


@dataclass
class Attitude:
    """
    Vehicle attitude received from the flight controller.
    """

    roll_rad: float = 0.0
    pitch_rad: float = 0.0
    yaw_rad: float = 0.0

    timestamp: float = 0.0

    @property
    def age_s(self) -> float:
        if self.timestamp <= 0:
            return float("inf")

        return max(
            0.0,
            time.monotonic() - self.timestamp,
        )


# ============================================================
# Motion command
# ============================================================

@dataclass
class BodyVelocity:
    """
    Velocity command using the SAME FLU convention as the old ROS FSM.

        +vx = forward
        +vy = left
        +vz = up

        +yaw_rate = counter-clockwise viewed from above

    MAVLink conversion happens later in mavlink_io.py.
    """

    vx_m_s: float = 0.0
    vy_m_s: float = 0.0
    vz_m_s: float = 0.0

    yaw_rate_rad_s: float = 0.0

    @classmethod
    def stop(cls) -> "BodyVelocity":
        return cls()


# ============================================================
# FSM interface
# ============================================================

class MissionState(str, Enum):

    PRE_ENTRY_GEOMETRY_LOCK = "PRE_ENTRY_GEOMETRY_LOCK"

    ENTER_CORRIDOR = "ENTER_CORRIDOR"

    CORRIDOR_CRUISE = "CORRIDOR_CRUISE"

    OBSTACLE_DECISION = "OBSTACLE_DECISION"

    AVOID_LEFT = "AVOID_LEFT"

    AVOID_RIGHT = "AVOID_RIGHT"

    EXIT_DETECTION = "EXIT_DETECTION"

    HOVER_AND_REASSESS = "HOVER_AND_REASSESS"

    CORRIDOR_EXITED = "CORRIDOR_EXITED"

    ABORT_CORRIDOR = "ABORT_CORRIDOR"


@dataclass
class ControllerOutput:
    """
    Every native FSM controller returns one of these.

    There is no publishing and no ROS topic involved.
    """

    command: BodyVelocity

    next_state: Optional[MissionState] = None

    status: str = ""

    reason: str = ""

    confidence: Optional[float] = None


@dataclass
class ControllerContext:
    """
    Sensor information available during one mission iteration.
    """

    scan: NativeScan

    attitude: Optional[Attitude] = None
