#!/usr/bin/env python3

"""
Dedicated EXIT_DETECTION state for SAE AeroTHON corridor navigation.

Purpose:
    This state ONLY handles physically leaving the corridor.

It is entered after CORRIDOR_CRUISE suspects that the corridor is ending.

Unlike CORRIDOR_CRUISE, this state deliberately does NOT require valid
left/right wall fits. Wall disappearance is expected here.

Sequence:

    IDLE
      |
      v
    PROBING
      |
      | front open + left open + right open
      | for several scans
      v
    CLEARING
      |
      | continue slowly forward for a short time
      v
    EXIT_CONFIRMED

Failure:
    obstacle / stale LiDAR / timeout
        -> HOVER_AND_REASSESS or OBSTACLE_DECISION

ROS FLU convention:
    +x forward
    +y left
    +z up
    +yaw CCW
"""

from __future__ import annotations

import json
import math
from enum import Enum, auto
from typing import Dict, Optional

import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String


# ======================================================================
# State
# ======================================================================


class ExitState(Enum):
    IDLE = auto()
    PROBING = auto()
    CLEARING = auto()
    EXIT_CONFIRMED = auto()
    TRANSITION_REQUESTED = auto()


# ======================================================================
# Node
# ======================================================================


class ExitDetection(Node):

    def __init__(self) -> None:

        super().__init__("exit_detection")

        # ==============================================================
        # Topics
        # ==============================================================

        self.declare_parameter(
            "scan_topic",
            "/scan",
        )

        self.declare_parameter(
            "enable_topic",
            "/corridor/exit_detection/enable",
        )

        self.declare_parameter(
            "cmd_topic",
            "/corridor/exit_detection/cmd_vel",
        )

        self.declare_parameter(
            "state_topic",
            "/corridor/exit_detection/state",
        )

        self.declare_parameter(
            "result_topic",
            "/corridor/exit_detection/result",
        )

        self.declare_parameter(
            "next_state_topic",
            "/corridor/exit_detection/next_state",
        )

        self.declare_parameter(
            "diagnostics_topic",
            "/corridor/exit_detection/diagnostics",
        )

        # ==============================================================
        # LiDAR orientation
        # ==============================================================

        self.declare_parameter(
            "invert_scan_angle",
            False,
        )

        self.declare_parameter(
            "lidar_yaw_offset_deg",
            0.0,
        )

        # ==============================================================
        # Exit geometry
        # ==============================================================

        # Width of side cone around +/-90 degrees.
        self.declare_parameter(
            "side_cone_deg",
            7.0,
        )

        # Front safety / opening cone.
        self.declare_parameter(
            "front_cone_deg",
            18.0,
        )

        # Ignore returns farther than this for side classification.
        self.declare_parameter(
            "sector_max_range_m",
            6.0,
        )

        # Side is considered open if median range is at least this.
        self.declare_parameter(
            "side_open_threshold_m",
            2.40,
        )

        # Forward path must also be open.
        self.declare_parameter(
            "front_open_threshold_m",
            3.00,
        )

        # Require several consecutive scans before declaring an exit.
        self.declare_parameter(
            "exit_confirm_scans",
            3,
        )

        # ==============================================================
        # Motion
        # ==============================================================

        # Slow controlled forward movement while looking for the exit.
        self.declare_parameter(
            "probe_speed_m_s",
            0.15,
        )

        # Continue forward after exit geometry has been confirmed so that
        # the aircraft clears the ends of the corridor walls.
        self.declare_parameter(
            "clear_speed_m_s",
            0.15,
        )

        self.declare_parameter(
            "clear_time_s",
            1.5,
        )

        # Maximum time we are willing to probe before reassessing.
        self.declare_parameter(
            "probe_timeout_s",
            10.0,
        )

        # ==============================================================
        # Safety
        # ==============================================================

        self.declare_parameter(
            "front_obstacle_trigger_m",
            1.35,
        )

        self.declare_parameter(
            "front_emergency_stop_m",
            0.75,
        )

        self.declare_parameter(
            "obstacle_confirm_scans",
            2,
        )

        self.declare_parameter(
            "scan_stale_s",
            0.30,
        )

        # ==============================================================
        # ROS
        # ==============================================================

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

        self.cmd_pub = self.create_publisher(
            TwistStamped,
            str(self.get_parameter("cmd_topic").value),
            10,
        )

        self.state_pub = self.create_publisher(
            String,
            str(self.get_parameter("state_topic").value),
            10,
        )

        self.result_pub = self.create_publisher(
            String,
            str(self.get_parameter("result_topic").value),
            10,
        )

        self.next_state_pub = self.create_publisher(
            String,
            str(self.get_parameter("next_state_topic").value),
            10,
        )

        self.diagnostics_pub = self.create_publisher(
            String,
            str(self.get_parameter("diagnostics_topic").value),
            10,
        )

        # 20 Hz command / status publication.
        self.create_timer(
            0.05,
            self.publish_outputs,
        )

        # ==============================================================
        # Runtime state
        # ==============================================================

        self.enabled = False

        self.state = ExitState.IDLE

        self.latest_command = self.zero_command()

        self.last_scan_time = None

        self.session_start_time = None

        self.clear_start_time = None

        self.exit_streak = 0

        self.obstacle_streak = 0

        self.transition_target: Optional[str] = None
        self.transition_reason: Optional[str] = None

        # Latest diagnostic values.
        self.front_clearance = 0.0
        self.left_range: Optional[float] = None
        self.right_range: Optional[float] = None

        self.left_open = False
        self.right_open = False
        self.front_open = False

        self.get_logger().info(
            "EXIT_DETECTION node started"
        )

    # ==================================================================
    # Lifecycle
    # ==================================================================

    def reset_session(self) -> None:

        self.state = ExitState.PROBING

        self.latest_command = self.zero_command()

        self.last_scan_time = None

        self.session_start_time = self.get_clock().now()

        self.clear_start_time = None

        self.exit_streak = 0

        self.obstacle_streak = 0

        self.transition_target = None
        self.transition_reason = None

        self.front_clearance = 0.0

        self.left_range = None
        self.right_range = None

        self.left_open = False
        self.right_open = False
        self.front_open = False

    def enable_callback(self, msg: Bool) -> None:

        if msg.data and not self.enabled:

            self.reset_session()

            self.enabled = True

            self.get_logger().info(
                "EXIT_DETECTION enabled -> PROBING"
            )

        elif not msg.data and self.enabled:

            self.enabled = False

            self.state = ExitState.IDLE

            self.latest_command = self.zero_command()

            self.transition_target = None
            self.transition_reason = None

            self.get_logger().info(
                "EXIT_DETECTION disabled"
            )

    # ==================================================================
    # Timing
    # ==================================================================

    def session_age(self) -> float:

        if self.session_start_time is None:
            return 0.0

        return (
            self.get_clock().now()
            - self.session_start_time
        ).nanoseconds * 1e-9

    def clearing_age(self) -> float:

        if self.clear_start_time is None:
            return 0.0

        return (
            self.get_clock().now()
            - self.clear_start_time
        ).nanoseconds * 1e-9

    # ==================================================================
    # Commands
    # ==================================================================

    def make_command(
        self,
        vx: float = 0.0,
        vy_left: float = 0.0,
        yaw_rate: float = 0.0,
    ) -> TwistStamped:

        msg = TwistStamped()

        msg.header.frame_id = "base_link"

        msg.twist.linear.x = float(vx)
        msg.twist.linear.y = float(vy_left)

        msg.twist.angular.z = float(yaw_rate)

        return msg

    def zero_command(self) -> TwistStamped:
        return self.make_command()

    # ==================================================================
    # Scan conversion
    # ==================================================================

    def corrected_scan_arrays(
        self,
        scan: LaserScan,
    ) -> tuple[np.ndarray, np.ndarray]:

        ranges = np.asarray(
            scan.ranges,
            dtype=np.float64,
        )

        count = ranges.size

        angles = (
            scan.angle_min
            + np.arange(
                count,
                dtype=np.float64,
            )
            * scan.angle_increment
        )

        angles = np.arctan2(
            np.sin(angles),
            np.cos(angles),
        )

        if bool(
            self.get_parameter(
                "invert_scan_angle"
            ).value
        ):
            angles = -angles

        angles += math.radians(
            float(
                self.get_parameter(
                    "lidar_yaw_offset_deg"
                ).value
            )
        )

        angles = np.arctan2(
            np.sin(angles),
            np.cos(angles),
        )

        return ranges, angles

    # ==================================================================
    # Sector measurement
    # ==================================================================

    def sector_range(
        self,
        ranges: np.ndarray,
        angles: np.ndarray,
        target_deg: float,
        cone_deg: float,
        range_min: float,
        range_max: float,
    ) -> Optional[float]:

        target = math.radians(target_deg)

        half_cone = math.radians(cone_deg)

        delta = np.abs(
            np.arctan2(
                np.sin(angles - target),
                np.cos(angles - target),
            )
        )

        valid = np.isfinite(ranges)

        valid &= ranges >= max(
            float(range_min),
            0.05,
        )

        valid &= ranges <= float(range_max)

        vals = ranges[
            valid
            & (delta <= half_cone)
        ]

        if vals.size == 0:
            return None

        return float(
            np.percentile(
                vals,
                50.0,
            )
        )

    def front_range(
        self,
        ranges: np.ndarray,
        angles: np.ndarray,
        scan: LaserScan,
    ) -> float:

        safe_ranges = ranges.copy()

        # +inf means no return, therefore free up to sensor range.
        safe_ranges[
            np.isposinf(safe_ranges)
        ] = float(scan.range_max)

        valid = np.isfinite(
            safe_ranges
        )

        valid &= safe_ranges >= max(
            float(scan.range_min),
            0.05,
        )

        valid &= safe_ranges <= float(
            scan.range_max
        )

        half_cone = math.radians(
            float(
                self.get_parameter(
                    "front_cone_deg"
                ).value
            )
        )

        front_mask = (
            valid
            & (np.abs(angles) <= half_cone)
        )

        vals = safe_ranges[
            front_mask
        ]

        if vals.size == 0:
            return 0.0

        # Conservative front clearance.
        return float(
            np.percentile(
                vals,
                10.0,
            )
        )

    # ==================================================================
    # Scan callback / EXIT FSM
    # ==================================================================

    def scan_callback(
        self,
        scan: LaserScan,
    ) -> None:

        self.last_scan_time = (
            self.get_clock().now()
        )

        if not self.enabled:
            return

        if self.state not in (
            ExitState.PROBING,
            ExitState.CLEARING,
        ):
            return

        ranges, angles = (
            self.corrected_scan_arrays(
                scan
            )
        )

        # --------------------------------------------------------------
        # Front
        # --------------------------------------------------------------

        self.front_clearance = (
            self.front_range(
                ranges,
                angles,
                scan,
            )
        )

        # --------------------------------------------------------------
        # Side sectors
        # --------------------------------------------------------------

        side_cone = float(
            self.get_parameter(
                "side_cone_deg"
            ).value
        )

        max_sector = min(
            float(scan.range_max),
            float(
                self.get_parameter(
                    "sector_max_range_m"
                ).value
            ),
        )

        self.left_range = (
            self.sector_range(
                ranges,
                angles,
                90.0,
                side_cone,
                scan.range_min,
                max_sector,
            )
        )

        self.right_range = (
            self.sector_range(
                ranges,
                angles,
                -90.0,
                side_cone,
                scan.range_min,
                max_sector,
            )
        )

        side_threshold = float(
            self.get_parameter(
                "side_open_threshold_m"
            ).value
        )

        # No return in the side cone means open.
        self.left_open = (
            self.left_range is None
            or self.left_range
            >= side_threshold
        )

        self.right_open = (
            self.right_range is None
            or self.right_range
            >= side_threshold
        )

        self.front_open = (
            self.front_clearance
            >= float(
                self.get_parameter(
                    "front_open_threshold_m"
                ).value
            )
        )

        # --------------------------------------------------------------
        # Safety: emergency obstacle
        # --------------------------------------------------------------

        emergency = float(
            self.get_parameter(
                "front_emergency_stop_m"
            ).value
        )

        obstacle_trigger = float(
            self.get_parameter(
                "front_obstacle_trigger_m"
            ).value
        )

        if (
            0.0
            < self.front_clearance
            <= emergency
        ):

            self.request_transition(
                "OBSTACLE_DECISION",
                (
                    "emergency obstacle during "
                    f"exit probe at "
                    f"{self.front_clearance:.2f} m"
                ),
            )

            return

        if (
            0.0
            < self.front_clearance
            <= obstacle_trigger
        ):

            self.obstacle_streak += 1

            self.latest_command = (
                self.zero_command()
            )

            if (
                self.obstacle_streak
                >= int(
                    self.get_parameter(
                        "obstacle_confirm_scans"
                    ).value
                )
            ):

                self.request_transition(
                    "OBSTACLE_DECISION",
                    (
                        "obstacle confirmed "
                        "during corridor exit"
                    ),
                )

            return

        self.obstacle_streak = 0

        # ==============================================================
        # PROBING
        # ==============================================================

        if self.state == ExitState.PROBING:

            # IMPORTANT:
            #
            # Unlike CORRIDOR_CRUISE, wall disappearance does NOT stop
            # this state.
            #
            # We intentionally continue slowly forward because the whole
            # purpose of this state is to pass beyond the wall endpoints.

            self.latest_command = (
                self.make_command(
                    vx=float(
                        self.get_parameter(
                            "probe_speed_m_s"
                        ).value
                    )
                )
            )

            strong_exit = (
                self.front_open
                and self.left_open
                and self.right_open
            )

            if strong_exit:

                self.exit_streak += 1

            else:

                self.exit_streak = 0

            if (
                self.exit_streak
                >= int(
                    self.get_parameter(
                        "exit_confirm_scans"
                    ).value
                )
            ):

                self.state = (
                    ExitState.CLEARING
                )

                self.clear_start_time = (
                    self.get_clock().now()
                )

                self.get_logger().info(
                    "EXIT opening confirmed "
                    "-> CLEARING corridor walls"
                )

                return

            # ----------------------------------------------------------
            # Probe timeout
            # ----------------------------------------------------------

            timeout = float(
                self.get_parameter(
                    "probe_timeout_s"
                ).value
            )

            if self.session_age() > timeout:

                self.request_transition(
                    "HOVER_AND_REASSESS",
                    (
                        "EXIT_DETECTION probe "
                        "timed out without "
                        "confirming corridor exit"
                    ),
                )

                return

            return

        # ==============================================================
        # CLEARING
        # ==============================================================

        if self.state == ExitState.CLEARING:

            # Continue forward briefly so the aircraft does not stop
            # exactly at the wall endpoints.

            self.latest_command = (
                self.make_command(
                    vx=float(
                        self.get_parameter(
                            "clear_speed_m_s"
                        ).value
                    )
                )
            )

            if (
                self.clearing_age()
                >= float(
                    self.get_parameter(
                        "clear_time_s"
                    ).value
                )
            ):

                self.state = (
                    ExitState.EXIT_CONFIRMED
                )

                self.latest_command = (
                    self.zero_command()
                )

                self.get_logger().info(
                    "CORRIDOR EXIT CONFIRMED"
                )

    # ==================================================================
    # External transition
    # ==================================================================

    def request_transition(
        self,
        target: str,
        reason: str,
    ) -> None:

        if (
            self.state
            == ExitState.TRANSITION_REQUESTED
        ):
            return

        self.transition_target = target
        self.transition_reason = reason

        self.state = (
            ExitState.TRANSITION_REQUESTED
        )

        self.latest_command = (
            self.zero_command()
        )

        self.get_logger().warning(
            f"EXIT_DETECTION -> "
            f"{target}: {reason}"
        )

    # ==================================================================
    # Diagnostics
    # ==================================================================

    def diagnostics_dict(
        self,
    ) -> Dict[str, object]:

        return {

            "state": self.state.name,

            "enabled": self.enabled,

            "session_age_s": round(
                self.session_age(),
                3,
            ),

            "front_clearance_m": round(
                self.front_clearance,
                3,
            ),

            "left_range_m": (
                round(
                    self.left_range,
                    3,
                )
                if self.left_range
                is not None
                else None
            ),

            "right_range_m": (
                round(
                    self.right_range,
                    3,
                )
                if self.right_range
                is not None
                else None
            ),

            "front_open": self.front_open,

            "left_open": self.left_open,

            "right_open": self.right_open,

            "exit_streak": self.exit_streak,

            "transition_target":
                self.transition_target,

            "transition_reason":
                self.transition_reason,

            "command_vx": round(
                float(
                    self.latest_command
                    .twist
                    .linear
                    .x
                ),
                3,
            ),
        }

    # ==================================================================
    # Publisher / watchdog
    # ==================================================================

    def publish_outputs(
        self,
    ) -> None:

        now = self.get_clock().now()

        # --------------------------------------------------------------
        # LiDAR watchdog
        # --------------------------------------------------------------

        if (
            self.enabled
            and self.state
            in (
                ExitState.PROBING,
                ExitState.CLEARING,
            )
        ):

            stale_limit = float(
                self.get_parameter(
                    "scan_stale_s"
                ).value
            )

            if self.last_scan_time is None:

                scan_stale = (
                    self.session_age()
                    > stale_limit
                )

            else:

                age = (
                    now
                    - self.last_scan_time
                ).nanoseconds * 1e-9

                scan_stale = (
                    age > stale_limit
                )

            if scan_stale:

                self.request_transition(
                    "HOVER_AND_REASSESS",
                    "LaserScan stale",
                )

        # --------------------------------------------------------------
        # Command
        # --------------------------------------------------------------

        self.latest_command.header.stamp = (
            now.to_msg()
        )

        self.cmd_pub.publish(
            self.latest_command
        )

        # --------------------------------------------------------------
        # State
        # --------------------------------------------------------------

        state_msg = String()

        state_msg.data = (
            self.state.name
        )

        self.state_pub.publish(
            state_msg
        )

        # --------------------------------------------------------------
        # Result / next state
        # --------------------------------------------------------------

        result_msg = String()
        next_msg = String()

        if (
            self.state
            == ExitState.EXIT_CONFIRMED
        ):

            result_msg.data = (
                "EXIT_CONFIRMED"
            )

            next_msg.data = (
                "CORRIDOR_EXITED"
            )

        elif (
            self.state
            == ExitState.TRANSITION_REQUESTED
        ):

            result_msg.data = (
                f"TRANSITION:"
                f"{self.transition_target}:"
                f"{self.transition_reason}"
            )

            next_msg.data = (
                self.transition_target
                or ""
            )

        elif self.enabled:

            result_msg.data = (
                f"ACTIVE:{self.state.name}"
            )

            next_msg.data = ""

        else:

            result_msg.data = "IDLE"
            next_msg.data = ""

        self.result_pub.publish(
            result_msg
        )

        self.next_state_pub.publish(
            next_msg
        )

        # --------------------------------------------------------------
        # Diagnostics
        # --------------------------------------------------------------

        diag = String()

        diag.data = json.dumps(
            self.diagnostics_dict(),
            separators=(",", ":"),
        )

        self.diagnostics_pub.publish(
            diag
        )


# ======================================================================
# Main
# ======================================================================


def main(args=None) -> None:

    rclpy.init(args=args)

    node = ExitDetection()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
