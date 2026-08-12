#!/usr/bin/env python3
"""EXIT_DETECTION V2 -- committed corridor-exit traversal.

This state is intentionally simple.

CORRIDOR_CRUISE has already done the actual exit-candidate detection and only
hands control to this node after a persistent opening has been observed.  Once
enabled, EXIT_DETECTION therefore commits straight through the corridor mouth
instead of hovering and attempting to rediscover the same geometry.

FSM contract
------------
    CORRIDOR_CRUISE
          -> EXIT_DETECTION
          -> CORRIDOR_EXITED

Exceptional path:
    EXIT_DETECTION -> HOVER_AND_REASSESS

Normal behaviour
----------------
* command body-frame +x at a slow constant velocity
* no lateral motion and no yaw command
* continue for a bounded commanded travel distance
* then publish CORRIDOR_EXITED

The completion distance is implemented as speed x elapsed active time.  It is
therefore a commanded-distance estimate, not odometry.  This is deliberate for
our current Gazebo bridge, where body-frame TwistStamped velocity is applied
directly to the test model.  The simulator should not be paused during this
state.

A fresh LaserScan is used only as an optional emergency front-stop gate.  Side
walls, wall confidence and exit geometry are NOT required once this state is
enabled; losing them is expected while leaving the corridor.

ROS FLU body convention:
    +x forward, +y left, +z up, +yaw counter-clockwise.
"""

from __future__ import annotations

import json
import math
import time
from enum import Enum, auto
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String


class ExitState(Enum):
    IDLE = auto()
    COMMIT_FORWARD = auto()
    COMPLETE = auto()
    HOLD = auto()


class ExitDetectionV2(Node):
    def __init__(self) -> None:
        super().__init__("exit_detection_v2")

        # --------------------------------------------------------------
        # ROS interface -- matches corridor_mission_bridge_v4 defaults.
        # --------------------------------------------------------------
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("enable_topic", "/corridor/exit/enable")
        self.declare_parameter("cmd_topic", "/corridor/exit/cmd_vel")
        self.declare_parameter("state_topic", "/corridor/exit/state")
        self.declare_parameter("result_topic", "/corridor/exit/result")
        self.declare_parameter("next_state_topic", "/corridor/exit/next_state")
        self.declare_parameter("diagnostics_topic", "/corridor/exit/diagnostics")

        # --------------------------------------------------------------
        # Exit commit behaviour.
        # --------------------------------------------------------------
        self.declare_parameter("exit_forward_speed_m_s", 0.15)
        self.declare_parameter("exit_commit_distance_m", 1.20)

        # --------------------------------------------------------------
        # Optional emergency front safety only.
        # A stale / missing scan does NOT block normal exit completion,
        # because CRUISE already confirmed the exit before this state starts.
        # --------------------------------------------------------------
        self.declare_parameter("use_front_safety", True)
        self.declare_parameter("front_cone_deg", 18.0)
        self.declare_parameter("front_stop_m", 0.60)
        self.declare_parameter("front_stop_confirm_scans", 2)
        self.declare_parameter("scan_fresh_for_safety_s", 0.50)
        self.declare_parameter("invert_scan_angle", False)
        self.declare_parameter("lidar_yaw_offset_deg", 0.0)

        # --------------------------------------------------------------
        # ROS wiring.
        # --------------------------------------------------------------
        self.create_subscription(
            LaserScan,
            str(self.get_parameter("scan_topic").value),
            self.scan_callback,
            10,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("enable_topic").value),
            self.enable_callback,
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

        # --------------------------------------------------------------
        # Runtime state.
        # --------------------------------------------------------------
        self.enabled = False
        self.state = ExitState.IDLE
        self.commit_start_monotonic: Optional[float] = None
        self.transition_target = ""
        self.transition_reason = ""

        self.last_scan_monotonic: Optional[float] = None
        self.front_clearance_m: Optional[float] = None
        self.front_stop_streak = 0

        # Publish command/status at 20 Hz, matching the bridge update rate.
        self.create_timer(0.05, self.timer_callback)

        self.get_logger().info(
            "EXIT_DETECTION V2 started: commit-forward exit behaviour ready; initially IDLE"
        )

    # ==================================================================
    # Lifecycle
    # ==================================================================

    def enable_callback(self, msg: Bool) -> None:
        if msg.data and not self.enabled:
            self.enabled = True
            self.state = ExitState.COMMIT_FORWARD
            self.commit_start_monotonic = time.monotonic()
            self.transition_target = ""
            self.transition_reason = ""
            self.front_stop_streak = 0

            speed = self.forward_speed()
            distance = self.commit_distance()
            duration = distance / speed
            self.get_logger().warning(
                "EXIT_DETECTION enabled -> COMMIT_FORWARD: "
                f"vx={speed:.2f} m/s, commanded distance={distance:.2f} m, "
                f"nominal duration={duration:.2f} s"
            )

        elif not msg.data and self.enabled:
            self.enabled = False
            self.state = ExitState.IDLE
            self.commit_start_monotonic = None
            self.transition_target = ""
            self.transition_reason = ""
            self.front_stop_streak = 0
            self.get_logger().info("EXIT_DETECTION disabled -> IDLE")

    # ==================================================================
    # LaserScan -- emergency front gate only
    # ==================================================================

    def scan_callback(self, scan: LaserScan) -> None:
        self.last_scan_monotonic = time.monotonic()
        self.front_clearance_m = self.extract_front_clearance(scan)

        if not self.enabled or self.state != ExitState.COMMIT_FORWARD:
            self.front_stop_streak = 0
            return

        if not bool(self.get_parameter("use_front_safety").value):
            self.front_stop_streak = 0
            return

        if self.front_clearance_m is None:
            self.front_stop_streak = 0
            return

        stop_m = float(self.get_parameter("front_stop_m").value)
        if 0.0 < self.front_clearance_m <= stop_m:
            self.front_stop_streak += 1
            if self.front_stop_streak >= max(
                1, int(self.get_parameter("front_stop_confirm_scans").value)
            ):
                self.request_transition(
                    "HOVER_AND_REASSESS",
                    f"unexpected front obstruction during exit commit: "
                    f"{self.front_clearance_m:.2f} m",
                )
        else:
            self.front_stop_streak = 0

    def extract_front_clearance(self, scan: LaserScan) -> Optional[float]:
        ranges = np.asarray(scan.ranges, dtype=np.float64)
        if ranges.size == 0:
            return None

        angles = (
            float(scan.angle_min)
            + np.arange(ranges.size, dtype=np.float64) * float(scan.angle_increment)
        )
        angles = np.arctan2(np.sin(angles), np.cos(angles))

        if bool(self.get_parameter("invert_scan_angle").value):
            angles = -angles

        angles += math.radians(float(self.get_parameter("lidar_yaw_offset_deg").value))
        angles = np.arctan2(np.sin(angles), np.cos(angles))

        # Treat +inf as clear to the scanner's maximum range.
        safe_ranges = ranges.copy()
        safe_ranges[np.isposinf(safe_ranges)] = float(scan.range_max)

        valid = np.isfinite(safe_ranges)
        valid &= safe_ranges >= max(float(scan.range_min), 0.05)
        valid &= safe_ranges <= float(scan.range_max)

        half = math.radians(float(self.get_parameter("front_cone_deg").value))
        vals = safe_ranges[valid & (np.abs(angles) <= half)]
        if vals.size == 0:
            return None

        # Conservative but spike-resistant front estimate.
        return float(np.percentile(vals, 10.0))

    # ==================================================================
    # Exit commit logic
    # ==================================================================

    def forward_speed(self) -> float:
        # Prevent zero / negative values from creating an infinite state.
        return max(0.05, float(self.get_parameter("exit_forward_speed_m_s").value))

    def commit_distance(self) -> float:
        return max(0.20, float(self.get_parameter("exit_commit_distance_m").value))

    def elapsed_s(self) -> float:
        if self.commit_start_monotonic is None:
            return 0.0
        return max(0.0, time.monotonic() - self.commit_start_monotonic)

    def estimated_travel_m(self) -> float:
        return self.forward_speed() * self.elapsed_s()

    def request_transition(self, target: str, reason: str) -> None:
        if self.transition_target:
            return
        self.transition_target = str(target).strip().upper()
        self.transition_reason = str(reason)
        self.state = ExitState.COMPLETE if self.transition_target == "CORRIDOR_EXITED" else ExitState.HOLD

        if self.transition_target == "CORRIDOR_EXITED":
            self.get_logger().warning(
                f"EXIT_DETECTION -> CORRIDOR_EXITED: {self.transition_reason}"
            )
        else:
            self.get_logger().error(
                f"EXIT_DETECTION -> {self.transition_target}: {self.transition_reason}"
            )

    def timer_callback(self) -> None:
        now = self.get_clock().now()
        cmd = TwistStamped()
        cmd.header.stamp = now.to_msg()
        cmd.header.frame_id = "base_link"

        if self.enabled and self.state == ExitState.COMMIT_FORWARD and not self.transition_target:
            # Critical design choice: no side-wall / confidence gate here.
            # Disappearing side geometry is expected once the vehicle leaves.
            cmd.twist.linear.x = self.forward_speed()

            if self.estimated_travel_m() >= self.commit_distance():
                self.request_transition(
                    "CORRIDOR_EXITED",
                    f"committed forward {self.estimated_travel_m():.2f} m "
                    f"(target {self.commit_distance():.2f} m)",
                )
                # Stop on the same timer tick that completion is latched.
                cmd.twist.linear.x = 0.0

        # IDLE, COMPLETE and HOLD all publish zero commands.
        self.cmd_pub.publish(cmd)
        self.publish_status()

    # ==================================================================
    # Status
    # ==================================================================

    def scan_age_s(self) -> Optional[float]:
        if self.last_scan_monotonic is None:
            return None
        return max(0.0, time.monotonic() - self.last_scan_monotonic)

    def fresh_front_safety_available(self) -> bool:
        age = self.scan_age_s()
        if age is None:
            return False
        return age <= float(self.get_parameter("scan_fresh_for_safety_s").value)

    def publish_status(self) -> None:
        state_msg = String()
        if self.state == ExitState.IDLE:
            state_msg.data = "IDLE"
        elif self.state == ExitState.COMMIT_FORWARD:
            state_msg.data = "EXIT_DETECTION"
        elif self.state == ExitState.COMPLETE:
            state_msg.data = "CORRIDOR_EXITED"
        else:
            state_msg.data = "HOLD"
        self.state_pub.publish(state_msg)

        result_msg = String()
        next_msg = String()

        if self.transition_target == "CORRIDOR_EXITED":
            # Either field is enough for Bridge V4; publishing both makes the
            # interface explicit and robust to message ordering.
            result_msg.data = "EXIT_CONFIRMED"
            next_msg.data = "CORRIDOR_EXITED"
        elif self.transition_target:
            result_msg.data = f"TRANSITION:{self.transition_target}:{self.transition_reason}"
            next_msg.data = self.transition_target
        elif self.enabled:
            result_msg.data = "IN_PROGRESS"
            next_msg.data = ""
        else:
            result_msg.data = "IDLE"
            next_msg.data = ""

        self.result_pub.publish(result_msg)
        self.next_state_pub.publish(next_msg)

        diag = {
            "state": state_msg.data,
            "enabled": self.enabled,
            "exit_forward_speed_m_s": round(self.forward_speed(), 3),
            "exit_commit_distance_m": round(self.commit_distance(), 3),
            "elapsed_s": round(self.elapsed_s(), 3),
            "estimated_commanded_travel_m": round(self.estimated_travel_m(), 3),
            "front_clearance_m": (
                round(self.front_clearance_m, 3)
                if self.front_clearance_m is not None
                else None
            ),
            "front_safety_scan_fresh": self.fresh_front_safety_available(),
            "front_stop_streak": self.front_stop_streak,
            "transition_target": self.transition_target or None,
            "transition_reason": self.transition_reason or None,
        }
        diag_msg = String()
        diag_msg.data = json.dumps(diag, separators=(",", ":"))
        self.diag_pub.publish(diag_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ExitDetectionV2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
