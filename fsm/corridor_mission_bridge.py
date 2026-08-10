#!/usr/bin/env python3

import math
import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TwistStamped
from gazebo_msgs.srv import GetEntityState, SetEntityState
from std_msgs.msg import Bool, String


def quaternion_to_yaw(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class CorridorMissionBridge(Node):

    def __init__(self):
        super().__init__("corridor_mission_bridge")

        self.model_name = "corridor_drone"

        # ============================================================
        # GLOBAL CORRIDOR FSM
        #
        # PRE_ENTRY
        #    ↓
        # ENTER_CORRIDOR
        #    ↓
        # CRUISE
        #    ↓
        # EXIT_DETECTION
        #    ↓
        # CORRIDOR_EXITED
        #
        # Any unsupported failure transition -> HOLD
        # ============================================================

        self.phase = "PRE_ENTRY"

        # Latest controller commands.
        self.pre_cmd = TwistStamped()
        self.cruise_cmd = TwistStamped()
        self.exit_cmd = TwistStamped()

        self.pre_cmd_time = 0.0
        self.cruise_cmd_time = 0.0
        self.exit_cmd_time = 0.0

        # ============================================================
        # Controller command subscriptions
        # ============================================================

        self.create_subscription(
            TwistStamped,
            "/corridor/pre_entry/cmd_vel",
            self.pre_cmd_callback,
            10,
        )

        self.create_subscription(
            TwistStamped,
            "/corridor/cruise/cmd_vel",
            self.cruise_cmd_callback,
            10,
        )

        self.create_subscription(
            TwistStamped,
            "/corridor/exit_detection/cmd_vel",
            self.exit_cmd_callback,
            10,
        )

        # ============================================================
        # State transition subscriptions
        # ============================================================

        # PRE_ENTRY completion
        self.create_subscription(
            Bool,
            "/corridor/pre_entry/locked",
            self.locked_callback,
            10,
        )

        # CORRIDOR_CRUISE requested next global state
        self.create_subscription(
            String,
            "/corridor/cruise/next_state",
            self.cruise_next_state_callback,
            10,
        )

        # EXIT_DETECTION requested next global state
        self.create_subscription(
            String,
            "/corridor/exit_detection/next_state",
            self.exit_next_state_callback,
            10,
        )

        # ============================================================
        # Controller enable publishers
        # ============================================================

        self.pre_enable_pub = self.create_publisher(
            Bool,
            "/corridor/pre_entry/enable",
            10,
        )

        self.cruise_enable_pub = self.create_publisher(
            Bool,
            "/corridor/cruise/enable",
            10,
        )

        self.exit_enable_pub = self.create_publisher(
            Bool,
            "/corridor/exit_detection/enable",
            10,
        )

        # Manager state for debugging.
        self.mission_state_pub = self.create_publisher(
            String,
            "/corridor/mission/state",
            10,
        )

        # ============================================================
        # Gazebo services
        # ============================================================

        self.get_state_client = self.create_client(
            GetEntityState,
            "/gazebo/get_entity_state",
        )

        self.set_state_client = self.create_client(
            SetEntityState,
            "/gazebo/set_entity_state",
        )

        self.get_logger().info(
            "Waiting for Gazebo state services..."
        )

        self.get_state_client.wait_for_service()
        self.set_state_client.wait_for_service()

        self.get_logger().info(
            "Gazebo connected"
        )

        self.request_in_progress = False

        # Model velocity update at 20 Hz.
        self.create_timer(
            0.05,
            self.update_motion,
        )

        # Re-publish enable and FSM state.
        self.create_timer(
            0.5,
            self.publish_enable_state,
        )

        self.publish_enable_state()

        self.get_logger().info(
            "MISSION START -> PRE_ENTRY"
        )

    # ================================================================
    # COMMAND CALLBACKS
    # ================================================================

    def pre_cmd_callback(self, msg):
        self.pre_cmd = msg
        self.pre_cmd_time = time.monotonic()

    def cruise_cmd_callback(self, msg):
        self.cruise_cmd = msg
        self.cruise_cmd_time = time.monotonic()

    def exit_cmd_callback(self, msg):
        self.exit_cmd = msg
        self.exit_cmd_time = time.monotonic()

    # ================================================================
    # GLOBAL FSM TRANSITIONS
    # ================================================================

    def locked_callback(self, msg):

        if (
            msg.data
            and self.phase == "PRE_ENTRY"
        ):

            self.phase = "ENTER_CORRIDOR"

            self.get_logger().info(
                "PRE_ENTRY -> ENTER_CORRIDOR"
            )

            self.publish_enable_state()

    def cruise_next_state_callback(self, msg):

        if self.phase != "CRUISE":
            return

        target = msg.data.strip()

        if not target:
            return

        # ------------------------------------------------------------
        # Normal corridor-end handoff
        # ------------------------------------------------------------

        if target == "EXIT_DETECTION":

            self.phase = "EXIT_DETECTION"

            self.get_logger().info(
                "CORRIDOR_CRUISE -> EXIT_DETECTION"
            )

            self.publish_enable_state()
            return

        # ------------------------------------------------------------
        # Failure states not yet implemented by this test manager
        # ------------------------------------------------------------

        if target in (
            "HOVER_AND_REASSESS",
            "OBSTACLE_DECISION",
            "ABORT_CORRIDOR",
        ):

            self.phase = "HOLD"

            self.get_logger().warning(
                f"CORRIDOR_CRUISE requested {target} "
                "-> HOLD for current Gazebo test"
            )

            self.publish_enable_state()

    def exit_next_state_callback(self, msg):

        if self.phase != "EXIT_DETECTION":
            return

        target = msg.data.strip()

        if not target:
            return

        # ------------------------------------------------------------
        # Successful corridor exit
        # ------------------------------------------------------------

        if target == "CORRIDOR_EXITED":

            self.phase = "CORRIDOR_EXITED"

            self.get_logger().info(
                "EXIT_DETECTION -> CORRIDOR_EXITED"
            )

            self.publish_enable_state()
            return

        # ------------------------------------------------------------
        # Exit-state failure
        # ------------------------------------------------------------

        if target in (
            "HOVER_AND_REASSESS",
            "OBSTACLE_DECISION",
            "ABORT_CORRIDOR",
        ):

            self.phase = "HOLD"

            self.get_logger().warning(
                f"EXIT_DETECTION requested {target} "
                "-> HOLD for current Gazebo test"
            )

            self.publish_enable_state()

    # ================================================================
    # ENABLE MANAGEMENT
    # ================================================================

    def publish_enable_state(self):

        pre = Bool()
        cruise = Bool()
        exit_detection = Bool()

        pre.data = False
        cruise.data = False
        exit_detection.data = False

        if self.phase == "PRE_ENTRY":

            pre.data = True

        elif self.phase == "ENTER_CORRIDOR":

            # Manager controls entry motion directly.
            pass

        elif self.phase == "CRUISE":

            cruise.data = True

        elif self.phase == "EXIT_DETECTION":

            exit_detection.data = True

        elif self.phase in (
            "CORRIDOR_EXITED",
            "HOLD",
        ):

            # Everything disabled.
            pass

        self.pre_enable_pub.publish(
            pre
        )

        self.cruise_enable_pub.publish(
            cruise
        )

        self.exit_enable_pub.publish(
            exit_detection
        )

        mission_msg = String()
        mission_msg.data = self.phase

        self.mission_state_pub.publish(
            mission_msg
        )

    # ================================================================
    # ACTIVE COMMAND SELECTION
    # ================================================================

    def active_command(self):

        now = time.monotonic()

        # ------------------------------------------------------------
        # PRE_ENTRY controller
        # ------------------------------------------------------------

        if self.phase == "PRE_ENTRY":

            if now - self.pre_cmd_time > 0.5:
                return TwistStamped()

            return self.pre_cmd

        # ------------------------------------------------------------
        # ENTER_CORRIDOR
        # ------------------------------------------------------------

        if self.phase == "ENTER_CORRIDOR":

            cmd = TwistStamped()

            # Slow straight entry after pre-entry lock.
            cmd.twist.linear.x = 0.20
            cmd.twist.linear.y = 0.0
            cmd.twist.linear.z = 0.0

            cmd.twist.angular.x = 0.0
            cmd.twist.angular.y = 0.0
            cmd.twist.angular.z = 0.0

            return cmd

        # ------------------------------------------------------------
        # CORRIDOR_CRUISE controller
        # ------------------------------------------------------------

        if self.phase == "CRUISE":

            if now - self.cruise_cmd_time > 0.5:
                return TwistStamped()

            return self.cruise_cmd

        # ------------------------------------------------------------
        # EXIT_DETECTION controller
        # ------------------------------------------------------------

        if self.phase == "EXIT_DETECTION":

            if now - self.exit_cmd_time > 0.5:
                return TwistStamped()

            return self.exit_cmd

        # ------------------------------------------------------------
        # CORRIDOR_EXITED / HOLD
        # ------------------------------------------------------------

        return TwistStamped()

    # ================================================================
    # GAZEBO MOTION
    # ================================================================

    def update_motion(self):

        if self.request_in_progress:
            return

        req = GetEntityState.Request()

        req.name = self.model_name
        req.reference_frame = "world"

        self.request_in_progress = True

        future = self.get_state_client.call_async(
            req
        )

        future.add_done_callback(
            self.state_received
        )

    def state_received(self, future):

        self.request_in_progress = False

        try:

            result = future.result()

        except Exception as e:

            self.get_logger().error(
                f"GetEntityState failed: {e}"
            )

            return

        if not result.success:

            self.get_logger().warning(
                f"Cannot find model {self.model_name}"
            )

            return

        pose = result.state.pose

        yaw = quaternion_to_yaw(
            pose.orientation
        )

        # ============================================================
        # ENTER_CORRIDOR -> CORRIDOR_CRUISE
        #
        # Simulation-only transition condition.
        #
        # Corridor begins at world x = 0.
        # Start cruise after travelling 0.75 m inside.
        # ============================================================

        if self.phase == "ENTER_CORRIDOR":

            if pose.position.x >= 0.75:

                self.phase = "CRUISE"

                self.get_logger().info(
                    "ENTER_CORRIDOR -> CORRIDOR_CRUISE"
                )

                self.publish_enable_state()

        # Select command AFTER any phase transition.
        cmd = self.active_command()

        # ============================================================
        # BODY FLU -> WORLD
        # ============================================================

        vx_body = cmd.twist.linear.x
        vy_body = cmd.twist.linear.y

        vx_world = (
            math.cos(yaw) * vx_body
            - math.sin(yaw) * vy_body
        )

        vy_world = (
            math.sin(yaw) * vx_body
            + math.cos(yaw) * vy_body
        )

        # ============================================================
        # SET GAZEBO VELOCITY
        # ============================================================

        req = SetEntityState.Request()

        req.state.name = self.model_name

        # Preserve current pose.
        req.state.pose = pose

        req.state.twist.linear.x = (
            vx_world
        )

        req.state.twist.linear.y = (
            vy_world
        )

        req.state.twist.linear.z = (
            cmd.twist.linear.z
        )

        req.state.twist.angular.x = 0.0
        req.state.twist.angular.y = 0.0

        req.state.twist.angular.z = (
            cmd.twist.angular.z
        )

        req.state.reference_frame = "world"

        self.set_state_client.call_async(
            req
        )


def main(args=None):

    rclpy.init(args=args)

    node = CorridorMissionBridge()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()