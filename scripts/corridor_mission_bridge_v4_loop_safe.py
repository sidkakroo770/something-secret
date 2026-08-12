#!/usr/bin/env python3
"""Gazebo Classic mission bridge V4 with loop-safe HOVER_AND_REASSESS recovery.

Controller sequence
-------------------
PRE_ENTRY_GEOMETRY_LOCK
    -> ENTER_CORRIDOR          (bridge-local short straight entry)
    -> CORRIDOR_CRUISE
    -> OBSTACLE_DECISION / AVOID_LEFT / AVOID_RIGHT
    -> CORRIDOR_CRUISE         (may repeat for multiple obstacles)
    -> EXIT_DETECTION
    -> CORRIDOR_EXITED

The bridge does two jobs only:
  1. Arbitrates which controller owns the body-frame TwistStamped command.
  2. Applies that command to the Gazebo Classic model through
     /gazebo/get_entity_state and /gazebo/set_entity_state.

ROS FLU body convention used by all corridor controllers:
    +x forward, +y left, +z up, +yaw counter-clockwise.

Recovery / safety behaviour
---------------------------
* Exactly one controller is enabled at a time.
* Commands older than command_timeout_s are replaced by zero velocity.
* HOVER_AND_REASSESS is now a real controller phase, not a terminal HOLD.
  The bridge publishes source_state + pause_reason context to the reassessment
  node and keeps all normal navigation controllers disabled while it runs.
* If reassessment cannot recover, exceeds its hard timeout, or repeats in a loop, ABORT_CORRIDOR is terminal.
* HOLD remains only for bridge-internal faults such as ENTER_CORRIDOR timeout.
* Recovery loops are bounded by chain/same-source/mission-wide retry budgets.
* State handoffs are latched: stale next_state messages from disabled nodes
  cannot steal command authority back.

EXIT_DETECTION interface
------------------------
The exact exit-node namespace is parameterised because older test versions may
use a different topic prefix. Defaults are:
    /corridor/exit/enable
    /corridor/exit/cmd_vel
    /corridor/exit/state
    /corridor/exit/result
    /corridor/exit/next_state
Override them with --ros-args -p ... if your verified exit detector differs.
"""

from __future__ import annotations

import json
import math
import time
from typing import Dict

import rclpy
from gazebo_msgs.srv import GetEntityState, SetEntityState
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from std_msgs.msg import Bool, String


def quaternion_to_yaw(q) -> float:
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class CorridorMissionBridgeV4(Node):
    """Command arbiter + Gazebo Classic mover for the corridor FSM."""

    PRE_ENTRY = "PRE_ENTRY"
    ENTER_CORRIDOR = "ENTER_CORRIDOR"
    CRUISE = "CRUISE"
    OBSTACLE = "OBSTACLE"
    EXIT = "EXIT_DETECTION"
    REASSESS = "HOVER_AND_REASSESS"
    COMPLETE = "CORRIDOR_EXITED"
    ABORT = "ABORT_CORRIDOR"
    HOLD = "HOLD"

    def __init__(self) -> None:
        super().__init__("corridor_mission_bridge_v4")

        # ------------------------------------------------------------------
        # Gazebo / bridge parameters
        # ------------------------------------------------------------------
        self.declare_parameter("model_name", "corridor_drone")
        self.declare_parameter("command_timeout_s", 0.50)
        self.declare_parameter("update_rate_hz", 20.0)

        # Recovery loop guards. HOVER_AND_REASSESS must always terminate in
        # a normal mission state or ABORT_CORRIDOR; it may never become a
        # livelock between recovery and the same failing controller.
        self.declare_parameter("reassess_bridge_hard_timeout_s", 12.0)
        self.declare_parameter("max_reassess_chain_cycles", 3)
        self.declare_parameter("max_reassess_same_source_cycles", 2)
        self.declare_parameter("max_reassess_total_cycles", 12)
        self.declare_parameter("reassess_chain_reset_after_s", 4.0)

        # Short bridge-local entry after PRE_ENTRY lock.  The obstacle world
        # entrance is at x=0 and the normal test spawn is around x=0.5.
        self.declare_parameter("enter_corridor_speed_m_s", 0.20)
        self.declare_parameter("enter_corridor_until_world_x_m", 0.75)
        self.declare_parameter("enter_corridor_timeout_s", 5.0)

        # ------------------------------------------------------------------
        # PRE_ENTRY interface
        # ------------------------------------------------------------------
        self.declare_parameter("pre_cmd_topic", "/corridor/pre_entry/cmd_vel")
        self.declare_parameter("pre_enable_topic", "/corridor/pre_entry/enable")
        self.declare_parameter("pre_locked_topic", "/corridor/pre_entry/locked")
        self.declare_parameter("pre_next_state_topic", "/corridor/pre_entry/next_state")

        # ------------------------------------------------------------------
        # CORRIDOR_CRUISE interface
        # ------------------------------------------------------------------
        self.declare_parameter("cruise_cmd_topic", "/corridor/cruise/cmd_vel")
        self.declare_parameter("cruise_enable_topic", "/corridor/cruise/enable")
        self.declare_parameter("cruise_next_state_topic", "/corridor/cruise/next_state")
        self.declare_parameter("cruise_state_topic", "/corridor/cruise/state")

        # ------------------------------------------------------------------
        # OBSTACLE_DECISION / AVOID_LEFT / AVOID_RIGHT interface
        # ------------------------------------------------------------------
        self.declare_parameter("obstacle_cmd_topic", "/corridor/obstacle/cmd_vel")
        self.declare_parameter("obstacle_enable_topic", "/corridor/obstacle/enable")
        self.declare_parameter("obstacle_next_state_topic", "/corridor/obstacle/next_state")
        self.declare_parameter("obstacle_state_topic", "/corridor/obstacle/state")
        self.declare_parameter("obstacle_result_topic", "/corridor/obstacle/result")

        # ------------------------------------------------------------------
        # EXIT_DETECTION interface -- parameterised deliberately
        # ------------------------------------------------------------------
        self.declare_parameter("exit_cmd_topic", "/corridor/exit/cmd_vel")
        self.declare_parameter("exit_enable_topic", "/corridor/exit/enable")
        self.declare_parameter("exit_state_topic", "/corridor/exit/state")
        self.declare_parameter("exit_result_topic", "/corridor/exit/result")
        self.declare_parameter("exit_next_state_topic", "/corridor/exit/next_state")

        # ------------------------------------------------------------------
        # HOVER_AND_REASSESS recovery supervisor interface
        # ------------------------------------------------------------------
        self.declare_parameter("reassess_cmd_topic", "/corridor/reassess/cmd_vel")
        self.declare_parameter("reassess_enable_topic", "/corridor/reassess/enable")
        self.declare_parameter("reassess_context_topic", "/corridor/reassess/context")
        self.declare_parameter("reassess_state_topic", "/corridor/reassess/state")
        self.declare_parameter("reassess_result_topic", "/corridor/reassess/result")
        self.declare_parameter("reassess_next_state_topic", "/corridor/reassess/next_state")

        # Bridge status topics
        self.declare_parameter("mission_state_topic", "/corridor/mission/state")
        self.declare_parameter("mission_result_topic", "/corridor/mission/result")
        self.declare_parameter("diagnostics_topic", "/corridor/mission/diagnostics")

        self.model_name = str(self.get_parameter("model_name").value)
        self.command_timeout_s = float(self.get_parameter("command_timeout_s").value)

        # ------------------------------------------------------------------
        # Runtime state
        # ------------------------------------------------------------------
        self.phase = self.PRE_ENTRY
        self.phase_enter_monotonic = time.monotonic()
        self.hold_reason = ""

        self.pre_cmd = TwistStamped()
        self.cruise_cmd = TwistStamped()
        self.obstacle_cmd = TwistStamped()
        self.exit_cmd = TwistStamped()
        self.reassess_cmd = TwistStamped()

        self.pre_cmd_time = 0.0
        self.cruise_cmd_time = 0.0
        self.obstacle_cmd_time = 0.0
        self.exit_cmd_time = 0.0
        self.reassess_cmd_time = 0.0

        # Most recent controller-local states are used only for readable
        # /corridor/mission/state output. They do not own transitions.
        self.cruise_state = ""
        self.obstacle_state = ""
        self.exit_state = ""
        self.exit_result = ""
        self.obstacle_result = ""
        self.reassess_state = ""
        self.reassess_result = ""

        # Recovery context is latched when a normal controller requests
        # HOVER_AND_REASSESS.  last_obstacle_operating_state preserves whether
        # the failure came from OBSTACLE_DECISION, AVOID_LEFT, or AVOID_RIGHT.
        self.reassess_source_state = ""
        self.reassess_pause_reason = ""
        self.last_obstacle_operating_state = "OBSTACLE_DECISION"

        # Recovery anti-loop bookkeeping.
        self.reassess_chain_cycles = 0
        self.reassess_total_cycles = 0
        self.reassess_same_source_cycles = 0
        self.reassess_last_source = ""
        self.reassess_last_exit_target = ""

        self.last_pose = None
        self.last_yaw = None
        self.request_in_progress = False

        # ------------------------------------------------------------------
        # Subscriptions: commands
        # ------------------------------------------------------------------
        self.create_subscription(
            TwistStamped,
            str(self.get_parameter("pre_cmd_topic").value),
            self.pre_cmd_callback,
            10,
        )
        self.create_subscription(
            TwistStamped,
            str(self.get_parameter("cruise_cmd_topic").value),
            self.cruise_cmd_callback,
            10,
        )
        self.create_subscription(
            TwistStamped,
            str(self.get_parameter("obstacle_cmd_topic").value),
            self.obstacle_cmd_callback,
            10,
        )
        self.create_subscription(
            TwistStamped,
            str(self.get_parameter("exit_cmd_topic").value),
            self.exit_cmd_callback,
            10,
        )
        self.create_subscription(
            TwistStamped,
            str(self.get_parameter("reassess_cmd_topic").value),
            self.reassess_cmd_callback,
            10,
        )

        # PRE_ENTRY completion
        self.create_subscription(
            Bool,
            str(self.get_parameter("pre_locked_topic").value),
            self.pre_locked_callback,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("pre_next_state_topic").value),
            self.pre_next_state_callback,
            10,
        )

        # CRUISE transitions + readable local state
        self.create_subscription(
            String,
            str(self.get_parameter("cruise_next_state_topic").value),
            self.cruise_next_state_callback,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("cruise_state_topic").value),
            self.cruise_state_callback,
            10,
        )

        # OBSTACLE transitions + readable local state/result
        self.create_subscription(
            String,
            str(self.get_parameter("obstacle_next_state_topic").value),
            self.obstacle_next_state_callback,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("obstacle_state_topic").value),
            self.obstacle_state_callback,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("obstacle_result_topic").value),
            self.obstacle_result_callback,
            10,
        )

        # EXIT transitions + readable local state/result
        self.create_subscription(
            String,
            str(self.get_parameter("exit_next_state_topic").value),
            self.exit_next_state_callback,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("exit_state_topic").value),
            self.exit_state_callback,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("exit_result_topic").value),
            self.exit_result_callback,
            10,
        )

        # HOVER_AND_REASSESS transitions + readable state/result
        self.create_subscription(
            String,
            str(self.get_parameter("reassess_next_state_topic").value),
            self.reassess_next_state_callback,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("reassess_state_topic").value),
            self.reassess_state_callback,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("reassess_result_topic").value),
            self.reassess_result_callback,
            10,
        )

        # ------------------------------------------------------------------
        # Controller enable publishers
        # ------------------------------------------------------------------
        self.pre_enable_pub = self.create_publisher(
            Bool, str(self.get_parameter("pre_enable_topic").value), 10
        )
        self.cruise_enable_pub = self.create_publisher(
            Bool, str(self.get_parameter("cruise_enable_topic").value), 10
        )
        self.obstacle_enable_pub = self.create_publisher(
            Bool, str(self.get_parameter("obstacle_enable_topic").value), 10
        )
        self.exit_enable_pub = self.create_publisher(
            Bool, str(self.get_parameter("exit_enable_topic").value), 10
        )
        self.reassess_enable_pub = self.create_publisher(
            Bool, str(self.get_parameter("reassess_enable_topic").value), 10
        )
        self.reassess_context_pub = self.create_publisher(
            String, str(self.get_parameter("reassess_context_topic").value), 10
        )

        # Bridge status
        self.mission_state_pub = self.create_publisher(
            String, str(self.get_parameter("mission_state_topic").value), 10
        )
        self.mission_result_pub = self.create_publisher(
            String, str(self.get_parameter("mission_result_topic").value), 10
        )
        self.diagnostics_pub = self.create_publisher(
            String, str(self.get_parameter("diagnostics_topic").value), 10
        )

        # ------------------------------------------------------------------
        # Gazebo Classic services
        # ------------------------------------------------------------------
        self.get_state_client = self.create_client(
            GetEntityState, "/gazebo/get_entity_state"
        )
        self.set_state_client = self.create_client(
            SetEntityState, "/gazebo/set_entity_state"
        )

        self.get_logger().info("Waiting for Gazebo Classic state services...")
        self.get_state_client.wait_for_service()
        self.set_state_client.wait_for_service()
        self.get_logger().info("Gazebo connected")

        update_rate_hz = max(1.0, float(self.get_parameter("update_rate_hz").value))
        self.create_timer(1.0 / update_rate_hz, self.update_motion)
        self.create_timer(0.50, self.publish_enable_state)
        self.create_timer(0.20, self.publish_status)

        # Immediate enable publication avoids waiting for the first 0.5 s tick.
        self.publish_enable_state()
        self.publish_status()

        self.get_logger().info(
            "MISSION START -> PRE_ENTRY_GEOMETRY_LOCK"
        )

    # ==================================================================
    # Small helpers
    # ==================================================================

    def phase_age(self) -> float:
        return time.monotonic() - self.phase_enter_monotonic

    @staticmethod
    def zero_command() -> TwistStamped:
        return TwistStamped()

    def set_phase(self, new_phase: str, reason: str) -> None:
        if new_phase == self.phase:
            return

        old = self.phase
        self.phase = new_phase
        self.phase_enter_monotonic = time.monotonic()

        # Never reuse a command produced before a controller was newly enabled.
        if new_phase == self.PRE_ENTRY:
            self.pre_cmd_time = 0.0
        elif new_phase == self.CRUISE:
            self.cruise_cmd_time = 0.0
        elif new_phase == self.OBSTACLE:
            self.obstacle_cmd_time = 0.0
            self.obstacle_state = "OBSTACLE_DECISION"
        elif new_phase == self.EXIT:
            self.exit_cmd_time = 0.0
            self.exit_state = "EXIT_DETECTION"
        elif new_phase == self.REASSESS:
            self.reassess_cmd_time = 0.0
            self.reassess_state = "HOVER_AND_REASSESS"

        if new_phase == self.HOLD:
            self.hold_reason = reason
        elif old == self.HOLD:
            self.hold_reason = ""

        self.publish_enable_state()
        self.publish_status()

        if new_phase in {self.HOLD, self.ABORT}:
            self.get_logger().error(f"FSM {old} -> {new_phase}: {reason}")
        else:
            self.get_logger().warning(f"FSM {old} -> {new_phase}: {reason}")

    def enter_hold(self, reason: str) -> None:
        self.set_phase(self.HOLD, reason)

    def enter_abort(self, reason: str) -> None:
        self.reassess_pause_reason = reason
        self.set_phase(self.ABORT, reason)

    def start_reassess(self, source_state: str, pause_reason: str) -> None:
        source = str(source_state).strip().upper()
        reason = str(pause_reason).strip()

        # Count recovery entries BEFORE enabling HOVER_AND_REASSESS.
        # A repeated bounce is therefore finite by construction:
        # normal_state -> REASSESS -> normal_state -> REASSESS -> ... -> ABORT.
        next_chain = self.reassess_chain_cycles + 1
        next_total = self.reassess_total_cycles + 1
        next_same = (
            self.reassess_same_source_cycles + 1
            if source == self.reassess_last_source
            else 1
        )

        max_chain = max(1, int(self.get_parameter("max_reassess_chain_cycles").value))
        max_same = max(1, int(self.get_parameter("max_reassess_same_source_cycles").value))
        max_total = max(1, int(self.get_parameter("max_reassess_total_cycles").value))

        if next_chain > max_chain:
            self.enter_abort(
                f"recovery loop guard: more than {max_chain} reassessment cycles "
                f"without stable mission progress (latest source={source})"
            )
            return

        if next_same > max_same:
            self.enter_abort(
                f"recovery loop guard: {source} requested HOVER_AND_REASSESS "
                f"more than {max_same} times in the same recovery chain"
            )
            return

        if next_total > max_total:
            self.enter_abort(
                f"mission recovery budget exhausted after {max_total} total "
                "HOVER_AND_REASSESS entries"
            )
            return

        self.reassess_chain_cycles = next_chain
        self.reassess_total_cycles = next_total
        self.reassess_same_source_cycles = next_same
        self.reassess_last_source = source
        self.reassess_source_state = source
        self.reassess_pause_reason = reason
        self.set_phase(
            self.REASSESS,
            f"source={self.reassess_source_state}; reason={self.reassess_pause_reason or 'unspecified'}",
        )
        self.publish_reassess_context()

    def publish_reassess_context(self) -> None:
        if not hasattr(self, "reassess_context_pub"):
            return
        msg = String()
        msg.data = json.dumps(
            {
                "source_state": self.reassess_source_state,
                "pause_reason": self.reassess_pause_reason,
            },
            separators=(",", ":"),
        )
        self.reassess_context_pub.publish(msg)

    # ==================================================================
    # Command callbacks
    # ==================================================================

    def pre_cmd_callback(self, msg: TwistStamped) -> None:
        self.pre_cmd = msg
        self.pre_cmd_time = time.monotonic()

    def cruise_cmd_callback(self, msg: TwistStamped) -> None:
        self.cruise_cmd = msg
        self.cruise_cmd_time = time.monotonic()

    def obstacle_cmd_callback(self, msg: TwistStamped) -> None:
        self.obstacle_cmd = msg
        self.obstacle_cmd_time = time.monotonic()

    def exit_cmd_callback(self, msg: TwistStamped) -> None:
        self.exit_cmd = msg
        self.exit_cmd_time = time.monotonic()

    def reassess_cmd_callback(self, msg: TwistStamped) -> None:
        self.reassess_cmd = msg
        self.reassess_cmd_time = time.monotonic()

    # ==================================================================
    # Controller-local state/result callbacks
    # ==================================================================

    def cruise_state_callback(self, msg: String) -> None:
        self.cruise_state = msg.data.strip()

    def obstacle_state_callback(self, msg: String) -> None:
        self.obstacle_state = msg.data.strip()
        value = self.obstacle_state.upper()
        if value in {"OBSTACLE_DECISION", "AVOID_LEFT", "AVOID_RIGHT"}:
            self.last_obstacle_operating_state = value

    def obstacle_result_callback(self, msg: String) -> None:
        self.obstacle_result = msg.data.strip()

    def exit_state_callback(self, msg: String) -> None:
        self.exit_state = msg.data.strip()

    def exit_result_callback(self, msg: String) -> None:
        self.exit_result = msg.data.strip()
        if self.phase != self.EXIT:
            return

        # Compatibility fallback for exit detectors that report completion in
        # result instead of next_state. next_state remains the preferred path.
        value = msg.data.strip().upper()
        if value in {
            "EXIT_CONFIRMED",
            "CORRIDOR_EXITED",
            "CONFIRMED",
            "SUCCESS",
            "COMPLETE",
        }:
            self.set_phase(self.COMPLETE, f"exit result: {msg.data.strip()}")


    def reassess_state_callback(self, msg: String) -> None:
        self.reassess_state = msg.data.strip()

    def reassess_result_callback(self, msg: String) -> None:
        self.reassess_result = msg.data.strip()

    def obstacle_failure_reason(self) -> str:
        value = self.obstacle_result.strip()
        prefix = "TRANSITION:HOVER_AND_REASSESS:"
        if value.upper().startswith(prefix):
            return value[len(prefix):].strip() or "NO_SAFE_SIDE_OR_BYPASS_FAILURE"
        return "NO_SAFE_SIDE_OR_BYPASS_FAILURE"

    # ==================================================================
    # FSM transition callbacks
    # ==================================================================

    def pre_locked_callback(self, msg: Bool) -> None:
        if msg.data and self.phase == self.PRE_ENTRY:
            self.set_phase(
                self.ENTER_CORRIDOR,
                "PRE_ENTRY_GEOMETRY_LOCK reported LOCKED",
            )

    def pre_next_state_callback(self, msg: String) -> None:
        if self.phase != self.PRE_ENTRY:
            return
        target = msg.data.strip().upper()
        if not target:
            return
        if target in {"ENTER_CORRIDOR", "CORRIDOR_CRUISE"}:
            self.set_phase(self.ENTER_CORRIDOR, f"pre-entry requested {target}")
        elif target == "HOVER_AND_REASSESS":
            self.start_reassess("PRE_ENTRY_GEOMETRY_LOCK", "ENTRY_LOCK_FAILED / LOW_CONFIDENCE_GEOMETRY")
        elif target == "ABORT_CORRIDOR":
            self.enter_abort("pre-entry requested ABORT_CORRIDOR")

    def cruise_next_state_callback(self, msg: String) -> None:
        if self.phase != self.CRUISE:
            return

        target = msg.data.strip().upper()
        if not target:
            return

        if target == "OBSTACLE_DECISION":
            self.set_phase(
                self.OBSTACLE,
                "CORRIDOR_CRUISE confirmed front obstacle",
            )
        elif target == "EXIT_DETECTION":
            self.set_phase(
                self.EXIT,
                "CORRIDOR_CRUISE confirmed persistent exit candidate",
            )
        elif target == "HOVER_AND_REASSESS":
            self.start_reassess("CORRIDOR_CRUISE", "LOW_CONFIDENCE_GEOMETRY")
        elif target == "ABORT_CORRIDOR":
            self.enter_abort("CORRIDOR_CRUISE requested ABORT_CORRIDOR")

    def obstacle_next_state_callback(self, msg: String) -> None:
        if self.phase != self.OBSTACLE:
            return

        target = msg.data.strip().upper()
        if not target:
            return

        if target == "CORRIDOR_CRUISE":
            self.set_phase(
                self.CRUISE,
                "obstacle bypass complete",
            )
        elif target == "EXIT_DETECTION":
            # Not expected in V1, but safe to support.
            self.set_phase(self.EXIT, "obstacle controller requested EXIT_DETECTION")
        elif target == "HOVER_AND_REASSESS":
            self.start_reassess(
                self.last_obstacle_operating_state,
                self.obstacle_failure_reason(),
            )
        elif target == "ABORT_CORRIDOR":
            self.enter_abort("obstacle controller requested ABORT_CORRIDOR")

    def exit_next_state_callback(self, msg: String) -> None:
        if self.phase != self.EXIT:
            return

        target = msg.data.strip().upper()
        if not target:
            return

        if target == "CORRIDOR_CRUISE":
            # Useful if EXIT_DETECTION rejects a false candidate and explicitly
            # asks to resume corridor tracking.
            self.set_phase(self.CRUISE, "EXIT_DETECTION rejected candidate")
        elif target in {
            "CORRIDOR_EXITED",
            "DELIVERY_ZONE_NAVIGATION",
            "DELIVERY_ZONE",
            "MISSION_COMPLETE",
            "COMPLETE",
        }:
            self.set_phase(self.COMPLETE, f"EXIT_DETECTION requested {target}")
        elif target == "HOVER_AND_REASSESS":
            self.start_reassess("EXIT_DETECTION", "AMBIGUOUS_EXIT")
        elif target == "ABORT_CORRIDOR":
            self.enter_abort("EXIT_DETECTION requested ABORT_CORRIDOR")

    def reassess_next_state_callback(self, msg: String) -> None:
        if self.phase != self.REASSESS:
            return
        target = msg.data.strip().upper()
        if not target:
            return

        # HOVER_AND_REASSESS is forbidden from requesting itself. Any unknown
        # or recursive target is treated as a safe terminal abort.
        if target == "HOVER_AND_REASSESS":
            self.enter_abort("HOVER_AND_REASSESS attempted a recursive self-transition")
            return

        self.reassess_last_exit_target = target

        if target == "PRE_ENTRY_GEOMETRY_LOCK":
            self.set_phase(self.PRE_ENTRY, "HOVER_AND_REASSESS recovered entry geometry")
        elif target == "ENTER_CORRIDOR":
            self.set_phase(self.ENTER_CORRIDOR, "HOVER_AND_REASSESS recovered entry path")
        elif target == "CORRIDOR_CRUISE":
            self.set_phase(self.CRUISE, "HOVER_AND_REASSESS recovered corridor cruise")
        elif target in {"OBSTACLE_DECISION", "AVOID_LEFT", "AVOID_RIGHT"}:
            # obstacle_avoidance_v1 intentionally starts from OBSTACLE_DECISION
            # whenever re-enabled, so AVOID_* is treated as a recovery
            # recommendation that must be independently revalidated.
            self.set_phase(
                self.OBSTACLE,
                f"HOVER_AND_REASSESS recommends {target}; obstacle controller will revalidate",
            )
        elif target == "EXIT_DETECTION":
            self.set_phase(self.EXIT, "HOVER_AND_REASSESS recovered exit evidence")
        elif target == "SEARCH_ENTRY_MARKER":
            # SEARCH_ENTRY_MARKER is in the spreadsheet but not part of this
            # current four-controller Gazebo bridge.
            self.enter_abort("SEARCH_ENTRY_MARKER recovery requested but marker controller is not wired")
        elif target == "ABORT_CORRIDOR":
            self.enter_abort("HOVER_AND_REASSESS exhausted safe recovery")
        else:
            self.enter_abort(f"HOVER_AND_REASSESS requested unsupported target {target}")

    def enforce_recovery_guards(self) -> None:
        """Guarantee HOVER_AND_REASSESS cannot become a terminal/livelock state."""
        if self.phase == self.REASSESS:
            hard_timeout = max(
                0.5,
                float(self.get_parameter("reassess_bridge_hard_timeout_s").value),
            )
            if self.phase_age() > hard_timeout:
                self.enter_abort(
                    f"HOVER_AND_REASSESS bridge hard-timeout after {hard_timeout:.1f} s "
                    f"(source={self.reassess_source_state or 'UNKNOWN'})"
                )
            return

        # A recovery chain is considered broken only after a normal mission
        # controller has remained active for a meaningful amount of time.
        # This prevents rapid A->REASSESS->B->REASSESS oscillations from
        # resetting the retry budget.
        if self.phase in {
            self.PRE_ENTRY,
            self.ENTER_CORRIDOR,
            self.CRUISE,
            self.OBSTACLE,
            self.EXIT,
        }:
            reset_after = max(
                0.5,
                float(self.get_parameter("reassess_chain_reset_after_s").value),
            )
            if self.reassess_chain_cycles > 0 and self.phase_age() >= reset_after:
                self.get_logger().info(
                    f"Recovery chain cleared after {self.phase_age():.1f} s stable in "
                    f"{self.public_state_name()}"
                )
                self.reassess_chain_cycles = 0
                self.reassess_same_source_cycles = 0
                self.reassess_last_source = ""

    # ==================================================================
    # Enable arbitration
    # ==================================================================

    def publish_enable_state(self) -> None:
        pre = Bool()
        cruise = Bool()
        obstacle = Bool()
        exit_enable = Bool()
        reassess = Bool()

        pre.data = self.phase == self.PRE_ENTRY
        cruise.data = self.phase == self.CRUISE
        obstacle.data = self.phase == self.OBSTACLE
        exit_enable.data = self.phase == self.EXIT
        reassess.data = self.phase == self.REASSESS

        self.pre_enable_pub.publish(pre)
        self.cruise_enable_pub.publish(cruise)
        self.obstacle_enable_pub.publish(obstacle)
        self.exit_enable_pub.publish(exit_enable)
        self.reassess_enable_pub.publish(reassess)
        if self.phase == self.REASSESS:
            self.publish_reassess_context()

    # ==================================================================
    # Command arbitration
    # ==================================================================

    def fresh_or_zero(self, cmd: TwistStamped, stamp: float) -> TwistStamped:
        if stamp <= 0.0:
            return self.zero_command()
        if time.monotonic() - stamp > self.command_timeout_s:
            return self.zero_command()
        return cmd

    def active_command(self) -> TwistStamped:
        if self.phase == self.PRE_ENTRY:
            return self.fresh_or_zero(self.pre_cmd, self.pre_cmd_time)

        if self.phase == self.ENTER_CORRIDOR:
            cmd = TwistStamped()
            cmd.header.frame_id = "base_link"
            cmd.twist.linear.x = max(
                0.0, float(self.get_parameter("enter_corridor_speed_m_s").value)
            )
            return cmd

        if self.phase == self.CRUISE:
            return self.fresh_or_zero(self.cruise_cmd, self.cruise_cmd_time)

        if self.phase == self.OBSTACLE:
            return self.fresh_or_zero(self.obstacle_cmd, self.obstacle_cmd_time)

        if self.phase == self.EXIT:
            return self.fresh_or_zero(self.exit_cmd, self.exit_cmd_time)

        if self.phase == self.REASSESS:
            # Reassess V1 itself publishes zero motion. Keeping it through the
            # same watchdog path preserves one-owner command arbitration.
            return self.fresh_or_zero(self.reassess_cmd, self.reassess_cmd_time)

        # COMPLETE, ABORT and HOLD are hard zero-motion states.
        return self.zero_command()

    # ==================================================================
    # Gazebo movement
    # ==================================================================

    def update_motion(self) -> None:
        self.enforce_recovery_guards()
        if self.request_in_progress:
            return

        req = GetEntityState.Request()
        req.name = self.model_name
        req.reference_frame = "world"

        self.request_in_progress = True
        future = self.get_state_client.call_async(req)
        future.add_done_callback(self.state_received)

    def state_received(self, future) -> None:
        self.request_in_progress = False

        try:
            result = future.result()
        except Exception as exc:
            self.get_logger().error(f"GetEntityState failed: {exc}")
            return

        if not result.success:
            self.get_logger().warning(f"Cannot find Gazebo model {self.model_name}")
            return

        pose = result.state.pose
        yaw = quaternion_to_yaw(pose.orientation)
        self.last_pose = pose
        self.last_yaw = yaw

        # Bridge-local ENTER_CORRIDOR completion check. Evaluate before command
        # selection so the controller handoff is immediate on this tick.
        if self.phase == self.ENTER_CORRIDOR:
            target_x = float(
                self.get_parameter("enter_corridor_until_world_x_m").value
            )
            timeout_s = float(self.get_parameter("enter_corridor_timeout_s").value)

            if pose.position.x >= target_x:
                self.set_phase(
                    self.CRUISE,
                    f"entry x={pose.position.x:.2f} m reached target {target_x:.2f} m",
                )
            elif self.phase_age() > timeout_s:
                self.enter_hold(
                    f"ENTER_CORRIDOR timeout: x={pose.position.x:.2f} m did not reach "
                    f"{target_x:.2f} m within {timeout_s:.1f} s"
                )

        cmd = self.active_command()

        # Controller commands are BODY-FRAME ROS FLU. Rotate body XY into the
        # Gazebo world frame while preserving z velocity and yaw rate.
        vx_body = float(cmd.twist.linear.x)
        vy_body = float(cmd.twist.linear.y)

        vx_world = math.cos(yaw) * vx_body - math.sin(yaw) * vy_body
        vy_world = math.sin(yaw) * vx_body + math.cos(yaw) * vy_body

        set_req = SetEntityState.Request()
        set_req.state.name = self.model_name
        set_req.state.pose = pose
        set_req.state.twist.linear.x = vx_world
        set_req.state.twist.linear.y = vy_world
        set_req.state.twist.linear.z = float(cmd.twist.linear.z)
        set_req.state.twist.angular.x = 0.0
        set_req.state.twist.angular.y = 0.0
        set_req.state.twist.angular.z = float(cmd.twist.angular.z)
        set_req.state.reference_frame = "world"

        self.set_state_client.call_async(set_req)

    # ==================================================================
    # Status / diagnostics
    # ==================================================================

    def public_state_name(self) -> str:
        if self.phase == self.PRE_ENTRY:
            return "PRE_ENTRY_GEOMETRY_LOCK"
        if self.phase == self.ENTER_CORRIDOR:
            return "ENTER_CORRIDOR"
        if self.phase == self.CRUISE:
            return "CORRIDOR_CRUISE"
        if self.phase == self.OBSTACLE:
            # obstacle_avoidance_v1 publishes OBSTACLE_DECISION, AVOID_LEFT,
            # AVOID_RIGHT, or a transition target on its state topic.
            if self.obstacle_state in {
                "OBSTACLE_DECISION",
                "AVOID_LEFT",
                "AVOID_RIGHT",
            }:
                return self.obstacle_state
            return "OBSTACLE_DECISION"
        if self.phase == self.EXIT:
            return "EXIT_DETECTION"
        if self.phase == self.REASSESS:
            return "HOVER_AND_REASSESS"
        if self.phase == self.COMPLETE:
            return "CORRIDOR_EXITED"
        if self.phase == self.ABORT:
            return "ABORT_CORRIDOR"
        return "HOLD"

    def current_command_source(self) -> str:
        return {
            self.PRE_ENTRY: "pre_entry",
            self.ENTER_CORRIDOR: "bridge_enter_corridor",
            self.CRUISE: "corridor_cruise",
            self.OBSTACLE: "obstacle_avoidance",
            self.EXIT: "exit_detection",
            self.REASSESS: "hover_and_reassess",
            self.COMPLETE: "zero",
            self.ABORT: "zero",
            self.HOLD: "zero",
        }.get(self.phase, "zero")

    def command_ages(self) -> Dict[str, float | None]:
        now = time.monotonic()

        def age(t: float):
            return round(now - t, 3) if t > 0.0 else None

        return {
            "pre_entry_s": age(self.pre_cmd_time),
            "cruise_s": age(self.cruise_cmd_time),
            "obstacle_s": age(self.obstacle_cmd_time),
            "exit_s": age(self.exit_cmd_time),
            "reassess_s": age(self.reassess_cmd_time),
        }

    def publish_status(self) -> None:
        state_msg = String()
        state_msg.data = self.public_state_name()
        self.mission_state_pub.publish(state_msg)

        result_msg = String()
        if self.phase == self.COMPLETE:
            result_msg.data = "SUCCESS:CORRIDOR_EXITED"
        elif self.phase == self.ABORT:
            result_msg.data = f"ABORTED:{self.reassess_pause_reason or 'ABORT_CORRIDOR'}"
        elif self.phase == self.HOLD:
            result_msg.data = f"HOLD:{self.hold_reason}"
        elif self.phase == self.REASSESS:
            result_msg.data = f"RECOVERING:{self.reassess_source_state}:{self.reassess_pause_reason}"
        else:
            result_msg.data = "IN_PROGRESS"
        self.mission_result_pub.publish(result_msg)

        diag = {
            "phase": self.phase,
            "public_state": self.public_state_name(),
            "command_source": self.current_command_source(),
            "command_timeout_s": self.command_timeout_s,
            "command_ages": self.command_ages(),
            "cruise_state": self.cruise_state,
            "obstacle_state": self.obstacle_state,
            "obstacle_result": self.obstacle_result,
            "exit_state": self.exit_state,
            "exit_result": self.exit_result,
            "reassess_state": self.reassess_state,
            "reassess_result": self.reassess_result,
            "reassess_source_state": self.reassess_source_state or None,
            "reassess_pause_reason": self.reassess_pause_reason or None,
            "reassess_chain_cycles": self.reassess_chain_cycles,
            "reassess_total_cycles": self.reassess_total_cycles,
            "reassess_same_source_cycles": self.reassess_same_source_cycles,
            "reassess_last_source": self.reassess_last_source or None,
            "reassess_last_exit_target": self.reassess_last_exit_target or None,
            "hold_reason": self.hold_reason or None,
        }

        if self.last_pose is not None:
            diag["gazebo_pose"] = {
                "x": round(float(self.last_pose.position.x), 3),
                "y": round(float(self.last_pose.position.y), 3),
                "z": round(float(self.last_pose.position.z), 3),
                "yaw_deg": round(math.degrees(float(self.last_yaw or 0.0)), 2),
            }

        msg = String()
        msg.data = json.dumps(diag, separators=(",", ":"))
        self.diagnostics_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CorridorMissionBridgeV4()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
