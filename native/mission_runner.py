#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Optional

from native.common.scan_adapter import ScanAdapter
from native.common.types import (
    Attitude,
    BodyVelocity,
    ControllerOutput,
    MissionState,
    NativeScan,
)

from native.controllers.pre_entry import PreEntryController
from native.controllers.corridor_cruise import CorridorCruiseController
from native.controllers.obstacle_avoidance import (
    ObstacleAvoidanceController,
    ObstacleState,
)
from native.controllers.exit_detection import ExitDetectionController
from native.controllers.hover_and_reassess import (
    HoverAndReassessController,
)
from native.controllers.abort_corridor import (
    AbortCorridorController,
)

from native.hardware.d500_driver import D500Driver


# ============================================================
# Vehicle state
# ============================================================

@dataclass
class VehiclePose:

    """
    Local horizontal position + yaw.

    The next MAVLink step will populate these fields from the
    Pixhawk. Until then the native runner deliberately receives None.
    """

    x_m: float
    y_m: float
    yaw_rad: float
    timestamp: float

    @property
    def age_s(self) -> float:
        return max(
            0.0,
            time.monotonic() - self.timestamp,
        )


# ============================================================
# Runner configuration
# ============================================================

@dataclass
class MissionRunnerConfig:

    # PRE_ENTRY transient HOLD is useful for brief obstructions, but
    # a fully autonomous mission must not remain there forever.
    pre_entry_hold_timeout_s: float = 8.0

    # ENTER_CORRIDOR
    enter_corridor_speed_m_s: float = 0.20

    # IMPORTANT:
    # This is a travelled distance, NOT the old Gazebo x=0.75 target.
    #
    # Leave None until we deliberately choose the physical entry
    # distance for the real aircraft.
    enter_corridor_distance_m: Optional[float] = None

    enter_corridor_pose_timeout_s: float = 2.0
    enter_corridor_timeout_s: float = 8.0

    # Recovery guards copied from final loop-safe bridge.
    reassess_hard_timeout_s: float = 12.0
    max_reassess_chain_cycles: int = 3
    max_reassess_same_source_cycles: int = 2
    max_reassess_total_cycles: int = 12
    reassess_chain_reset_after_s: float = 4.0

    # Runner
    diagnostics_period_s: float = 0.50


# ============================================================
# Mission runner
# ============================================================

class NativeMissionRunner:

    def __init__(
        self,
        config: Optional[MissionRunnerConfig] = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else MissionRunnerConfig()
        )

        # ----------------------------------------------------
        # Controllers
        # ----------------------------------------------------

        self.pre_entry = PreEntryController()
        self.cruise = CorridorCruiseController()
        self.obstacle = ObstacleAvoidanceController()
        self.exit = ExitDetectionController()
        self.reassess = HoverAndReassessController()
        self.abort_controller = AbortCorridorController()

        # ----------------------------------------------------
        # Mission state
        # ----------------------------------------------------

        self.state = (
            MissionState.PRE_ENTRY_GEOMETRY_LOCK
        )

        self.state_enter_time = (
            time.monotonic()
        )

        self.latest_command = (
            BodyVelocity.stop()
        )

        self.terminal_reason = ""

        # ----------------------------------------------------
        # ENTER_CORRIDOR tracking
        # ----------------------------------------------------

        self.enter_start_pose: Optional[
            VehiclePose
        ] = None

        # ----------------------------------------------------
        # Recovery bookkeeping
        # ----------------------------------------------------

        self.reassess_source_state: Optional[
            MissionState
        ] = None

        self.reassess_pause_reason = ""

        self.reassess_chain_cycles = 0
        self.reassess_total_cycles = 0
        self.reassess_same_source_cycles = 0

        self.reassess_last_source: Optional[
            MissionState
        ] = None

        self.last_normal_progress_time = (
            time.monotonic()
        )

        self.last_diagnostics_time = 0.0

        # Start first controller.
        self.pre_entry.enter()

        print()
        print(
            "[MISSION] START -> "
            "PRE_ENTRY_GEOMETRY_LOCK"
        )

    # ========================================================
    # Timing
    # ========================================================

    def state_age(self) -> float:

        return max(
            0.0,
            time.monotonic()
            - self.state_enter_time,
        )

    # ========================================================
    # Public mission state
    # ========================================================

    def public_state(self) -> MissionState:

        # While obstacle controller owns the mission, expose its
        # actual AVOID_LEFT / AVOID_RIGHT state.
        if self.state == MissionState.OBSTACLE_DECISION:

            if (
                self.obstacle.state
                == ObstacleState.AVOID_LEFT
            ):
                return MissionState.AVOID_LEFT

            if (
                self.obstacle.state
                == ObstacleState.AVOID_RIGHT
            ):
                return MissionState.AVOID_RIGHT

            return MissionState.OBSTACLE_DECISION

        return self.state

    # ========================================================
    # State transitions
    # ========================================================

    def transition(
        self,
        new_state: MissionState,
        reason: str,
    ) -> None:

        if new_state == self.state:
            return

        old_public = self.public_state()
        old_internal = self.state

        self.state = new_state

        self.state_enter_time = (
            time.monotonic()
        )

        self.latest_command = (
            BodyVelocity.stop()
        )

        self.enter_start_pose = None

        # ----------------------------------------------------
        # Start appropriate controller fresh.
        # ----------------------------------------------------

        if (
            new_state
            == MissionState.PRE_ENTRY_GEOMETRY_LOCK
        ):

            self.pre_entry.enter()

        elif (
            new_state
            == MissionState.CORRIDOR_CRUISE
        ):

            self.cruise.enter()

        elif (
            new_state
            == MissionState.OBSTACLE_DECISION
        ):

            # Important:
            # every re-entry starts from OBSTACLE_DECISION.
            # Never resume stale SHIFT/PASS state.
            self.obstacle.enter()

        elif (
            new_state
            == MissionState.EXIT_DETECTION
        ):

            self.exit.enter()

        # HOVER_AND_REASSESS is entered through
        # start_reassess(), not directly here.

        if new_state not in (
            MissionState.HOVER_AND_REASSESS,
            MissionState.ABORT_CORRIDOR,
            MissionState.CORRIDOR_EXITED,
        ):

            self.last_normal_progress_time = (
                time.monotonic()
            )

        print(
            f"[MISSION] "
            f"{old_public.value} -> "
            f"{new_state.value}: "
            f"{reason}"
        )

    # ========================================================
    # Recovery supervisor
    # ========================================================

    def start_reassess(
        self,
        source_state: MissionState,
        reason: str,
    ) -> None:

        c = self.config

        # -----------------------------------------------
        # Reproduce bridge anti-loop accounting.
        # -----------------------------------------------

        next_chain = (
            self.reassess_chain_cycles + 1
        )

        next_total = (
            self.reassess_total_cycles + 1
        )

        next_same = (
            self.reassess_same_source_cycles + 1
            if source_state
            == self.reassess_last_source
            else 1
        )

        if (
            next_chain
            > c.max_reassess_chain_cycles
        ):

            self.abort(
                "recovery loop guard: "
                f"more than "
                f"{c.max_reassess_chain_cycles} "
                "reassessment cycles without "
                "stable mission progress"
            )

            return

        if (
            next_same
            > c.max_reassess_same_source_cycles
        ):

            self.abort(
                "recovery loop guard: "
                f"{source_state.value} requested "
                "HOVER_AND_REASSESS more than "
                f"{c.max_reassess_same_source_cycles} "
                "times in the same recovery chain"
            )

            return

        if (
            next_total
            > c.max_reassess_total_cycles
        ):

            self.abort(
                "mission recovery budget exhausted "
                f"after "
                f"{c.max_reassess_total_cycles} "
                "total reassessment entries"
            )

            return

        self.reassess_chain_cycles = next_chain
        self.reassess_total_cycles = next_total
        self.reassess_same_source_cycles = next_same

        self.reassess_last_source = (
            source_state
        )

        self.reassess_source_state = (
            source_state
        )

        self.reassess_pause_reason = (
            reason
        )

        old = self.public_state()

        self.state = (
            MissionState.HOVER_AND_REASSESS
        )

        self.state_enter_time = (
            time.monotonic()
        )

        self.latest_command = (
            BodyVelocity.stop()
        )

        self.reassess.enter(
            source_state,
            reason,
        )

        print(
            f"[MISSION] {old.value} -> "
            "HOVER_AND_REASSESS: "
            f"source={source_state.value}; "
            f"reason={reason}"
        )

    def maybe_reset_recovery_chain(
        self,
    ) -> None:

        if self.state in (
            MissionState.HOVER_AND_REASSESS,
            MissionState.ABORT_CORRIDOR,
            MissionState.CORRIDOR_EXITED,
        ):
            return

        stable_time = (
            time.monotonic()
            - self.last_normal_progress_time
        )

        if (
            stable_time
            >= self.config.reassess_chain_reset_after_s
            and self.reassess_chain_cycles > 0
        ):

            self.reassess_chain_cycles = 0
            self.reassess_same_source_cycles = 0
            self.reassess_last_source = None

    # ========================================================
    # Terminal states
    # ========================================================

    def abort(
        self,
        reason: str,
    ) -> None:

        old = self.public_state()

        self.state = (
            MissionState.ABORT_CORRIDOR
        )

        self.state_enter_time = (
            time.monotonic()
        )

        self.latest_command = (
            BodyVelocity.stop()
        )

        self.terminal_reason = reason

        self.abort_controller.enter(
            reason
        )

        print(
            f"[MISSION] "
            f"{old.value} -> "
            f"ABORT_CORRIDOR: "
            f"{reason}"
        )

    def complete(
        self,
        reason: str,
    ) -> None:

        old = self.public_state()

        self.state = (
            MissionState.CORRIDOR_EXITED
        )

        self.state_enter_time = (
            time.monotonic()
        )

        self.latest_command = (
            BodyVelocity.stop()
        )

        self.terminal_reason = reason

        print(
            f"[MISSION] "
            f"{old.value} -> "
            f"CORRIDOR_EXITED: "
            f"{reason}"
        )

    # ========================================================
    # ENTER_CORRIDOR
    # ========================================================

    @staticmethod
    def forward_distance_from_pose(
        start: VehiclePose,
        current: VehiclePose,
    ) -> float:

        dx = current.x_m - start.x_m
        dy = current.y_m - start.y_m

        # Project local displacement onto heading that was present
        # when ENTER_CORRIDOR began.
        return (
            math.cos(start.yaw_rad) * dx
            + math.sin(start.yaw_rad) * dy
        )

    def step_enter_corridor(
        self,
        pose: Optional[VehiclePose],
    ) -> ControllerOutput:

        c = self.config

        # Until MAVLink is connected we have deliberately not selected a
        # physical real-world entry distance.
        if c.enter_corridor_distance_m is None:

            self.latest_command = (
                BodyVelocity.stop()
            )

            return ControllerOutput(
                command=self.latest_command,
                next_state=None,
                status="WAITING_FOR_ENTER_DISTANCE_CONFIG",
                reason=(
                    "real ENTER_CORRIDOR distance "
                    "has not been configured"
                ),
                confidence=None,
            )

        if pose is None or pose.age_s > 0.50:

            self.latest_command = (
                BodyVelocity.stop()
            )

            if (
                self.state_age()
                > c.enter_corridor_pose_timeout_s
            ):

                self.abort(
                    "ENTER_CORRIDOR has no fresh "
                    "vehicle local-position feedback"
                )

            return ControllerOutput(
                command=self.latest_command,
                status="WAITING_FOR_POSE",
                reason="no fresh local-position feedback",
            )

        if self.enter_start_pose is None:

            self.enter_start_pose = pose

            print(
                "[ENTER] local start pose latched"
            )

        travelled = (
            self.forward_distance_from_pose(
                self.enter_start_pose,
                pose,
            )
        )

        if (
            travelled
            >= c.enter_corridor_distance_m
        ):

            self.transition(
                MissionState.CORRIDOR_CRUISE,
                (
                    "entered corridor by "
                    f"{travelled:.2f} m "
                    f"(target "
                    f"{c.enter_corridor_distance_m:.2f} m)"
                ),
            )

            return ControllerOutput(
                command=BodyVelocity.stop(),
                status="COMPLETE",
            )

        if (
            self.state_age()
            > c.enter_corridor_timeout_s
        ):

            self.abort(
                "ENTER_CORRIDOR timeout: "
                f"travelled {travelled:.2f} m "
                f"of required "
                f"{c.enter_corridor_distance_m:.2f} m"
            )

            return ControllerOutput(
                command=BodyVelocity.stop(),
                status="TIMEOUT",
            )

        self.latest_command = BodyVelocity(
            vx_m_s=max(
                0.0,
                c.enter_corridor_speed_m_s,
            )
        )

        return ControllerOutput(
            command=self.latest_command,
            status="ENTER_CORRIDOR",
            reason=(
                f"travelled={travelled:.2f} m"
            ),
        )

    # ========================================================
    # Route controller result
    # ========================================================

    def handle_controller_transition(
        self,
        output: ControllerOutput,
    ) -> None:

        target = output.next_state

        if target is None:
            return

        source = self.public_state()

        # ----------------------------------------------------
        # PRE_ENTRY
        # ----------------------------------------------------

        if (
            self.state
            == MissionState.PRE_ENTRY_GEOMETRY_LOCK
        ):

            if target == MissionState.ENTER_CORRIDOR:

                self.transition(
                    MissionState.ENTER_CORRIDOR,
                    output.reason
                    or "entry geometry locked",
                )

                return

            if (
                target
                == MissionState.HOVER_AND_REASSESS
            ):

                self.start_reassess(
                    MissionState.PRE_ENTRY_GEOMETRY_LOCK,
                    output.reason
                    or "ENTRY_LOCK_FAILED",
                )

                return

        # ----------------------------------------------------
        # CRUISE
        # ----------------------------------------------------

        if (
            self.state
            == MissionState.CORRIDOR_CRUISE
        ):

            if (
                target
                == MissionState.OBSTACLE_DECISION
            ):

                self.transition(
                    MissionState.OBSTACLE_DECISION,
                    output.reason
                    or "front obstacle confirmed",
                )

                return

            if (
                target
                == MissionState.EXIT_DETECTION
            ):

                self.transition(
                    MissionState.EXIT_DETECTION,
                    output.reason
                    or "exit candidate confirmed",
                )

                return

            if (
                target
                == MissionState.HOVER_AND_REASSESS
            ):

                self.start_reassess(
                    MissionState.CORRIDOR_CRUISE,
                    output.reason
                    or "LOW_CONFIDENCE_GEOMETRY",
                )

                return

        # ----------------------------------------------------
        # OBSTACLE controller
        # ----------------------------------------------------

        if (
            self.state
            == MissionState.OBSTACLE_DECISION
        ):

            if (
                target
                == MissionState.CORRIDOR_CRUISE
            ):

                self.transition(
                    MissionState.CORRIDOR_CRUISE,
                    output.reason
                    or "obstacle bypass complete",
                )

                return

            if (
                target
                == MissionState.EXIT_DETECTION
            ):

                self.transition(
                    MissionState.EXIT_DETECTION,
                    output.reason,
                )

                return

            if (
                target
                == MissionState.HOVER_AND_REASSESS
            ):

                self.start_reassess(
                    source,
                    output.reason
                    or "NO_SAFE_SIDE_OR_BYPASS_FAILURE",
                )

                return

        # ----------------------------------------------------
        # EXIT
        # ----------------------------------------------------

        if (
            self.state
            == MissionState.EXIT_DETECTION
        ):

            if (
                target
                == MissionState.CORRIDOR_EXITED
            ):

                self.complete(
                    output.reason
                    or "corridor exit complete"
                )

                return

            if (
                target
                == MissionState.CORRIDOR_CRUISE
            ):

                self.transition(
                    MissionState.CORRIDOR_CRUISE,
                    output.reason
                    or "exit candidate rejected",
                )

                return

            if (
                target
                == MissionState.HOVER_AND_REASSESS
            ):

                self.start_reassess(
                    MissionState.EXIT_DETECTION,
                    output.reason
                    or "AMBIGUOUS_EXIT",
                )

                return

        # ----------------------------------------------------
        # REASSESS
        # ----------------------------------------------------

        if (
            self.state
            == MissionState.HOVER_AND_REASSESS
        ):

            if (
                target
                == MissionState.ABORT_CORRIDOR
            ):

                self.abort(
                    output.reason
                    or "reassessment exhausted safe recovery"
                )

                return

            if target in (
                MissionState.AVOID_LEFT,
                MissionState.AVOID_RIGHT,
                MissionState.OBSTACLE_DECISION,
            ):

                # Reassessment recommendation must be revalidated by
                # obstacle controller from OBSTACLE_DECISION.
                self.transition(
                    MissionState.OBSTACLE_DECISION,
                    (
                        f"reassessment recommends "
                        f"{target.value}; "
                        "obstacle controller will revalidate"
                    ),
                )

                return

            if target in (
                MissionState.PRE_ENTRY_GEOMETRY_LOCK,
                MissionState.ENTER_CORRIDOR,
                MissionState.CORRIDOR_CRUISE,
                MissionState.EXIT_DETECTION,
            ):

                self.transition(
                    target,
                    output.reason
                    or "reassessment recovered mission state",
                )

                return

        # Anything unexpected fails closed.
        self.abort(
            "unsupported mission transition: "
            f"{source.value} -> {target.value}"
        )

    # ========================================================
    # One FSM update
    # ========================================================

    def step(
        self,
        scan: Optional[NativeScan],
        attitude: Optional[Attitude] = None,
        pose: Optional[VehiclePose] = None,
    ) -> ControllerOutput:

        # ----------------------------------------------------
        # Terminal success
        # ----------------------------------------------------

        if (
            self.state
            == MissionState.CORRIDOR_EXITED
        ):

            self.latest_command = (
                BodyVelocity.stop()
            )

            return ControllerOutput(
                command=self.latest_command,
                status=self.state.value,
                reason=self.terminal_reason,
            )

        # ----------------------------------------------------
        # Terminal abort
        #
        # Navigation is permanently over. The only remaining
        # semantic request is LAND.
        #
        # Nothing here transmits anything to the Pixhawk.
        # ----------------------------------------------------

        if (
            self.state
            == MissionState.ABORT_CORRIDOR
        ):

            abort_output = (
                self.abort_controller.step(
                    landed=False
                )
            )

            self.latest_command = (
                BodyVelocity.stop()
            )

            return ControllerOutput(
                command=self.latest_command,
                status=MissionState.ABORT_CORRIDOR.value,
                reason=self.terminal_reason,
                action=abort_output.action,
            )

        self.maybe_reset_recovery_chain()

        # Recovery bridge hard timeout.
        if (
            self.state
            == MissionState.HOVER_AND_REASSESS
            and self.state_age()
            > self.config.reassess_hard_timeout_s
        ):

            self.abort(
                "HOVER_AND_REASSESS hard timeout "
                f"after "
                f"{self.config.reassess_hard_timeout_s:.1f} s"
            )

            return ControllerOutput(
                command=BodyVelocity.stop(),
                status="ABORT_CORRIDOR",
            )

        # ----------------------------------------------------
        # PRE_ENTRY
        # ----------------------------------------------------

        if (
            self.state
            == MissionState.PRE_ENTRY_GEOMETRY_LOCK
        ):

            if scan is None:

                output = ControllerOutput(
                    command=BodyVelocity.stop(),
                    status="WAITING_FOR_SCAN",
                )

            else:

                output = self.pre_entry.step(
                    scan,
                    attitude,
                )

            # ------------------------------------------------
            # PRE_ENTRY HOLD liveness guard
            #
            # The controller itself intentionally allows a
            # transient HOLD to self-recover. However, the
            # original HOLD has no hard timeout. A fully
            # autonomous mission must not remain here forever.
            # ------------------------------------------------

            if (
                getattr(self.pre_entry.state, "name", "")
                == "HOLD"
                and self.pre_entry.failure_reason is None
                and self.pre_entry.state_age()
                >= self.config.pre_entry_hold_timeout_s
            ):

                hold_age = self.pre_entry.state_age()

                self.start_reassess(
                    MissionState.PRE_ENTRY_GEOMETRY_LOCK,
                    (
                        "PRE_ENTRY transient HOLD persisted "
                        f"{hold_age:.1f} s without safe recovery"
                    ),
                )

                return ControllerOutput(
                    command=BodyVelocity.stop(),
                    status=self.public_state().value,
                    reason="PRE_ENTRY HOLD timeout",
                )

        # ----------------------------------------------------
        # ENTER
        # ----------------------------------------------------

        elif (
            self.state
            == MissionState.ENTER_CORRIDOR
        ):

            output = (
                self.step_enter_corridor(
                    pose
                )
            )

        # ----------------------------------------------------
        # CRUISE
        # ----------------------------------------------------

        elif (
            self.state
            == MissionState.CORRIDOR_CRUISE
        ):

            if scan is None:

                output = ControllerOutput(
                    command=BodyVelocity.stop(),
                    status="WAITING_FOR_SCAN",
                )

            else:

                output = self.cruise.step(
                    scan,
                    attitude,
                )

        # ----------------------------------------------------
        # OBSTACLE / AVOID
        # ----------------------------------------------------

        elif (
            self.state
            == MissionState.OBSTACLE_DECISION
        ):

            if scan is None:

                output = ControllerOutput(
                    command=BodyVelocity.stop(),
                    status="WAITING_FOR_SCAN",
                )

            else:

                output = self.obstacle.step(
                    scan,
                    attitude,
                )

        # ----------------------------------------------------
        # EXIT
        # ----------------------------------------------------

        elif (
            self.state
            == MissionState.EXIT_DETECTION
        ):

            output = self.exit.step(
                scan=scan,
                pose=pose,
            )

        # ----------------------------------------------------
        # REASSESS
        # ----------------------------------------------------

        elif (
            self.state
            == MissionState.HOVER_AND_REASSESS
        ):

            output = self.reassess.step(
                scan
            )

        else:

            self.abort(
                "mission runner entered unsupported state "
                f"{self.state.value}"
            )

            output = ControllerOutput(
                command=BodyVelocity.stop(),
                status="ABORT_CORRIDOR",
            )

        self.latest_command = (
            output.command
        )

        self.handle_controller_transition(
            output
        )

        # A controller may have transitioned us into ABORT during
        # this exact update. Expose LAND immediately so callers
        # never miss the terminal action.
        if (
            self.state
            == MissionState.ABORT_CORRIDOR
        ):

            abort_output = (
                self.abort_controller.step(
                    landed=False
                )
            )

            self.latest_command = (
                BodyVelocity.stop()
            )

            return ControllerOutput(
                command=self.latest_command,
                next_state=None,
                status=MissionState.ABORT_CORRIDOR.value,
                reason=self.terminal_reason,
                confidence=None,
                action=abort_output.action,
            )

        # If transition occurred, do NOT leak the old
        # controller's command across the handoff.
        if self.state != self.public_state():
            pass

        if output.next_state is not None:
            self.latest_command = (
                BodyVelocity.stop()
            )

        return ControllerOutput(
            command=self.latest_command,
            next_state=None,
            status=self.public_state().value,
            reason=output.reason,
            confidence=output.confidence,
            action=output.action,
        )

    # ========================================================
    # Diagnostics
    # ========================================================

    def diagnostics(self) -> dict:

        command = self.latest_command

        return {

            "mission_state":
                self.public_state().value,

            "internal_phase":
                self.state.value,

            "state_age_s":
                round(
                    self.state_age(),
                    3,
                ),

            "command": {

                "vx":
                    round(
                        command.vx_m_s,
                        4,
                    ),

                "vy_left":
                    round(
                        command.vy_m_s,
                        4,
                    ),

                "vz_up":
                    round(
                        command.vz_m_s,
                        4,
                    ),

                "yaw_rate_rad_s":
                    round(
                        command.yaw_rate_rad_s,
                        4,
                    ),
            },

            "recovery": {

                "source":
                    (
                        self.reassess_source_state.value
                        if self.reassess_source_state
                        is not None
                        else None
                    ),

                "reason":
                    self.reassess_pause_reason
                    or None,

                "chain_cycles":
                    self.reassess_chain_cycles,

                "same_source_cycles":
                    self.reassess_same_source_cycles,

                "total_cycles":
                    self.reassess_total_cycles,
            },

            "terminal_reason":
                self.terminal_reason
                or None,
        }


# ============================================================
# Standalone dry-run
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--port",
        default="/dev/ttyUSB0",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=20.0,
    )

    parser.add_argument(
        "--yaw-offset",
        type=float,
        default=0.0,
    )

    # D500 sensor angles increase clockwise.
    # For the final forward-mounted installation this will normally
    # be enabled to convert to the FSM's CCW/FLU convention.
    parser.add_argument(
        "--invert-angle",
        action="store_true",
    )

    args = parser.parse_args()

    adapter = ScanAdapter(
        invert_angle=args.invert_angle,
        yaw_offset_deg=args.yaw_offset,
    )

    runner = NativeMissionRunner()

    print()
    print("==================================================")
    print(" NATIVE CORRIDOR MISSION RUNNER")
    print(" *** DRY RUN: NO PIXHAWK COMMAND OUTPUT ***")
    print("==================================================")
    print()

    started = time.monotonic()
    last_print = 0.0

    with D500Driver(
        port=args.port
    ) as lidar:

        while (
            time.monotonic()
            - started
            < args.duration
        ):

            raw = lidar.get_scan(
                timeout_s=1.0
            )

            scan = (
                adapter.convert(raw)
                if raw is not None
                else None
            )

            # No MAVLink yet:
            attitude = None
            pose = None

            output = runner.step(
                scan=scan,
                attitude=attitude,
                pose=pose,
            )

            now = time.monotonic()

            if (
                now - last_print
                >= 0.50
            ):

                d = runner.diagnostics()

                cmd = output.command

                print(
                    f"{d['mission_state']:<25} "
                    f"vx={cmd.vx_m_s:+.3f} "
                    f"vy={cmd.vy_m_s:+.3f} "
                    f"vz={cmd.vz_m_s:+.3f} "
                    f"yaw={cmd.yaw_rate_rad_s:+.3f} "
                    f"| recovery="
                    f"{d['recovery']['chain_cycles']}/"
                    f"{d['recovery']['total_cycles']}"
                )

                last_print = now

            if runner.state in (
                MissionState.ABORT_CORRIDOR,
                MissionState.CORRIDOR_EXITED,
            ):

                print()
                print(
                    "FINAL STATE:",
                    runner.state.value,
                )

                print(
                    "REASON:",
                    runner.terminal_reason,
                )

                break


if __name__ == "__main__":
    main()
