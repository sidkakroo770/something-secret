#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import threading
import time
from dataclasses import dataclass
from typing import Optional

from native.common.scan_adapter import ScanAdapter
from native.common.types import (
    BodyVelocity,
    MissionState,
    VehicleAction,
)

from native.hardware.d500_driver import D500Driver
from native.hardware.mavlink_io import MavlinkIO
from native.hardware.mavlink_sender import MavlinkCommandSender
from native.hardware.flight_controller_manager import (
    FlightControllerManager,
)

from native.mission_runner import (
    NativeMissionRunner,
    VehiclePose,
)


# ============================================================
# Runtime command cache
# ============================================================

@dataclass
class CachedCommand:
    command: BodyVelocity
    action: Optional[VehicleAction]
    timestamp: float
    sequence: int


class CommandCache:

    def __init__(self) -> None:

        self._lock = threading.Lock()

        self._value = CachedCommand(
            command=BodyVelocity.stop(),
            action=None,
            timestamp=time.monotonic(),
            sequence=0,
        )

    def update(
        self,
        command: BodyVelocity,
        action: Optional[VehicleAction],
    ) -> None:

        with self._lock:

            self._value = CachedCommand(
                command=command,
                action=action,
                timestamp=time.monotonic(),
                sequence=self._value.sequence + 1,
            )

    def snapshot(self) -> CachedCommand:

        with self._lock:

            value = self._value

            return CachedCommand(
                command=value.command,
                action=value.action,
                timestamp=value.timestamp,
                sequence=value.sequence,
            )


# ============================================================
# MAVLink output thread
# ============================================================

class CommandOutputThread:

    def __init__(
        self,
        sender: MavlinkCommandSender,
        cache: CommandCache,
        rate_hz: float = 20.0,
        command_timeout_s: float = 0.35,
    ) -> None:

        self.sender = sender
        self.cache = cache

        self.period_s = (
            1.0 / max(1.0, rate_hz)
        )

        self.command_timeout_s = (
            command_timeout_s
        )

        self.running = False

        self.thread: Optional[
            threading.Thread
        ] = None

        self.last_action_sequence = -1
        self.last_action_time = 0.0

    def start(self) -> None:

        self.running = True

        self.thread = threading.Thread(
            target=self._loop,
            name="mavlink-command-output",
            daemon=True,
        )

        self.thread.start()

    def stop(self) -> None:

        self.running = False

        if self.thread is not None:

            self.thread.join(
                timeout=1.0
            )

    def _loop(self) -> None:

        while self.running:

            started = time.monotonic()

            value = self.cache.snapshot()

            age = (
                time.monotonic()
                - value.timestamp
            )

            # ------------------------------------------------
            # LAND has priority over velocity.
            # ------------------------------------------------

            if value.action is not None:

                should_send = (
                    value.sequence
                    != self.last_action_sequence
                    or
                    time.monotonic()
                    - self.last_action_time
                    >= 1.0
                )

                if should_send:

                    result = (
                        self.sender.send_action(
                            value.action
                        )
                    )

                    print(
                        "[TX ACTION] "
                        f"{value.action.value}: "
                        f"tx={result.transmitted} "
                        f"reason={result.reason}"
                    )

                    self.last_action_sequence = (
                        value.sequence
                    )

                    self.last_action_time = (
                        time.monotonic()
                    )

            else:

                # --------------------------------------------
                # Command watchdog.
                #
                # Never repeat an old motion command forever.
                # --------------------------------------------

                if age > self.command_timeout_s:

                    command = (
                        BodyVelocity.stop()
                    )

                else:

                    command = value.command

                self.sender.send_velocity(
                    command
                )

            elapsed = (
                time.monotonic()
                - started
            )

            time.sleep(
                max(
                    0.0,
                    self.period_s - elapsed,
                )
            )


# ============================================================
# Pose adapter
# ============================================================

def get_vehicle_pose(
    fc: MavlinkIO,
) -> Optional[VehiclePose]:

    position = fc.local_position()
    attitude = fc.attitude()

    if position is None:
        return None

    if attitude is None:
        return None

    if position.age_s > 0.50:
        return None

    if attitude.age_s > 0.50:
        return None

    # LOCAL_POSITION_NED:
    #
    # x = North
    # y = East
    #
    # ATTITUDE yaw follows the NED navigation convention.
    #
    # MissionRunner's projection:
    #
    # cos(yaw)*dN + sin(yaw)*dE
    #
    # therefore directly gives displacement along the
    # vehicle's initial heading.

    return VehiclePose(
        x_m=position.x_m,
        y_m=position.y_m,
        yaw_rad=attitude.yaw_rad,
        timestamp=position.timestamp,
    )


# ============================================================
# Startup readiness
# ============================================================

def wait_for_attitude(
    fc: MavlinkIO,
    timeout_s: float = 5.0,
) -> bool:

    deadline = (
        time.monotonic()
        + timeout_s
    )

    while time.monotonic() < deadline:

        attitude = fc.attitude()

        if (
            attitude is not None
            and attitude.age_s <= 0.5
        ):

            return True

        time.sleep(0.05)

    return False


def wait_for_xy(
    fc: MavlinkIO,
    timeout_s: float,
) -> bool:

    deadline = (
        time.monotonic()
        + timeout_s
    )

    while time.monotonic() < deadline:

        if fc.horizontal_position_ok():
            return True

        time.sleep(0.1)

    return False


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lidar-port",
        default="/dev/ttyUSB0",
    )

    parser.add_argument(
        "--pixhawk-port",
        required=True,
    )

    parser.add_argument(
        "--invert-angle",
        action="store_true",
    )

    parser.add_argument(
        "--yaw-offset",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="0 means unlimited",
    )

    # --------------------------------------------------------
    # REAL CONTROL MUST BE EXPLICIT.
    # --------------------------------------------------------

    parser.add_argument(
        "--control",
        action="store_true",
        help="Allow real MAVLink flight-control transmission",
    )

    parser.add_argument(
        "--arm",
        action="store_true",
        help="Normal-arm automatically after GUIDED is confirmed",
    )

    args = parser.parse_args()

    print()
    print("======================================================")
    print(" SAE AEROTHON NATIVE CORRIDOR RUNTIME")
    print("======================================================")
    print()

    print(
        "Control:",
        "REAL" if args.control else "DRY RUN",
    )

    print(
        "Auto arm:",
        args.arm,
    )

    print()

    adapter = ScanAdapter(
        invert_angle=args.invert_angle,
        yaw_offset_deg=args.yaw_offset,
    )

    runner = NativeMissionRunner()

    cache = CommandCache()

    output_thread = None
    manager = None
    sender = None

    started = time.monotonic()

    last_scan_time = time.monotonic()

    xy_bad_since: Optional[float] = None

    terminal_seen = False

    try:

        with MavlinkIO(
            args.pixhawk_port
        ) as fc:

            manager = (
                FlightControllerManager(fc)
            )

            sender = (
                MavlinkCommandSender(fc)
            )

            print(
                "[STARTUP] waiting for Pixhawk attitude..."
            )

            if not wait_for_attitude(fc):

                raise RuntimeError(
                    "No fresh Pixhawk attitude"
                )

            print(
                "[STARTUP] attitude OK"
            )

            # ------------------------------------------------
            # REAL CONTROL STARTUP
            # ------------------------------------------------

            if args.control:

                print(
                    "[STARTUP] waiting for valid EKF XY..."
                )

                if not wait_for_xy(
                    fc,
                    timeout_s=10.0,
                ):

                    raise RuntimeError(
                        "No valid horizontal EKF position. "
                        "Real control refused."
                    )

                print(
                    "[STARTUP] EKF XY OK"
                )

                if not manager.request_mode(
                    "GUIDED"
                ):

                    raise RuntimeError(
                        "GUIDED mode could not be confirmed"
                    )

                if args.arm:

                    if not manager.arm():

                        raise RuntimeError(
                            "Normal arm was rejected"
                        )

                sender.enable_control(
                    sender.ENABLE_CONFIRMATION
                )

            else:

                print()
                print(
                    "*** DRY RUN ***"
                )

                print(
                    "FSM output will reach the MAVLink "
                    "sender, but transmission is disabled."
                )

            # ------------------------------------------------
            # Start independent 20 Hz output thread.
            # ------------------------------------------------

            output_thread = (
                CommandOutputThread(
                    sender=sender,
                    cache=cache,
                    rate_hz=20.0,
                    command_timeout_s=0.35,
                )
            )

            output_thread.start()

            # ------------------------------------------------
            # LiDAR / FSM loop
            #
            # One FSM iteration per NEW D500 scan.
            # ------------------------------------------------

            with D500Driver(
                port=args.lidar_port
            ) as lidar:

                print()
                print(
                    "[STARTUP] D500 connected"
                )

                print(
                    "[MISSION] corridor runtime active"
                )

                print()

                while True:

                    if (
                        args.duration > 0.0
                        and
                        time.monotonic()
                        - started
                        >= args.duration
                    ):

                        print(
                            "[RUNTIME] duration complete"
                        )

                        break

                    raw = lidar.get_scan(
                        timeout_s=0.25
                    )

                    now = time.monotonic()

                    # ----------------------------------------
                    # LiDAR watchdog
                    # ----------------------------------------

                    if raw is None:

                        lidar_age = (
                            now - last_scan_time
                        )

                        if (
                            lidar_age > 0.50
                            and runner.state
                            not in (
                                MissionState.HOVER_AND_REASSESS,
                                MissionState.ABORT_CORRIDOR,
                                MissionState.CORRIDOR_EXITED,
                            )
                        ):

                            print(
                                "[SAFETY] D500 stale "
                                f"{lidar_age:.2f}s"
                            )

                            runner.start_reassess(
                                runner.public_state(),
                                (
                                    "D500 scan stale "
                                    f"{lidar_age:.2f}s"
                                ),
                            )

                            cache.update(
                                BodyVelocity.stop(),
                                None,
                            )

                        continue

                    last_scan_time = now

                    scan = adapter.convert(raw)

                    attitude = fc.attitude()

                    pose = get_vehicle_pose(fc)

                    # ----------------------------------------
                    # Pixhawk heartbeat
                    # ----------------------------------------

                    if not fc.heartbeat_ok():

                        print(
                            "[SAFETY] Pixhawk heartbeat stale"
                        )

                        cache.update(
                            BodyVelocity.stop(),
                            None,
                        )

                        continue

                    # ----------------------------------------
                    # XY-loss safety during REAL control
                    # ----------------------------------------

                    if args.control:

                        if not fc.horizontal_position_ok():

                            if xy_bad_since is None:

                                xy_bad_since = (
                                    time.monotonic()
                                )

                            xy_bad_age = (
                                time.monotonic()
                                - xy_bad_since
                            )

                            cache.update(
                                BodyVelocity.stop(),
                                None,
                            )

                            if (
                                xy_bad_age >= 1.0
                                and runner.state
                                != MissionState.ABORT_CORRIDOR
                            ):

                                runner.abort(
                                    "horizontal EKF position "
                                    "lost for >=1.0 s"
                                )

                        else:

                            xy_bad_since = None

                    # ----------------------------------------
                    # FSM
                    # ----------------------------------------

                    output = runner.step(
                        scan=scan,
                        attitude=attitude,
                        pose=pose,
                    )

                    cache.update(
                        output.command,
                        output.action,
                    )

                    d = runner.diagnostics()

                    print(
                        f"{d['mission_state']:<25} "
                        f"vx={output.command.vx_m_s:+.3f} "
                        f"vy={output.command.vy_m_s:+.3f} "
                        f"vz={output.command.vz_m_s:+.3f} "
                        f"yaw="
                        f"{math.degrees(output.command.yaw_rate_rad_s):+.1f}°/s "
                        f"| XY={fc.horizontal_position_ok()} "
                        f"| mode={fc.status().mode}"
                    )

                    # ----------------------------------------
                    # ABORT
                    # ----------------------------------------

                    if (
                        runner.state
                        == MissionState.ABORT_CORRIDOR
                    ):

                        print()
                        print(
                            "[MISSION] ABORT_CORRIDOR"
                        )

                        print(
                            "[MISSION] LAND action handed "
                            "to output thread"
                        )

                        terminal_seen = True

                        # Give output thread time to dispatch LAND.
                        time.sleep(1.5)

                        break

                    # ----------------------------------------
                    # Corridor success
                    # ----------------------------------------

                    if (
                        runner.state
                        == MissionState.CORRIDOR_EXITED
                    ):

                        cache.update(
                            BodyVelocity.stop(),
                            None,
                        )

                        print()
                        print(
                            "[MISSION] CORRIDOR_EXITED"
                        )

                        print(
                            "[MISSION] corridor module complete"
                        )

                        terminal_seen = True

                        time.sleep(0.5)

                        break

    except KeyboardInterrupt:

        print()
        print(
            "[RUNTIME] KeyboardInterrupt"
        )

        # Never disarm a potentially flying aircraft here.
        #
        # If real control is active and aircraft is armed,
        # request LAND instead.

        if (
            args.control
            and manager is not None
            and manager.fc.status().armed
        ):

            print(
                "[SAFETY] requesting LAND"
            )

            try:
                manager.land()
            except Exception as exc:
                print(
                    "[SAFETY] LAND request failed:",
                    exc,
                )

    except Exception as exc:

        print()
        print(
            "[RUNTIME ERROR]",
            exc,
        )

        # If we have already armed, do not respond to an exception
        # by disarming in flight. LAND is the safe terminal action.

        if (
            args.control
            and manager is not None
        ):

            try:

                if manager.fc.status().armed:

                    print(
                        "[SAFETY] runtime failure "
                        "while armed -> LAND"
                    )

                    manager.land()

            except Exception as land_exc:

                print(
                    "[SAFETY] LAND request failed:",
                    land_exc,
                )

    finally:

        if output_thread is not None:

            output_thread.stop()

        if sender is not None:

            sender.disable_control()

        print()
        print(
            "[RUNTIME] shutdown complete"
        )


if __name__ == "__main__":
    main()
