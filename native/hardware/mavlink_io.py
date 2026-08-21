#!/usr/bin/env python3

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

from pymavlink import mavutil

from native.common.types import Attitude


# ============================================================
# Telemetry data
# ============================================================

@dataclass(frozen=True)
class LocalPositionNED:
    x_m: float
    y_m: float
    z_m: float

    vx_m_s: float
    vy_m_s: float
    vz_m_s: float

    timestamp: float

    @property
    def age_s(self) -> float:
        return max(
            0.0,
            time.monotonic() - self.timestamp,
        )


@dataclass(frozen=True)
class MavlinkStatus:
    connected: bool = False

    armed: bool = False
    mode: str = "UNKNOWN"

    landed_state: Optional[int] = None

    gps_fix_type: Optional[int] = None
    satellites_visible: Optional[int] = None

    ekf_flags: int = 0

    battery_voltage_v: Optional[float] = None

    heartbeat_timestamp: float = 0.0

    @property
    def heartbeat_age_s(self) -> float:
        if self.heartbeat_timestamp <= 0.0:
            return float("inf")

        return max(
            0.0,
            time.monotonic()
            - self.heartbeat_timestamp,
        )


# ============================================================
# MAVLink interface
# ============================================================

class MavlinkIO:
    """
    Read-only Pixhawk interface.

    CURRENT VERSION:
        - reads telemetry
        - requests telemetry rates
        - exposes Attitude
        - exposes LOCAL_POSITION_NED if available
        - exposes EKF/GPS/landed state

    CURRENT VERSION DOES NOT:
        - arm
        - disarm
        - change mode
        - send velocity
        - send attitude
        - command LAND
        - command motors
    """

    def __init__(
        self,
        port: str,
        baud: int = 115200,
    ) -> None:

        self.port = port
        self.baud = int(baud)

        self.master = None

        self._lock = threading.Lock()
        self._running = False
        self._thread = None

        self._attitude: Optional[Attitude] = None
        self._local_position: Optional[
            LocalPositionNED
        ] = None

        self._status = MavlinkStatus()

    # ========================================================
    # Connection
    # ========================================================

    def connect(
        self,
        heartbeat_timeout_s: float = 10.0,
    ) -> None:

        print(
            f"[MAVLINK] opening {self.port}"
        )

        self.master = mavutil.mavlink_connection(
            self.port,
            baud=self.baud,
            autoreconnect=True,
            source_system=255,
        )

        heartbeat = (
            self.master.wait_heartbeat(
                timeout=heartbeat_timeout_s
            )
        )

        if heartbeat is None:
            raise RuntimeError(
                "Pixhawk heartbeat timeout"
            )

        now = time.monotonic()

        try:
            mode = mavutil.mode_string_v10(
                heartbeat
            )
        except Exception:
            mode = "UNKNOWN"

        armed = bool(
            heartbeat.base_mode
            & mavutil.mavlink.
            MAV_MODE_FLAG_SAFETY_ARMED
        )

        with self._lock:
            self._status = MavlinkStatus(
                connected=True,
                armed=armed,
                mode=mode,
                heartbeat_timestamp=now,
            )

        print(
            "[MAVLINK] heartbeat received "
            f"sys={self.master.target_system} "
            f"comp={self.master.target_component} "
            f"mode={mode} "
            f"armed={armed}"
        )

        self._request_telemetry()

        self._running = True

        self._thread = threading.Thread(
            target=self._reader_loop,
            name="mavlink-reader",
            daemon=True,
        )

        self._thread.start()

    def close(self) -> None:

        self._running = False

        if self._thread is not None:
            self._thread.join(
                timeout=1.0
            )

        if self.master is not None:
            try:
                self.master.close()
            except Exception:
                pass

        print(
            "[MAVLINK] connection closed"
        )

    def __enter__(self):
        self.connect()
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        self.close()

    # ========================================================
    # Telemetry requests
    # ========================================================

    def _request_message(
        self,
        message_id: int,
        hz: float,
    ) -> None:

        interval_us = int(
            1_000_000 / hz
        )

        self.master.mav.command_long_send(
            self.master.target_system,
            0,
            mavutil.mavlink.
            MAV_CMD_SET_MESSAGE_INTERVAL,
            0,
            message_id,
            interval_us,
            0,
            0,
            0,
            0,
            0,
        )

    def _request_telemetry(self) -> None:

        requests = (
            (
                mavutil.mavlink.
                MAVLINK_MSG_ID_ATTITUDE,
                10.0,
            ),
            (
                mavutil.mavlink.
                MAVLINK_MSG_ID_LOCAL_POSITION_NED,
                10.0,
            ),
            (
                mavutil.mavlink.
                MAVLINK_MSG_ID_GPS_RAW_INT,
                2.0,
            ),
            (
                mavutil.mavlink.
                MAVLINK_MSG_ID_EXTENDED_SYS_STATE,
                2.0,
            ),
            (
                mavutil.mavlink.
                MAVLINK_MSG_ID_EKF_STATUS_REPORT,
                2.0,
            ),
            (
                mavutil.mavlink.
                MAVLINK_MSG_ID_SYS_STATUS,
                1.0,
            ),
        )

        for message_id, hz in requests:
            self._request_message(
                message_id,
                hz,
            )

            time.sleep(0.03)

    # ========================================================
    # Reader
    # ========================================================

    def _reader_loop(self) -> None:

        while self._running:

            try:

                msg = self.master.recv_match(
                    blocking=True,
                    timeout=0.25,
                )

                if msg is None:
                    continue

                if msg.get_type() == "BAD_DATA":
                    continue

                self._handle_message(msg)

            except Exception as exc:

                print(
                    "[MAVLINK] reader error:",
                    exc,
                )

                time.sleep(0.1)

    def _handle_message(
        self,
        msg,
    ) -> None:

        msg_type = msg.get_type()
        now = time.monotonic()

        with self._lock:

            s = self._status

            if msg_type == "HEARTBEAT":

                try:
                    mode = (
                        mavutil.mode_string_v10(
                            msg
                        )
                    )
                except Exception:
                    mode = "UNKNOWN"

                armed = bool(
                    msg.base_mode
                    & mavutil.mavlink.
                    MAV_MODE_FLAG_SAFETY_ARMED
                )

                self._status = MavlinkStatus(
                    connected=True,
                    armed=armed,
                    mode=mode,
                    landed_state=s.landed_state,
                    gps_fix_type=s.gps_fix_type,
                    satellites_visible=(
                        s.satellites_visible
                    ),
                    ekf_flags=s.ekf_flags,
                    battery_voltage_v=(
                        s.battery_voltage_v
                    ),
                    heartbeat_timestamp=now,
                )

            elif msg_type == "ATTITUDE":

                self._attitude = Attitude(
                    roll_rad=float(msg.roll),
                    pitch_rad=float(msg.pitch),
                    yaw_rad=float(msg.yaw),
                    timestamp=now,
                )

            elif (
                msg_type
                == "LOCAL_POSITION_NED"
            ):

                self._local_position = (
                    LocalPositionNED(
                        x_m=float(msg.x),
                        y_m=float(msg.y),
                        z_m=float(msg.z),
                        vx_m_s=float(msg.vx),
                        vy_m_s=float(msg.vy),
                        vz_m_s=float(msg.vz),
                        timestamp=now,
                    )
                )

            elif msg_type == "GPS_RAW_INT":

                self._status = MavlinkStatus(
                    connected=s.connected,
                    armed=s.armed,
                    mode=s.mode,
                    landed_state=s.landed_state,
                    gps_fix_type=int(
                        msg.fix_type
                    ),
                    satellites_visible=int(
                        getattr(
                            msg,
                            "satellites_visible",
                            255,
                        )
                    ),
                    ekf_flags=s.ekf_flags,
                    battery_voltage_v=(
                        s.battery_voltage_v
                    ),
                    heartbeat_timestamp=(
                        s.heartbeat_timestamp
                    ),
                )

            elif (
                msg_type
                == "EKF_STATUS_REPORT"
            ):

                self._status = MavlinkStatus(
                    connected=s.connected,
                    armed=s.armed,
                    mode=s.mode,
                    landed_state=s.landed_state,
                    gps_fix_type=s.gps_fix_type,
                    satellites_visible=(
                        s.satellites_visible
                    ),
                    ekf_flags=int(msg.flags),
                    battery_voltage_v=(
                        s.battery_voltage_v
                    ),
                    heartbeat_timestamp=(
                        s.heartbeat_timestamp
                    ),
                )

            elif (
                msg_type
                == "EXTENDED_SYS_STATE"
            ):

                self._status = MavlinkStatus(
                    connected=s.connected,
                    armed=s.armed,
                    mode=s.mode,
                    landed_state=int(
                        msg.landed_state
                    ),
                    gps_fix_type=s.gps_fix_type,
                    satellites_visible=(
                        s.satellites_visible
                    ),
                    ekf_flags=s.ekf_flags,
                    battery_voltage_v=(
                        s.battery_voltage_v
                    ),
                    heartbeat_timestamp=(
                        s.heartbeat_timestamp
                    ),
                )

            elif msg_type == "SYS_STATUS":

                voltage = float(
                    msg.voltage_battery
                ) / 1000.0

                self._status = MavlinkStatus(
                    connected=s.connected,
                    armed=s.armed,
                    mode=s.mode,
                    landed_state=s.landed_state,
                    gps_fix_type=s.gps_fix_type,
                    satellites_visible=(
                        s.satellites_visible
                    ),
                    ekf_flags=s.ekf_flags,
                    battery_voltage_v=voltage,
                    heartbeat_timestamp=(
                        s.heartbeat_timestamp
                    ),
                )

    # ========================================================
    # Public telemetry API
    # ========================================================

    def attitude(
        self,
    ) -> Optional[Attitude]:

        with self._lock:

            if self._attitude is None:
                return None

            a = self._attitude

            return Attitude(
                roll_rad=a.roll_rad,
                pitch_rad=a.pitch_rad,
                yaw_rad=a.yaw_rad,
                timestamp=a.timestamp,
            )

    def local_position(
        self,
    ) -> Optional[LocalPositionNED]:

        with self._lock:
            return self._local_position

    def status(
        self,
    ) -> MavlinkStatus:

        with self._lock:
            return self._status

    # ========================================================
    # Navigation-health checks
    # ========================================================

    def horizontal_position_ok(
        self,
    ) -> bool:
        """
        Require EKF horizontal position validity AND
        a recent LOCAL_POSITION_NED message.
        """

        with self._lock:

            flags = self._status.ekf_flags
            position = self._local_position

        # ArduPilot EKF_STATUS_FLAGS:
        # bit 3 = relative horizontal position
        # bit 4 = absolute horizontal position
        horizontal_position_valid = bool(
            flags & (8 | 16)
        )

        if not horizontal_position_valid:
            return False

        if position is None:
            return False

        if position.age_s > 0.5:
            return False

        return True

    def heartbeat_ok(
        self,
        timeout_s: float = 2.5,
    ) -> bool:

        status = self.status()

        return (
            status.connected
            and status.heartbeat_age_s
            <= timeout_s
        )
