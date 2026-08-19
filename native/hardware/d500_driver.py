#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import struct
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import numpy as np
import serial


# ============================================================
# D500 / LDROBOT packet protocol
# ============================================================

PACKET_HEADER = 0x54
PACKET_VER_LEN = 0x2C
PACKET_SIZE = 47
POINTS_PER_PACKET = 12

DEFAULT_BAUD = 230400

# Values previously seen from your D500 ROS driver.
DEFAULT_RANGE_MIN_M = 0.02
DEFAULT_RANGE_MAX_M = 12.0


# ============================================================
# Data types used by the NON-ROS corridor stack
# ============================================================

@dataclass(frozen=True)
class LidarPoint:
    angle_deg: float
    distance_m: float
    intensity: int


@dataclass
class ScanFrame:
    """
    One assembled 360-degree LiDAR revolution.

    angles_rad and ranges_m have the same length.

    Sensor angular convention is deliberately left RAW here.
    Mounting correction / FLU conversion will be applied later
    in the corridor perception layer.
    """

    angles_rad: np.ndarray
    ranges_m: np.ndarray
    intensities: np.ndarray

    stamp_monotonic: float
    scan_time_s: float

    range_min_m: float = DEFAULT_RANGE_MIN_M
    range_max_m: float = DEFAULT_RANGE_MAX_M

    @property
    def size(self) -> int:
        return int(self.ranges_m.size)

    @property
    def age_s(self) -> float:
        return max(0.0, time.monotonic() - self.stamp_monotonic)

    def valid_mask(self) -> np.ndarray:
        return (
            np.isfinite(self.ranges_m)
            & (self.ranges_m >= self.range_min_m)
            & (self.ranges_m <= self.range_max_m)
        )

    def sector_range(
        self,
        centre_deg: float,
        half_width_deg: float = 4.0,
        percentile: float = 25.0,
    ) -> Optional[float]:
        """
        Return a robust range estimate inside an angular sector.

        Example:
            centre_deg = 0    -> nominal forward sector
            centre_deg = 90   -> nominal +90 degree sector

        This does NOT yet guarantee those directions correspond to the
        aircraft's forward/left axes. We verify mounting orientation later.
        """

        if self.size == 0:
            return None

        centre = math.radians(centre_deg)
        half_width = math.radians(half_width_deg)

        angular_error = np.arctan2(
            np.sin(self.angles_rad - centre),
            np.cos(self.angles_rad - centre),
        )

        mask = (
            (np.abs(angular_error) <= half_width)
            & self.valid_mask()
        )

        values = self.ranges_m[mask]

        if values.size == 0:
            return None

        return float(np.percentile(values, percentile))


# ============================================================
# CRC-8
# ============================================================

def crc8_ldrobot(data: bytes) -> int:
    """
    LDROBOT CRC-8.

    Polynomial: 0x4D
    Initial value: 0x00

    Equivalent to the vendor lookup table implementation.
    """

    crc = 0

    for byte in data:
        crc ^= byte

        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x4D) & 0xFF
            else:
                crc = (crc << 1) & 0xFF

    return crc


# ============================================================
# D500 serial driver
# ============================================================

class D500Driver:
    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baudrate: int = DEFAULT_BAUD,
        range_min_m: float = DEFAULT_RANGE_MIN_M,
        range_max_m: float = DEFAULT_RANGE_MAX_M,
    ) -> None:

        self.port = port
        self.baudrate = int(baudrate)

        self.range_min_m = float(range_min_m)
        self.range_max_m = float(range_max_m)

        self.serial: Optional[serial.Serial] = None

        self.rx_buffer = bytearray()

        # Points currently being assembled into one revolution.
        self.current_scan_points: List[LidarPoint] = []

        # Fully completed scans waiting for the caller.
        self.scan_queue: Deque[ScanFrame] = deque(maxlen=3)

        self.last_point_angle_deg: Optional[float] = None
        self.have_seen_first_wrap = False

        self.last_scan_stamp: Optional[float] = None

        # Diagnostics.
        self.valid_packets = 0
        self.bad_crc_packets = 0
        self.discarded_bytes = 0

    # --------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------

    def open(self) -> None:
        if self.serial is not None and self.serial.is_open:
            return

        self.serial = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=0.05,
        )

        # Throw away anything left from an old partial read.
        self.serial.reset_input_buffer()

        self.rx_buffer.clear()
        self.current_scan_points.clear()
        self.scan_queue.clear()

        self.last_point_angle_deg = None
        self.have_seen_first_wrap = False
        self.last_scan_stamp = None

    def close(self) -> None:
        if self.serial is not None:
            self.serial.close()

        self.serial = None

    def __enter__(self) -> "D500Driver":
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    # --------------------------------------------------------
    # Public scan interface
    # --------------------------------------------------------

    def get_scan(self, timeout_s: float = 1.0) -> Optional[ScanFrame]:
        """
        Wait for one complete 360-degree scan.

        Returns None if a complete scan is not received before timeout.
        """

        if self.serial is None or not self.serial.is_open:
            raise RuntimeError("D500 serial port is not open")

        # A previously assembled scan may already be waiting.
        if self.scan_queue:
            return self.scan_queue.popleft()

        deadline = time.monotonic() + timeout_s

        while time.monotonic() < deadline:

            packet = self._read_packet(deadline)

            if packet is None:
                continue

            points = self._decode_packet(packet)

            if points:
                self._consume_points(points)

            if self.scan_queue:
                return self.scan_queue.popleft()

        return None

    # --------------------------------------------------------
    # Serial packet extraction
    # --------------------------------------------------------

    def _read_packet(self, deadline: float) -> Optional[bytes]:

        assert self.serial is not None

        while time.monotonic() < deadline:

            # First see whether a complete frame already exists in buffer.
            packet = self._extract_packet_from_buffer()

            if packet is not None:
                return packet

            waiting = self.serial.in_waiting

            if waiting > 0:
                chunk = self.serial.read(min(waiting, 4096))
            else:
                chunk = self.serial.read(1)

            if chunk:
                self.rx_buffer.extend(chunk)

        return None

    def _extract_packet_from_buffer(self) -> Optional[bytes]:

        header = bytes((PACKET_HEADER, PACKET_VER_LEN))

        while True:

            if len(self.rx_buffer) < 2:
                return None

            index = self.rx_buffer.find(header)

            if index < 0:
                # Preserve a trailing 0x54 because it may be the first
                # byte of a header split across two serial reads.
                if self.rx_buffer[-1] == PACKET_HEADER:
                    self.discarded_bytes += len(self.rx_buffer) - 1
                    self.rx_buffer[:] = self.rx_buffer[-1:]
                else:
                    self.discarded_bytes += len(self.rx_buffer)
                    self.rx_buffer.clear()

                return None

            if index > 0:
                self.discarded_bytes += index
                del self.rx_buffer[:index]

            if len(self.rx_buffer) < PACKET_SIZE:
                return None

            candidate = bytes(self.rx_buffer[:PACKET_SIZE])

            calculated_crc = crc8_ldrobot(candidate[:-1])
            received_crc = candidate[-1]

            if calculated_crc != received_crc:
                self.bad_crc_packets += 1

                # Shift one byte and search for a new header.
                del self.rx_buffer[0]

                continue

            del self.rx_buffer[:PACKET_SIZE]

            self.valid_packets += 1

            return candidate

    # --------------------------------------------------------
    # Packet -> points
    # --------------------------------------------------------

    def _decode_packet(self, packet: bytes) -> List[LidarPoint]:

        if len(packet) != PACKET_SIZE:
            return []

        header, ver_len, speed, start_angle_raw = struct.unpack_from(
            "<BBHH",
            packet,
            0,
        )

        if header != PACKET_HEADER or ver_len != PACKET_VER_LEN:
            return []

        offset = 6

        measurements = []

        for _ in range(POINTS_PER_PACKET):
            distance_mm, intensity = struct.unpack_from(
                "<HB",
                packet,
                offset,
            )

            offset += 3

            measurements.append((distance_mm, intensity))

        end_angle_raw, sensor_timestamp_ms = struct.unpack_from(
            "<HH",
            packet,
            offset,
        )

        # Angles are represented in hundredths of a degree.
        angular_difference_raw = (
            end_angle_raw
            + 36000
            - start_angle_raw
        ) % 36000

        step_deg = (
            angular_difference_raw
            / 100.0
            / (POINTS_PER_PACKET - 1)
        )

        start_deg = start_angle_raw / 100.0

        points: List[LidarPoint] = []

        for index, (distance_mm, intensity) in enumerate(measurements):

            angle_deg = start_deg + index * step_deg

            if angle_deg >= 360.0:
                angle_deg -= 360.0

            distance_m = distance_mm / 1000.0

            points.append(
                LidarPoint(
                    angle_deg=angle_deg,
                    distance_m=distance_m,
                    intensity=intensity,
                )
            )

        return points

    # --------------------------------------------------------
    # Points -> complete 360-degree scan
    # --------------------------------------------------------

    def _consume_points(self, points: List[LidarPoint]) -> None:

        for point in points:

            wrapped = (
                self.last_point_angle_deg is not None
                and point.angle_deg < 20.0
                and self.last_point_angle_deg > 340.0
            )

            if wrapped:

                if self.have_seen_first_wrap:

                    if len(self.current_scan_points) >= 100:
                        scan = self._build_scan(
                            self.current_scan_points
                        )

                        if scan is not None:
                            self.scan_queue.append(scan)

                else:
                    # Ignore the initial partial revolution after opening
                    # the serial connection.
                    self.have_seen_first_wrap = True

                self.current_scan_points = []

            self.current_scan_points.append(point)
            self.last_point_angle_deg = point.angle_deg

    def _build_scan(
        self,
        points: List[LidarPoint],
    ) -> Optional[ScanFrame]:

        if len(points) < 100:
            return None

        # Sort by angle to guarantee deterministic geometry input.
        ordered = sorted(
            points,
            key=lambda p: p.angle_deg,
        )

        angles = np.asarray(
            [math.radians(p.angle_deg) for p in ordered],
            dtype=np.float64,
        )

        ranges = np.asarray(
            [p.distance_m for p in ordered],
            dtype=np.float64,
        )

        intensities = np.asarray(
            [p.intensity for p in ordered],
            dtype=np.uint8,
        )

        now = time.monotonic()

        if self.last_scan_stamp is None:
            scan_time = 0.0
        else:
            scan_time = now - self.last_scan_stamp

        self.last_scan_stamp = now

        return ScanFrame(
            angles_rad=angles,
            ranges_m=ranges,
            intensities=intensities,
            stamp_monotonic=now,
            scan_time_s=scan_time,
            range_min_m=self.range_min_m,
            range_max_m=self.range_max_m,
        )


# ============================================================
# Standalone hardware test
# ============================================================

def fmt(value: Optional[float]) -> str:
    if value is None:
        return "----"

    return f"{value:5.2f} m"


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--port",
        default="/dev/ttyUSB0",
    )

    parser.add_argument(
        "--baud",
        type=int,
        default=DEFAULT_BAUD,
    )

    parser.add_argument(
        "--scans",
        type=int,
        default=30,
    )

    args = parser.parse_args()

    print()
    print("D500 native driver test")
    print("=======================")
    print(f"Port : {args.port}")
    print(f"Baud : {args.baud}")
    print()

    timestamps: Deque[float] = deque(maxlen=20)

    try:

        with D500Driver(
            port=args.port,
            baudrate=args.baud,
        ) as lidar:

            for scan_number in range(1, args.scans + 1):

                scan = lidar.get_scan(timeout_s=2.0)

                if scan is None:
                    print(
                        f"[{scan_number:03d}] "
                        f"TIMEOUT waiting for complete revolution"
                    )
                    continue

                timestamps.append(scan.stamp_monotonic)

                if len(timestamps) >= 2:
                    elapsed = timestamps[-1] - timestamps[0]

                    if elapsed > 0:
                        hz = (len(timestamps) - 1) / elapsed
                    else:
                        hz = 0.0
                else:
                    hz = 0.0

                valid = scan.ranges_m[scan.valid_mask()]

                if valid.size:
                    nearest = float(np.min(valid))
                    farthest = float(np.max(valid))
                else:
                    nearest = math.nan
                    farthest = math.nan

                sector_0 = scan.sector_range(0.0)
                sector_90 = scan.sector_range(90.0)
                sector_180 = scan.sector_range(180.0)
                sector_270 = scan.sector_range(270.0)

                print(
                    f"[{scan_number:03d}] "
                    f"{hz:5.2f} Hz | "
                    f"{scan.size:4d} pts | "
                    f"min={nearest:5.2f} m | "
                    f"max={farthest:5.2f} m | "
                    f"0°={fmt(sector_0)} | "
                    f"90°={fmt(sector_90)} | "
                    f"180°={fmt(sector_180)} | "
                    f"270°={fmt(sector_270)}"
                )

            print()
            print("Packet diagnostics")
            print("------------------")
            print(f"Valid packets : {lidar.valid_packets}")
            print(f"Bad CRC       : {lidar.bad_crc_packets}")
            print(f"Discarded B   : {lidar.discarded_bytes}")

    except serial.SerialException as exc:
        print()
        print(f"SERIAL ERROR: {exc}")
        raise SystemExit(1)

    except KeyboardInterrupt:
        print()
        print("Stopped.")


if __name__ == "__main__":
    main()
