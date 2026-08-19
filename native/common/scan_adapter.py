#!/usr/bin/env python3

from __future__ import annotations

import math

import numpy as np

from native.common.types import NativeScan
from native.hardware.d500_driver import ScanFrame


class ScanAdapter:
    """
    Converts the raw D500 scan into the aircraft-body angular convention
    used by all corridor perception code.

    Raw D500 mounting can later be corrected using:

        invert_angle
        yaw_offset_deg
    """

    def __init__(
        self,
        invert_angle: bool = False,
        yaw_offset_deg: float = 0.0,
    ) -> None:

        self.invert_angle = bool(invert_angle)

        self.yaw_offset_rad = math.radians(
            float(yaw_offset_deg)
        )

    @staticmethod
    def wrap_pi(angle: np.ndarray) -> np.ndarray:

        return np.arctan2(
            np.sin(angle),
            np.cos(angle),
        )

    def convert(
        self,
        raw_scan: ScanFrame,
    ) -> NativeScan:

        angles = np.asarray(
            raw_scan.angles_rad,
            dtype=np.float64,
        ).copy()

        ranges = np.asarray(
            raw_scan.ranges_m,
            dtype=np.float64,
        ).copy()

        intensities = np.asarray(
            raw_scan.intensities
        ).copy()

        if self.invert_angle:
            angles *= -1.0

        angles += self.yaw_offset_rad

        angles = self.wrap_pi(angles)

        # Sort so angle ordering is deterministic:
        #
        # -pi ... 0 ... +pi
        #
        order = np.argsort(angles)

        angles = angles[order]
        ranges = ranges[order]
        intensities = intensities[order]

        return NativeScan(
            angles_rad=angles,
            ranges_m=ranges,
            intensities=intensities,
            timestamp=raw_scan.stamp_monotonic,
            range_min_m=raw_scan.range_min_m,
            range_max_m=raw_scan.range_max_m,
        )

