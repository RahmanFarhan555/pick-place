"""Pixel to robot coordinates.

The camera looks straight down, so the image and the table are two flat
planes. I calculate a homography from the 4 calibration points. For these
points it gives the same result as:

    x = 150 + 0.5 * (v - 100)
    y = -150 + 0.5 * (u - 100)

So 1 pixel is 0.5 mm, image v is robot x and image u is robot y.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

# Calibration points from the assignment: pixel (u, v) -> robot (x, y) in mm
DEFAULT_PIXEL_POINTS: list[tuple[float, float]] = [
    (100, 100), (540, 100), (540, 380), (100, 380),
]
DEFAULT_ROBOT_POINTS: list[tuple[float, float]] = [
    (150, -150), (150, 70), (290, 70), (290, -150),
]

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480


class CalibrationError(ValueError):
    """The calibration points can not be used."""


def solve_homography(src: Sequence[Sequence[float]],
                     dst: Sequence[Sequence[float]]) -> np.ndarray:
    """Calculate the 3x3 homography from the point pairs (needs at least 4)."""
    src_a = np.asarray(src, dtype=float)
    dst_a = np.asarray(dst, dtype=float)
    if src_a.shape != dst_a.shape or src_a.ndim != 2 or src_a.shape[1] != 2:
        raise CalibrationError("src and dst must both be Nx2 arrays of equal size")
    if len(src_a) < 4:
        raise CalibrationError("at least 4 point pairs are needed")

    rows = []
    for (u, v), (x, y) in zip(src_a, dst_a):
        rows.append([-u, -v, -1, 0, 0, 0, u * x, v * x, x])
        rows.append([0, 0, 0, -u, -v, -1, u * y, v * y, y])
    a = np.asarray(rows)

    # points on one line can not be solved
    if np.linalg.matrix_rank(a) < 8:
        raise CalibrationError("calibration points are degenerate (collinear?)")

    _, _, vt = np.linalg.svd(a)
    h = vt[-1].reshape(3, 3)
    if abs(h[2, 2]) < 1e-12:
        raise CalibrationError("homography is singular")
    return h / h[2, 2]


@dataclass(frozen=True)
class Pose:
    """Tool pose in the robot frame. x, y, z in mm, rz in degrees."""
    x: float
    y: float
    z: float
    rz: float = 0.0

    def offset(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> "Pose":
        return Pose(self.x + dx, self.y + dy, self.z + dz, self.rz)


@dataclass(frozen=True)
class PickTarget:
    pick: Pose
    approach: Pose


class PixelToRobot:
    """Converts pixel coordinates to robot coordinates in mm."""

    def __init__(self,
                 pixel_points: Iterable[Sequence[float]] = DEFAULT_PIXEL_POINTS,
                 robot_points: Iterable[Sequence[float]] = DEFAULT_ROBOT_POINTS,
                 image_size: tuple[int, int] = (IMAGE_WIDTH, IMAGE_HEIGHT)):
        self.pixel_points = np.asarray(list(pixel_points), dtype=float)
        self.robot_points = np.asarray(list(robot_points), dtype=float)
        self.image_size = image_size
        self.H = solve_homography(self.pixel_points, self.robot_points)

    def to_robot(self, u: float, v: float, *, strict: bool = True) -> tuple[float, float]:
        """Convert one pixel (u, v) to robot (x, y) in mm.

        With strict=True a pixel outside the image gives an error, because a
        real detection can not be outside the image.
        """
        if not (np.isfinite(u) and np.isfinite(v)):
            raise ValueError(f"pixel coordinate is not finite: ({u}, {v})")
        w, h = self.image_size
        if strict and not (0 <= u <= w - 1 and 0 <= v <= h - 1):
            raise ValueError(f"pixel ({u}, {v}) is outside the {w}x{h} image")
        p = self.H @ np.array([u, v, 1.0])
        return float(p[0] / p[2]), float(p[1] / p[2])

    def reprojection_error(self) -> float:
        """Biggest error in mm when the calibration pixels are converted."""
        mapped = np.array([self.to_robot(u, v, strict=False) for u, v in self.pixel_points])
        return float(np.max(np.linalg.norm(mapped - self.robot_points, axis=1)))

    def angle_to_robot(self, u: float, v: float, angle_img_deg: float) -> float:
        """Convert an angle in the image to an angle in the robot frame.

        The axes are swapped, so the angle can not be copied. I convert two
        points on the long axis and measure the angle between them. The result
        is between -90 and 90, because a rectangle is the same after 180 deg.
        """
        a = np.deg2rad(angle_img_deg)
        step = 10.0
        x0, y0 = self.to_robot(u, v, strict=False)
        x1, y1 = self.to_robot(u + step * np.cos(a), v + step * np.sin(a), strict=False)
        yaw = np.rad2deg(np.arctan2(y1 - y0, x1 - x0))
        # round first, so 89.9999999 does not jump to -90
        return normalize_angle_180(round(float(yaw), 6))

    def pick_target(self, u: float, v: float, angle_img_deg: float = 0.0,
                    pick_z: float = 20.0, approach_offset: float = 50.0) -> PickTarget:
        """Pick pose at pick_z and approach pose approach_offset mm above it."""
        if approach_offset <= 0:
            raise ValueError("approach offset must be positive (above the part)")
        x, y = self.to_robot(u, v)
        rz = self.angle_to_robot(u, v, angle_img_deg)
        pick = Pose(x, y, pick_z, rz)
        return PickTarget(pick=pick, approach=pick.offset(dz=approach_offset))


def normalize_angle_180(angle_deg: float) -> float:
    """Bring an angle into the range -90 to 90 degrees."""
    return float((angle_deg + 90.0) % 180.0 - 90.0)
