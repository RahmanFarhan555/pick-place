import math

import numpy as np
import pytest

from pickplace.calibration import (
    DEFAULT_PIXEL_POINTS,
    DEFAULT_ROBOT_POINTS,
    CalibrationError,
    PixelToRobot,
    normalize_angle_180,
    solve_homography,
)


@pytest.fixture
def cal():
    return PixelToRobot()


@pytest.mark.parametrize("pixel, robot", list(zip(DEFAULT_PIXEL_POINTS, DEFAULT_ROBOT_POINTS)))
def test_calibration_points_map_exactly(cal, pixel, robot):
    x, y = cal.to_robot(*pixel)
    assert x == pytest.approx(robot[0], abs=1e-6)
    assert y == pytest.approx(robot[1], abs=1e-6)


def test_reprojection_error_is_zero(cal):
    assert cal.reprojection_error() < 1e-6


def test_image_center(cal):
    # 0.5 mm/px, v -> x, u -> y
    assert cal.to_robot(320, 240) == pytest.approx((220.0, -40.0))


@pytest.mark.parametrize("u, v", [(0, 0), (639, 479), (17.5, 333.25), (600, 20)])
def test_matches_closed_form_affine(cal, u, v):
    expected = (150 + 0.5 * (v - 100), -150 + 0.5 * (u - 100))
    assert cal.to_robot(u, v) == pytest.approx(expected)


def test_axes_are_swapped(cal):
    """Right in the image is robot +y, down in the image is robot +x."""
    x0, y0 = cal.to_robot(300, 200)
    x_right, y_right = cal.to_robot(310, 200)
    x_down, y_down = cal.to_robot(300, 210)
    assert (x_right - x0, y_right - y0) == pytest.approx((0, 5), abs=1e-9)
    assert (x_down - x0, y_down - y0) == pytest.approx((5, 0), abs=1e-9)


@pytest.mark.parametrize("u, v", [(-1, 10), (10, 480), (640, 0), (float("nan"), 3)])
def test_rejects_pixels_outside_image(cal, u, v):
    with pytest.raises(ValueError):
        cal.to_robot(u, v)


def test_non_strict_allows_extrapolation(cal):
    assert cal.to_robot(-100, 100, strict=False) == pytest.approx((150, -250))


def test_degenerate_calibration_rejected():
    with pytest.raises(CalibrationError):
        solve_homography([(0, 0), (1, 1), (2, 2), (3, 3)], [(0, 0), (1, 1), (2, 2), (3, 3)])
    with pytest.raises(CalibrationError):
        solve_homography([(0, 0), (1, 0), (0, 1)], [(0, 0), (1, 0), (0, 1)])


def test_homography_handles_perspective():
    """It also works for a tilted camera."""
    h_true = np.array([[1.2, 0.1, 5.0], [-0.05, 0.9, -3.0], [1e-4, 2e-4, 1.0]])
    src = np.array([(10, 10), (600, 20), (620, 470), (5, 460), (320, 240)], float)
    hom = np.c_[src, np.ones(len(src))] @ h_true.T
    dst = hom[:, :2] / hom[:, 2:]
    h = solve_homography(src, dst)
    assert h == pytest.approx(h_true, rel=1e-6, abs=1e-9)


@pytest.mark.parametrize("img_angle, robot_yaw", [
    (0, 90 - 180),   # image right is robot +y: 90 deg, same as -90
    (90, 0),         # image down is robot +x: 0 deg
    (45, 45),        # diagonal
    (-45, -45),      # 135 deg, same as -45
])
def test_angle_conversion(cal, img_angle, robot_yaw):
    assert cal.angle_to_robot(320, 240, img_angle) == pytest.approx(robot_yaw, abs=1e-9)


def test_pick_and_approach_poses(cal):
    target = cal.pick_target(320, 240, angle_img_deg=90)
    assert (target.pick.x, target.pick.y, target.pick.z) == pytest.approx((220, -40, 20))
    assert (target.approach.x, target.approach.y, target.approach.z) == pytest.approx((220, -40, 70))
    assert target.approach.rz == target.pick.rz == pytest.approx(0, abs=1e-9)


def test_custom_heights(cal):
    target = cal.pick_target(100, 100, pick_z=5, approach_offset=30)
    assert target.pick.z == 5 and target.approach.z == 35


def test_approach_must_be_above(cal):
    with pytest.raises(ValueError):
        cal.pick_target(100, 100, approach_offset=-10)


@pytest.mark.parametrize("a, expected", [(0, 0), (90, -90), (-90, -90), (180, 0), (135, -45), (-100, 80)])
def test_normalize_angle(a, expected):
    assert normalize_angle_180(a) == pytest.approx(expected)
    assert -90 <= normalize_angle_180(a) < 90
    assert not math.isnan(normalize_angle_180(a))
