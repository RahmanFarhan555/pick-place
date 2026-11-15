"""Tests for the paper crop."""
import cv2
import numpy as np
import pytest

from pickplace.paper import PaperNotFound, order_corners, rectify_paper
from pickplace.synthetic import DEFAULT_PARTS, render_scene
from pickplace.vision import detect_objects


def handheld_photo(scene):
    """Put the scene on an orange desk, turned and a little tilted."""
    rng = np.random.default_rng(1)
    desk = np.zeros((1600, 1200, 3), np.uint8)
    desk[:] = (90, 160, 210)                                   # orange desk (BGR)
    desk = np.clip(desk + rng.normal(0, 6, desk.shape), 0, 255).astype(np.uint8)
    cv2.rectangle(desk, (40, 40), (200, 180), (40, 200, 230), -1)   # something on the desk
    portrait = cv2.rotate(scene, cv2.ROTATE_90_COUNTERCLOCKWISE)   # 480 x 640
    h, w = portrait.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[250, 330], [960, 300], [990, 1290], [220, 1310]])  # tilted
    m = cv2.getPerspectiveTransform(src, dst)
    cv2.warpPerspective(portrait, m, (1200, 1600), dst=desk, borderMode=cv2.BORDER_TRANSPARENT)
    return desk


def test_rectified_handheld_photo_gives_same_detections():
    scene = render_scene()
    out = rectify_paper(handheld_photo(scene))
    assert out.shape == (480, 640, 3)
    dets = detect_objects(out)
    assert len(dets) == len(DEFAULT_PARTS)
    # every true angle is found again
    for part in DEFAULT_PARTS:
        diffs = [abs((d.angle_deg - part.angle_deg + 90) % 180 - 90) for d in dets]
        assert min(diffs) < 3.0


def test_desk_is_not_detected_as_a_part():
    out = rectify_paper(handheld_photo(render_scene([])))
    assert detect_objects(out) == []


def test_no_paper_raises():
    with pytest.raises(PaperNotFound):
        rectify_paper(np.full((600, 800, 3), (40, 90, 140), np.uint8))


def test_order_corners():
    pts = np.float32([[10, 90], [100, 100], [0, 0], [90, 5]])
    assert order_corners(pts).tolist() == [[0, 0], [90, 5], [100, 100], [10, 90]]
