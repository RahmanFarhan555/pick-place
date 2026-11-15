import numpy as np
import pytest

from pickplace.synthetic import DEFAULT_PARTS, Part, render_scene
from pickplace.vision import annotate, detect_objects


def angle_diff(a, b):
    """Difference between two angles of a long axis (180 deg is the same)."""
    d = (a - b + 90) % 180 - 90
    return abs(d)


def match(detections, part):
    return min(detections, key=lambda d: np.hypot(d.center[0] - part.center[0],
                                                   d.center[1] - part.center[1]))


def test_detects_all_parts_with_center_and_angle():
    img = render_scene()
    dets = detect_objects(img)
    assert len(dets) == len(DEFAULT_PARTS)
    for part in DEFAULT_PARTS:
        d = match(dets, part)
        assert np.hypot(d.center[0] - part.center[0], d.center[1] - part.center[1]) < 2.0
        assert angle_diff(d.angle_deg, part.angle_deg) < 2.0
        assert d.size[0] == pytest.approx(part.size[0], abs=4)
        assert d.size[1] == pytest.approx(part.size[1], abs=4)
        assert -90 <= d.angle_deg < 90


@pytest.mark.parametrize("angle", [-89, -60, -30, 0, 15, 45, 89])
def test_angle_range(angle):
    img = render_scene([Part((320, 240), (120, 50), angle, (50, 50, 50))], seed=angle + 100)
    (d,) = detect_objects(img)
    assert angle_diff(d.angle_deg, angle) < 2.0


def test_empty_paper_gives_no_detections():
    img = render_scene([])
    assert detect_objects(img) == []


def test_part_touching_border_is_rejected():
    parts = [Part((30, 240), (100, 40), 0, (50, 50, 50)),   # cut by left border
             Part((320, 240), (100, 40), 0, (50, 50, 50))]
    dets = detect_objects(render_scene(parts))
    assert len(dets) == 1
    assert dets[0].center[0] == pytest.approx(320, abs=2)


def test_tiny_noise_blob_is_rejected():
    parts = [Part((100, 100), (8, 6), 0, (30, 30, 30)), Part((320, 240), (100, 40), 0, (30, 30, 30))]
    assert len(detect_objects(render_scene(parts))) == 1


def test_touching_parts_are_not_reported_as_one_rectangle():
    # two parts in an L shape are not a rectangle
    parts = [Part((300, 240), (120, 40), 0, (30, 30, 30)), Part((340, 280), (120, 40), 90, (30, 30, 30))]
    assert detect_objects(render_scene(parts)) == []


def test_annotate_returns_same_size_image_and_keeps_input():
    img = render_scene()
    before = img.copy()
    out = annotate(img, detect_objects(img))
    assert out.shape == img.shape
    assert np.array_equal(img, before)
    assert not np.array_equal(out, img)


def test_rejects_non_colour_input():
    with pytest.raises(ValueError):
        detect_objects(np.zeros((480, 640), np.uint8))
