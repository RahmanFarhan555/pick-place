"""Test of the whole program on a generated image."""
import json

import cv2
import numpy as np

from pickplace.main import main
from pickplace.synthetic import DEFAULT_PARTS, render_scene


def test_end_to_end(tmp_path, capsys):
    img = tmp_path / "photo.png"
    cv2.imwrite(str(img), render_scene())
    out = tmp_path / "out"
    assert main(["--image", str(img), "--out", str(out), "--grip-fail", "3"]) == 0
    assert (out / "annotated.png").is_file()
    data = json.loads((out / "result.json").read_text())
    assert data["source"] == str(img) and len(data["parts"]) == len(DEFAULT_PARTS)
    results = [c["result"] for c in data["cycle"]]
    assert results[0] == "grip_failed" and all(r == "success" for r in results[1:])
    for p in data["parts"]:
        assert p["approach"]["z"] - p["pick"]["z"] == 50


def test_missing_image_returns_error_code(tmp_path):
    assert main(["--image", str(tmp_path / "missing.jpg"), "--out", str(tmp_path)]) == 2


class FakeCapture:
    def __init__(self, frame):
        self.frame = frame

    def isOpened(self):
        return True

    def read(self):
        return True, self.frame

    def release(self):
        pass


def test_camera_without_paper_uses_the_photo(tmp_path, monkeypatch):
    img = tmp_path / "photo.png"
    cv2.imwrite(str(img), render_scene())
    room = np.full((480, 640, 3), (60, 90, 140), np.uint8)       # webcam looking at a room
    monkeypatch.setattr(cv2, "VideoCapture", lambda i: FakeCapture(room))
    out = tmp_path / "out"
    assert main(["--camera", "0", "--image", str(img), "--paper", "--out", str(out), "--no-sim"]) == 0
    data = json.loads((out / "result.json").read_text())
    assert data["source"] == str(img) and len(data["parts"]) == len(DEFAULT_PARTS)


def test_camera_with_empty_paper_uses_the_photo(tmp_path, monkeypatch):
    img = tmp_path / "photo.png"
    cv2.imwrite(str(img), render_scene())
    empty = render_scene([])                                      # white paper, no objects
    monkeypatch.setattr(cv2, "VideoCapture", lambda i: FakeCapture(empty))
    out = tmp_path / "out"
    assert main(["--camera", "0", "--image", str(img), "--out", str(out), "--no-sim"]) == 0
    data = json.loads((out / "result.json").read_text())
    assert data["source"] == str(img) and len(data["parts"]) == len(DEFAULT_PARTS)
