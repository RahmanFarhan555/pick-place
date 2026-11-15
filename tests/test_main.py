"""Test of the whole program on a generated image."""
import json

import cv2

from pickplace.main import main
from pickplace.synthetic import DEFAULT_PARTS, render_scene


def test_end_to_end(tmp_path, capsys):
    img = tmp_path / "photo.png"
    cv2.imwrite(str(img), render_scene())
    out = tmp_path / "out"
    assert main(["--image", str(img), "--out", str(out), "--grip-fail", "3"]) == 0
    assert (out / "annotated.png").is_file()
    data = json.loads((out / "result.json").read_text())
    assert len(data["parts"]) == len(DEFAULT_PARTS)
    results = [c["result"] for c in data["cycle"]]
    assert results[0] == "grip_failed" and all(r == "success" for r in results[1:])
    for p in data["parts"]:
        assert p["approach"]["z"] - p["pick"]["z"] == 50


def test_missing_image_returns_error_code(tmp_path):
    assert main(["--image", str(tmp_path / "missing.jpg"), "--out", str(tmp_path)]) == 2
