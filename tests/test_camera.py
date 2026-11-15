import cv2
import numpy as np
import pytest

from pickplace.camera import CameraError, FrameSource


class FakeCapture:
    def __init__(self, opened=True, ok=True, frame=None):
        self._opened, self._ok = opened, ok
        self._frame = frame if frame is not None else np.full((480, 640, 3), 255, np.uint8)
        self.released = False

    def isOpened(self):
        return self._opened

    def read(self):
        return (self._ok, self._frame if self._ok else None)

    def release(self):
        self.released = True


def factory_from(seq):
    """Returns the given fake cameras one after another."""
    it = iter(seq)
    created = []

    def make(_index):
        cap = next(it)
        created.append(cap)
        return cap
    make.created = created
    return make


def no_sleep(_):
    pass


def test_reads_from_camera():
    src = FrameSource(0, capture_factory=factory_from([FakeCapture()]), sleep=no_sleep)
    assert src.read().shape == (480, 640, 3)
    assert src.last_source == "camera 0"


def test_retries_then_recovers(caplog):
    sleeps = []
    factory = factory_from([FakeCapture(opened=False), FakeCapture(ok=False), FakeCapture()])
    src = FrameSource(0, retries=3, retry_delay_s=0.5, capture_factory=factory, sleep=sleeps.append)
    frame = src.read()
    assert frame.shape == (480, 640, 3)
    assert sleeps == [0.5, 0.5]
    assert "not found or cannot be opened" in caplog.text
    assert "returned no valid frame" in caplog.text
    assert all(c.released for c in factory.created)


def test_missing_camera_without_fallback_raises_clear_error():
    factory = factory_from([FakeCapture(opened=False)] * 2)
    src = FrameSource(0, retries=2, capture_factory=factory, sleep=no_sleep)
    with pytest.raises(CameraError, match="failed after 2 attempts"):
        src.read()


def test_backend_exception_is_handled():
    def boom(_):
        raise RuntimeError("driver crashed")
    src = FrameSource(0, retries=2, capture_factory=boom, sleep=no_sleep)
    with pytest.raises(CameraError):
        src.read()


def test_falls_back_to_image_file(tmp_path):
    path = tmp_path / "photo.png"
    cv2.imwrite(str(path), np.full((480, 640, 3), 200, np.uint8))
    src = FrameSource(0, image_path=path, retries=1,
                      capture_factory=factory_from([FakeCapture(opened=False)]), sleep=no_sleep)
    assert src.read()[0, 0, 0] == 200
    assert src.last_source == str(path)


def test_missing_file_raises(tmp_path):
    src = FrameSource(None, image_path=tmp_path / "nope.jpg")
    with pytest.raises(CameraError, match="not found"):
        src.read()


def test_corrupt_file_raises(tmp_path):
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"not an image")
    with pytest.raises(CameraError, match="could not be decoded"):
        FrameSource(None, image_path=path).read()


def test_large_photo_is_resized_to_calibrated_resolution(tmp_path):
    path = tmp_path / "phone.jpg"
    cv2.imwrite(str(path), np.full((1200, 1600, 3), 255, np.uint8))  # 4:3 like 640x480
    assert FrameSource(None, image_path=path).read().shape == (480, 640, 3)


def test_needs_some_source():
    with pytest.raises(ValueError):
        FrameSource(None, None)
