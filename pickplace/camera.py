"""Read a frame from the webcam or from an image file.

If the camera is missing or a frame can not be read, the program prints an
error and tries again. If it still fails and an image file is given, the file
is used. The camera and the sleep function can be replaced in the tests.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from .calibration import IMAGE_HEIGHT, IMAGE_WIDTH

log = logging.getLogger(__name__)


class CameraError(RuntimeError):
    """No frame could be read."""


def _default_capture_factory(index: int):
    return cv2.VideoCapture(index)


class FrameSource:
    def __init__(self,
                 camera_index: Optional[int] = 0,
                 image_path: Optional[str | Path] = None,
                 retries: int = 3,
                 retry_delay_s: float = 1.0,
                 target_size: tuple[int, int] = (IMAGE_WIDTH, IMAGE_HEIGHT),
                 capture_factory: Callable[[int], object] = _default_capture_factory,
                 sleep: Callable[[float], None] = time.sleep,
                 preprocess: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        if camera_index is None and image_path is None:
            raise ValueError("configure a camera index, an image path, or both")
        if retries < 1:
            raise ValueError("retries must be >= 1")
        self.camera_index = camera_index
        self.image_path = Path(image_path) if image_path is not None else None
        self.retries = retries
        self.retry_delay_s = retry_delay_s
        self.target_size = target_size
        self._capture_factory = capture_factory
        self._sleep = sleep
        self._preprocess = preprocess   # for example the paper crop
        self.last_source: Optional[str] = None

    def read(self) -> np.ndarray:
        """Return one frame in the target size, or raise CameraError."""
        errors: list[str] = []
        if self.camera_index is not None:
            try:
                frame = self._read_camera()
                self.last_source = f"camera {self.camera_index}"
                return self._normalize(frame)
            except CameraError as exc:
                errors.append(str(exc))
                if self.image_path is not None:
                    log.warning("%s -> falling back to image file %s", exc, self.image_path)
        if self.image_path is not None:
            try:
                frame = self._read_file()
                self.last_source = str(self.image_path)
                return self._normalize(frame)
            except CameraError as exc:
                errors.append(str(exc))
        raise CameraError("no frame available: " + " | ".join(errors))

    def _read_camera(self) -> np.ndarray:
        for attempt in range(1, self.retries + 1):
            cap = None
            try:
                cap = self._capture_factory(self.camera_index)
                if cap is None or not cap.isOpened():
                    reason = f"camera {self.camera_index} not found or cannot be opened"
                else:
                    ok, frame = cap.read()
                    if ok and _is_valid_frame(frame):
                        if attempt > 1:
                            log.info("camera %s recovered on attempt %d", self.camera_index, attempt)
                        return frame
                    reason = f"camera {self.camera_index} opened but returned no valid frame"
            except Exception as exc:  # a camera driver error must not stop the program
                reason = f"camera {self.camera_index} error: {exc}"
            finally:
                if cap is not None and hasattr(cap, "release"):
                    cap.release()
            log.error("%s (attempt %d/%d)", reason, attempt, self.retries)
            if attempt < self.retries:
                log.info("retrying in %.1f s ...", self.retry_delay_s)
                self._sleep(self.retry_delay_s)
        raise CameraError(f"camera {self.camera_index} failed after {self.retries} attempts")

    def _read_file(self) -> np.ndarray:
        path = self.image_path
        if not path.is_file():
            raise CameraError(f"image file not found: {path}")
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if not _is_valid_frame(frame):
            raise CameraError(f"image file could not be decoded: {path}")
        return frame

    def _normalize(self, frame: np.ndarray) -> np.ndarray:
        """Resize the frame to 640x480, because the calibration is for this size."""
        if self._preprocess is not None:
            frame = self._preprocess(frame)
        w, h = self.target_size
        fh, fw = frame.shape[:2]
        if (fw, fh) == (w, h):
            return frame
        if abs(fw / fh - w / h) > 0.02:
            log.warning("frame aspect %dx%d differs from calibrated %dx%d; "
                        "pixel->robot mapping will be distorted", fw, fh, w, h)
        else:
            log.info("resizing frame %dx%d -> %dx%d", fw, fh, w, h)
        return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)


def _is_valid_frame(frame) -> bool:
    return (isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.shape[2] == 3
            and frame.size > 0)
