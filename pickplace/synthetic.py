"""Generated test images where the true center and angle are known."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Part:
    center: tuple[float, float]
    size: tuple[float, float]   # (long, short) px
    angle_deg: float            # long axis, image frame
    color: tuple[int, int, int]  # BGR


DEFAULT_PARTS = [
    Part((170, 140), (90, 45), 20, (60, 60, 200)),     # red eraser
    Part((430, 130), (110, 60), -35, (40, 40, 40)),    # black domino
    Part((300, 300), (100, 50), 75, (180, 120, 40)),   # blue matchbox
    Part((500, 360), (80, 40), 0, (60, 170, 60)),      # green eraser
]


def render_scene(parts=DEFAULT_PARTS, size=(640, 480), seed=0,
                 noise_sigma=4.0, shadow=True) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w, h = size
    # white paper with a little uneven light
    yy, xx = np.mgrid[0:h, 0:w]
    light = 245 - 18 * (((xx - w * 0.4) / w) ** 2 + ((yy - h * 0.3) / h) ** 2)
    img = np.dstack([light, light - 2, light - 5]).astype(np.float32)

    for p in parts:
        box = cv2.boxPoints(((p.center[0], p.center[1]), p.size, p.angle_deg)).astype(np.int32)
        if shadow:
            sh = np.zeros((h, w), np.float32)
            cv2.fillPoly(sh, [box + np.array([5, 6])], 1.0)
            sh = cv2.GaussianBlur(sh, (15, 15), 0)
            img *= (1 - 0.25 * sh)[:, :, None]
        cv2.fillPoly(img, [box], p.color, lineType=cv2.LINE_AA)

    img += rng.normal(0, noise_sigma, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)
