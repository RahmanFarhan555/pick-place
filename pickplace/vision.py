"""Find rectangular parts on white paper.

Steps:
  1. score for every pixel: how colourful or how dark it is
  2. Otsu threshold on the score
  3. open and close to remove noise and small holes
  4. contours, filtered by area, image border and rectangle shape
  5. minAreaRect gives the center, the long side gives the angle

The angle is the direction of the long side, from image right towards image
down, between -90 and 90 degrees.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import cv2
import numpy as np

from .calibration import PickTarget, normalize_angle_180


@dataclass
class Detection:
    center: tuple[float, float]          # pixel (u, v)
    angle_deg: float                     # long axis, image frame, [-90, 90)
    size: tuple[float, float]            # (long, short) side in px
    area: float                          # contour area px^2
    rectangularity: float                # contour area / box area (1.0 = perfect)
    box: np.ndarray = field(repr=False)  # 4x2 corner points


@dataclass
class DetectionConfig:
    min_area: float = 400.0       # px^2, rejects dust/noise
    max_area_frac: float = 0.35   # part of the image, rejects the desk and big shadows
    min_rectangularity: float = 0.80       # contour area / box area
    min_hull_rectangularity: float = 0.92  # hull area / box area (boxes with white labels)
    border_margin: int = 3        # a part at the image border is cut off
    blur_ksize: int = 5
    morph_ksize: int = 5


def foreground_mask(frame: np.ndarray, cfg: DetectionConfig = DetectionConfig()) -> np.ndarray:
    if frame is None or frame.ndim != 3:
        raise ValueError("expected a BGR colour image")
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    score = np.maximum(hsv[:, :, 1], 255 - hsv[:, :, 2])
    score = cv2.GaussianBlur(score, (cfg.blur_ksize, cfg.blur_ksize), 0)
    _, mask = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Otsu always gives a threshold. On empty paper it only splits noise,
    # so I check that the two groups are really different.
    fg, bg = score[mask > 0], score[mask == 0]
    if fg.size == 0 or bg.size == 0 or float(fg.mean()) - float(bg.mean()) < 40:
        return np.zeros_like(mask)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.morph_ksize, cfg.morph_ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask


def detect_objects(frame: np.ndarray, cfg: DetectionConfig = DetectionConfig()) -> list[Detection]:
    mask = foreground_mask(frame, cfg)
    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections: list[Detection] = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (cfg.min_area <= area <= cfg.max_area_frac * h * w):
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        m = cfg.border_margin
        if x <= m or y <= m or x + bw >= w - m or y + bh >= h - m:
            continue  # part is cut off, center would be wrong
        rect = cv2.minAreaRect(c)
        (cu, cv_), (rw, rh), _ = rect
        box_area = rw * rh
        if box_area <= 0:
            continue
        rectangularity = area / box_area
        hull_rect = cv2.contourArea(cv2.convexHull(c)) / box_area
        if rectangularity < cfg.min_rectangularity and hull_rect < cfg.min_hull_rectangularity:
            continue  # not a rectangle, for example two touching parts
        box = cv2.boxPoints(rect)
        detections.append(Detection(
            center=(float(cu), float(cv_)),
            angle_deg=_long_axis_angle(box),
            size=(float(max(rw, rh)), float(min(rw, rh))),
            area=float(area),
            rectangularity=float(rectangularity),
            box=box,
        ))
    # sort: rows from top to bottom, inside a row from left to right
    detections.sort(key=lambda d: (int(d.center[1] // 60), d.center[0]))
    return detections


def _long_axis_angle(box: np.ndarray) -> float:
    p0, p1, p2 = box[0], box[1], box[2]
    e1, e2 = p1 - p0, p2 - p1
    long_edge = e1 if np.hypot(*e1) >= np.hypot(*e2) else e2
    return normalize_angle_180(math.degrees(math.atan2(long_edge[1], long_edge[0])))


def annotate(frame: np.ndarray,
             detections: Sequence[Detection],
             targets: Optional[Sequence[PickTarget]] = None,
             calibration_pixels: Optional[Sequence[Sequence[float]]] = None) -> np.ndarray:
    """Draw boxes, centers, long axis and labels on a copy of the image."""
    out = frame.copy()
    if calibration_pixels is not None:
        for (u, v) in calibration_pixels:
            cv2.drawMarker(out, (int(u), int(v)), (200, 120, 0), cv2.MARKER_TILTED_CROSS, 12, 1)
    _draw_robot_axes(out)

    for i, d in enumerate(detections):
        cv2.drawContours(out, [np.intp(d.box)], 0, (0, 200, 0), 2)
        cu, cv_ = d.center
        a = math.radians(d.angle_deg)
        half = d.size[0] / 2
        p1 = (int(cu - half * math.cos(a)), int(cv_ - half * math.sin(a)))
        p2 = (int(cu + half * math.cos(a)), int(cv_ + half * math.sin(a)))
        cv2.line(out, p1, p2, (255, 0, 255), 1)
        cv2.circle(out, (int(cu), int(cv_)), 4, (0, 0, 255), -1)

        lines = [f"#{i}  px ({cu:.0f}, {cv_:.0f})  {d.angle_deg:+.1f} deg"]
        if targets is not None and i < len(targets):
            p = targets[i].pick
            lines.append(f"robot ({p.x:.1f}, {p.y:.1f}) mm  rz {p.rz:+.1f}")
        _label_box(out, lines, (int(cu), int(cv_) + 12))

    _put_label(out, f"{len(detections)} part(s) detected", (8, 18))
    return out


def _label_box(img, lines, anchor, scale=0.4):
    """Draw text lines on a dark box below the anchor point."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    sizes = [cv2.getTextSize(t, font, scale, 1)[0] for t in lines]
    w = max(s[0] for s in sizes) + 8
    lh = max(s[1] for s in sizes) + 6
    h = lh * len(lines) + 4
    x = min(max(anchor[0] - w // 2, 2), img.shape[1] - w - 2)
    y = min(max(anchor[1], 2), img.shape[0] - h - 2)
    roi = img[y:y + h, x:x + w]
    roi[:] = (roi * 0.35).astype(img.dtype)
    for j, t in enumerate(lines):
        cv2.putText(img, t, (x + 4, y + lh * (j + 1)), font, scale, (255, 255, 255), 1, cv2.LINE_AA)


def _put_label(img, text, org):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1, cv2.LINE_AA)


def _draw_robot_axes(img):
    """Draw the robot axes in the corner (x down, y right)."""
    o = (img.shape[1] - 60, img.shape[0] - 60)
    cv2.arrowedLine(img, o, (o[0], o[1] + 40), (0, 0, 255), 2, tipLength=0.25)
    cv2.arrowedLine(img, o, (o[0] + 40, o[1]), (0, 160, 0), 2, tipLength=0.25)
    _put_label(img, "x_rob", (o[0] - 45, o[1] + 45))
    _put_label(img, "y_rob", (o[0] + 5, o[1] - 6))
