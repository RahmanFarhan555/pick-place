"""Crop a hand-held photo to the white paper.

In the assignment the camera is fixed and its 640x480 image is the work area.
My phone photo also shows the desk and is a little tilted. This file finds
the sheet, makes it straight and fits it into 640x480.
"""
from __future__ import annotations

import cv2
import numpy as np


class PaperNotFound(RuntimeError):
    pass


A4_ASPECT = 297.0 / 210.0


def _paper_mask(hsv: np.ndarray, s_thr: float, v_thr: float, k: int) -> np.ndarray:
    mask = ((hsv[:, :, 2] >= v_thr) & (hsv[:, :, 1] <= s_thr)).astype(np.uint8) * 255
    # opening removes small dots and thin connections to the desk
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))


def _quad_from_mask(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.1 * mask.size:
        return None
    # parts at the paper edge cut into the outline, the convex hull fixes this
    hull = cv2.convexHull(c)
    quad = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
    pts = quad.reshape(-1, 2).astype(np.float32) if len(quad) == 4 else cv2.boxPoints(cv2.minAreaRect(hull))
    return order_corners(pts)


def _paper_score(quad: np.ndarray, shape) -> float:
    """Score how much the shape looks like a full A4 sheet."""
    h, w = shape[:2]
    area = cv2.contourArea(quad)
    (_, _), (rw, rh), _ = cv2.minAreaRect(quad)
    if area <= 0 or rw * rh <= 0:
        return -1.0
    rectangularity = area / (rw * rh)
    aspect = max(rw, rh) / min(rw, rh)
    touches = np.any(quad[:, 0] <= 2) or np.any(quad[:, 1] <= 2) or \
        np.any(quad[:, 0] >= w - 3) or np.any(quad[:, 1] >= h - 3)
    return rectangularity - 0.5 * abs(aspect - A4_ASPECT) / A4_ASPECT - (0.3 if touches else 0.0)


def find_paper(img: np.ndarray) -> np.ndarray:
    """Find the 4 corners of the paper (top-left, top-right, bottom-right, bottom-left).

    Paper is bright and has almost no colour. I try two thresholds and keep
    the shape that looks most like an A4 sheet.
    """
    scale = 800.0 / max(img.shape[:2])
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    brightest = hsv[v >= np.percentile(v, 75)]
    paper_s, paper_v = float(np.median(brightest[:, 1])), float(np.median(brightest[:, 2]))
    if paper_s > 80 or paper_v < 90:
        raise PaperNotFound("no bright, colourless paper region found")

    candidates = []
    for s_thr, v_thr, k in ((max(30.0, paper_s + 20), max(90.0, paper_v - 45), 15),
                            (70.0, max(120.0, float(np.percentile(v, 50))), 7)):
        quad = _quad_from_mask(_paper_mask(hsv, s_thr, v_thr, k))
        if quad is not None:
            candidates.append((_paper_score(quad, small.shape), quad))
    if not candidates:
        raise PaperNotFound("paper region too small - is the sheet in the picture?")
    best = max(candidates, key=lambda c: c[0])[1]
    return best / scale


def order_corners(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float32)
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def rectify_paper(img: np.ndarray, size=(640, 480), margin: int = 10, edge_px: int = 4) -> np.ndarray:
    """Make the paper straight and fit it into the output image.

    Everything outside the sheet is painted in paper colour.
    """
    tl, tr, br, bl = find_paper(img)
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    src = np.array([tl, tr, br, bl], np.float32)
    if h > w:  # portrait sheet: turn it to landscape
        src = np.array([bl, tl, tr, br], np.float32)
        w, h = h, w
    out_w, out_h = size
    fit = min((out_w - 2 * margin) / w, (out_h - 2 * margin) / h)   # keep aspect ratio
    pw, ph = w * fit, h * fit
    x0, y0 = (out_w - pw) / 2, (out_h - ph) / 2
    dst = np.array([[x0, y0], [x0 + pw, y0], [x0 + pw, y0 + ph], [x0, y0 + ph]], np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, m, size, flags=cv2.INTER_AREA)

    # keep the photo inside the sheet, paint the rest in paper colour
    inside = np.zeros((out_h, out_w), np.uint8)
    cv2.rectangle(inside, (int(x0) + edge_px, int(y0) + edge_px),
                  (int(x0 + pw) - edge_px, int(y0 + ph) - edge_px), 255, -1)
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    paper_px = warped[(inside > 0) & (hsv[:, :, 1] < 60) & (hsv[:, :, 2] > 120)]
    fill = np.median(paper_px, axis=0) if len(paper_px) else np.array([235, 235, 235])
    warped[inside == 0] = fill.astype(np.uint8)
    return warped
