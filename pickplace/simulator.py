"""Simulator window for the pick and place cycle.

It runs the same PickPlaceCycle as the tests. SimRobot is a FakeRobot that
draws every move. The view shows the table from above in robot coordinates.

    python -m pickplace.simulator --image data/test_photo.jpg --paper
    python -m pickplace.simulator --image data/test_photo.jpg --paper --demo

Buttons or keys: E-STOP (E), RELEASE + RESET (R), FAIL NEXT GRIP (F),
PAUSE (SPACE), QUIT (Q or ESC).
"""
from __future__ import annotations

import argparse
import logging
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .calibration import PickTarget, PixelToRobot, Pose
from .camera import CameraError, FrameSource
from .cycle import CycleConfig, CycleResult, PickPlaceCycle
from .robot import EmergencyStopError, FakeRobot
from .paper import rectify_paper
from .vision import Detection, detect_objects, foreground_mask

log = logging.getLogger(__name__)

FPS = 30
# top view: robot x (down) 90..350 mm, robot y (right) -230..270 mm, 1.6 px/mm
X0, Y0, S = 90.0, -230.0, 1.6
VIEW_W, VIEW_H = 800, 416
SIDE_W, STATUS_H = 200, 130
CANVAS_W, CANVAS_H = VIEW_W + SIDE_W, VIEW_H + STATUS_H
GRIPPER_OPEN_MM = 40.0


class QuitSim(Exception):
    pass


def to_canvas(x: float, y: float) -> tuple[int, int]:
    """Robot (x, y) in mm to a pixel in the window. Robot y is right, x is down."""
    return int(round((y - Y0) * S)), int(round((x - X0) * S))


def rot(rz_deg: float) -> np.ndarray:
    a = math.radians(rz_deg)
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


@dataclass
class SimPart:
    home_poly: np.ndarray            # 4x2 robot (x, y) where it lies on the table
    color: tuple[int, int, int]
    width_mm: float                  # short side: the gripper closes across it
    state: str = "table"             # table | held | placed
    poly: np.ndarray = None          # current 4x2 robot polygon
    rel: Optional[np.ndarray] = None  # polygon in tool frame while held

    def __post_init__(self):
        self.poly = self.home_poly.copy()


@dataclass
class Button:
    action: str                        # estop | reset | fail | pause | quit
    rect: tuple[int, int, int, int]    # x, y, w, h on the canvas

    def contains(self, x: int, y: int) -> bool:
        bx, by, bw, bh = self.rect
        return bx <= x < bx + bw and by <= y < by + bh


def make_buttons() -> list[Button]:
    y, h, gap = VIEW_H + 84, 38, 10
    widths = [("estop", 210), ("reset", 200), ("fail", 190), ("pause", 150), ("quit", 100)]
    buttons, x = [], 12
    for action, w in widths:
        buttons.append(Button(action, (x, y, w, h)))
        x += w + gap
    return buttons


@dataclass
class Script:
    """Events for the automatic demo and the tests."""
    grip_failures: dict[int, int] = field(default_factory=dict)   # part -> forced fails
    estop: Optional[tuple[int, str, float]] = None                 # (part, step, fraction)
    estop_hold_s: float = 1.5


class SimRobot(FakeRobot):
    def __init__(self, parts: list[SimPart], home: Pose, place: Pose, speed: float = 1.0):
        super().__init__()
        self.parts = parts
        self.place = place
        self.pose = home
        self.speed = speed
        # open wide enough for the widest part, plus 10 mm
        self.open_gap = max([GRIPPER_OPEN_MM] + [p.width_mm + 10 for p in parts])
        self.gap = self.open_gap
        self.current_part = 0
        self.held: Optional[SimPart] = None
        self.forced_fails: dict[int, int] = {}
        self.view: Optional["SimView"] = None
        self.estop_hook = None        # callable(step, fraction) -> bool
        self.trail: list[tuple[float, float]] = []

    def move_joint(self, pose: Pose, speed_pct: float = 50.0) -> None:
        self._require_no_estop("move_joint")
        self._record("move_joint", pose)
        self._animate(pose, mm_per_s=5.0 * speed_pct)  # 50 % ~ 250 mm/s

    def move_linear(self, pose: Pose, speed_mm_s: float = 100.0) -> None:
        self._require_no_estop("move_linear")
        self._record("move_linear", pose)
        self._animate(pose, mm_per_s=speed_mm_s)

    def _animate(self, target: Pose, mm_per_s: float) -> None:
        start = self.pose
        dist = math.dist((start.x, start.y, start.z), (target.x, target.y, target.z))
        drz = (target.rz - start.rz + 180) % 360 - 180
        duration = max(dist / mm_per_s, abs(drz) / 180.0, 0.1) / self.speed
        n = max(2, int(duration * FPS))
        for i in range(1, n + 1):
            t = i / n
            s = 0.5 - 0.5 * math.cos(math.pi * t)  # smooth accel / decel
            self.pose = Pose(start.x + (target.x - start.x) * s,
                             start.y + (target.y - start.y) * s,
                             start.z + (target.z - start.z) * s,
                             start.rz + drz * s)
            self._carry()
            self.trail.append((self.pose.x, self.pose.y))
            self._frame()
            if self.estop_hook and self.estop_hook(t):
                self._estop = True
            if self._estop:   # stop at once, like a real robot
                raise EmergencyStopError("e-stop pressed during motion")
        self.pose = target

    def _carry(self) -> None:
        if self.held is not None:
            self.held.poly = np.array([self.pose.x, self.pose.y]) + self.held.rel @ rot(self.pose.rz).T

    def close_gripper(self) -> None:
        self._require_no_estop("close_gripper")
        self._record("close_gripper")
        self._gripper_closed = True
        part = self._part_under_gripper()
        forced = self.forced_fails.get(self.current_part, 0) > 0
        if forced:
            self.forced_fails[self.current_part] -= 1
        if part is not None and not forced:
            self._animate_gap(part.width_mm)
            self._holding = True
            self.held = part
            part.state = "held"
            part.rel = (part.poly - np.array([self.pose.x, self.pose.y])) @ rot(self.pose.rz)
        else:
            self._animate_gap(0.0)   # nothing in the gripper, the grip check will fail
            self._holding = False

    def open_gripper(self) -> None:
        self._require_no_estop("open_gripper")
        self._record("open_gripper")
        self._gripper_closed = False
        self._holding = False
        if self.held is not None:
            near_place = math.dist((self.pose.x, self.pose.y), (self.place.x, self.place.y)) < 15
            self.held.state = "placed" if near_place else "table"
            self.held.rel = None
            self.held = None
        self._animate_gap(self.open_gap)

    def _animate_gap(self, target: float) -> None:
        start = self.gap
        n = max(2, int(0.3 * FPS / self.speed))
        for i in range(1, n + 1):
            self.gap = start + (target - start) * i / n
            self._frame()
            if self._estop:
                raise EmergencyStopError("e-stop pressed during gripper motion")

    def _part_under_gripper(self) -> Optional[SimPart]:
        if self.pose.z > 22.0:   # gripper is not down at pick height (20 mm)
            return None
        for p in self.parts:
            if p.state == "table" and cv2.pointPolygonTest(
                    p.poly.astype(np.float32), (self.pose.x, self.pose.y), False) >= 0:
                return p
        return None

    def read_input(self, name: str) -> bool:
        # the part is present if it is still on the table
        return self.parts[self.current_part].state == "table"

    def set_alarm(self, message: str) -> None:
        super().set_alarm(message)
        if self.view:
            self.view.alarm(message)

    def operator_clears_gripper(self) -> None:
        """After an e-stop the operator puts a held part back on the table."""
        if self.held is not None:
            self.held.state = "table"
            self.held.poly = self.held.home_poly.copy()
            self.held.rel = None
            self.held = None
        self._holding = False

    def _frame(self) -> None:
        if self.view:
            self.view.tick()


class SimView:
    def __init__(self, photo: np.ndarray, cal: PixelToRobot, detections: list[Detection],
                 targets: list[PickTarget], robot: SimRobot, cycle: PickPlaceCycle,
                 live: bool = True, record: Optional[Path] = None):
        self.cal, self.targets, self.robot, self.cycle = cal, targets, robot, cycle
        self.live, self.record = live, record
        self.paused = False
        self.reset_requested = False
        self.alarm_text, self.alarm_frames = "", 0
        self.results: list[str] = []
        self.frames: list[np.ndarray] = []
        self.frame_no = 0

        # show the photo in robot coordinates, the area outside the camera is grey
        m = np.array([[0, S, -Y0 * S], [S, 0, -X0 * S], [0, 0, 1]])
        self.background = cv2.warpPerspective(photo, m @ cal.H, (VIEW_W, VIEW_H),
                                              flags=cv2.INTER_LINEAR, borderValue=(70, 70, 70))
        mask = foreground_mask(photo)
        paper = photo[mask == 0]
        self.paper = tuple(int(c) for c in np.median(paper, axis=0)) if len(paper) else (235, 235, 235)
        robot.view = self
        self.buttons = make_buttons()
        self.mouse = (-1, -1)
        self.quit_requested = False
        self.flash: dict[str, int] = {}   # action -> frames left of "pressed" highlight
        if live:
            cv2.namedWindow("pick and place simulator", cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback("pick and place simulator", self.on_mouse)

    def alarm(self, text: str) -> None:
        self.alarm_text, self.alarm_frames = text, FPS * 4

    def tick(self) -> None:
        frame = self.render()
        self.frame_no += 1
        if self.record is not None:
            self.frames.append(frame)
        if self.live:
            self._show(frame)

    # buttons and keys use the same function
    def handle(self, action: str) -> None:
        if action == "estop":
            self.robot.trigger_estop()
        elif action == "reset":
            self.robot.release_estop()
            self.reset_requested = True
        elif action == "fail":
            i = self.robot.current_part
            self.robot.forced_fails[i] = self.robot.forced_fails.get(i, 0) + 1
        elif action == "pause":
            self.paused = not self.paused
        elif action == "quit":
            self.quit_requested = True
        else:
            raise ValueError(f"unknown action {action!r}")
        self.flash[action] = 6

    def on_mouse(self, event, x, y, flags=None, param=None) -> None:
        """Mouse click in the window (the tests call it directly)."""
        self.mouse = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            for b in self.buttons:
                if b.contains(x, y):
                    self.handle(b.action)
                    break

    KEYS = {ord("e"): "estop", ord("r"): "reset", ord("f"): "fail", ord(" "): "pause",
            ord("q"): "quit", 27: "quit"}

    def _show(self, frame) -> None:
        cv2.imshow("pick and place simulator", frame)
        while True:
            key = cv2.waitKey(int(1000 / FPS)) & 0xFF
            if key in self.KEYS:
                self.handle(self.KEYS[key])
            if self.quit_requested:
                raise QuitSim()
            if not self.paused:
                return
            cv2.imshow("pick and place simulator", self.render(paused=True))

    def hold(self, seconds: float) -> None:
        for _ in range(int(seconds * FPS / max(self.robot.speed, 1e-6))):
            self.tick()

    def render(self, paused: bool = False) -> np.ndarray:
        canvas = np.full((CANVAS_H, CANVAS_W, 3), 30, np.uint8)
        view = self.background.copy()

        # erase parts that left their spot on the table
        for p in self.robot.parts:
            if p.state != "table" or not np.allclose(p.poly, p.home_poly):
                poly = np.array([to_canvas(x, y) for x, y in p.home_poly], np.int32)
                cv2.fillPoly(view, [poly], self.paper)
                cv2.polylines(view, [poly], True, self.paper, 18)  # also covers the shadow

        self._draw_static(view)
        # moved / held / placed parts
        for p in self.robot.parts:
            if p.state in ("held", "placed"):
                poly = np.array([to_canvas(x, y) for x, y in p.poly], np.int32)
                cv2.fillPoly(view, [poly], p.color, cv2.LINE_AA)
                cv2.polylines(view, [poly], True, (20, 20, 20), 1, cv2.LINE_AA)
        self._draw_gripper(view)
        canvas[:VIEW_H, :VIEW_W] = view
        self._draw_side(canvas)
        self._draw_status(canvas, paused)
        return canvas

    def _draw_static(self, view) -> None:
        # targets
        for i, t in enumerate(self.targets):
            c = to_canvas(t.pick.x, t.pick.y)
            cv2.drawMarker(view, c, (0, 0, 220), cv2.MARKER_CROSS, 12, 1)
            _label(view, f"#{i}", (c[0] + 6, c[1] - 6))
        # place area and home
        pc = to_canvas(self.cycle.cfg.place.x, self.cycle.cfg.place.y)
        r = int(40 * S)
        cv2.rectangle(view, (pc[0] - r, pc[1] - r), (pc[0] + r, pc[1] + r), (0, 140, 255), 2)
        placed = sum(p.state == "placed" for p in self.robot.parts)
        _label(view, f"PLACE ({placed} placed)", (pc[0] - r, pc[1] + r + 16))
        hc = to_canvas(self.cycle.cfg.home.x, self.cycle.cfg.home.y)
        cv2.drawMarker(view, hc, (255, 120, 0), cv2.MARKER_DIAMOND, 14, 2)
        _label(view, "HOME", (hc[0] + 8, hc[1] + 4))
        # trail
        if len(self.robot.trail) > 1:
            pts = np.array([to_canvas(x, y) for x, y in self.robot.trail[-400:]], np.int32)
            cv2.polylines(view, [pts], False, (255, 160, 60), 1, cv2.LINE_AA)
        # robot axes
        o = (20, 20)
        cv2.arrowedLine(view, o, (o[0], o[1] + 50), (0, 0, 255), 2, tipLength=0.2)
        cv2.arrowedLine(view, o, (o[0] + 50, o[1]), (0, 170, 0), 2, tipLength=0.2)
        _label(view, "x", (o[0] + 4, o[1] + 62))
        _label(view, "y", (o[0] + 56, o[1] + 4))
        _label(view, "robot frame, top view (mm)", (70, VIEW_H - 8))

    def _draw_gripper(self, view) -> None:
        p = self.robot.pose
        c = np.array([p.x, p.y])
        r = rot(p.rz)
        long_ax, short_ax = r @ np.array([1.0, 0.0]), r @ np.array([0.0, 1.0])
        # the shadow is bigger when the tool is higher
        shadow_r = int((8 + p.z * 0.08) * S)
        sc = to_canvas(*c)
        overlay = view.copy()
        cv2.circle(overlay, sc, shadow_r, (60, 60, 60), -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.25, view, 0.75, 0, view)
        # two fingers, closing along the part's short axis
        for sign in (-1, 1):
            fc = c + short_ax * sign * (self.robot.gap / 2 + 3)
            corners = [fc + long_ax * a * 9 + short_ax * b * 3 for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            poly = np.array([to_canvas(*q) for q in corners], np.int32)
            col = (0, 0, 230) if self.robot.is_estop_active() else (40, 40, 40)
            cv2.fillPoly(view, [poly], col, cv2.LINE_AA)
        cv2.circle(view, sc, 4, (255, 120, 0), -1, cv2.LINE_AA)

    def _draw_side(self, canvas) -> None:
        x0 = VIEW_W
        cv2.rectangle(canvas, (x0, 0), (CANVAS_W, VIEW_H), (45, 45, 45), -1)
        _label(canvas, "TOOL HEIGHT z (mm)", (x0 + 30, 22), light=True)
        top, bottom = 50, VIEW_H - 60

        def zrow(z):
            return int(bottom - (bottom - top) * z / 220.0)
        cv2.rectangle(canvas, (x0 + 20, zrow(0)), (x0 + 180, zrow(0) + 12), (120, 120, 120), -1)
        _label(canvas, "table", (x0 + 140, zrow(0) + 28), light=True)
        for z, name, col in ((200, "home 200", (255, 120, 0)), (70, "approach 70", (0, 200, 255)),
                             (20, "pick 20", (0, 0, 255))):
            cv2.line(canvas, (x0 + 20, zrow(z)), (x0 + 180, zrow(z)), col, 1)
            _label(canvas, name, (x0 + 100, zrow(z) - 4), light=True)
        p = self.robot.pose
        zr = zrow(p.z)
        cv2.line(canvas, (x0 + 60, top - 10), (x0 + 60, zr), (180, 180, 180), 3)
        g = int(self.robot.gap / self.robot.open_gap * 18)
        for sign in (-1, 1):
            cv2.rectangle(canvas, (x0 + 60 + sign * (g + 3) - 3, zr), (x0 + 60 + sign * (g + 3) + 3, zr + 22),
                          (230, 230, 230), -1)
        if self.robot.held is not None:
            cv2.rectangle(canvas, (x0 + 60 - g, zr + 8), (x0 + 60 + g, zr + 20), self.robot.held.color, -1)
        _label(canvas, f"x {p.x:6.1f}", (x0 + 20, VIEW_H - 40), light=True)
        _label(canvas, f"y {p.y:6.1f}", (x0 + 20, VIEW_H - 24), light=True)
        _label(canvas, f"z {p.z:6.1f}  rz {p.rz:+.0f}", (x0 + 20, VIEW_H - 8), light=True)

    def _draw_status(self, canvas, paused) -> None:
        y0 = VIEW_H
        n = len(self.targets)
        i = min(self.robot.current_part, n - 1) if n else 0
        step = self.cycle.current_step or "-"
        gr = "CLOSED" if self.robot._gripper_closed else "OPEN"
        if self.robot.held is not None:
            gr += " (holding part)"
        _label(canvas, f"part {i + 1}/{n}   step: {step}   gripper: {gr}", (12, y0 + 22), light=True, scale=0.5)
        _label(canvas, "results: " + "  ".join(self.results[-6:]), (12, y0 + 46), light=True, scale=0.45)
        if self.alarm_frames > 0:
            self.alarm_frames -= 1
            _label(canvas, "ALARM: " + self.alarm_text, (12, y0 + 72), color=(60, 60, 255), scale=0.5)
        self._draw_buttons(canvas)
        if self.robot.is_estop_active() or self.cycle.estop_latched:
            blink = (self.frame_no // 8) % 2 == 0
            cv2.rectangle(canvas, (VIEW_W // 2 - 230, 180), (VIEW_W // 2 + 230, 240),
                          (0, 0, 200) if blink else (0, 0, 140), -1)
            msg = ("EMERGENCY STOP - pressed" if self.robot.is_estop_active()
                   else "E-STOP released - waiting for reset")
            cv2.putText(canvas, msg, (VIEW_W // 2 - 215, 218), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (255, 255, 255), 2, cv2.LINE_AA)
        if paused:
            _label(canvas, "PAUSED", (VIEW_W - 90, y0 + 22), color=(0, 220, 255), scale=0.6)

    def _draw_buttons(self, canvas) -> None:
        estop_on = self.robot.is_estop_active()
        stopped = estop_on or self.cycle.estop_latched
        pending = self.robot.forced_fails.get(self.robot.current_part, 0)
        for b in self.buttons:
            x, y, w, h = b.rect
            enabled = True
            if b.action == "estop":
                text = "E-STOP ACTIVE" if estop_on else "E-STOP  (E)"
                base = (0, 0, 120) if estop_on else (0, 0, 210)
            elif b.action == "reset":
                text, base, enabled = "RELEASE + RESET  (R)", (40, 140, 40), stopped
            elif b.action == "fail":
                text = f"FAIL NEXT GRIP  (F){f' x{pending}' if pending else ''}"
                base = (0, 120, 200)
            elif b.action == "pause":
                text, base = ("RESUME" if self.paused else "PAUSE") + "  (SPACE)", (120, 90, 40)
            else:
                text, base = "QUIT  (Q)", (80, 80, 80)
            col = base if enabled else (70, 70, 70)
            if enabled and (b.contains(*self.mouse) or self.flash.get(b.action, 0) > 0):
                col = tuple(min(255, c + 60) for c in col)       # hover / just clicked
            cv2.rectangle(canvas, (x, y), (x + w, y + h), col, -1)
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (230, 230, 230) if enabled else (110, 110, 110), 1)
            scale = 0.5 if b.action == "estop" else 0.42
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            cv2.putText(canvas, text, (x + (w - tw) // 2, y + (h + th) // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, (255, 255, 255) if enabled else (150, 150, 150),
                        2 if b.action == "estop" else 1, cv2.LINE_AA)
        for k in list(self.flash):
            self.flash[k] -= 1
            if self.flash[k] <= 0:
                del self.flash[k]

    def save(self) -> None:
        if self.record is None or not self.frames:
            return
        self.record.parent.mkdir(parents=True, exist_ok=True)
        raw = self.record.with_suffix(".raw.mp4")
        writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (CANVAS_W, CANVAS_H))
        for f in self.frames:
            writer.write(f)
        writer.release()
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:  # convert to H.264 so the browser can play it
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264",
                            "-preset", "veryfast", "-pix_fmt", "yuv420p", str(self.record)], check=True)
            raw.unlink()
        else:
            raw.replace(self.record)
        log.info("recorded %d frames to %s", len(self.frames), self.record)


def _label(img, text, org, light=False, color=None, scale=0.4):
    if color is None:
        color = (230, 230, 230) if light else (20, 20, 20)
    if not light and color == (20, 20, 20):
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def build_parts(photo, detections, cal) -> list[SimPart]:
    parts = []
    for d in detections:
        poly = np.array([cal.to_robot(float(u), float(v), strict=False) for u, v in d.box])
        m = np.zeros(photo.shape[:2], np.uint8)
        cv2.fillPoly(m, [np.intp(d.box)], 255)
        color = tuple(int(c) for c in cv2.mean(photo, mask=m)[:3])
        parts.append(SimPart(poly, color, width_mm=d.size[1] * 0.5))
    return parts


def run_simulation(photo: np.ndarray, live: bool = True, record: Optional[Path] = None,
                   script: Optional[Script] = None, speed: float = 1.0) -> dict:
    cal = PixelToRobot()
    detections = detect_objects(photo)
    targets = [cal.pick_target(*d.center, d.angle_deg) for d in detections]
    cfg = CycleConfig()
    parts = build_parts(photo, detections, cal)
    robot = SimRobot(parts, cfg.home, cfg.place, speed=speed)
    cycle = PickPlaceCycle(robot, cfg)
    view = SimView(photo, cal, detections, targets, robot, cycle, live=live, record=record)
    script = script or Script()
    robot.forced_fails = dict(script.grip_failures)
    estop_done = False

    def estop_hook(fraction: float) -> bool:
        nonlocal estop_done
        if script.estop and not estop_done:
            part, step, frac = script.estop
            if robot.current_part == part and cycle.current_step == step and fraction >= frac:
                estop_done = True
                view.flash["estop"] = 12        # highlight the button in the video
                return True
        return False
    robot.estop_hook = estop_hook

    reports = []
    try:
        view.hold(0.8)
        queue = list(range(len(targets)))
        while queue:
            i = queue[0]
            robot.current_part = i
            rep = cycle.run(targets[i])
            if rep.result is CycleResult.ESTOP:
                view.results.append(f"#{i}:ESTOP")
                _wait_for_reset(view, robot, cycle, live, script)
                robot.operator_clears_gripper()
                continue                      # retry the same part after reset
            view.results.append(f"#{i}:{rep.result.value}")
            reports.append((i, rep))
            queue.pop(0)
        view.hold(1.5)
    except QuitSim:
        log.info("quit by user")
    finally:
        view.save()
        if live:
            cv2.destroyAllWindows()
    return {"reports": reports, "parts": parts, "robot": robot, "view": view}


def _wait_for_reset(view, robot, cycle, live, script) -> None:
    """After an e-stop: wait until it is released and reset."""
    if live:
        view.reset_requested = False
        while True:
            view.tick()
            if view.reset_requested and cycle.reset():
                return
            view.reset_requested = False
    else:  # demo: keep pressed, release, wait, reset
        view.hold(script.estop_hold_s)
        robot.release_estop()
        view.hold(script.estop_hold_s / 2)
        view.flash["reset"] = 12
        view.hold(0.4)
        cycle.reset()


DEMO_SCRIPT = Script(grip_failures={1: 1, 2: 3}, estop=(3, "descend_pick", 0.5))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", type=Path, default=Path("data/test_photo.jpg"))
    p.add_argument("--paper", action="store_true",
                   help="hand-held photo: find the white sheet, remove perspective, crop to it")
    p.add_argument("--camera", type=int, default=None)
    p.add_argument("--speed", type=float, default=1.0, help="animation speed factor")
    p.add_argument("--demo", action="store_true",
                   help="scripted: part 1 fails once, part 2 fails 3x (skipped), e-stop on part 3")
    p.add_argument("--record", type=Path, default=None, help="save an .mp4 of the run")
    p.add_argument("--headless", action="store_true", help="no window (use with --record)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s: %(message)s")

    try:
        photo = FrameSource(args.camera, args.image,
                            preprocess=rectify_paper if args.paper else None).read()
    except CameraError as exc:
        log.error("%s", exc)
        return 2
    script = DEMO_SCRIPT if args.demo else Script()
    out = run_simulation(photo, live=not args.headless, record=args.record, script=script, speed=args.speed)
    for i, rep in out["reports"]:
        print(f"part {i}: {rep.result.value:<12} attempts={rep.grip_attempts} {rep.message}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
