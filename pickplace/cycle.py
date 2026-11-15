"""Pick and place cycle.

Steps for one part:
    1. wait for the part (timeout)
    2. open gripper, move to the approach pose
    3. move down to the pick pose, close gripper, check the grip
       (if it fails: open, move up, try again, max 2 retries)
    4. move up, move above the place position, check the part is still held
    5. move down, open gripper, move up, go home

The e-stop is checked before and after every step. After an e-stop the cycle
stays locked until the e-stop is released and reset() is called.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable, Optional

from .calibration import PickTarget, Pose
from .robot import EmergencyStopError, RobotError, RobotInterface

log = logging.getLogger(__name__)


class CycleResult(Enum):
    SUCCESS = "success"
    NO_PART = "no_part"            # timeout waiting for a part
    GRIP_FAILED = "grip_failed"    # skipped after retries, alarm raised
    PART_LOST = "part_lost"        # dropped during transfer, alarm raised
    OUT_OF_REACH = "out_of_reach"  # target outside workspace, alarm raised
    ESTOP = "estop"                # aborted by emergency stop
    FAULT = "fault"                # other robot error


@dataclass
class CycleReport:
    result: CycleResult
    grip_attempts: int = 0
    message: str = ""
    aborted_at: Optional[str] = None


@dataclass
class Workspace:
    """Allowed work area of the robot in mm."""
    x: tuple[float, float] = (100.0, 350.0)
    y: tuple[float, float] = (-220.0, 260.0)
    z: tuple[float, float] = (10.0, 300.0)   # not below 10 mm, the table is there

    def contains(self, p: Pose) -> bool:
        return (self.x[0] <= p.x <= self.x[1] and self.y[0] <= p.y <= self.y[1]
                and self.z[0] <= p.z <= self.z[1])


@dataclass
class CycleConfig:
    home: Pose = field(default_factory=lambda: Pose(200.0, -40.0, 200.0, 0.0))
    place: Pose = field(default_factory=lambda: Pose(220.0, 200.0, 20.0, 0.0))  # beside the camera area
    place_approach_offset: float = 50.0
    part_timeout_s: float = 3.0
    poll_interval_s: float = 0.05
    max_grip_retries: int = 2          # 3 grip attempts in total
    approach_speed_pct: float = 50.0   # joint moves in free space
    linear_speed_mm_s: float = 50.0    # slow linear moves near the table
    workspace: Workspace = field(default_factory=Workspace)


class PickPlaceCycle:
    def __init__(self, robot: RobotInterface, config: CycleConfig = CycleConfig(),
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.robot = robot
        self.cfg = config
        self._clock = clock
        self._sleep = sleep
        self.estop_latched = False
        self._needs_homing = False
        self.current_step: Optional[str] = None

    def reset(self) -> bool:
        """Operator reset after an e-stop. Works only when the e-stop is released."""
        if self.robot.is_estop_active():
            log.warning("reset refused: e-stop is still active")
            return False
        if self.estop_latched:
            log.info("e-stop acknowledged; robot will home slowly before the next part")
            self.estop_latched = False
            self._needs_homing = True
        return True

    def run(self, target: PickTarget) -> CycleReport:
        if self.estop_latched or self.robot.is_estop_active():
            self.estop_latched = True
            return CycleReport(CycleResult.ESTOP, message="e-stop latched: release and reset() first")

        bad = self._unreachable(target)
        if bad:
            self.robot.set_alarm(bad)
            return CycleReport(CycleResult.OUT_OF_REACH, message=bad)

        attempts = 0
        try:
            if self._needs_homing:
                # after an e-stop the gripper may still hold a part, so do not open it.
                # Go home slowly first. The operator removes the part.
                self._step("recover_home", self.robot.move_joint, self.cfg.home, 10.0)
                self._needs_homing = False

            if not self._wait_for_part():
                log.warning("no part within %.1f s", self.cfg.part_timeout_s)
                return CycleReport(CycleResult.NO_PART, message="timeout waiting for part")

            self._step("open_gripper", self.robot.open_gripper)
            self._step("move_approach", self.robot.move_joint, target.approach, self.cfg.approach_speed_pct)

            max_attempts = 1 + self.cfg.max_grip_retries
            gripped = False
            while attempts < max_attempts and not gripped:
                attempts += 1
                self._step("descend_pick", self.robot.move_linear, target.pick, self.cfg.linear_speed_mm_s)
                self._step("close_gripper", self.robot.close_gripper)
                gripped = self._step("check_grip", self.robot.gripper_holds_part)
                if not gripped:
                    log.warning("grip check failed (attempt %d/%d)", attempts, max_attempts)
                    self._step("open_gripper", self.robot.open_gripper)
                self._step("retract_pick", self.robot.move_linear, target.approach, self.cfg.linear_speed_mm_s)

            if not gripped:
                msg = f"grip failed after {attempts} attempts - part skipped"
                self._step("move_home", self.robot.move_joint, self.cfg.home, self.cfg.approach_speed_pct)
                self.robot.set_alarm(msg)
                return CycleReport(CycleResult.GRIP_FAILED, attempts, msg)

            place_approach = self.cfg.place.offset(dz=self.cfg.place_approach_offset)
            self._step("move_place_approach", self.robot.move_joint, place_approach, self.cfg.approach_speed_pct)
            if not self._step("verify_grip", self.robot.gripper_holds_part):
                msg = "part lost during transfer"
                self._step("open_gripper", self.robot.open_gripper)
                self._step("move_home", self.robot.move_joint, self.cfg.home, self.cfg.approach_speed_pct)
                self.robot.set_alarm(msg)
                return CycleReport(CycleResult.PART_LOST, attempts, msg)
            self._step("descend_place", self.robot.move_linear, self.cfg.place, self.cfg.linear_speed_mm_s)
            self._step("release", self.robot.open_gripper)
            self._step("retract_place", self.robot.move_linear, place_approach, self.cfg.linear_speed_mm_s)
            self._step("move_home", self.robot.move_joint, self.cfg.home, self.cfg.approach_speed_pct)
            return CycleReport(CycleResult.SUCCESS, attempts, "part placed")

        except EmergencyStopError as exc:
            return self._abort_estop(exc, attempts)
        except RobotError as exc:
            self._safe_stop()
            msg = f"robot fault in step {self.current_step}: {exc}"
            self.robot.set_alarm(msg)
            return CycleReport(CycleResult.FAULT, attempts, msg, aborted_at=self.current_step)

    def run_all(self, targets: Iterable[PickTarget]) -> list[CycleReport]:
        """Run the cycle for every part. Stops on e-stop or robot error."""
        reports = []
        for i, t in enumerate(targets):
            rep = self.run(t)
            log.info("part %d: %s %s", i, rep.result.value, rep.message)
            reports.append(rep)
            if rep.result in (CycleResult.ESTOP, CycleResult.FAULT):
                break
        return reports

    def _step(self, name: str, fn, *args):
        """Run one step and check the e-stop before and after it."""
        self.current_step = name
        self._check_estop()
        result = fn(*args)
        self._check_estop()
        return result

    def _check_estop(self) -> None:
        if self.robot.is_estop_active():
            raise EmergencyStopError(f"e-stop active at step {self.current_step}")

    def _wait_for_part(self) -> bool:
        self.current_step = "wait_for_part"
        deadline = self._clock() + self.cfg.part_timeout_s
        while True:
            self._check_estop()
            if self.robot.read_input("part_present"):
                return True
            if self._clock() >= deadline:
                return False
            self._sleep(self.cfg.poll_interval_s)

    def _abort_estop(self, exc: Exception, attempts: int) -> CycleReport:
        self._safe_stop()
        self.estop_latched = True
        msg = f"EMERGENCY STOP during {self.current_step}: cycle aborted"
        log.critical(msg)
        try:
            self.robot.set_alarm(msg)
        except Exception:  # an alarm error must not hide the e-stop
            log.exception("could not set alarm")
        return CycleReport(CycleResult.ESTOP, attempts, msg, aborted_at=self.current_step)

    def _safe_stop(self) -> None:
        try:
            self.robot.stop()
        except Exception:
            log.exception("robot.stop() failed")

    def _unreachable(self, t: PickTarget) -> Optional[str]:
        ws = self.cfg.workspace
        place_app = self.cfg.place.offset(dz=self.cfg.place_approach_offset)
        for name, p in (("pick", t.pick), ("approach", t.approach), ("place", self.cfg.place),
                        ("place approach", place_app), ("home", self.cfg.home)):
            if not ws.contains(p):
                return f"{name} pose {p} outside workspace - part skipped"
        return None
