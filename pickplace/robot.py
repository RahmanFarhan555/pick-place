"""Robot interface and a fake robot for the tests."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Callable, Iterable, Optional

from .calibration import Pose

log = logging.getLogger(__name__)


class RobotError(RuntimeError):
    """Robot error, for example connection lost."""


class EmergencyStopError(RobotError):
    """The emergency stop is active."""


class RobotInterface(ABC):
    @abstractmethod
    def move_joint(self, pose: Pose, speed_pct: float = 50.0) -> None:
        """Fast move, the path is not a straight line. Used in free space."""

    @abstractmethod
    def move_linear(self, pose: Pose, speed_mm_s: float = 100.0) -> None:
        """Move in a straight line. Used near the part."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the robot now."""

    @abstractmethod
    def open_gripper(self) -> None: ...

    @abstractmethod
    def close_gripper(self) -> None: ...

    @abstractmethod
    def gripper_holds_part(self) -> bool:
        """Gripper feedback: True if a part is in the gripper."""

    @abstractmethod
    def read_input(self, name: str) -> bool:
        """Read a digital input, for example 'part_present'."""

    @abstractmethod
    def is_estop_active(self) -> bool: ...

    @abstractmethod
    def set_alarm(self, message: str) -> None:
        """Raise an alarm for the operator."""


class FakeRobot(RobotInterface):
    """Fake robot for the tests. It saves every command in .calls.

    grip_results  list of True/False for the next grip checks
    part_present  True/False or a function, for the 'part_present' input
    estop_on      (command, n): the e-stop is pressed at the n-th call
    drop_part_on  command name after which the part is lost
    """

    def __init__(self,
                 grip_results: Iterable[bool] = (),
                 part_present: bool | Callable[[], bool] = True,
                 estop_on: Optional[tuple[str, int]] = None,
                 drop_part_on: Optional[str] = None,
                 default_grip: bool = True):
        self._grip_results = list(grip_results)
        self._default_grip = default_grip
        self._part_present = part_present
        self._estop_on = estop_on
        self._drop_part_on = drop_part_on
        self._estop = False
        self._holding = False
        self._gripper_closed = False
        self._counts: dict[str, int] = {}
        self.pose: Optional[Pose] = None
        self.calls: list[tuple[str, object]] = []
        self.alarms: list[str] = []
        self.stopped = False

    def _record(self, name: str, arg=None) -> None:
        self._counts[name] = self._counts.get(name, 0) + 1
        self.calls.append((name, arg))
        if self._estop_on and self._estop_on == (name, self._counts[name]):
            self._estop = True
            raise EmergencyStopError(f"e-stop pressed during {name}")

    def _require_no_estop(self, name: str) -> None:
        if self._estop:
            raise EmergencyStopError(f"{name} rejected: e-stop active")

    def trigger_estop(self) -> None:
        self._estop = True

    def release_estop(self) -> None:
        self._estop = False

    def command_names(self) -> list[str]:
        return [c[0] for c in self.calls]

    def move_joint(self, pose: Pose, speed_pct: float = 50.0) -> None:
        self._require_no_estop("move_joint")
        self._record("move_joint", pose)
        self.pose = pose
        self._maybe_drop("move_joint")

    def move_linear(self, pose: Pose, speed_mm_s: float = 100.0) -> None:
        self._require_no_estop("move_linear")
        self._record("move_linear", pose)
        self.pose = pose
        self._maybe_drop("move_linear")

    def stop(self) -> None:
        self.stopped = True
        self.calls.append(("stop", None))

    def _maybe_drop(self, name):
        if self._drop_part_on == name and self._holding:
            self._holding = False

    def open_gripper(self) -> None:
        self._require_no_estop("open_gripper")
        self._record("open_gripper")
        self._gripper_closed = False
        self._holding = False

    def close_gripper(self) -> None:
        self._require_no_estop("close_gripper")
        self._record("close_gripper")
        self._gripper_closed = True
        self._holding = self._grip_results.pop(0) if self._grip_results else self._default_grip

    def gripper_holds_part(self) -> bool:
        self._record("gripper_holds_part")
        return self._gripper_closed and self._holding

    def read_input(self, name: str) -> bool:
        if name != "part_present":
            raise RobotError(f"unknown input {name!r}")
        p = self._part_present
        return bool(p() if callable(p) else p)

    def is_estop_active(self) -> bool:
        return self._estop

    def set_alarm(self, message: str) -> None:
        log.error("ALARM: %s", message)
        self.alarms.append(message)
        self.calls.append(("set_alarm", message))
