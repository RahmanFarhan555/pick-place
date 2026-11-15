import pytest

from pickplace.calibration import PixelToRobot, Pose
from pickplace.cycle import CycleConfig, CycleResult, PickPlaceCycle
from pickplace.robot import FakeRobot, RobotError


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


@pytest.fixture
def target():
    return PixelToRobot().pick_target(320, 240, angle_img_deg=30)


def make_cycle(robot, **cfg):
    clock = FakeClock()
    return PickPlaceCycle(robot, CycleConfig(**cfg), clock=clock, sleep=clock.sleep), clock


SUCCESS_SEQUENCE = [
    "open_gripper",
    "move_joint",          # approach
    "move_linear",         # descend to pick
    "close_gripper",
    "gripper_holds_part",  # grip check
    "move_linear",         # retract
    "move_joint",          # above place
    "gripper_holds_part",  # verify still holding
    "move_linear",         # descend place
    "open_gripper",        # release
    "move_linear",         # retract
    "move_joint",          # home
]


def test_successful_cycle(target):
    robot = FakeRobot()
    cycle, _ = make_cycle(robot)
    rep = cycle.run(target)
    assert rep.result is CycleResult.SUCCESS
    assert rep.grip_attempts == 1
    assert robot.command_names() == SUCCESS_SEQUENCE
    assert robot.alarms == []


def test_successful_cycle_visits_correct_poses(target):
    robot = FakeRobot()
    cfg = CycleConfig()
    cycle, _ = make_cycle(robot)
    cycle.run(target)
    poses = [arg for name, arg in robot.calls if name.startswith("move_")]
    place_app = cfg.place.offset(dz=50)
    assert poses == [target.approach, target.pick, target.approach,
                     place_app, cfg.place, place_app, cfg.home]
    assert target.approach.z - target.pick.z == 50
    assert target.pick.z == 20


def test_approach_and_descend_motion_types(target):
    """The move to the approach pose is joint, the last 50 mm are linear."""
    robot = FakeRobot()
    make_cycle(robot)[0].run(target)
    moves = [(n, a) for n, a in robot.calls if n.startswith("move_")]
    assert moves[0] == ("move_joint", target.approach)
    assert moves[1] == ("move_linear", target.pick)


def test_grip_fails_once_then_succeeds(target):
    robot = FakeRobot(grip_results=[False, True])
    rep = make_cycle(robot)[0].run(target)
    assert rep.result is CycleResult.SUCCESS
    assert rep.grip_attempts == 2
    assert robot.command_names().count("close_gripper") == 2
    assert robot.alarms == []


def test_grip_fails_three_times_skips_part_and_raises_alarm(target):
    robot = FakeRobot(grip_results=[False, False, False])
    cfg = CycleConfig()
    rep = make_cycle(robot)[0].run(target)
    assert rep.result is CycleResult.GRIP_FAILED
    assert rep.grip_attempts == 3                       # 1 try + 2 retries
    assert robot.command_names().count("close_gripper") == 3
    assert len(robot.alarms) == 1 and "grip failed" in robot.alarms[0]
    # the robot did not go to the place position and ended at home
    assert all(arg != cfg.place for _, arg in robot.calls)
    assert robot.pose == cfg.home
    # after a failed grip the gripper opens and the robot moves up
    names = robot.command_names()
    first_fail = names.index("gripper_holds_part")
    assert names[first_fail + 1:first_fail + 3] == ["open_gripper", "move_linear"]


def test_retry_count_is_configurable(target):
    robot = FakeRobot(grip_results=[False] * 10)
    rep = make_cycle(robot, max_grip_retries=0)[0].run(target)
    assert rep.result is CycleResult.GRIP_FAILED and rep.grip_attempts == 1


def test_batch_continues_after_skipped_part(target):
    robot = FakeRobot(grip_results=[False, False, False, True])
    reports = make_cycle(robot)[0].run_all([target, target])
    assert [r.result for r in reports] == [CycleResult.GRIP_FAILED, CycleResult.SUCCESS]


def test_part_lost_during_transfer(target):
    robot = FakeRobot(drop_part_on="move_joint")
    rep = make_cycle(robot)[0].run(target)
    assert rep.result is CycleResult.PART_LOST
    assert robot.alarms and "lost" in robot.alarms[0]
    assert all(arg != CycleConfig().place for _, arg in robot.calls)


ESTOP_POINTS = [
    ("open_gripper", 1),
    ("move_joint", 1),          # during approach
    ("move_linear", 1),         # during descend
    ("close_gripper", 1),
    ("gripper_holds_part", 1),
    ("move_linear", 2),         # retract with part
    ("move_joint", 2),          # transfer
    ("move_linear", 3),         # descend at place
    ("open_gripper", 2),        # release
    ("move_joint", 3),          # going home
]


@pytest.mark.parametrize("estop_on", ESTOP_POINTS)
def test_estop_aborts_at_any_step(target, estop_on):
    robot = FakeRobot(estop_on=estop_on)
    cycle, _ = make_cycle(robot)
    rep = cycle.run(target)
    assert rep.result is CycleResult.ESTOP
    assert robot.stopped
    assert cycle.estop_latched
    # after the e-stop only stop() and the alarm are sent
    names = robot.command_names()
    hit = [i for i, n in enumerate(names) if n == estop_on[0]][estop_on[1] - 1]
    assert names[hit + 1:] == ["stop", "set_alarm"]


def test_estop_while_waiting_for_part(target):
    clock_calls = {"n": 0}
    robot = FakeRobot(part_present=False)

    def part_present():
        clock_calls["n"] += 1
        if clock_calls["n"] == 3:
            robot.trigger_estop()
        return False
    robot._part_present = part_present
    rep = make_cycle(robot)[0].run(target)
    assert rep.result is CycleResult.ESTOP
    assert rep.aborted_at == "wait_for_part"
    assert not any(n.startswith("move_") for n in robot.command_names())


def test_estop_active_before_start_refuses_to_move(target):
    robot = FakeRobot()
    robot.trigger_estop()
    rep = make_cycle(robot)[0].run(target)
    assert rep.result is CycleResult.ESTOP
    assert robot.command_names() == []


def test_no_auto_restart_after_estop_release(target):
    robot = FakeRobot(estop_on=("move_linear", 1))
    cycle, _ = make_cycle(robot)
    assert cycle.run(target).result is CycleResult.ESTOP

    robot.release_estop()
    n = len(robot.calls)
    assert cycle.run(target).result is CycleResult.ESTOP   # still locked
    assert len(robot.calls) == n                            # and did not move

    assert cycle.reset()
    rep = cycle.run(target)
    assert rep.result is CycleResult.SUCCESS
    # the first move after reset is home, and the gripper is not opened before
    new = robot.calls[n:]
    assert new[0][0] == "move_joint" and new[0][1] == CycleConfig().home


def test_reset_refused_while_estop_pressed(target):
    robot = FakeRobot(estop_on=("move_joint", 1))
    cycle, _ = make_cycle(robot)
    cycle.run(target)
    assert cycle.reset() is False
    assert cycle.estop_latched


def test_batch_stops_on_estop(target):
    robot = FakeRobot(estop_on=("close_gripper", 1))
    reports = make_cycle(robot)[0].run_all([target, target, target])
    assert len(reports) == 1 and reports[0].result is CycleResult.ESTOP


def test_timeout_when_no_part(target):
    robot = FakeRobot(part_present=False)
    cycle, clock = make_cycle(robot, part_timeout_s=2.0)
    rep = cycle.run(target)
    assert rep.result is CycleResult.NO_PART
    assert clock.t == pytest.approx(2.0, abs=0.06)
    assert robot.command_names() == []


def test_part_arrives_before_timeout(target):
    clock = FakeClock()
    robot = FakeRobot(part_present=lambda: clock.t >= 1.0)
    cycle = PickPlaceCycle(robot, CycleConfig(part_timeout_s=3.0), clock=clock, sleep=clock.sleep)
    assert cycle.run(target).result is CycleResult.SUCCESS
    assert 1.0 <= clock.t < 1.1


def test_target_outside_workspace_is_rejected_before_motion():
    robot = FakeRobot()
    far = PixelToRobot().pick_target(320, 240)
    from pickplace.calibration import PickTarget
    bad = PickTarget(pick=Pose(900, 0, 20), approach=Pose(900, 0, 70))
    rep = make_cycle(robot)[0].run(bad)
    assert rep.result is CycleResult.OUT_OF_REACH
    assert robot.command_names() == ["set_alarm"]
    assert make_cycle(FakeRobot())[0].run(far).result is CycleResult.SUCCESS


def test_robot_fault_stops_and_alarms(target):
    class FaultyRobot(FakeRobot):
        def move_linear(self, pose, speed_mm_s=100.0):
            raise RobotError("following error")
    robot = FaultyRobot()
    rep = make_cycle(robot)[0].run(target)
    assert rep.result is CycleResult.FAULT
    assert rep.aborted_at == "descend_pick"
    assert robot.stopped and robot.alarms
