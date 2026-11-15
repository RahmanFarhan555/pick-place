"""Tests for the simulator. They run without a window."""
import pytest

from pickplace.calibration import Pose
from pickplace.cycle import CycleResult
from pickplace.robot import EmergencyStopError
from pickplace.simulator import Script, SimPart, SimRobot, run_simulation
from pickplace.synthetic import render_scene

import numpy as np


def test_scripted_demo_runs_the_real_cycle():
    script = Script(grip_failures={1: 1, 2: 3}, estop=(3, "descend_pick", 0.5), estop_hold_s=0.1)
    out = run_simulation(render_scene(), live=False, script=script, speed=50)
    results = {i: rep for i, rep in out["reports"]}
    assert results[0].result is CycleResult.SUCCESS
    assert results[1].result is CycleResult.SUCCESS and results[1].grip_attempts == 2
    assert results[2].result is CycleResult.GRIP_FAILED
    assert results[3].result is CycleResult.SUCCESS          # retried after e-stop + reset
    assert "#3:ESTOP" in out["view"].results
    states = [p.state for p in out["parts"]]
    assert states == ["placed", "placed", "table", "placed"]  # skipped part stays on the table


def test_geometric_grip_check_fails_when_not_over_part():
    part = SimPart(np.array([[0, 0], [0, 10], [10, 10], [10, 0]], float), (0, 0, 0), 10)
    robot = SimRobot([part], home=Pose(50, 50, 20), place=Pose(300, 0, 20), speed=1000)
    robot.close_gripper()
    assert not robot.gripper_holds_part()
    robot.open_gripper()
    robot.pose = Pose(5, 5, 20)
    robot.close_gripper()
    assert robot.gripper_holds_part() and part.state == "held"


def test_estop_freezes_robot_mid_motion():
    robot = SimRobot([], home=Pose(100, 0, 100), place=Pose(300, 0, 20), speed=1)
    robot.estop_hook = lambda fraction: fraction >= 0.5
    with pytest.raises(EmergencyStopError):
        robot.move_linear(Pose(100, 0, 0), 100)
    assert 0 < robot.pose.z < 100        # stopped on the way, not at the target


def make_view():
    from pickplace.calibration import PixelToRobot
    from pickplace.cycle import CycleConfig, PickPlaceCycle
    from pickplace.simulator import SimView, build_parts
    from pickplace.vision import detect_objects
    photo = render_scene()
    cal = PixelToRobot()
    dets = detect_objects(photo)
    targets = [cal.pick_target(*d.center, d.angle_deg) for d in dets]
    cfg = CycleConfig()
    robot = SimRobot(build_parts(photo, dets, cal), cfg.home, cfg.place, speed=100)
    cycle = PickPlaceCycle(robot, cfg)
    return SimView(photo, cal, dets, targets, robot, cycle, live=False), robot


def click(view, action):
    import cv2
    b = next(b for b in view.buttons if b.action == action)
    x, y, w, h = b.rect
    view.on_mouse(cv2.EVENT_LBUTTONDOWN, x + w // 2, y + h // 2)


def test_estop_and_reset_buttons():
    view, robot = make_view()
    click(view, "estop")
    assert robot.is_estop_active()
    click(view, "reset")
    assert not robot.is_estop_active() and view.reset_requested


def test_fail_pause_quit_buttons():
    view, robot = make_view()
    click(view, "fail")
    assert robot.forced_fails[robot.current_part] == 1
    click(view, "pause")
    assert view.paused
    click(view, "pause")
    assert not view.paused
    click(view, "quit")
    assert view.quit_requested


def test_click_outside_buttons_does_nothing():
    import cv2
    view, robot = make_view()
    view.on_mouse(cv2.EVENT_LBUTTONDOWN, 5, 5)          # on the top view
    assert not robot.is_estop_active() and not view.paused


def test_estop_button_is_drawn_red():
    view, _ = make_view()
    frame = view.render()
    b = next(b for b in view.buttons if b.action == "estop")
    x, y, w, h = b.rect
    bgr = frame[y + 3, x + 5]
    assert bgr[2] > 150 and bgr[0] < 60 and bgr[1] < 60


def test_clicking_estop_during_a_running_cycle_stops_the_robot_mid_move():
    """Click E-STOP while the robot is moving. It must stop at once."""
    view, robot = make_view()
    cycle = view.cycle
    target = view.targets[0]
    frames = {"n": 0}
    original_tick = view.tick

    def tick_and_click():
        original_tick()
        frames["n"] += 1
        # click while descending to the part (mid-move)
        if cycle.current_step == "descend_pick" and robot.pose.z < 50 and not robot.is_estop_active():
            click(view, "estop")
    view.tick = tick_and_click

    report = cycle.run(target)
    assert report.result is CycleResult.ESTOP
    assert report.aborted_at == "descend_pick"
    assert target.pick.z < robot.pose.z < 50          # stopped between approach and pick height
    assert robot.stopped and cycle.estop_latched
    frozen, n_calls = robot.pose, len(robot.calls)

    # still stopped until RELEASE + RESET is clicked
    assert cycle.run(target).result is CycleResult.ESTOP
    assert robot.pose == frozen and len(robot.calls) == n_calls

    view.tick = original_tick                        # the user stops clicking
    click(view, "reset")
    assert cycle.reset()
    assert cycle.run(target).result is CycleResult.SUCCESS
