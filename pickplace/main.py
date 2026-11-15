"""Run everything: read frame, find parts, convert to robot poses, run the cycle.

    python -m pickplace.main --image data/test_photo.jpg --paper
    python -m pickplace.main --camera 0 --image data/test_photo.jpg --paper
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import cv2

from .calibration import PixelToRobot
from .camera import CameraError, FrameSource
from .cycle import CycleConfig, PickPlaceCycle
from .robot import FakeRobot
from .paper import PaperNotFound, rectify_paper
from .vision import annotate, detect_objects

log = logging.getLogger("pickplace")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--paper", action="store_true",
                   help="hand-held photo: find the white sheet, remove perspective, crop to it")
    p.add_argument("--camera", type=int, default=None, help="webcam index (omit to use only --image)")
    p.add_argument("--image", type=Path, default=Path("data/test_photo.jpg"), help="image file / fallback")
    p.add_argument("--retries", type=int, default=3, help="camera read attempts")
    p.add_argument("--out", type=Path, default=Path("results"), help="output directory")
    p.add_argument("--no-sim", action="store_true", help="skip the simulated pick and place cycle")
    p.add_argument("--grip-fail", type=int, default=0,
                   help="simulate N failed grips on the first part (demo of retry/alarm)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)-8s %(name)s: %(message)s")

    # read the frame
    source = FrameSource(camera_index=args.camera,
                         image_path=args.image if args.image else None,
                         retries=args.retries,
                         preprocess=rectify_paper if args.paper else None)
    try:
        frame = source.read()
    except (CameraError, PaperNotFound) as exc:
        log.error("%s", exc)
        log.error("Connect a camera (--camera 0) or pass an existing image (--image path).")
        return 2
    log.info("frame from %s", source.last_source)

    # find the parts
    detections = detect_objects(frame)
    if not detections:
        log.warning("no parts detected")

    # pixel to robot
    cal = PixelToRobot()
    targets = [cal.pick_target(*d.center, d.angle_deg) for d in detections]

    print(f"\n{'#':>2} {'u':>7} {'v':>7} {'ang_img':>8} | {'x_mm':>7} {'y_mm':>7} {'rz':>7} "
          f"| pick z | appr z")
    for i, (d, t) in enumerate(zip(detections, targets)):
        print(f"{i:>2} {d.center[0]:7.1f} {d.center[1]:7.1f} {d.angle_deg:8.1f} | "
              f"{t.pick.x:7.1f} {t.pick.y:7.1f} {t.pick.rz:7.1f} | {t.pick.z:6.0f} | {t.approach.z:6.0f}")
    print()

    args.out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out / "input_640x480.png"), frame)   # the image used for detection
    out_img = args.out / "annotated.png"
    cv2.imwrite(str(out_img), annotate(frame, detections, targets, cal.pixel_points))
    log.info("annotated image saved to %s", out_img)

    result = {
        "source": source.last_source,
        "parts": [
            {"pixel": {"u": d.center[0], "v": d.center[1], "angle_deg": d.angle_deg},
             "pick": asdict(t.pick), "approach": asdict(t.approach)}
            for d, t in zip(detections, targets)
        ],
    }

    # run the cycle on the fake robot
    if not args.no_sim and targets:
        robot = FakeRobot(grip_results=[False] * args.grip_fail)
        cycle = PickPlaceCycle(robot, CycleConfig())
        reports = cycle.run_all(targets)
        result["cycle"] = [{"result": r.result.value, "grip_attempts": r.grip_attempts,
                            "message": r.message} for r in reports]
        for i, r in enumerate(reports):
            print(f"part {i}: {r.result.value:<12} attempts={r.grip_attempts} {r.message}")

    (args.out / "result.json").write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
