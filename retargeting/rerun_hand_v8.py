from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import rerun as rr

from retargeting.hand_v8_mapping import FINGER_ROBOT, robot_keypoints
from retargeting.manus_keypoints import FINGER_ORDER, frame_points_in_wrist, iter_manus_frames
from retargeting.retarget_hand_v8 import DEFAULT_URDF, HandV8Retargeter


COLORS = {
    "thumb": (245, 180, 70),
    "index": (80, 190, 255),
    "middle": (90, 230, 120),
    "ring": (220, 120, 255),
    "pinky": (255, 100, 110),
}


def _flatten(points: dict[str, list[np.ndarray]]) -> tuple[list[np.ndarray], list[tuple[int, int, int]], list[str]]:
    flat: list[np.ndarray] = []
    colors: list[tuple[int, int, int]] = []
    labels: list[str] = []
    for finger in FINGER_ORDER:
        for i, p in enumerate(points.get(finger, [])):
            flat.append(p)
            colors.append(COLORS[finger])
            labels.append(f"{finger}_{i}")
    return flat, colors, labels


def _strips(points: dict[str, list[np.ndarray]]) -> tuple[list[list[list[float]]], list[tuple[int, int, int]]]:
    strips = []
    colors = []
    for finger in FINGER_ORDER:
        pts = points.get(finger, [])
        if len(pts) >= 2:
            strips.append([p.tolist() for p in pts])
            colors.append(COLORS[finger])
    return strips, colors


def log_hand(entity: str, points: dict[str, list[np.ndarray]], radius: float = 0.006) -> None:
    flat, colors, labels = _flatten(points)
    strips, strip_colors = _strips(points)
    if flat:
        rr.log(f"{entity}/points", rr.Points3D([p.tolist() for p in flat], colors=colors, radii=radius, labels=labels))
    if strips:
        rr.log(f"{entity}/segments", rr.LineStrips3D(strips, colors=strip_colors, radii=radius * 0.55))


def log_robot_joint_scalars(joint_names: list[str], q: np.ndarray) -> None:
    for name, value in zip(joint_names, q):
        rr.log(f"retarget/joints/{name}", rr.Scalars([float(value)]))


def run_once(
    manus_csv: Path,
    urdf_path: Path,
    max_frames: int,
    calibration_frame: int,
    realtime: bool,
) -> int:
    retargeter = HandV8Retargeter(urdf_path)
    frames = iter_manus_frames(manus_csv)
    for i, frame in enumerate(frames):
        points = frame_points_in_wrist(frame)
        if i < calibration_frame:
            continue
        retargeter.calibrate(points)
        first_frame = frame
        break
    else:
        raise RuntimeError(f"No MANUS frames found in {manus_csv}")

    first_t = first_frame.t_wall_ns
    last_t = first_t
    count = 0
    pending = [first_frame]
    for frame in frames:
        pending.append(frame)
        if max_frames and len(pending) >= max_frames:
            break

    for frame in pending:
        if realtime and count > 0:
            dt = max(0.0, min(0.05, (frame.t_wall_ns - last_t) / 1e9))
            time.sleep(dt)
        last_t = frame.t_wall_ns
        count += log_frame(retargeter, frame, first_t)
    return count


def log_frame(retargeter: HandV8Retargeter, frame, first_t_wall_ns: int) -> int:
    rr.set_time("manus_frame", sequence=frame.frame_seq)
    rr.set_time("capture_time", duration=(frame.t_wall_ns - first_t_wall_ns) / 1e9)
    manus_points = frame_points_in_wrist(frame)
    target_points = retargeter.target_points(manus_points)
    q, stats = retargeter.solve(manus_points)
    robot_points = robot_keypoints(retargeter.kin, q)

    log_hand("retarget/manus_wrist_points", manus_points, radius=0.004)
    log_hand("retarget/robot_hand", robot_points, radius=0.006)
    log_hand("retarget/robot_targets", target_points, radius=0.003)
    log_robot_joint_scalars(retargeter.joint_names, q)
    rr.log("retarget/error/mean_tip_m", rr.Scalars([stats["mean_tip_error_m"]]))
    rr.log("retarget/error/ik_cost", rr.Scalars([stats["cost"]]))
    return 1


def run_follow(manus_csv: Path, urdf_path: Path, calibration_frame: int, poll_s: float) -> None:
    retargeter = HandV8Retargeter(urdf_path)
    processed: set[int] = set()
    first_t_wall_ns: int | None = None
    calibrated = False
    print(f"Following {manus_csv}")
    while True:
        if not manus_csv.exists():
            time.sleep(poll_s)
            continue
        frames = list(iter_manus_frames(manus_csv))
        for idx, frame in enumerate(frames):
            if frame.frame_seq in processed:
                continue
            # Avoid likely partial tail frames while the logger is writing.
            if idx == len(frames) - 1:
                continue
            points = frame_points_in_wrist(frame)
            if not calibrated:
                if len(processed) < calibration_frame:
                    processed.add(frame.frame_seq)
                    continue
                retargeter.calibrate(points)
                first_t_wall_ns = frame.t_wall_ns
                calibrated = True
            assert first_t_wall_ns is not None
            log_frame(retargeter, frame, first_t_wall_ns)
            processed.add(frame.frame_seq)
        time.sleep(poll_s)


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize MANUS to Hand V8 retargeting in Rerun.")
    parser.add_argument("--manus-csv", required=True, type=Path)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--max-frames", type=int, default=500, help="0 = all frames in replay mode")
    parser.add_argument("--calibration-frame", type=int, default=0)
    parser.add_argument("--follow", action="store_true", help="Follow a live, growing MANUS CSV")
    parser.add_argument("--realtime", action="store_true", help="Replay with capture timing")
    parser.add_argument("--connect", action="store_true", help="Connect to existing Rerun viewer on 127.0.0.1:9876")
    parser.add_argument("--save", type=Path, default=None, help="Optional .rrd output")
    args = parser.parse_args()

    rr.init("hand_capture_retarget")
    if args.connect:
        rr.connect_grpc("rerun+http://127.0.0.1:9876/proxy")
    else:
        rr.spawn()
    if args.save:
        rr.save(str(args.save))
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("retarget/urdf", rr.Asset3D(path=args.urdf), static=True)

    if args.follow:
        run_follow(args.manus_csv, args.urdf, args.calibration_frame, poll_s=0.2)
        return 0
    count = run_once(args.manus_csv, args.urdf, args.max_frames, args.calibration_frame, args.realtime)
    print(f"Logged {count} retargeted frames to Rerun")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
