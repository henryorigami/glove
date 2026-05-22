from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation

from retargeting.hand_v8_mapping import FINGER_ROBOT, robot_keypoints
from retargeting.manus_keypoints import FINGER_ORDER, frame_points, iter_manus_frames, _rows_to_frame
from retargeting.retarget_hand_v8 import DEFAULT_URDF, HandV8Retargeter


COLORS = {
    "thumb": (245, 180, 70),
    "index": (80, 190, 255),
    "middle": (90, 230, 120),
    "ring": (220, 120, 255),
    "pinky": (255, 100, 110),
}


def _flatten(points: dict[str, list[np.ndarray]]) -> tuple[list[np.ndarray], list[tuple[int, int, int]]]:
    flat: list[np.ndarray] = []
    colors: list[tuple[int, int, int]] = []
    for finger in FINGER_ORDER:
        for p in points.get(finger, []):
            flat.append(p)
            colors.append(COLORS[finger])
    return flat, colors


def _strips(points: dict[str, list[np.ndarray]]) -> tuple[list[list[list[float]]], list[tuple[int, int, int]]]:
    strips = []
    colors = []
    for finger in FINGER_ORDER:
        pts = points.get(finger, [])
        if len(pts) >= 2:
            strips.append([p.tolist() for p in pts])
            colors.append(COLORS[finger])
    return strips, colors


def log_hand(entity: str, points: dict[str, list[np.ndarray]], radius: float = 0.006, labels: bool = False) -> None:
    flat, colors = _flatten(points)
    strips, strip_colors = _strips(points)
    if flat:
        rr.log(f"{entity}/points", rr.Points3D([p.tolist() for p in flat], colors=colors, radii=radius, show_labels=labels))
    if strips:
        rr.log(f"{entity}/segments", rr.LineStrips3D(strips, colors=strip_colors, radii=radius * 0.55))


def log_robot_joint_scalars(joint_names: list[str], q: np.ndarray) -> None:
    for name, value in zip(joint_names, q):
        rr.log(f"retarget/joints/{name}", rr.Scalars([float(value)]))


class RobotMeshLogger:
    def __init__(self, retargeter: HandV8Retargeter):
        self.retargeter = retargeter
        self.logged_static = False
        self.logged_assets: set[str] = set()

    def log_static_meshes(self) -> None:
        if self.logged_static:
            return
        self.logged_static = True

    def log_instance_transforms(self, q: np.ndarray) -> None:
        self.log_static_meshes()
        poses = self.retargeter.kin.forward(q)
        for idx, (link, mesh_path, visual_origin) in enumerate(self.retargeter.kin.visual_meshes()):
            if link not in poses:
                continue
            transform = poses[link] @ visual_origin
            rot = Rotation.from_matrix(transform[:3, :3]).as_quat()
            entity = f"retarget/robot_mesh/{link}/visual_{idx}_{mesh_path.stem}"
            if entity not in self.logged_assets:
                rr.log(entity, rr.Asset3D(path=mesh_path), static=True)
                self.logged_assets.add(entity)
            rr.log(
                entity,
                rr.Transform3D(
                    translation=transform[:3, 3].tolist(),
                    rotation=rr.Quaternion(xyzw=rot.tolist()),
                ),
            )


def run_once(
    manus_csv: Path,
    urdf_path: Path,
    max_frames: int,
    calibration_frame: int,
    realtime: bool,
    sample_every: int,
    log_mesh: bool,
    wrist_mode: str,
) -> int:
    retargeter = HandV8Retargeter(urdf_path)
    mesh_logger = RobotMeshLogger(retargeter) if log_mesh else None
    frames = iter_manus_frames(manus_csv)
    for i, frame in enumerate(frames):
        points = frame_points(frame, wrist_mode)
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
        if frame.frame_seq % sample_every == 0:
            count += log_frame(retargeter, frame, first_t, mesh_logger, wrist_mode)
    return count


def log_frame(retargeter: HandV8Retargeter, frame, first_t_wall_ns: int, mesh_logger: RobotMeshLogger | None = None, wrist_mode: str = "local") -> int:
    rr.set_time("manus_frame", sequence=frame.frame_seq)
    rr.set_time("capture_time", duration=(frame.t_wall_ns - first_t_wall_ns) / 1e9)
    manus_points = frame_points(frame, wrist_mode)
    target_points = retargeter.target_points(manus_points)
    q, stats = retargeter.solve(manus_points)
    robot_points = robot_keypoints(retargeter.kin, q)

    # Keep MANUS/target paths available, but de-emphasized and unlabeled.
    log_hand("retarget/manus_wrist_points", manus_points, radius=0.002, labels=False)
    log_hand("retarget/robot_hand", robot_points, radius=0.004, labels=False)
    log_hand("retarget/robot_targets", target_points, radius=0.002, labels=False)
    if mesh_logger is not None:
        mesh_logger.log_instance_transforms(q)
    log_robot_joint_scalars(retargeter.joint_names, q)
    rr.log("retarget/error/mean_tip_m", rr.Scalars([stats["mean_tip_error_m"]]))
    rr.log("retarget/error/ik_cost", rr.Scalars([stats["cost"]]))
    return 1


def _stream_frames(manus_csv: Path, poll_s: float):
    """Yield completed MANUS frames from a growing CSV without rereading it."""
    while not manus_csv.exists():
        time.sleep(poll_s)
    with manus_csv.open(newline="", encoding="utf-8") as f:
        header_line = f.readline()
        while not header_line:
            time.sleep(poll_s)
            header_line = f.readline()
        fieldnames = next(csv.reader([header_line]))
        current_seq: str | None = None
        rows: list[dict[str, str]] = []
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                time.sleep(poll_s)
                f.seek(pos)
                continue
            try:
                values = next(csv.reader([line]))
                if len(values) != len(fieldnames):
                    continue
                row = dict(zip(fieldnames, values))
            except Exception:
                continue
            if row is None or not row.get("frame_seq"):
                continue
            seq = row["frame_seq"]
            if current_seq is None:
                current_seq = seq
            if seq != current_seq:
                frame = _rows_to_frame(rows)
                if frame is not None:
                    yield frame
                rows = []
                current_seq = seq
            rows.append(row)


def run_follow(manus_csv: Path, urdf_path: Path, calibration_frame: int, poll_s: float, sample_every: int, log_mesh: bool, wrist_mode: str) -> None:
    retargeter = HandV8Retargeter(urdf_path)
    mesh_logger = RobotMeshLogger(retargeter) if log_mesh else None
    first_t_wall_ns: int | None = None
    calibrated = False
    seen = 0
    logged = 0
    print(f"Following {manus_csv}")
    for frame in _stream_frames(manus_csv, poll_s):
        points = frame_points(frame, wrist_mode)
        if not calibrated:
            if seen < calibration_frame:
                seen += 1
                continue
            retargeter.calibrate(points)
            first_t_wall_ns = frame.t_wall_ns
            calibrated = True
        assert first_t_wall_ns is not None
        if seen % sample_every == 0:
            log_frame(retargeter, frame, first_t_wall_ns, mesh_logger, wrist_mode)
            logged += 1
            if logged % 30 == 0:
                print(f"logged={logged} seen={seen} frame={frame.frame_seq}", flush=True)
        seen += 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize MANUS to Hand V8 retargeting in Rerun.")
    parser.add_argument("--manus-csv", required=True, type=Path)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--max-frames", type=int, default=500, help="0 = all frames in replay mode")
    parser.add_argument("--calibration-frame", type=int, default=0)
    parser.add_argument("--follow", action="store_true", help="Follow a live, growing MANUS CSV")
    parser.add_argument("--realtime", action="store_true", help="Replay with capture timing")
    parser.add_argument("--sample-every", type=int, default=10, help="Only log every Nth MANUS frame")
    parser.add_argument("--no-mesh", action="store_true", help="Only log keypoint skeleton, not URDF STL meshes")
    parser.add_argument("--wrist-mode", choices=["local", "world"], default="local", help="local avoids double-applying MANUS wrist IMU rotation")
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
        run_follow(args.manus_csv, args.urdf, args.calibration_frame, poll_s=0.005, sample_every=max(1, args.sample_every), log_mesh=not args.no_mesh, wrist_mode=args.wrist_mode)
        return 0
    count = run_once(args.manus_csv, args.urdf, args.max_frames, args.calibration_frame, args.realtime, sample_every=max(1, args.sample_every), log_mesh=not args.no_mesh, wrist_mode=args.wrist_mode)
    print(f"Logged {count} retargeted frames to Rerun")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
