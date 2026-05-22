from __future__ import annotations

import argparse
import json
import shlex
import signal
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation

from pipeline.manus_linux_logger import (
    LINUX_MANUS_DIR,
    WSL_DISTRO,
    WSL_USER,
    attach_manus_dongle,
    count_manus_in_wsl,
    wsl_path,
)
from retargeting.hand_v8_mapping import robot_keypoints
from retargeting.manus_keypoints import FINGER_ORDER, frame_from_jsonl, frame_points
from retargeting.retarget_hand_v8 import DEFAULT_URDF, HandV8Retargeter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SESSION_ROOT = PROJECT_ROOT / "recordings"

COLORS = {
    "thumb": (245, 180, 70),
    "index": (80, 190, 255),
    "middle": (90, 230, 120),
    "ring": (220, 120, 255),
    "pinky": (255, 100, 110),
}


def _session_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("live_retarget_%Y%m%d_%H%M%SZ")
    return DEFAULT_SESSION_ROOT / stamp


def _flatten(points: dict[str, list[np.ndarray]]) -> tuple[list[list[float]], list[tuple[int, int, int]]]:
    pts: list[list[float]] = []
    colors: list[tuple[int, int, int]] = []
    for finger in FINGER_ORDER:
        for p in points.get(finger, []):
            pts.append(p.tolist())
            colors.append(COLORS[finger])
    return pts, colors


def _strips(points: dict[str, list[np.ndarray]]) -> tuple[list[list[list[float]]], list[tuple[int, int, int]]]:
    strips: list[list[list[float]]] = []
    colors: list[tuple[int, int, int]] = []
    for finger in FINGER_ORDER:
        pts = points.get(finger, [])
        if len(pts) >= 2:
            strips.append([p.tolist() for p in pts])
            colors.append(COLORS[finger])
    return strips, colors


def log_hand(entity: str, points: dict[str, list[np.ndarray]], radius: float) -> None:
    pts, colors = _flatten(points)
    strips, strip_colors = _strips(points)
    if pts:
        rr.log(f"{entity}/points", rr.Points3D(pts, colors=colors, radii=radius, show_labels=False))
    if strips:
        rr.log(f"{entity}/segments", rr.LineStrips3D(strips, colors=strip_colors, radii=radius * 0.55))


class RobotMeshLogger:
    def __init__(self, retargeter: HandV8Retargeter):
        self.retargeter = retargeter
        self.logged_assets: set[str] = set()

    def log_instance_transforms(self, q: np.ndarray) -> None:
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


def start_manus_process(session_dir: Path, duration: int) -> subprocess.Popen:
    session_dir.mkdir(parents=True, exist_ok=True)
    print("Attaching MANUS dongle to WSL...")
    for busid, status in attach_manus_dongle().items():
        print(f"  {busid}: {status}")
    print(f"WSL sees {count_manus_in_wsl()} MANUS dongle(s)")

    session_wsl = wsl_path(session_dir)
    logger_dir_wsl = wsl_path(LINUX_MANUS_DIR)
    duration_arg = f" --duration {duration}" if duration > 0 else ""
    cmd = (
        f"cd {shlex.quote(logger_dir_wsl)} && "
        f"./manus_integrated_logger --session-dir {shlex.quote(session_wsl)} "
        f"--prefix manus_ --stream-jsonl{duration_arg}"
    )
    return subprocess.Popen(
        ["wsl", "-d", WSL_DISTRO, "-u", WSL_USER, "--exec", "bash", "-lc", cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )


def stderr_printer(proc: subprocess.Popen, stop_evt: threading.Event) -> None:
    if proc.stderr is None:
        return
    for line in proc.stderr:
        if stop_evt.is_set():
            break
        clean = line.rstrip()
        if clean:
            print(f"[manus] {clean}", flush=True)


def log_frame(
    retargeter: HandV8Retargeter,
    mesh_logger: RobotMeshLogger | None,
    frame,
    first_t_wall_ns: int,
    wrist_mode: str,
    logged: int,
    mesh_every: int,
    show_manus: bool,
) -> dict:
    rr.set_time("manus_frame", sequence=frame.frame_seq)
    rr.set_time("capture_time", duration=(frame.t_wall_ns - first_t_wall_ns) / 1e9)

    manus_points = frame_points(frame, wrist_mode)
    q, stats = retargeter.solve(manus_points)
    robot_points = robot_keypoints(retargeter.kin, q)

    if show_manus:
        log_hand("retarget/manus_points", manus_points, radius=0.002)
    log_hand("retarget/robot_hand", robot_points, radius=0.004)
    rr.log("retarget/error/mean_tip_m", rr.Scalars([stats["mean_tip_error_m"]]))
    if mesh_logger is not None and (mesh_every <= 1 or logged % mesh_every == 0):
        mesh_logger.log_instance_transforms(q)
    return stats


def run(args: argparse.Namespace) -> int:
    rr.init("hand_capture_live_retarget")
    if args.connect:
        rr.connect_grpc("rerun+http://127.0.0.1:9876/proxy")
    else:
        rr.spawn()
    if args.save:
        rr.save(str(args.save))
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("retarget/urdf", rr.Asset3D(path=args.urdf), static=True)

    retargeter = HandV8Retargeter(
        args.urdf,
        regularization=args.regularization,
        smoothness=args.smoothness,
        max_nfev=args.max_nfev,
    )
    mesh_logger = None if args.no_mesh else RobotMeshLogger(retargeter)
    proc = start_manus_process(args.session_dir, args.duration)
    stop_evt = threading.Event()
    stderr_thread = threading.Thread(target=stderr_printer, args=(proc, stop_evt), daemon=True)
    stderr_thread.start()

    calibrated = False
    seen = 0
    logged = 0
    first_t_wall_ns: int | None = None
    t0 = time.perf_counter()
    last_report = t0

    def stop_process(*_ignored) -> None:
        stop_evt.set()
        if proc.poll() is None:
            proc.terminate()

    signal.signal(signal.SIGINT, stop_process)
    signal.signal(signal.SIGTERM, stop_process)

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            clean = line.strip()
            if not clean:
                continue
            try:
                obj = json.loads(clean)
            except json.JSONDecodeError:
                print(f"[skip non-json stdout] {clean[:160]}", flush=True)
                continue
            frame = frame_from_jsonl(obj)
            if frame is None:
                continue
            if args.glove_id is not None and frame.glove_id != args.glove_id:
                continue

            points = frame_points(frame, args.wrist_mode)
            if not calibrated:
                if seen < args.calibration_frames:
                    seen += 1
                    continue
                calib = retargeter.calibrate(points)
                first_t_wall_ns = frame.t_wall_ns
                calibrated = True
                print(f"Calibrated on MANUS frame {frame.frame_seq}: {calib}", flush=True)

            assert first_t_wall_ns is not None
            if seen % args.sample_every == 0:
                try:
                    stats = log_frame(
                        retargeter,
                        mesh_logger,
                        frame,
                        first_t_wall_ns,
                        args.wrist_mode,
                        logged,
                        args.mesh_every,
                        args.show_manus,
                    )
                except Exception as exc:
                    print(f"[retarget skip] frame={frame.frame_seq} {exc}", flush=True)
                    seen += 1
                    continue
                logged += 1
                now = time.perf_counter()
                if logged % 30 == 0 or now - last_report > 5:
                    hz = logged / max(1e-6, now - t0)
                    print(
                        f"logged={logged} seen={seen} frame={frame.frame_seq} "
                        f"viewer_hz={hz:.1f} tip_err={stats['mean_tip_error_m']:.4f}m",
                        flush=True,
                    )
                    last_report = now
            seen += 1
    finally:
        stop_process()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        stop_evt.set()
        stderr_thread.join(timeout=2)
    return proc.returncode or 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Live MANUS Integrated -> Hand V8 retargeting in Rerun.")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--session-dir", type=Path, default=None)
    parser.add_argument("--duration", type=int, default=0, help="0 = run until Ctrl-C")
    parser.add_argument("--connect", action="store_true", help="Connect to existing Rerun viewer on 127.0.0.1:9876")
    parser.add_argument("--save", type=Path, default=None, help="Optional .rrd recording")
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--mesh-every", type=int, default=3, help="Log STL mesh transforms every N logged frames")
    parser.add_argument("--no-mesh", action="store_true")
    parser.add_argument("--show-manus", action="store_true", help="Also show raw MANUS keypoints")
    parser.add_argument("--wrist-mode", choices=["local", "world"], default="local")
    parser.add_argument("--calibration-frames", type=int, default=10)
    parser.add_argument("--glove-id", type=int, default=None)
    parser.add_argument("--max-nfev", type=int, default=25)
    parser.add_argument("--regularization", type=float, default=0.03)
    parser.add_argument("--smoothness", type=float, default=0.18)
    args = parser.parse_args()
    if args.session_dir is None:
        args.session_dir = _session_dir()
    args.sample_every = max(1, args.sample_every)
    args.mesh_every = max(1, args.mesh_every)
    print(f"Session dir: {args.session_dir}")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
