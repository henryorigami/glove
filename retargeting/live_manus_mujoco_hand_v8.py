from __future__ import annotations

import argparse
import json
import signal
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from retargeting.joint_calibration import DEFAULT_JOINT_CALIBRATION, apply_joint_calibration, load_joint_calibration
from retargeting.manus_stream import start_manus_process, stderr_printer
from retargeting.manus_keypoints import frame_from_jsonl, frame_points
from retargeting.angle_retarget import AngleRetargeter
from retargeting.retarget_hand_v8 import DEFAULT_URDF, HandV8Retargeter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SESSION_ROOT = PROJECT_ROOT / "recordings"
DEFAULT_MODEL_XML = Path(r"C:\Users\henry\Downloads\Hand_V9_add_weight\Hand_V9_add_weight\robot.urdf")


def _session_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("live_mujoco_%Y%m%d_%H%M%SZ")
    return DEFAULT_SESSION_ROOT / stamp


def prepare_mujoco_xml(path: Path, session_dir: Path) -> Path:
    """Return a MuJoCo-loadable XML path.

    Onshape URDF exports here use `package:///mesh.stl`, which MuJoCo does not
    resolve on Windows. For live viewing, create a session-local URDF copy with
    those mesh URLs rewritten to absolute mesh paths.
    """
    text = path.read_text(encoding="utf-8", errors="ignore")
    if "package:///" not in text and "package://" not in text:
        return path
    mesh_root = path.parent.as_posix().rstrip("/") + "/"
    fixed = text.replace("package:///", mesh_root).replace("package://", mesh_root)
    out = session_dir / f"{path.stem}_mujoco{path.suffix}"
    out.write_text(fixed, encoding="utf-8", newline="\n")
    return out


class LiveLog:
    def __init__(self, path: Path, echo: bool = True):
        self.path = path
        self.echo = echo
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, message: str) -> None:
        line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
        if self.echo:
            print(message, flush=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(line + "\n")


def mujoco_joint_qpos_addresses(model: mujoco.MjModel) -> dict[str, int]:
    out: dict[str, int] = {}
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name:
            out[name] = int(model.jnt_qposadr[j])
    return out


def apply_named_qpos(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_addresses: dict[str, int],
    joint_names: list[str],
    q: np.ndarray,
    alpha: float,
    limits: bool,
) -> None:
    target = data.qpos.copy()
    for name, value in zip(joint_names, q):
        adr = joint_addresses.get(name)
        if adr is None:
            continue
        v = float(value)
        if limits:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id >= 0 and model.jnt_limited[joint_id]:
                lo, hi = model.jnt_range[joint_id]
                v = float(np.clip(v, lo, hi))
        target[adr] = v
    data.qpos[:] = (1.0 - alpha) * data.qpos + alpha * target
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def run(args: argparse.Namespace) -> int:
    log = LiveLog(args.session_dir / "live_mujoco.log")
    log.write(f"Starting live MuJoCo retargeter session_dir={args.session_dir}")
    model_xml = prepare_mujoco_xml(args.model_xml, args.session_dir)
    log.write(f"Loading MuJoCo model: {model_xml}")

    model = mujoco.MjModel.from_xml_path(str(model_xml))
    data = mujoco.MjData(model)
    joint_addresses = mujoco_joint_qpos_addresses(model)
    log.write(f"MuJoCo joints: {', '.join(joint_addresses.keys())}")

    if args.retarget_mode == "angle":
        retargeter = AngleRetargeter(
            args.urdf,
            max_curl_rad=args.max_curl_rad,
            thumb_max_curl_rad=args.thumb_max_curl_rad,
            smoothness=args.angle_smoothness,
            finger_sign=args.finger_sign,
            thumb_sign=args.thumb_sign,
        )
        log.write("Retarget mode: angle/curl-only")
    else:
        retargeter = HandV8Retargeter(
            args.urdf,
            regularization=args.regularization,
            smoothness=args.smoothness,
            max_nfev=args.max_nfev,
        )
        log.write("Retarget mode: legacy point IK")
    missing = [name for name in retargeter.joint_names if name not in joint_addresses]
    if missing:
        raise RuntimeError(f"MuJoCo model is missing retarget joints: {missing}")
    joint_calibration = load_joint_calibration(args.joint_calibration)
    if joint_calibration is None:
        log.write("Joint calibration: none")
    elif joint_calibration.get("kind") != "hand_v9_angle_calibration":
        log.write(f"Joint calibration ignored: unsupported kind={joint_calibration.get('kind')!r}")
        joint_calibration = None
    elif int(joint_calibration.get("angle_profile", 0)) < 2:
        log.write("Joint calibration ignored: stale angle profile; rerun calibrate_hand_v9")
        joint_calibration = None
    else:
        log.write(f"Joint calibration loaded: {args.joint_calibration}")

    proc = start_manus_process(args.session_dir, args.duration, log)
    stop_evt = threading.Event()
    stderr_thread = threading.Thread(target=stderr_printer, args=(proc, stop_evt, log), daemon=True)
    stderr_thread.start()

    latest_q = np.zeros(len(retargeter.joint_names), dtype=float)
    latest_stats: dict | None = None
    latest_frame = None
    latest_frame_seq = -1
    lock = threading.Lock()
    calibrated = threading.Event()
    fatal: list[BaseException] = []
    counts = {"received": 0, "solved": 0, "skipped": 0, "dropped": 0, "latest_frame_seq": -1, "solved_frame_seq": -1}
    start_time = time.perf_counter()

    def stop_process(*_ignored) -> None:
        stop_evt.set()
        if proc.poll() is None:
            proc.terminate()

    def reader_loop() -> None:
        nonlocal latest_frame, latest_frame_seq
        received = 0
        last_report = time.perf_counter()
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                if stop_evt.is_set():
                    break
                clean = line.strip()
                if not clean:
                    continue
                try:
                    obj = json.loads(clean)
                except json.JSONDecodeError:
                    log.write(f"[skip non-json stdout] {clean[:160]}")
                    continue
                frame = frame_from_jsonl(obj)
                if frame is None:
                    continue
                if args.glove_id is not None and frame.glove_id != args.glove_id:
                    continue

                received += 1
                with lock:
                    if latest_frame is not None and latest_frame_seq != counts["solved_frame_seq"]:
                        counts["dropped"] += 1
                    latest_frame = frame
                    latest_frame_seq = frame.frame_seq
                    counts["received"] = received
                    counts["latest_frame_seq"] = frame.frame_seq

                now = time.perf_counter()
                if now - last_report >= 2.0:
                    elapsed = max(1e-6, now - start_time)
                    with lock:
                        solved = counts["solved"]
                        dropped = counts["dropped"]
                        solved_seq = counts["solved_frame_seq"]
                        latest_seq = counts["latest_frame_seq"]
                        err = latest_stats.get("mean_tip_error_m", float("nan")) if latest_stats else float("nan")
                        mode = latest_stats.get("mode", args.retarget_mode) if latest_stats else args.retarget_mode
                    lag_frames = max(0, latest_seq - solved_seq)
                    log.write(
                        f"frames mode={mode} received={received} solved={solved} dropped={dropped} "
                        f"solve_hz={solved / elapsed:.1f} lag_frames={lag_frames} tip_err={err:.4f}m"
                    )
                    last_report = now
            stop_evt.set()
        except BaseException as exc:
            fatal.append(exc)
            stop_evt.set()

    def solver_loop() -> None:
        nonlocal latest_q, latest_stats
        solved = 0
        skipped = 0
        calibration_seen = 0
        last_seq = -1
        while not stop_evt.is_set():
            with lock:
                frame = latest_frame
                seq = latest_frame_seq
            if frame is None or seq == last_seq:
                time.sleep(0.001)
                continue
            last_seq = seq
            points = frame_points(frame, args.wrist_mode)
            if not calibrated.is_set():
                if calibration_seen < args.calibration_frames:
                    calibration_seen += 1
                    continue
                calib = retargeter.calibrate(points)
                log.write(f"Calibrated on MANUS frame {frame.frame_seq}: {calib}")
                calibrated.set()

            if frame.frame_seq % args.solve_every != 0:
                continue
            try:
                q, stats = retargeter.solve(points)
            except Exception as exc:
                skipped += 1
                log.write(f"[retarget skip] frame={frame.frame_seq} {exc}")
                with lock:
                    counts["skipped"] = skipped
                    counts["solved_frame_seq"] = frame.frame_seq
                continue
            solved += 1
            with lock:
                latest_q = apply_joint_calibration(q, retargeter.joint_names, joint_calibration)
                latest_stats = stats
                counts["solved"] = solved
                counts["skipped"] = skipped
                counts["solved_frame_seq"] = frame.frame_seq

    signal.signal(signal.SIGINT, stop_process)
    signal.signal(signal.SIGTERM, stop_process)

    reader_thread = threading.Thread(target=reader_loop, name="manus-retarget-reader", daemon=True)
    solver_thread = threading.Thread(target=solver_loop, name="manus-retarget-solver", daemon=True)
    reader_thread.start()
    solver_thread.start()

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            log.write("MuJoCo viewer opened")
            viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.06])
            viewer.cam.distance = 0.28
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -25
            while viewer.is_running() and not stop_evt.is_set():
                if fatal:
                    raise fatal[0]
                with lock:
                    q = latest_q.copy()
                    solved = counts["solved"]
                apply_named_qpos(
                    model,
                    data,
                    joint_addresses,
                    retargeter.joint_names,
                    q,
                    alpha=args.display_alpha,
                    limits=True,
                )
                viewer.sync()
                if solved == 0 and int(time.perf_counter() - start_time) % 5 == 0:
                    time.sleep(0.001)
                else:
                    time.sleep(max(0.0, 1.0 / args.display_hz))
    finally:
        stop_process()
        reader_thread.join(timeout=3)
        solver_thread.join(timeout=3)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        stop_evt.set()
        stderr_thread.join(timeout=2)
        log.write("Stopped live MuJoCo retargeter")

    return proc.returncode or 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Live MANUS Integrated -> Hand V9 MuJoCo viewer.")
    parser.add_argument("--model-xml", "--mjcf", dest="model_xml", type=Path, default=DEFAULT_MODEL_XML)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--session-dir", type=Path, default=None)
    parser.add_argument("--duration", type=int, default=0, help="0 = run until viewer closes or Ctrl-C")
    parser.add_argument("--wrist-mode", choices=["local", "world"], default="world")
    parser.add_argument("--retarget-mode", choices=["angle", "ik"], default="angle")
    parser.add_argument("--calibration-frames", type=int, default=10)
    parser.add_argument("--glove-id", type=int, default=None)
    parser.add_argument("--solve-every", type=int, default=1, help="Solve IK every N MANUS frames")
    parser.add_argument("--max-nfev", type=int, default=8)
    parser.add_argument("--regularization", type=float, default=0.03)
    parser.add_argument("--smoothness", type=float, default=0.2)
    parser.add_argument("--angle-smoothness", type=float, default=0.25)
    parser.add_argument("--max-curl-rad", type=float, default=2.75)
    parser.add_argument("--thumb-max-curl-rad", type=float, default=2.45)
    parser.add_argument("--finger-sign", type=float, choices=[-1.0, 1.0], default=-1.0)
    parser.add_argument("--thumb-sign", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--display-alpha", type=float, default=0.9, help="0..1 smoothing for displayed qpos")
    parser.add_argument("--display-hz", type=float, default=120.0)
    parser.add_argument(
        "--joint-calibration",
        type=Path,
        default=DEFAULT_JOINT_CALIBRATION,
        help="Angle calibration JSON. Legacy IK-derived calibrations are ignored.",
    )
    args = parser.parse_args()
    if args.session_dir is None:
        args.session_dir = _session_dir()
    args.solve_every = max(1, args.solve_every)
    args.display_alpha = float(np.clip(args.display_alpha, 0.01, 1.0))
    args.display_hz = max(15.0, float(args.display_hz))
    args.angle_smoothness = float(np.clip(args.angle_smoothness, 0.0, 0.95))
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
