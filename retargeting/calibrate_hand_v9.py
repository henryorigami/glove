from __future__ import annotations

import argparse
import json
import signal
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from retargeting.hand_v8_mapping import FINGER_ROBOT
from retargeting.joint_calibration import (
    DEFAULT_JOINT_CALIBRATION,
    model_forward_signs,
    save_joint_calibration,
    summarize_joint_samples,
)
from retargeting.live_manus_mujoco_hand_v8 import DEFAULT_MODEL_XML, DEFAULT_SESSION_ROOT, LiveLog, prepare_mujoco_xml
from retargeting.manus_keypoints import frame_from_jsonl, frame_points
from retargeting.manus_stream import start_manus_process, stderr_printer
from retargeting.retarget_hand_v8 import DEFAULT_URDF, HandV8Retargeter


POSES = [
    ("neutral", "Open your hand naturally. Fingers straight and relaxed."),
    ("fist", "Make a full fist. Curl every finger as far forward as feels safe."),
    ("index_curl", "Only curl INDEX. Keep the other fingers as straight as you can."),
    ("middle_curl", "Only curl MIDDLE. Keep the other fingers as straight as you can."),
    ("ring_curl", "Only curl RING. Keep the other fingers as straight as you can."),
    ("pinky_curl", "Only curl PINKY. Keep the other fingers as straight as you can."),
    ("thumb_curl", "Curl the THUMB across/inward."),
    ("spread", "Spread all fingers apart, then hold that spread."),
]

POSE_TO_FINGER = {
    "index_curl": "index",
    "middle_curl": "middle",
    "ring_curl": "ring",
    "pinky_curl": "pinky",
    "thumb_curl": "thumb",
}


def _session_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("joint_calibration_%Y%m%d_%H%M%SZ")
    return DEFAULT_SESSION_ROOT / stamp


class LatestFrame:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.seq = -1
        self.count = 0

    def set(self, frame) -> None:
        with self.lock:
            self.frame = frame
            self.seq = frame.frame_seq
            self.count += 1

    def get(self):
        with self.lock:
            return self.frame, self.seq, self.count


def reader_loop(proc, stop_evt: threading.Event, latest: LatestFrame, log: LiveLog, glove_id: int | None) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        if stop_evt.is_set():
            break
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        frame = frame_from_jsonl(obj)
        if frame is None:
            continue
        if glove_id is not None and frame.glove_id != glove_id:
            continue
        latest.set(frame)
    stop_evt.set()


def collect_pose(
    pose_name: str,
    latest: LatestFrame,
    retargeter: HandV8Retargeter,
    duration_s: float,
    wrist_mode: str,
    log: LiveLog,
) -> list[np.ndarray]:
    samples: list[np.ndarray] = []
    start = time.perf_counter()
    last_seq = -1
    while time.perf_counter() - start < duration_s:
        frame, seq, _ = latest.get()
        if frame is None or seq == last_seq:
            time.sleep(0.003)
            continue
        last_seq = seq
        try:
            q, _stats = retargeter.solve(frame_points(frame, wrist_mode))
        except Exception as exc:
            log.write(f"[pose {pose_name}] skip frame={seq}: {exc}")
            continue
        samples.append(q)
    log.write(f"[pose {pose_name}] collected {len(samples)} solved samples")
    return samples


def countdown(seconds: float, label: str) -> None:
    whole = max(1, int(round(seconds)))
    for i in range(whole, 0, -1):
        print(f"{label} in {i}...", flush=True)
        time.sleep(1.0)


def pose_medians(samples_by_pose: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    medians = {}
    for pose, samples in samples_by_pose.items():
        if samples:
            medians[pose] = np.median(np.vstack(samples), axis=0)
    return medians


def build_calibration(
    retargeter: HandV8Retargeter,
    samples_by_pose: dict[str, list[np.ndarray]],
    wrist_mode: str,
) -> dict:
    med = pose_medians(samples_by_pose)
    if "neutral" not in med:
        raise RuntimeError("neutral pose has no solved samples")
    neutral = med["neutral"]
    forward = model_forward_signs(retargeter)
    name_to_idx = {name: i for i, name in enumerate(retargeter.joint_names)}
    all_samples = [q for samples in samples_by_pose.values() for q in samples]
    joints = {}

    for name in retargeter.joint_names:
        idx = name_to_idx[name]
        finger = next((f for f, spec in FINGER_ROBOT.items() if name in spec["joints"]), None)
        pose_name = f"{finger}_curl" if finger != "thumb" else "thumb_curl"
        curl_pose = med.get(pose_name, med.get("fist", neutral))
        observed_delta = float(curl_pose[idx] - neutral[idx])
        observed_sign = 1.0 if observed_delta >= 0 else -1.0
        desired_sign = float(forward.get(name, 1.0))
        sign = desired_sign * observed_sign

        transformed = [sign * (q[idx] - neutral[idx]) for q in all_samples]
        stats = summarize_joint_samples(transformed)
        lower = min(0.0, stats["p05"] * 1.25)
        upper = max(0.0, stats["p95"] * 1.25)
        if upper - lower < 0.08:
            lower, upper = -0.05, 0.05
        lower = float(np.clip(lower, -2.2, 2.2))
        upper = float(np.clip(upper, -2.2, 2.2))

        joints[name] = {
            "enabled": True,
            "finger": finger,
            "neutral_raw": float(neutral[idx]),
            "display_neutral": 0.0,
            "sign": float(sign),
            "scale": 1.0,
            "lower": lower,
            "upper": upper,
            "observed_curl_delta_raw": observed_delta,
            "model_forward_sign": desired_sign,
            "observed_samples": stats,
        }

    pose_summary = {
        pose: {
            joint: float(q[i])
            for i, joint in enumerate(retargeter.joint_names)
        }
        for pose, q in med.items()
    }
    return {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "hand_v9_joint_calibration",
        "wrist_mode": wrist_mode,
        "joint_names": retargeter.joint_names,
        "poses": list(samples_by_pose.keys()),
        "pose_medians_raw": pose_summary,
        "joints": joints,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Guided MANUS -> Hand V9 joint direction/limit calibration.")
    parser.add_argument("--out", type=Path, default=DEFAULT_JOINT_CALIBRATION)
    parser.add_argument("--session-dir", type=Path, default=None)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--model-xml", type=Path, default=DEFAULT_MODEL_XML)
    parser.add_argument("--wrist-mode", choices=["local", "world"], default="world")
    parser.add_argument("--pose-seconds", type=float, default=3.0)
    parser.add_argument("--calibration-frames", type=int, default=10)
    parser.add_argument("--max-nfev", type=int, default=16)
    parser.add_argument("--glove-id", type=int, default=None)
    parser.add_argument("--prep-seconds", type=float, default=5.0)
    parser.add_argument("--manual-enter", action="store_true", help="Require Enter before each pose instead of timed countdowns")
    args = parser.parse_args()

    if args.session_dir is None:
        args.session_dir = _session_dir()
    log = LiveLog(args.session_dir / "joint_calibration.log", echo=False)
    log.write(f"Joint calibration session: {args.session_dir}")
    print(f"Joint calibration session: {args.session_dir}", flush=True)
    # Validate the MuJoCo XML early; this also writes the session-local fixed URDF.
    prepare_mujoco_xml(args.model_xml, args.session_dir)

    retargeter = HandV8Retargeter(args.urdf, max_nfev=args.max_nfev, smoothness=0.0)
    proc = start_manus_process(args.session_dir, duration=0, log=log)
    stop_evt = threading.Event()
    latest = LatestFrame()
    threads = [
        threading.Thread(target=stderr_printer, args=(proc, stop_evt, log), daemon=True),
        threading.Thread(target=reader_loop, args=(proc, stop_evt, latest, log, args.glove_id), daemon=True),
    ]
    for t in threads:
        t.start()

    def stop_process(*_ignored) -> None:
        stop_evt.set()
        if proc.poll() is None:
            proc.terminate()

    signal.signal(signal.SIGINT, stop_process)
    signal.signal(signal.SIGTERM, stop_process)

    try:
        print("\nWaiting for MANUS skeleton frames...", flush=True)
        while not stop_evt.is_set():
            frame, _seq, count = latest.get()
            if frame is not None and count >= args.calibration_frames:
                break
            time.sleep(0.05)
        frame, seq, _count = latest.get()
        if frame is None:
            raise RuntimeError("No MANUS frames arrived")

        print("\nFirst: hold a relaxed open hand for neutral calibration.", flush=True)
        if args.manual_enter:
            input("Press Enter when ready...")
            time.sleep(0.25)
        else:
            countdown(args.prep_seconds, "Neutral capture starts")
        frame, seq, _count = latest.get()
        if frame is None:
            raise RuntimeError("No MANUS frame available for neutral calibration")
        retargeter.calibrate(frame_points(frame, args.wrist_mode))
        samples_by_pose: dict[str, list[np.ndarray]] = {}

        for pose_name, prompt in POSES:
            print(f"\nPOSE: {pose_name}", flush=True)
            print(prompt, flush=True)
            if args.manual_enter:
                input("Hold the pose, then press Enter to collect...")
                countdown(2.0, "Collecting")
            else:
                countdown(args.prep_seconds, "Collecting")
            samples_by_pose[pose_name] = collect_pose(
                pose_name,
                latest,
                retargeter,
                args.pose_seconds,
                args.wrist_mode,
                log,
            )

        calibration = build_calibration(retargeter, samples_by_pose, args.wrist_mode)
        save_joint_calibration(args.out, calibration)
        log.write(f"Wrote joint calibration: {args.out}")
        print(f"\nWrote calibration:\n{args.out}")
        print("\nRestart live viewer; it loads this config by default.")
        return 0
    finally:
        stop_process()
        for t in threads:
            t.join(timeout=3)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
