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

from retargeting.geort_prepare import DEFAULT_GEORT_ROOT, manus_points_to_geort21
from retargeting.live_manus_mujoco_hand_v8 import DEFAULT_SESSION_ROOT, LiveLog
from retargeting.manus_keypoints import frame_from_jsonl, frame_points
from retargeting.manus_stream import start_manus_process, stderr_printer


DEFAULT_OUT = DEFAULT_GEORT_ROOT / "data" / "manus_v9.npy"


class Collector:
    def __init__(
        self,
        session_dir: Path,
        out: Path,
        wrist_mode: str,
        sample_every: int,
        max_frames: int,
        glove_id: int | None,
        log: LiveLog,
    ):
        self.session_dir = session_dir
        self.out = out
        self.wrist_mode = wrist_mode
        self.sample_every = max(1, sample_every)
        self.max_frames = max(0, max_frames)
        self.glove_id = glove_id
        self.log = log

        self.stop_evt = threading.Event()
        self.lock = threading.Lock()
        self.samples: list[np.ndarray] = []
        self.raw_frames = 0
        self.skipped = 0
        self.first_frame_wall_ns: int | None = None
        self.last_frame_wall_ns: int | None = None

    def read_stdout(self, proc: subprocess.Popen) -> None:
        if proc.stdout is None:
            return
        for line in proc.stdout:
            if self.stop_evt.is_set():
                break
            clean = line.strip()
            if not clean or not clean.startswith("{"):
                continue
            try:
                obj = json.loads(clean)
                frame = frame_from_jsonl(obj)
            except Exception as exc:
                self.skipped += 1
                self.log.write(f"skipped malformed MANUS frame: {exc}")
                continue
            if frame is None:
                continue
            if self.glove_id is not None and frame.glove_id != self.glove_id:
                continue

            self.raw_frames += 1
            if self.raw_frames % self.sample_every:
                continue

            try:
                points = frame_points(frame, self.wrist_mode)
                arr = manus_points_to_geort21(points)
            except Exception as exc:
                self.skipped += 1
                self.log.write(f"skipped MANUS frame {frame.frame_seq}: {exc}")
                continue
            if arr is None or not np.isfinite(arr).all():
                self.skipped += 1
                continue

            with self.lock:
                if self.max_frames and len(self.samples) >= self.max_frames:
                    self.stop_evt.set()
                    break
                self.samples.append(arr.astype(np.float32))
                self.first_frame_wall_ns = self.first_frame_wall_ns or frame.t_wall_ns
                self.last_frame_wall_ns = frame.t_wall_ns

    def sample_count(self) -> int:
        with self.lock:
            return len(self.samples)

    def save(self, duration_s: float, started_at: str, stopped_at: str) -> dict:
        with self.lock:
            samples = list(self.samples)
        if not samples:
            raise RuntimeError("No usable MANUS samples were collected.")

        data = np.stack(samples, axis=0).astype(np.float32)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.out, data)

        session_copy = self.session_dir / self.out.name
        if session_copy.resolve() != self.out.resolve():
            np.save(session_copy, data)

        meta = {
            "started_at": started_at,
            "stopped_at": stopped_at,
            "session_dir": str(self.session_dir),
            "out": str(self.out),
            "session_copy": str(session_copy),
            "shape": list(data.shape),
            "dtype": str(data.dtype),
            "wrist_mode": self.wrist_mode,
            "requested_duration_s": duration_s,
            "sample_every": self.sample_every,
            "max_frames": self.max_frames,
            "glove_id": self.glove_id,
            "raw_frames_seen": self.raw_frames,
            "skipped": self.skipped,
            "first_frame_wall_ns": self.first_frame_wall_ns,
            "last_frame_wall_ns": self.last_frame_wall_ns,
        }
        meta_path = self.out.with_suffix(".metadata.json")
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8", newline="\n")
        (self.session_dir / meta_path.name).write_text(json.dumps(meta, indent=2), encoding="utf-8", newline="\n")
        return meta


def _session_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("geort_collect_%Y%m%d_%H%M%SZ")
    return DEFAULT_SESSION_ROOT / stamp


def _collect_for(seconds: float, collector: Collector, log: LiveLog, proc: subprocess.Popen) -> None:
    end = time.monotonic() + seconds
    next_print = 0
    while not collector.stop_evt.is_set():
        if proc.poll() is not None:
            log.write(f"MANUS logger exited with code {proc.returncode}")
            break
        remaining = end - time.monotonic()
        if remaining <= 0:
            break
        whole = int(round(remaining))
        if whole != next_print and (whole <= 10 or whole % 10 == 0):
            next_print = whole
            log.write(f"{whole:>3}s left | samples={collector.sample_count()} | raw_frames={collector.raw_frames}")
        time.sleep(0.1)


def _terminate_process(proc: subprocess.Popen, log: LiveLog) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        log.write("MANUS logger did not exit after terminate; killing it.")
        proc.kill()
        proc.wait(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect live MANUS glove data into a GeoRT-ready 21-keypoint .npy file."
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="GeoRT .npy output path.")
    parser.add_argument("--session-dir", type=Path, default=None, help="Recording/log directory.")
    parser.add_argument("--wrist-mode", choices=["world", "local"], default="world")
    parser.add_argument("--sample-every", type=int, default=2, help="Keep every Nth MANUS skeleton frame.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many saved samples; 0 = no limit.")
    parser.add_argument("--glove-id", type=int, default=None, help="Only collect one MANUS glove id.")
    parser.add_argument("--seconds", type=float, default=180.0, help="Collection duration. Default: 180 seconds.")
    args = parser.parse_args()

    session_dir = args.session_dir or _session_dir()
    log = LiveLog(session_dir / "collect_geort_manus.log", echo=True)
    collector = Collector(
        session_dir=session_dir,
        out=args.out,
        wrist_mode=args.wrist_mode,
        sample_every=args.sample_every,
        max_frames=args.max_frames,
        glove_id=args.glove_id,
        log=log,
    )

    started_at = datetime.now(timezone.utc).isoformat()
    log.write(f"Session dir: {session_dir}")
    log.write(f"Output: {args.out}")
    log.write(f"Duration: {args.seconds:.1f}s")
    log.write("Starting MANUS integrated logger. Close MANUS Core first if it is running.")

    proc = start_manus_process(session_dir, duration=0, log=log)
    stderr_thread = threading.Thread(target=stderr_printer, args=(proc, collector.stop_evt, log), daemon=True)
    stdout_thread = threading.Thread(target=collector.read_stdout, args=(proc,), daemon=True)
    stderr_thread.start()
    stdout_thread.start()

    old_sigint = signal.getsignal(signal.SIGINT)

    def _handle_sigint(signum, frame):
        log.write("Ctrl-C received; saving partial collection.")
        collector.stop_evt.set()

    signal.signal(signal.SIGINT, _handle_sigint)
    try:
        log.write("Waiting for first usable MANUS frame...")
        wait_start = time.monotonic()
        while collector.sample_count() == 0 and not collector.stop_evt.is_set():
            if proc.poll() is not None:
                raise RuntimeError(f"MANUS logger exited early with code {proc.returncode}")
            if time.monotonic() - wait_start > 45:
                log.write("Still no usable frame. Check dongle, glove power, license, and that no other Core is running.")
                wait_start = time.monotonic()
            time.sleep(0.2)

        log.write(f"Collecting for {args.seconds:.1f}s. Move however you want.")
        _collect_for(max(0.1, args.seconds), collector, log, proc)
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        collector.stop_evt.set()
        _terminate_process(proc, log)

    stopped_at = datetime.now(timezone.utc).isoformat()
    meta = collector.save(args.seconds, started_at, stopped_at)
    log.write("")
    log.write(f"Saved {meta['shape']} to {meta['out']}")
    log.write(f"Session copy: {meta['session_copy']}")
    log.write(f"Metadata: {args.out.with_suffix('.metadata.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
