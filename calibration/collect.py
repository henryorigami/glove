"""
Capture a two-phase calibration session for solving the rigid transform
between a Vive tracker and the Manus wrist node.

Phase 1 (rotation, ~30s):
    Move the hand through varied rotations on all three axes.
    Solves R_X via Tsai-Lenz hand-eye on paired IMU sequences.

Phase 2 (translation, ~10s):
    Rest the forearm on a table, keep the wrist position stationary,
    rotate the hand around the wrist. The tracker traces a sphere;
    sphere fit gives |t_X| and combined with R_X yields the full vector.

Usage:
    python -m calibration.collect --side right --out-dir <dir>
    python -m calibration.collect --side right --auto    # no prompts; fixed timings

Writes:
    <out-dir>/
        vive_poses.csv             (from ViveLogger)
        manus_raw_devices.csv      (from manus_logger)
        manus_raw_skeleton.csv     (for cross-check)
        manus_ergonomics.csv
        manus_manifest.json
        calibration_meta.json      <-- phase boundaries and side
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import wake_basestations  # noqa: E402
from pipeline.vive_logger import (  # noqa: E402
    ViveLogger,
    attach_dongles,
    count_watchman_in_wsl,
    start_wsl_keepalive,
)


MANUS_LOGGER_EXE = Path(
    r"C:\Users\henry\Desktop\manus_logger\build\Release\manus_logger.exe"
)

PHASE1_DEFAULT_SECONDS = 30.0
PHASE2_DEFAULT_SECONDS = 12.0


def prompt(msg: str) -> None:
    print(f"\n>>> {msg}")
    input("    Press Enter when ready... ")


def countdown(seconds: float, label: str) -> None:
    print(f"\n  {label} — {seconds:.0f}s")
    start = time.monotonic()
    while True:
        elapsed = time.monotonic() - start
        remaining = max(0.0, seconds - elapsed)
        sys.stdout.write(f"\r    {remaining:5.1f}s remaining... ")
        sys.stdout.flush()
        if remaining <= 0:
            break
        time.sleep(0.1)
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--side", choices=["left", "right"], required=True)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output dir. Default: calibration/raw/calib_<UTC>_<side>")
    ap.add_argument("--phase1-seconds", type=float, default=PHASE1_DEFAULT_SECONDS)
    ap.add_argument("--phase2-seconds", type=float, default=PHASE2_DEFAULT_SECONDS)
    ap.add_argument("--auto", action="store_true",
                    help="Run phases back-to-back without prompts.")
    ap.add_argument("--no-wake", action="store_true",
                    help="Skip BLE wake of base stations.")
    args = ap.parse_args()

    if args.out_dir is None:
        stamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%SZ")
        args.out_dir = (PROJECT_ROOT / "calibration" / "raw"
                        / f"calib_{stamp}_{args.side}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {args.out_dir}")

    # --- environment setup ---
    if not args.no_wake:
        print("Waking base station(s)...")
        try:
            woken = asyncio.run(wake_basestations.wake_all(timeout=8.0))
            print(f"  woken: {woken or 'none'}")
        except Exception as e:
            print(f"  WARN: BLE wake failed: {e}")

    print("Starting WSL keepalive...")
    wsl_proc = start_wsl_keepalive()
    time.sleep(3)

    print("Attaching dongles...")
    for bid, msg in attach_dongles().items():
        print(f"  {bid}: {msg}")
    time.sleep(2)
    if count_watchman_in_wsl() == 0:
        print("ERROR: no Watchman dongles visible in WSL. Aborting.")
        wsl_proc.terminate()
        return 1

    # --- start loggers ---
    print("Starting ViveLogger...")
    vive = ViveLogger(args.out_dir)
    vive.start()

    print("Starting manus_logger...")
    if not MANUS_LOGGER_EXE.exists():
        print(f"ERROR: {MANUS_LOGGER_EXE} not found.")
        vive.stop()
        wsl_proc.terminate()
        return 1
    manus = subprocess.Popen(
        [str(MANUS_LOGGER_EXE),
         "--session-dir", str(args.out_dir),
         "--prefix", "manus_"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=MANUS_LOGGER_EXE.parent,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        text=True,
    )

    # Give both loggers a few seconds to actually start streaming.
    print("Letting streams warm up (5s)...")
    time.sleep(5)

    phases: list[dict] = []

    # --- Phase 1 ---
    if not args.auto:
        prompt(
            f"PHASE 1 (rotation, {args.phase1_seconds:.0f}s):\n"
            "    Move your hand through ALL rotation axes:\n"
            "    - rotate up/down (pitch)\n"
            "    - left/right (yaw)\n"
            "    - tilt side-to-side (roll)\n"
            "    Slow, deliberate, large motions. Cover the full range."
        )
    phase1_start = time.time_ns()
    countdown(args.phase1_seconds, "ROTATING")
    phase1_end = time.time_ns()
    phases.append({
        "name": "rotation",
        "start_wall_ns": phase1_start,
        "end_wall_ns": phase1_end,
        "purpose": "Tsai-Lenz hand-eye rotation R_X solve.",
    })

    # Brief gap between phases.
    print("Phase 1 done. Take 3 seconds before Phase 2...")
    time.sleep(3)

    # --- Phase 2 ---
    if not args.auto:
        prompt(
            f"PHASE 2 (wrist stationary, {args.phase2_seconds:.0f}s):\n"
            "    Rest your forearm flat on a table.\n"
            "    Keep your wrist position FIXED.\n"
            "    Rotate just the hand (palm up/down/left/right) around the wrist.\n"
            "    The tracker should trace a sphere around the wrist joint."
        )
    phase2_start = time.time_ns()
    countdown(args.phase2_seconds, "WRIST STATIONARY")
    phase2_end = time.time_ns()
    phases.append({
        "name": "translation",
        "start_wall_ns": phase2_start,
        "end_wall_ns": phase2_end,
        "purpose": "Sphere fit on tracker positions for |t_X|.",
    })

    # --- Cleanup ---
    print("\nStopping loggers...")
    vive.stop()
    if manus.poll() is None:
        try:
            import signal
            manus.send_signal(getattr(signal, "CTRL_BREAK_EVENT", 1))
            manus.wait(timeout=3)
        except Exception:
            manus.kill()
    wsl_proc.terminate()

    # --- Metadata ---
    meta = {
        "side": args.side,
        "session_dir": str(args.out_dir),
        "phase1_seconds": args.phase1_seconds,
        "phase2_seconds": args.phase2_seconds,
        "phases": phases,
        "vive_pose_count": vive._pose_count,
        "notes": (
            "Two-phase calibration. Use calibration.solve to compute "
            "tracker_to_wrist_<side>.json from this data."
        ),
    }
    (args.out_dir / "calibration_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"\nDone. Metadata at {args.out_dir / 'calibration_meta.json'}")
    print(f"Next step:  python -m calibration.solve {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
