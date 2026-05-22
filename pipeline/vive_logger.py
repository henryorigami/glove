"""
ViveLogger — wraps the libsurvive api_example running in WSL.

Reads its pose stdout, parses each line, writes CSV, and emits PoseEvent
objects to an optional callback (used by the orchestrator for Rerun
visualization and dashboard stats).

Designed to be embedded in the orchestrator. Standalone CLI also works:
    python -m pipeline.vive_logger --session-dir recordings/test
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime
import re
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Optional


# --- Configuration ---------------------------------------------------------

USBIPD_EXE = r"C:\Program Files\usbipd-win\usbipd.exe"
WSL_DISTRO = "Ubuntu"
WSL_USER = "root"
WSL_API_EXAMPLE = "/root/libsurvive/build/api_example"
# VID:PID of the Vive Watchman dongle (Tracker 3.0 wireless receiver).
WATCHMAN_VID_PID = "28de:2101"

POSE_RE = re.compile(
    r"^(?P<name>\w+)\s+(?P<serial>\S+)\s+\(\s*(?P<time>[\d.]+)\):\s+"
    r"pos=\s*(?P<x>[+-][\d.]+)\s+(?P<y>[+-][\d.]+)\s+(?P<z>[+-][\d.]+)\s+"
    r"rot=\s*(?P<rw>[+-][\d.]+)\s+(?P<rx>[+-][\d.]+)\s+(?P<ry>[+-][\d.]+)\s+(?P<rz>[+-][\d.]+)"
)


# --- Public dataclasses ----------------------------------------------------


@dataclasses.dataclass
class PoseEvent:
    """One 6-DOF pose sample from a Vive device.

    Quaternion order: (w, x, y, z). Position in meters in libsurvive world frame.
    """
    name: str               # "LH0", "WM0", ...
    serial: str             # "LHB-..." or "LHR-..."
    t_libsurvive: float     # libsurvive internal timestamp (monotonic-ish, seconds)
    t_wall_ns: int          # local wall clock at the moment we parsed the line, ns
    t_wall_iso: str         # human-readable form of t_wall_ns
    x: float
    y: float
    z: float
    qw: float
    qx: float
    qy: float
    qz: float


# --- WSL plumbing helpers --------------------------------------------------


def start_wsl_keepalive() -> subprocess.Popen:
    """Start a long-running WSL process so the VM stays up."""
    return subprocess.Popen(
        ["wsl", "-d", WSL_DISTRO, "-u", WSL_USER, "--exec", "sleep", "3600"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def discover_watchman_busids() -> list[tuple[str, str]]:
    """Return [(busid, state), ...] for every Watchman dongle visible to usbipd.

    state is one of "Shared", "Not shared", "Attached" (verbatim from usbipd).
    """
    out: list[tuple[str, str]] = []
    try:
        result = subprocess.run(
            [USBIPD_EXE, "list"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:
        return out
    for line in result.stdout.splitlines():
        # Lines look like:  "2-3    28de:2101  Watchman Dongle    Shared"
        if WATCHMAN_VID_PID not in line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        busid = parts[0]
        # The state token(s) come at the end. Could be "Shared", "Not shared",
        # or "Attached - <distro>". Join everything after the device name.
        state_idx = line.rfind("  ")
        state = line[state_idx:].strip() if state_idx > 0 else ""
        out.append((busid, state))
    return out


def attach_dongles(busids: list[str] | None = None) -> dict[str, str]:
    """Detach + re-attach Watchman dongles to WSL.

    If busids is None, auto-discovers them by VID:PID via `usbipd list`.
    Robust against:
      - dongles moving to different USB ports (re-discovers each call)
      - dongles being "already attached" (detaches first, attaches fresh)
      - "no device" errors right after attach (waits + re-discovers once)

    Returns dict {busid: short_status_line}.
    """
    if busids is None:
        busids = [b for b, _ in discover_watchman_busids()]

    statuses: dict[str, str] = {}
    if not busids:
        statuses["(none)"] = (
            f"no Watchman dongles ({WATCHMAN_VID_PID}) visible. "
            "Plug in the dongle(s) and try again."
        )
        return statuses

    # Detach everything first (idempotent — no error if already detached).
    for bid in busids:
        subprocess.run(
            [USBIPD_EXE, "detach", "--busid", bid],
            capture_output=True,
        )
    # Wait briefly so Windows can reclaim the device cleanly.
    time.sleep(1.0)

    # Re-discover in case BUSIDs shifted during detach.
    fresh = [b for b, _ in discover_watchman_busids()]
    busids = fresh or busids

    for bid in busids:
        result = subprocess.run(
            [USBIPD_EXE, "attach", "--wsl", "--busid", bid],
            capture_output=True, text=True,
        )
        combined = (result.stdout + result.stderr).strip()
        lines = combined.splitlines()
        last = lines[-1] if lines else "(no output)"

        # Retry once on transient "no device" race.
        if "no device" in last.lower():
            time.sleep(0.7)
            result = subprocess.run(
                [USBIPD_EXE, "attach", "--wsl", "--busid", bid],
                capture_output=True, text=True,
            )
            combined = (result.stdout + result.stderr).strip()
            lines = combined.splitlines()
            last = lines[-1] if lines else "(no output)"

        # "already attached" is success for our purposes.
        if "already attached" in last.lower():
            last = "ok (already attached)"
        elif result.returncode == 0:
            last = "ok"

        statuses[bid] = last
    return statuses


def count_watchman_in_wsl() -> int:
    """Return the number of Watchman dongles visible to lsusb in WSL."""
    result = subprocess.run(
        ["wsl", "-d", WSL_DISTRO, "-u", WSL_USER, "--exec",
         "bash", "-c", "lsusb | grep -c 28de:2101 || true"],
        capture_output=True, text=True,
    )
    try:
        return int(result.stdout.strip())
    except ValueError:
        return 0


# --- Logger ----------------------------------------------------------------


class ViveLogger:
    """Manage api_example running in WSL and stream poses out.

    Writes a CSV at session_dir/vive_poses.csv.
    Calls on_pose(event) for every pose if provided.
    Public methods are thread-safe.
    """

    def __init__(
        self,
        session_dir: Path,
        on_pose: Optional[Callable[[PoseEvent], None]] = None,
        on_log_line: Optional[Callable[[str], None]] = None,
        lighthouse_gen: int = 2,
        force_ootx: bool = False,
        force_calibrate: bool = False,
    ):
        self.session_dir = Path(session_dir)
        self.on_pose = on_pose
        self.on_log_line = on_log_line
        self.lighthouse_gen = lighthouse_gen
        self.force_ootx = force_ootx
        self.force_calibrate = force_calibrate
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self._csv_path = self.session_dir / "vive_poses.csv"
        self._log_path = self.session_dir / "vive_libsurvive.log"
        self._csv_file = None
        self._csv_writer = None
        self._log_file = None

        self._proc: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()

        self._first_libsurvive_t: Optional[float] = None
        self._pose_count = 0
        self._last_pose_wall_ns = 0
        self._last_libsurvive_log_line = ""
        self._vive_stage = "starting"
        self._vive_tracker_seen = False
        self._vive_lh_seen: set[str] = set()
        self._vive_ootx_seen: set[str] = set()
        self._vive_global_solve = False
        self._vive_last_warning = ""
        self._start_wall_ns = 0
        self._device_seen: dict[str, int] = {}   # name -> last pose wall ns
        self._rate_window: deque = deque(maxlen=240)  # ~2 sec of stamps for Hz

    # ---- lifecycle --------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self._proc is not None:
                raise RuntimeError("ViveLogger already started")
            self._csv_file = self._csv_path.open("w", newline="", encoding="utf-8")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow([
                "wall_time_iso", "wall_time_ns", "libsurvive_time",
                "name", "serial",
                "x", "y", "z", "qw", "qx", "qy", "qz",
            ])
            self._csv_file.flush()
            self._start_wall_ns = time.time_ns()
            self._vive_stage = "starting"
            self._vive_tracker_seen = False
            self._vive_lh_seen = set()
            self._vive_ootx_seen = set()
            self._vive_global_solve = False
            self._vive_last_warning = ""

            self._log_file = self._log_path.open(
                "w", encoding="utf-8", newline="\n",
            )

            argv = [
                "wsl", "-d", WSL_DISTRO, "-u", WSL_USER, "--exec",
                WSL_API_EXAMPLE,
                "--lighthouse-gen", str(self.lighthouse_gen),
            ]
            if self.force_ootx:
                argv += ["--force-ootx", "1"]
            if self.force_calibrate:
                argv += ["--force-calibrate", "1"]
            self._proc = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            self._stop_evt.clear()
            self._reader_thread = threading.Thread(
                target=self._reader_loop, name="vive-stdout-reader", daemon=True,
            )
            self._reader_thread.start()
            self._stderr_thread = threading.Thread(
                target=self._stderr_loop, name="vive-stderr-reader", daemon=True,
            )
            self._stderr_thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop_evt.set()
        proc = self._proc
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except Exception:
                    pass
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=timeout)
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=timeout)
        with self._lock:
            self._proc = None
            self._reader_thread = None
            self._stderr_thread = None
            if self._csv_file is not None:
                self._csv_file.flush()
                self._csv_file.close()
                self._csv_file = None
                self._csv_writer = None
            if self._log_file is not None:
                self._log_file.flush()
                self._log_file.close()
                self._log_file = None

    def is_alive(self) -> bool:
        proc = self._proc
        return proc is not None and proc.poll() is None

    # ---- stats ------------------------------------------------------------

    def stats(self) -> dict:
        """Snapshot of current logger state for the dashboard."""
        now_ns = time.time_ns()
        hz = 0.0
        if len(self._rate_window) >= 2:
            dt = (self._rate_window[-1] - self._rate_window[0]) / 1e9
            if dt > 0:
                hz = (len(self._rate_window) - 1) / dt
        return {
            "alive": self.is_alive(),
            "pose_count": self._pose_count,
            "last_pose_wall_ns": self._last_pose_wall_ns,
            "rate_hz": hz,
            "devices_seen": dict(self._device_seen),
            "csv_path": str(self._csv_path),
            "log_path": str(self._log_path),
            "last_libsurvive_log_line": self._last_libsurvive_log_line,
            "stage": self._current_stage(),
            "tracker_seen": self._vive_tracker_seen,
            "lighthouses_seen": sorted(self._vive_lh_seen),
            "ootx_seen": sorted(self._vive_ootx_seen),
            "global_solve": self._vive_global_solve,
            "last_warning": self._vive_last_warning,
            "warmup_s": (
                (now_ns - self._start_wall_ns) / 1e9
                if self._start_wall_ns else 0.0
            ),
        }

    # ---- internals --------------------------------------------------------

    def _reader_loop(self) -> None:
        assert self._proc is not None
        stdout = self._proc.stdout
        if stdout is None:
            return
        try:
            for line in stdout:
                if self._stop_evt.is_set():
                    break
                self._handle_line(line)
        except Exception as e:  # pragma: no cover - defensive
            print(f"[vive_logger] reader thread crashed: {e}", file=sys.stderr)

    def _stderr_loop(self) -> None:
        """libsurvive prints diagnostics to stderr — capture them too."""
        assert self._proc is not None
        stderr = self._proc.stderr
        if stderr is None:
            return
        try:
            for line in stderr:
                if self._stop_evt.is_set():
                    break
                self._record_log_line(line.rstrip("\n"), source="stderr")
        except Exception:
            pass

    _ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

    def _record_log_line(self, line: str, source: str = "stdout") -> None:
        if not line:
            return
        clean = self._ANSI_RE.sub("", line)
        self._update_stage_from_log(clean)
        if self._log_file is not None:
            try:
                self._log_file.write(f"[{source}] {clean}\n")
                self._log_file.flush()
            except Exception:
                pass
        self._last_libsurvive_log_line = clean[:200]
        cb = self.on_log_line
        if cb is not None:
            try:
                cb(clean)
            except Exception:
                pass

    def _update_stage_from_log(self, line: str) -> None:
        low = line.lower()
        if "adding tracked object" in low or "device wm" in low:
            self._vive_tracker_seen = True
        if "adding lighthouse" in low:
            m = re.search(r"ch\s+(\d+)", line)
            self._vive_lh_seen.add(m.group(1) if m else "?")
        if "preamble found" in low:
            self._vive_stage = "capturing OOTX"
        if "got ootx packet" in low:
            m = re.search(r"Got OOTX packet\s+\d+\s+([0-9a-fA-F]+)", line)
            self._vive_ootx_seen.add(m.group(1).lower() if m else "?")
            self._vive_stage = "waiting for global solve"
        if "mpfit success" in low or "global solve" in low:
            self._vive_global_solve = True
            self._vive_stage = "waiting for poses"
        if (
            "bad sync" in low
            or "can't solve" in low
            or "read light data error" in low
            or "ootx not set" in low
        ):
            self._vive_last_warning = line[:200]

    def _current_stage(self) -> str:
        if self._pose_count > 0:
            return "streaming poses"
        if self._vive_global_solve:
            return "waiting for poses"
        if self._vive_ootx_seen:
            return "waiting for global solve"
        if self._vive_lh_seen:
            return "capturing OOTX"
        if self._vive_tracker_seen:
            return "tracker seen; looking for lighthouse"
        return self._vive_stage

    def _handle_line(self, line: str) -> None:
        m = POSE_RE.match(line.strip())
        if not m:
            # Not a pose line — but still surface to the log so the dashboard
            # / log file can show what libsurvive is doing.
            self._record_log_line(line.rstrip("\n"), source="stdout")
            return

        t_lib = float(m["time"])
        if self._first_libsurvive_t is None:
            self._first_libsurvive_t = t_lib
        now_ns = time.time_ns()
        iso = datetime.datetime.fromtimestamp(now_ns / 1e9).isoformat(
            timespec="microseconds"
        )

        ev = PoseEvent(
            name=m["name"],
            serial=m["serial"],
            t_libsurvive=t_lib,
            t_wall_ns=now_ns,
            t_wall_iso=iso,
            x=float(m["x"]), y=float(m["y"]), z=float(m["z"]),
            qw=float(m["rw"]), qx=float(m["rx"]),
            qy=float(m["ry"]), qz=float(m["rz"]),
        )

        # CSV write (cheap, line buffered)
        if self._csv_writer is not None:
            self._csv_writer.writerow([
                ev.t_wall_iso, ev.t_wall_ns, f"{ev.t_libsurvive:.6f}",
                ev.name, ev.serial,
                f"{ev.x:.6f}", f"{ev.y:.6f}", f"{ev.z:.6f}",
                f"{ev.qw:.6f}", f"{ev.qx:.6f}", f"{ev.qy:.6f}", f"{ev.qz:.6f}",
            ])
            # flush periodically (every 50 poses)
            if self._pose_count % 50 == 0 and self._csv_file is not None:
                self._csv_file.flush()

        self._pose_count += 1
        self._last_pose_wall_ns = now_ns
        self._device_seen[ev.name] = now_ns
        self._rate_window.append(now_ns)
        self._vive_stage = "streaming poses"

        cb = self.on_pose
        if cb is not None:
            try:
                cb(ev)
            except Exception as e:  # pragma: no cover
                print(f"[vive_logger] on_pose callback raised: {e}",
                      file=sys.stderr)


# --- CLI -------------------------------------------------------------------


def _cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", required=True, type=Path,
                        help="Where to write vive_poses.csv")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="Auto-stop after N seconds; 0 = run until Ctrl-C")
    parser.add_argument("--no-attach", action="store_true",
                        help="Skip usbipd attach (assume already attached)")
    args = parser.parse_args()

    if not args.no_attach:
        ka = start_wsl_keepalive()
        time.sleep(3)
        statuses = attach_dongles()
        for bid, msg in statuses.items():
            print(f"[setup] {bid}: {msg}", flush=True)
        if count_watchman_in_wsl() == 0:
            print("[setup] ERROR: no Watchman dongle visible in WSL.",
                  flush=True)
            ka.terminate()
            return 1
    else:
        ka = None

    def on_pose(ev: PoseEvent):
        if logger._pose_count <= 3 or logger._pose_count % 250 == 0:
            print(f"[pose #{logger._pose_count}] {ev.name} "
                  f"({ev.x:+.3f},{ev.y:+.3f},{ev.z:+.3f})", flush=True)

    logger = ViveLogger(args.session_dir, on_pose=on_pose)
    logger.start()
    print(f"[run] Logging to {logger._csv_path}", flush=True)

    t0 = time.monotonic()
    try:
        while logger.is_alive():
            if args.duration > 0 and time.monotonic() - t0 >= args.duration:
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[run] Stopped by user.", flush=True)
    finally:
        logger.stop()
        if ka is not None:
            ka.terminate()
        print(f"[run] {logger._pose_count} poses logged.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
