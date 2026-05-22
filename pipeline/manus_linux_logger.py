"""
LinuxManusLogger - wraps the MANUS SDK Core Integrated logger in WSL.

This path does not require MANUS Core on Windows. It attaches the MANUS dongle
to WSL through usbipd, runs linux_manus_logger/manus_integrated_logger, and
surfaces stdout/stderr to the dashboard.
"""

from __future__ import annotations

import dataclasses
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from pipeline.vive_logger import USBIPD_EXE, WSL_DISTRO, WSL_USER


MANUS_VID_PID = "3325:0049"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LINUX_MANUS_DIR = PROJECT_ROOT / "linux_manus_logger"
LINUX_MANUS_EXE = "/mnt/c/Users/henry/Desktop/hand_capture/linux_manus_logger/manus_integrated_logger"

STATS_RE = re.compile(
    r"^stats\s+skeleton_frames=(?P<skel>\d+)\s+ergonomics_frames=(?P<ergo>\d+)\s+"
    r"raw_device_frames=(?P<raw>\d+)"
)


def discover_manus_busids() -> list[tuple[str, str]]:
    """Return [(busid, state), ...] for MANUS USB devices visible to usbipd."""
    out: list[tuple[str, str]] = []
    try:
        result = subprocess.run(
            [USBIPD_EXE, "list"], capture_output=True, text=True, timeout=10
        )
    except Exception:
        return out
    for line in result.stdout.splitlines():
        if MANUS_VID_PID not in line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        state_idx = line.rfind("  ")
        state = line[state_idx:].strip() if state_idx > 0 else ""
        out.append((parts[0], state))
    return out


def attach_manus_dongle() -> dict[str, str]:
    """Attach MANUS dongle(s) to WSL.

    The first bind must be done from an elevated shell. If the device is not
    shared yet, this returns a clear message instead of silently failing.
    """
    busids = [b for b, _ in discover_manus_busids()]
    statuses: dict[str, str] = {}
    if not busids:
        return {"(none)": f"no MANUS dongle ({MANUS_VID_PID}) visible"}

    keepalive = subprocess.Popen(
        ["wsl", "-d", WSL_DISTRO, "-u", WSL_USER, "--exec", "sleep", "30"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1.0)

    for bid in busids:
        subprocess.run(
            [USBIPD_EXE, "detach", "--busid", bid],
            capture_output=True, text=True,
        )
        time.sleep(0.5)
        result = subprocess.run(
            [USBIPD_EXE, "attach", "--wsl", "--busid", bid],
            capture_output=True, text=True,
        )
        combined = (result.stdout + result.stderr).strip()
        last = combined.splitlines()[-1] if combined else "(no output)"
        low = last.lower()
        if "already attached" in low:
            last = "ok (already attached)"
        elif result.returncode == 0:
            last = "ok"
        elif "not shared" in low or "bind" in low:
            last = "needs admin bind: run tools\\share_manus_usb.ps1"
        time.sleep(0.5)
        seen = count_manus_in_wsl()
        if seen <= 0 and last == "ok":
            last = "attach returned ok, but WSL does not see MANUS yet"
        statuses[bid] = last
    # Let the caller's real WSL keepalive take over if this is a session start.
    try:
        keepalive.terminate()
    except Exception:
        pass
    return statuses


def count_manus_in_wsl() -> int:
    try:
        result = subprocess.run(
            [
                "wsl", "-d", WSL_DISTRO, "-u", WSL_USER, "--exec",
                "bash", "-lc", f"lsusb | grep -c {MANUS_VID_PID} || true",
            ],
            capture_output=True, text=True, timeout=5,
        )
    except subprocess.TimeoutExpired:
        return 0
    try:
        return int(result.stdout.strip())
    except ValueError:
        return 0


def wsl_path(path: Path) -> str:
    result = subprocess.run(
        ["wsl", "-d", WSL_DISTRO, "-u", WSL_USER, "--exec", "wslpath", "-a", str(path)],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


@dataclasses.dataclass
class ManusLinuxStats:
    alive: bool = False
    skeleton_frames: int = 0
    ergonomics_frames: int = 0
    raw_device_frames: int = 0
    last_frame_wall_ns: int = 0
    rate_hz: float = 0.0
    log_path: str = ""


class LinuxManusLogger:
    def __init__(
        self,
        session_dir: Path,
        on_log_line: Optional[Callable[[str], None]] = None,
    ):
        self.session_dir = Path(session_dir)
        self.on_log_line = on_log_line
        self._proc: Optional[subprocess.Popen] = None
        self._stdout_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._log_path = self.session_dir / "manus_linux_logger.log"
        self._log_file = None
        self._skel = 0
        self._ergo = 0
        self._raw = 0
        self._last_frame_wall_ns = 0
        self._last_stats_wall_ns = 0
        self._last_stats_skel = 0
        self._rate_hz = 0.0

    def start(self) -> None:
        if self._proc is not None:
            raise RuntimeError("LinuxManusLogger already started")
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = self._log_path.open("w", encoding="utf-8", newline="\n")
        session_wsl = wsl_path(self.session_dir)
        argv = [
            "wsl", "-d", WSL_DISTRO, "-u", WSL_USER, "--exec",
            "bash", "-lc",
            (
                "cd /mnt/c/Users/henry/Desktop/hand_capture/linux_manus_logger && "
                f"./manus_integrated_logger --session-dir '{session_wsl}' --prefix manus_"
            ),
        ]
        self._proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        self._stop_evt.clear()
        self._stdout_thread = threading.Thread(
            target=self._reader_loop, args=("stdout", self._proc.stdout),
            name="manus-linux-stdout", daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._reader_loop, args=("stderr", self._proc.stderr),
            name="manus-linux-stderr", daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

    def stop(self, timeout: float = 4.0) -> None:
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
        for t in (self._stdout_thread, self._stderr_thread):
            if t is not None:
                t.join(timeout=timeout)
        if self._log_file is not None:
            self._log_file.flush()
            self._log_file.close()
            self._log_file = None
        self._proc = None
        self._stdout_thread = None
        self._stderr_thread = None

    def is_alive(self) -> bool:
        proc = self._proc
        return proc is not None and proc.poll() is None

    def stats(self) -> dict:
        with self._lock:
            return dataclasses.asdict(ManusLinuxStats(
                alive=self.is_alive(),
                skeleton_frames=self._skel,
                ergonomics_frames=self._ergo,
                raw_device_frames=self._raw,
                last_frame_wall_ns=self._last_frame_wall_ns,
                rate_hz=self._rate_hz,
                log_path=str(self._log_path),
            ))

    def _reader_loop(self, source: str, stream) -> None:
        if stream is None:
            return
        for line in stream:
            if self._stop_evt.is_set():
                break
            clean = line.rstrip("\n")
            if not clean:
                continue
            if self._log_file is not None:
                try:
                    self._log_file.write(f"[{source}] {clean}\n")
                    self._log_file.flush()
                except Exception:
                    pass
            self._handle_stats(clean)
            if self.on_log_line is not None:
                self.on_log_line(clean)

    def _handle_stats(self, line: str) -> None:
        m = STATS_RE.match(line.strip())
        if not m:
            return
        now = time.time_ns()
        with self._lock:
            new_skel = int(m["skel"])
            new_ergo = int(m["ergo"])
            new_raw = int(m["raw"])
            if new_skel > self._skel or new_ergo > self._ergo or new_raw > self._raw:
                self._last_frame_wall_ns = now
            if self._last_stats_wall_ns and new_skel >= self._last_stats_skel:
                dt = (now - self._last_stats_wall_ns) / 1e9
                if dt > 0:
                    self._rate_hz = (new_skel - self._last_stats_skel) / dt
            self._last_stats_wall_ns = now
            self._last_stats_skel = new_skel
            self._skel = new_skel
            self._ergo = new_ergo
            self._raw = new_raw
