"""
record_session.py — master orchestrator.

Coordinates: WSL/usbipd setup, base station wake, ViveLogger, manus_logger.exe,
Rerun viewer, and episode bookkeeping. Designed to be controlled either
directly from the CLI or via the FastAPI dashboard (which embeds this module).

Layout per session:

    recordings/session_<UTC>/
        manifest.json               # session-level info
        episodes.jsonl              # one JSON line per episode
        vive_poses.csv              # ALL Vive poses for the session
        manus_raw_skeleton.csv      # ALL Manus skeleton samples
        manus_ergonomics.csv
        manus_raw_devices.csv
        manus_manifest.json
        session.rrd                 # combined Rerun recording (optional)
        calibration/                # snapshot of profiles in use
            tracker_to_wrist_<side>.json

Episodes are metadata-only: a time range over the continuous stream. Episode
extraction is done post-hoc by filtering by wall_time_ns.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

# Make sure bundled rerun.exe is on PATH so rr.spawn works on Windows.
_scripts_dir = Path(sys.executable).parent / "Scripts"
if _scripts_dir.exists():
    os.environ["PATH"] = str(_scripts_dir) + os.pathsep + os.environ.get("PATH", "")

import rerun as rr  # noqa: E402

from pipeline.vive_logger import (  # noqa: E402
    ViveLogger,
    PoseEvent,
    start_wsl_keepalive,
    attach_dongles,
    count_watchman_in_wsl,
)
from pipeline.manus_linux_logger import (  # noqa: E402
    LinuxManusLogger,
    attach_manus_dongle,
    count_manus_in_wsl,
)
from pipeline import wake_basestations  # noqa: E402
from pipeline.rerun_bridge import FusionBridge  # noqa: E402
from pipeline import vive_config_cache  # noqa: E402

import numpy as np  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent
RECORDINGS_DIR = PROJECT_ROOT / "recordings"
CALIBRATION_DIR = PROJECT_ROOT / "calibration" / "profiles"
MANUS_LOGGER_EXE = Path(
    r"C:\Users\henry\Desktop\manus_logger\build\Release\manus_logger.exe"
)
MANUS_MODE = os.environ.get("HAND_CAPTURE_MANUS_MODE", "linux").lower()
APP_ID = "hand_capture"


@dataclass
class Episode:
    idx: int
    episode_id: str           # uuid4 hex
    label: str
    notes: str
    requested_duration_s: float
    start_wall_ns: int
    end_wall_ns: int = 0
    actual_duration_s: float = 0.0


@dataclass
class SessionStatus:
    session_dir: str = ""
    session_started_wall_ns: int = 0
    wsl_alive: bool = False
    dongles_attached: int = 0
    vive_alive: bool = False
    manus_alive: bool = False
    manus_mode: str = MANUS_MODE
    manus_usb_attached: int = 0
    manus_stats: dict = field(default_factory=dict)
    base_stations_woken: list[str] = field(default_factory=list)
    rerun_spawned: bool = False
    recording: bool = False
    current_episode_idx: Optional[int] = None
    current_episode_elapsed_s: float = 0.0
    total_episodes: int = 0
    vive_stats: dict = field(default_factory=dict)
    recent_log: list[str] = field(default_factory=list)
    calibration_loaded: list[str] = field(default_factory=list)


class SessionOrchestrator:
    """Long-lived object coordinating one recording session."""

    def __init__(
        self,
        recordings_dir: Path = RECORDINGS_DIR,
        spawn_rerun: bool = True,
    ):
        self._recordings_dir = recordings_dir
        self._spawn_rerun = spawn_rerun

        self._wsl_proc: Optional[subprocess.Popen] = None
        self._manus_proc: Optional[subprocess.Popen] = None
        self._manus_linux: Optional[LinuxManusLogger] = None
        self._vive: Optional[ViveLogger] = None

        self._session_dir: Optional[Path] = None
        self._session_started_wall_ns: int = 0
        self._episodes: list[Episode] = []
        self._current_episode: Optional[Episode] = None
        self._episode_timer: Optional[threading.Timer] = None
        self._calibration_loaded: list[str] = []
        self._lock = threading.Lock()
        self._log_buf: list[str] = []
        self._rerun_started = False
        self._fusion = FusionBridge(CALIBRATION_DIR)
        self._manus_log_path: Optional[Path] = None
        self._manus_log_file = None
        self._manus_reader_threads: list[threading.Thread] = []

    # -- helpers ------------------------------------------------------------

    def _log(self, msg: str) -> None:
        line = f"[{datetime.datetime.now().isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        with self._lock:
            self._log_buf.append(line)
            if len(self._log_buf) > 200:
                self._log_buf = self._log_buf[-200:]

    def _now_ns(self) -> int:
        return time.time_ns()

    # -- session lifecycle --------------------------------------------------

    def setup(self, wake_basestations_first: bool = True) -> Path:
        """Bring up WSL, attach USB, wake BS, init Rerun, start loggers."""
        if self._session_dir is not None:
            raise RuntimeError("Session already set up.")

        stamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%SZ")
        self._session_dir = self._recordings_dir / f"session_{stamp}"
        self._session_dir.mkdir(parents=True, exist_ok=True)
        (self._session_dir / "calibration").mkdir(exist_ok=True)
        self._session_started_wall_ns = self._now_ns()

        self._log(f"Session dir: {self._session_dir}")

        if wake_basestations_first:
            self._log("Waking base stations...")
            try:
                woken = asyncio.run(wake_basestations.wake_all(timeout=8.0))
                self._log(f"  woken: {woken or 'none'}")
            except Exception as e:
                self._log(f"  BLE wake failed: {e}")

        self._log("Starting WSL keepalive...")
        self._wsl_proc = start_wsl_keepalive()
        time.sleep(3)

        restored = vive_config_cache.restore_config_if_missing()
        if restored.get("restored"):
            self._log("Restored libsurvive cache from project backup")
        cfg = vive_config_cache.config_summary()
        if cfg.get("exists"):
            self._log(
                "libsurvive cache: "
                f"OOTX={cfg.get('ootx_set_count', 0)} "
                f"position={cfg.get('position_set_count', 0)}"
            )

        self._log("Attaching Watchman dongles to WSL...")
        statuses = attach_dongles()
        for bid, msg in statuses.items():
            self._log(f"  {bid}: {msg}")
        time.sleep(2)
        seen = count_watchman_in_wsl()
        self._log(f"  WSL sees {seen} dongle(s)")

        if MANUS_MODE == "linux":
            self._log("Attaching MANUS dongle to WSL...")
            manus_statuses = attach_manus_dongle()
            for bid, msg in manus_statuses.items():
                self._log(f"  manus {bid}: {msg}")
            time.sleep(1)
            manus_seen = count_manus_in_wsl()
            self._log(f"  WSL sees {manus_seen} MANUS dongle(s)")

        if self._spawn_rerun:
            self._log("Spawning Rerun viewer...")
            rr.init(APP_ID, spawn=True)
            rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
            rr.save(str(self._session_dir / "session.rrd"))
            self._rerun_started = True

        self._copy_calibration_profiles()
        self._write_manifest()
        self._start_loggers()
        return self._session_dir

    def _copy_calibration_profiles(self) -> None:
        """Snapshot the active calibration JSONs into the session dir."""
        if not CALIBRATION_DIR.exists():
            return
        dest = self._session_dir / "calibration"
        for f in CALIBRATION_DIR.glob("*.json"):
            (dest / f.name).write_text(f.read_text(encoding="utf-8"),
                                       encoding="utf-8")
            self._calibration_loaded.append(f.name)
            self._log(f"  loaded calibration: {f.name}")

    def _start_loggers(self, force_ootx: bool = False,
                       force_calibrate: bool = False) -> None:
        self._log("Starting ViveLogger"
                  + (" [force_ootx]" if force_ootx else "")
                  + (" [force_calibrate]" if force_calibrate else "")
                  + "...")
        self._vive = ViveLogger(
            self._session_dir,
            on_pose=self._on_vive_pose,
            on_log_line=self._on_vive_log_line,
            force_ootx=force_ootx,
            force_calibrate=force_calibrate,
        )
        self._vive.start()

        if MANUS_MODE == "linux":
            self._log("Starting MANUS integrated logger in WSL...")
            self._manus_linux = LinuxManusLogger(
                self._session_dir,
                on_log_line=self._on_manus_linux_log_line,
            )
            self._manus_linux.start()
            return

        self._log("Starting manus_logger.exe...")
        if not MANUS_LOGGER_EXE.exists():
            self._log(f"  WARN: {MANUS_LOGGER_EXE} not found; "
                      "Manus stream will be unavailable.")
        else:
            self._manus_proc = subprocess.Popen(
                [str(MANUS_LOGGER_EXE),
                 "--session-dir", str(self._session_dir),
                 "--prefix", "manus_"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=MANUS_LOGGER_EXE.parent,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
                text=True,
                bufsize=1,
            )
            # Drain manus_logger's stdout + stderr so it doesn't block on a
            # full pipe, and surface diagnostics to the dashboard log.
            self._manus_log_path = self._session_dir / "manus_logger.log"
            self._manus_log_file = self._manus_log_path.open(
                "w", encoding="utf-8", newline="\n"
            )
            self._manus_reader_threads = []
            for src, stream in (("stdout", self._manus_proc.stdout),
                                ("stderr", self._manus_proc.stderr)):
                if stream is None:
                    continue
                t = threading.Thread(
                    target=self._manus_reader_loop,
                    args=(src, stream),
                    name=f"manus-{src}-reader",
                    daemon=True,
                )
                t.start()
                self._manus_reader_threads.append(t)

    def _on_manus_linux_log_line(self, line: str) -> None:
        self._log(f"[manus-linux] {line}")

    def _manus_reader_loop(self, source: str, stream) -> None:
        try:
            for line in stream:
                clean = line.rstrip("\n")
                if not clean:
                    continue
                try:
                    if self._manus_log_file is not None:
                        self._manus_log_file.write(f"[{source}] {clean}\n")
                        self._manus_log_file.flush()
                except Exception:
                    pass
                self._log(f"[manus] {clean}")
        except Exception:
            pass

    def _on_vive_log_line(self, line: str) -> None:
        """libsurvive diagnostic lines forwarded from the Vive subprocess."""
        # Filter the obvious libusb noise so the dashboard log stays readable.
        low = line.lower()
        if "libusb: debug" in low or "handle_events" in low:
            return
        if "libusb_get_next_timeout" in low or "libusb_handle_events" in low:
            return
        if "full payload" in low:
            return
        self._log(f"[vive] {line}")

    def _on_vive_pose(self, ev: PoseEvent) -> None:
        if not self._rerun_started:
            return
        rr.set_time("session_wall",
                    duration=(ev.t_wall_ns - self._session_started_wall_ns) / 1e9)
        path = f"world/{ev.name}"
        rr.log(
            path,
            rr.Transform3D(
                translation=[ev.x, ev.y, ev.z],
                rotation=rr.Quaternion(xyzw=[ev.qx, ev.qy, ev.qz, ev.qw]),
            ),
        )
        color = (220, 70, 220) if ev.name.startswith("LH") else (70, 220, 90)
        half = 0.04 if ev.name.startswith("LH") else 0.025
        rr.log(
            f"{path}/body",
            rr.Boxes3D(half_sizes=[[half, half, half]], colors=[color]),
        )

        # Apply tracker→wrist calibration if one is loaded for this tracker.
        if self._fusion.has_any() and ev.name.startswith("WM"):
            self._fusion.log_world_wrist(
                ev.serial,
                np.array([ev.x, ev.y, ev.z]),
                np.array([ev.qx, ev.qy, ev.qz, ev.qw]),
            )

    def force_recalibrate(self) -> dict:
        """Wipe libsurvive's cached config and restart the Vive logger with
        --force-ootx + --force-calibrate. Use this after lighthouse changes
        (channel reassignment, repositioning, new station added) when poses
        stop being computed despite USB / lighthouses being healthy.
        """
        self._log("FORCE RECALIBRATE: wiping libsurvive cache + force-ootx")
        # Wipe config inside WSL.
        try:
            subprocess.run(
                ["wsl", "-d", "Ubuntu", "-u", "root", "--exec",
                 "rm", "-f", "/root/.config/libsurvive/config.json"],
                check=False, capture_output=True,
            )
            self._log("  cache wiped")
        except Exception as e:
            self._log(f"  cache wipe failed: {e}")
        if self._vive is not None:
            self._vive.stop()
        self._vive = ViveLogger(
            self._session_dir,
            on_pose=self._on_vive_pose,
            on_log_line=self._on_vive_log_line,
            force_ootx=True,
            force_calibrate=True,
        )
        self._vive.start()
        self._log("  Vive logger restarted with force flags")
        self._log("  >>> MOVE the tracker continuously for ~20s for OOTX <<<")
        self._log("  Vive can take 1-4 minutes to finish OOTX/global solve after a cache wipe.")
        return {"ok": True}

    def reattach_usb_and_restart_vive(self) -> dict:
        """Re-discover + re-attach Watchman dongles AND restart the Vive logger
        so it actually picks them up. The orchestrator stays alive.
        """
        from pipeline.vive_logger import attach_dongles, count_watchman_in_wsl
        self._log("Re-attaching USB dongles...")
        statuses = attach_dongles()
        for k, v in statuses.items():
            self._log(f"  {k}: {v}")
        time.sleep(2)
        seen = count_watchman_in_wsl()
        self._log(f"  WSL sees {seen} dongle(s)")

        # Restart Vive logger so api_example re-enumerates.
        if self._vive is not None:
            self._log("Restarting Vive logger...")
            self._vive.stop()
            self._vive = ViveLogger(
                self._session_dir, on_pose=self._on_vive_pose,
            )
            self._vive.start()
        return {"ok": True, "dongles_in_wsl": seen, "attach_status": statuses}

    def reattach_usb_and_restart_manus(self) -> dict:
        """Re-attach the MANUS dongle and restart the Linux integrated logger.

        Core Integrated checks its license/device state during startup. If the
        dongle was not visible then, attaching USB later is not enough; restart
        the MANUS logger after the dongle is attached.
        """
        self._log("Re-attaching MANUS dongle...")
        if self._manus_linux is not None:
            self._log("  stopping MANUS integrated logger")
            self._manus_linux.stop()
            self._manus_linux = None

        statuses = attach_manus_dongle()
        for k, v in statuses.items():
            self._log(f"  manus {k}: {v}")
        time.sleep(1)
        seen = count_manus_in_wsl()
        self._log(f"  WSL sees {seen} MANUS dongle(s)")

        if MANUS_MODE == "linux" and self._session_dir is not None:
            self._log("  restarting MANUS integrated logger")
            self._manus_linux = LinuxManusLogger(
                self._session_dir,
                on_log_line=self._on_manus_linux_log_line,
            )
            self._manus_linux.start()
        return {"ok": True, "manus_in_wsl": seen, "attach_status": statuses}

    def pair_tracker(self, duration_s: float = 30.0) -> dict:
        """Put the Watchman dongle into pairing mode for `duration_s` seconds.

        Stops the Vive logger temporarily (USB exclusive access), runs
        `survive-cli --pair-device 1`, then resumes. The user must hold the
        tracker's power button until its LED is solid (no longer blinking).
        """
        self._log(f"Pairing tracker (duration={duration_s:.0f}s)...")
        self._log("  >>> press and hold the tracker button now <<<")
        was_running = self._vive is not None and self._vive.is_alive()
        if was_running:
            assert self._vive is not None
            self._vive.stop()
            self._log("  paused Vive logger")

        try:
            proc = subprocess.run(
                ["wsl", "-d", "Ubuntu", "-u", "root", "--exec",
                 "/root/libsurvive/build/survive-cli",
                 "--pair-device", "1",
                 "--run-time", str(duration_s),
                 "--lighthouse-gen", "2"],
                capture_output=True, text=True,
                timeout=duration_s + 15,
            )
            out = (proc.stdout or "") + (proc.stderr or "")
        except subprocess.TimeoutExpired:
            out = "(timeout)"
        except Exception as e:
            out = f"(error: {e})"

        lines = out.splitlines()
        # Look for pairing success markers in libsurvive's chatty output.
        joined = out.lower()
        paired_hint = any(k in joined for k in (
            "added tracked object", "wm0 from htc", "wm1 from htc",
            "successfully paired", "device added")
        )
        for ln in lines[-30:]:
            if ln.strip():
                self._log(f"  pair> {ln[:140]}")

        if was_running:
            self._log("  resuming Vive logger")
            self._vive = ViveLogger(
                self._session_dir, on_pose=self._on_vive_pose,
            )
            self._vive.start()
        return {
            "ok": True,
            "paired_hint": paired_hint,
            "output_tail": lines[-30:],
        }

    def shutdown(self) -> None:
        self._log("Shutting down...")
        if self._current_episode is not None:
            self.stop_episode()
        if self._vive is not None:
            try:
                stats = self._vive.stats()
                if stats.get("pose_count", 0) > 0:
                    snap = vive_config_cache.snapshot_config()
                    if snap.get("ok"):
                        self._log(f"Saved libsurvive cache backup: {snap.get('path')}")
            except Exception as e:
                self._log(f"libsurvive cache backup failed: {e}")
            self._vive.stop()
            self._vive = None
        if self._manus_linux is not None:
            self._manus_linux.stop()
            self._manus_linux = None
        if self._manus_proc is not None:
            try:
                # Send Ctrl-Break to the new process group so manus_logger
                # gets its console signal handler.
                self._manus_proc.send_signal(getattr(__import__("signal"),
                                                    "CTRL_BREAK_EVENT", 1))
            except Exception:
                pass
            try:
                self._manus_proc.wait(timeout=3)
            except Exception:
                try:
                    self._manus_proc.kill()
                except Exception:
                    pass
            self._manus_proc = None
        if self._manus_log_file is not None:
            try:
                self._manus_log_file.close()
            except Exception:
                pass
            self._manus_log_file = None
        if self._wsl_proc is not None:
            try:
                self._wsl_proc.terminate()
            except Exception:
                pass
            self._wsl_proc = None
        self._write_manifest(final=True)
        self._log("Done.")

    # -- episodes -----------------------------------------------------------

    def start_episode(self, label: str = "",
                      notes: str = "",
                      duration_s: float = 30.0) -> Episode:
        with self._lock:
            if self._current_episode is not None:
                raise RuntimeError("Episode already running.")
            idx = len(self._episodes)
            ep = Episode(
                idx=idx,
                episode_id=uuid.uuid4().hex,
                label=label, notes=notes,
                requested_duration_s=duration_s,
                start_wall_ns=self._now_ns(),
            )
            self._current_episode = ep
        self._log(f"Episode {ep.idx} START label={label!r} duration={duration_s}s")
        if duration_s > 0:
            t = threading.Timer(duration_s, self._auto_stop_episode, args=(ep.idx,))
            t.daemon = True
            t.start()
            self._episode_timer = t
        return ep

    def _auto_stop_episode(self, idx: int) -> None:
        with self._lock:
            ep = self._current_episode
            if ep is None or ep.idx != idx:
                return
        self.stop_episode()

    def stop_episode(self) -> Optional[Episode]:
        with self._lock:
            ep = self._current_episode
            if ep is None:
                return None
            ep.end_wall_ns = self._now_ns()
            ep.actual_duration_s = (ep.end_wall_ns - ep.start_wall_ns) / 1e9
            self._episodes.append(ep)
            self._current_episode = None
        if self._episode_timer is not None:
            self._episode_timer.cancel()
            self._episode_timer = None
        self._append_episode(ep)
        self._log(f"Episode {ep.idx} STOP "
                  f"({ep.actual_duration_s:.2f}s actual)")
        return ep

    def _append_episode(self, ep: Episode) -> None:
        line = json.dumps(asdict(ep))
        (self._session_dir / "episodes.jsonl").open(
            "a", encoding="utf-8"
        ).write(line + "\n")

    # -- status -------------------------------------------------------------

    def status(self) -> SessionStatus:
        s = SessionStatus()
        s.session_dir = str(self._session_dir) if self._session_dir else ""
        s.session_started_wall_ns = self._session_started_wall_ns
        s.wsl_alive = self._wsl_proc is not None and self._wsl_proc.poll() is None
        s.dongles_attached = count_watchman_in_wsl() if s.wsl_alive else 0
        s.vive_alive = self._vive.is_alive() if self._vive is not None else False
        s.manus_alive = (
            (self._manus_linux is not None and self._manus_linux.is_alive())
            if MANUS_MODE == "linux"
            else (self._manus_proc is not None and self._manus_proc.poll() is None)
        )
        s.manus_mode = MANUS_MODE
        s.manus_usb_attached = count_manus_in_wsl() if s.wsl_alive else 0
        if self._manus_linux is not None:
            s.manus_stats = self._manus_linux.stats()
        s.rerun_spawned = self._rerun_started
        s.recording = self._current_episode is not None
        s.calibration_loaded = list(self._calibration_loaded)
        s.total_episodes = len(self._episodes) + (1 if s.recording else 0)
        if self._vive is not None:
            s.vive_stats = self._vive.stats()
        if self._current_episode is not None:
            s.current_episode_idx = self._current_episode.idx
            s.current_episode_elapsed_s = (
                self._now_ns() - self._current_episode.start_wall_ns
            ) / 1e9
        with self._lock:
            s.recent_log = list(self._log_buf[-30:])
        return s

    # -- manifest -----------------------------------------------------------

    def _write_manifest(self, final: bool = False) -> None:
        if self._session_dir is None:
            return
        manifest = {
            "session_id": self._session_dir.name,
            "session_started_wall_ns": self._session_started_wall_ns,
            "session_started_iso": datetime.datetime.fromtimestamp(
                self._session_started_wall_ns / 1e9,
                tz=datetime.timezone.utc,
            ).isoformat(),
            "app_id": APP_ID,
            "calibration_loaded": list(self._calibration_loaded),
            "streams": {
                "vive": "vive_poses.csv",
                "manus_skeleton": "manus_raw_skeleton.csv",
                "manus_ergonomics": "manus_ergonomics.csv",
                "manus_raw_devices": "manus_raw_devices.csv",
            },
            "manus_mode": MANUS_MODE,
            "rerun_recording": "session.rrd",
            "episodes_index": "episodes.jsonl",
            "final": final,
        }
        (self._session_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )


# --- CLI -------------------------------------------------------------------


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-rerun", action="store_true",
                        help="Don't spawn the Rerun viewer.")
    parser.add_argument("--no-wake", action="store_true",
                        help="Skip base station BLE wake.")
    parser.add_argument("--episode-duration", type=float, default=30.0,
                        help="Default episode duration in seconds.")
    parser.add_argument("--auto-episode", action="store_true",
                        help="Immediately start a single episode and exit when done.")
    parser.add_argument("--label", default="", help="Episode label (with --auto-episode).")
    args = parser.parse_args()

    orch = SessionOrchestrator(spawn_rerun=not args.no_rerun)
    orch.setup(wake_basestations_first=not args.no_wake)

    try:
        if args.auto_episode:
            orch.start_episode(label=args.label, duration_s=args.episode_duration)
            # Wait for episode to finish.
            while orch.status().recording:
                time.sleep(0.2)
        else:
            print("\nSession ready. Commands: e=start episode, s=stop, q=quit",
                  flush=True)
            while True:
                cmd = input("> ").strip().lower()
                if cmd == "q":
                    break
                elif cmd == "e":
                    orch.start_episode(duration_s=args.episode_duration)
                elif cmd == "s":
                    orch.stop_episode()
                elif cmd == "?":
                    print(json.dumps(asdict(orch.status()), indent=2,
                                     default=str))
    except KeyboardInterrupt:
        print("\nInterrupted.", flush=True)
    finally:
        orch.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
