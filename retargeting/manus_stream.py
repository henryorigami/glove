from __future__ import annotations

import shlex
import subprocess
import threading
from pathlib import Path

from pipeline.manus_linux_logger import (
    LINUX_MANUS_DIR,
    WSL_DISTRO,
    WSL_USER,
    attach_manus_dongle,
    count_manus_in_wsl,
    wsl_path,
)


def start_manus_process(session_dir: Path, duration: int, log) -> subprocess.Popen:
    session_dir.mkdir(parents=True, exist_ok=True)
    log.write("Attaching MANUS dongle to WSL...")
    for busid, status in attach_manus_dongle().items():
        log.write(f"  {busid}: {status}")
    log.write(f"WSL sees {count_manus_in_wsl()} MANUS dongle(s)")

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


def stderr_printer(proc: subprocess.Popen, stop_evt: threading.Event, log) -> None:
    if proc.stderr is None:
        return
    for line in proc.stderr:
        if stop_evt.is_set():
            break
        clean = line.rstrip()
        if clean:
            log.write(f"[manus] {clean}")
