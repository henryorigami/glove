"""Helpers for preserving libsurvive's lighthouse calibration cache."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from pipeline.vive_logger import WSL_DISTRO, WSL_USER


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "calibration" / "libsurvive_cache"
BACKUP_PATH = CACHE_DIR / "config.json"
WSL_CONFIG = "/root/.config/libsurvive/config.json"


def _run_wsl(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["wsl", "-d", WSL_DISTRO, "-u", WSL_USER, "--exec", "bash", "-lc", script],
        capture_output=True,
        text=True,
        timeout=10,
    )


def snapshot_config() -> dict:
    """Copy the current WSL libsurvive config into the project cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    result = _run_wsl(f"test -s {WSL_CONFIG} && cat {WSL_CONFIG}")
    if result.returncode != 0 or not result.stdout.strip():
        return {"ok": False, "error": "no libsurvive config found"}
    BACKUP_PATH.write_text(result.stdout, encoding="utf-8")
    return {"ok": True, "path": str(BACKUP_PATH), "bytes": len(result.stdout)}


def restore_config_if_missing() -> dict:
    """Restore cached config into WSL only if WSL has no config."""
    if not BACKUP_PATH.exists():
        return {"ok": False, "restored": False, "reason": "no project cache"}
    check = _run_wsl(f"test -s {WSL_CONFIG}")
    if check.returncode == 0:
        return {"ok": True, "restored": False, "reason": "WSL config already exists"}
    text = BACKUP_PATH.read_text(encoding="utf-8")
    escaped = text.replace("'", "'\"'\"'")
    result = _run_wsl(
        f"mkdir -p /root/.config/libsurvive && printf '%s' '{escaped}' > {WSL_CONFIG}"
    )
    return {
        "ok": result.returncode == 0,
        "restored": result.returncode == 0,
        "stderr": result.stderr.strip(),
    }


def config_summary() -> dict:
    """Return a small summary of cached lighthouse state from WSL config."""
    result = _run_wsl(f"test -s {WSL_CONFIG} && cat {WSL_CONFIG}")
    text = result.stdout if result.returncode == 0 else ""
    # libsurvive's config file is not strict JSON; summarize by simple markers.
    return {
        "exists": bool(text.strip()),
        "ootx_set_count": text.count('"OOTXSet":"1"'),
        "position_set_count": text.count('"PositionSet":"1"'),
        "project_backup_exists": BACKUP_PATH.exists(),
        "project_backup": str(BACKUP_PATH),
    }
