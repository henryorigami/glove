"""
FastAPI dashboard for hand_capture.

Hosts a single-page UI at http://localhost:8080/ that:
- Shows live status of every subsystem (BS, dongles, Vive, Manus, Rerun)
- Lets you start/stop episodes, set duration / label / notes
- Provides recovery buttons (wake BS, re-attach USB)
- Tails the orchestrator log

Run with:
    python -m dashboard.server
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from record_session import SessionOrchestrator  # noqa: E402
from pipeline.vive_logger import attach_dongles  # noqa: E402
from pipeline.manus_linux_logger import attach_manus_dongle  # noqa: E402
from pipeline import wake_basestations  # noqa: E402


# --- Globals (single orchestrator per server) ---


class _State:
    orchestrator: SessionOrchestrator | None = None
    setup_lock = asyncio.Lock()


state = _State()
app = FastAPI(title="hand_capture")

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# --- Models ---


class EpisodeStartReq(BaseModel):
    label: str = ""
    notes: str = ""
    duration_s: float = 30.0


# --- Routes ---


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/status")
async def get_status():
    if state.orchestrator is None:
        return {"orchestrator_running": False}
    s = state.orchestrator.status()
    out = asdict(s)
    out["orchestrator_running"] = True
    return out


@app.post("/session/start")
async def session_start(spawn_rerun: bool = True, wake: bool = True):
    async with state.setup_lock:
        if state.orchestrator is not None:
            return {"ok": True, "already_running": True}
        orch = SessionOrchestrator(spawn_rerun=spawn_rerun)
        # setup() does blocking IO; run in thread to keep the event loop free.
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: orch.setup(wake_basestations_first=wake)
        )
        state.orchestrator = orch
    return {"ok": True, "session_dir": str(orch._session_dir)}


@app.post("/session/stop")
async def session_stop():
    async with state.setup_lock:
        if state.orchestrator is None:
            return {"ok": True, "already_stopped": True}
        orch = state.orchestrator
        await asyncio.get_event_loop().run_in_executor(None, orch.shutdown)
        state.orchestrator = None
    return {"ok": True}


@app.post("/episode/start")
async def episode_start(req: EpisodeStartReq):
    if state.orchestrator is None:
        return {"ok": False, "error": "session not started"}
    ep = state.orchestrator.start_episode(
        label=req.label, notes=req.notes,
        duration_s=req.duration_s,
    )
    return {"ok": True, "episode": asdict(ep)}


@app.post("/episode/stop")
async def episode_stop():
    if state.orchestrator is None:
        return {"ok": False, "error": "session not started"}
    ep = state.orchestrator.stop_episode()
    return {"ok": True, "episode": asdict(ep) if ep else None}


@app.post("/actions/wake-bs")
async def action_wake_bs():
    try:
        woken = await wake_basestations.wake_all(timeout=8.0)
        return {"ok": True, "woken": woken}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/actions/reattach-usb")
async def action_reattach_usb():
    try:
        if state.orchestrator is not None:
            # Re-attach AND restart Vive logger so it sees the dongles.
            result = await asyncio.get_event_loop().run_in_executor(
                None, state.orchestrator.reattach_usb_and_restart_vive,
            )
            return result
        # No session yet — just do the attach standalone.
        statuses = await asyncio.get_event_loop().run_in_executor(
            None, attach_dongles,
        )
        return {"ok": True, "attach_status": statuses}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/actions/reattach-manus")
async def action_reattach_manus():
    try:
        if state.orchestrator is not None:
            result = await asyncio.get_event_loop().run_in_executor(
                None, state.orchestrator.reattach_usb_and_restart_manus,
            )
            return result
        statuses = await asyncio.get_event_loop().run_in_executor(
            None, attach_manus_dongle,
        )
        return {"ok": True, "attach_status": statuses}
    except Exception as e:
        return {"ok": False, "error": str(e)}


class PairReq(BaseModel):
    duration_s: float = 30.0


@app.post("/actions/pair-tracker")
async def action_pair_tracker(req: PairReq):
    if state.orchestrator is None:
        return {"ok": False, "error": "Start session first."}
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, state.orchestrator.pair_tracker, req.duration_s,
        )
        return result
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/actions/force-recalibrate")
async def action_force_recalibrate():
    if state.orchestrator is None:
        return {"ok": False, "error": "Start session first."}
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, state.orchestrator.force_recalibrate,
        )
        return result
    except Exception as e:
        return {"ok": False, "error": str(e)}


# --- WebSocket: live status broadcast ---


@app.websocket("/ws")
async def ws_status(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            if state.orchestrator is None:
                payload = {"orchestrator_running": False}
            else:
                s = state.orchestrator.status()
                payload = asdict(s)
                payload["orchestrator_running"] = True
            await ws.send_text(json.dumps(payload, default=str))
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_text(json.dumps({"error": str(e)}))
        except Exception:
            pass


# --- Shutdown hook ---


@app.on_event("shutdown")
async def on_shutdown():
    if state.orchestrator is not None:
        state.orchestrator.shutdown()


def main():
    # Bundled rerun.exe needs to be reachable from spawned child processes.
    scripts_dir = Path(sys.executable).parent / "Scripts"
    if scripts_dir.exists():
        os.environ["PATH"] = str(scripts_dir) + os.pathsep + os.environ.get("PATH", "")

    uvicorn.run(
        "dashboard.server:app",
        host="127.0.0.1", port=8080,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()
