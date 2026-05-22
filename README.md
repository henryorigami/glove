# hand_capture

Unified data collection for ML hand-tracking training. Fuses:

- **Vive Tracker 3.0** (6-DOF wrist pose via libsurvive in WSL)
- **Manus glove** (finger articulation + IMU via MANUS SDK Core Integrated in WSL)
- **2× Arducam fisheye on Raspberry Pi CM5** (object cameras, planned)

## Quick start

```
python record_session.py
```

Opens the dashboard at <http://localhost:8080>, brings up Rerun viewer, starts both loggers.

MANUS defaults to the Linux Core Integrated logger, so MANUS Core does not need
to be running. If Windows has not shared the MANUS USB dongle yet, run this once
from an elevated PowerShell:

```
tools\share_manus_usb.ps1
```

To temporarily use the old Windows/Core logger:

```
$env:HAND_CAPTURE_MANUS_MODE="windows"
python record_session.py
```

The repo does not commit MANUS runtime `.so` files. Put the local MANUS SDK
Linux libraries here before building or running on a new machine:

```
linux_manus_logger/ManusSDK/lib/libManusSDK_Integrated.so
linux_manus_logger/ManusSDK/lib/libManusSDK.so
```

## Layout

```
hand_capture/
├── record_session.py         # main orchestrator
├── pipeline/
│   ├── vive_logger.py        # Vive → CSV + Rerun (subprocess wrapper around WSL libsurvive)
│   ├── manus_runner.py       # spawns manus_logger.exe
│   ├── camera_receiver.py    # Pi CM5 ingest (placeholder)
│   └── rerun_bridge.py       # shared Rerun stream
├── calibration/
│   ├── collect.py            # records 2-phase calibration sequence
│   ├── solve.py              # Tsai-Lenz + sphere fit → tracker_to_wrist
│   └── profiles/             # saved JSONs
├── dashboard/
│   ├── server.py             # FastAPI + websocket
│   └── static/index.html     # control surface
├── tools/
│   └── validate_session.py
├── linux_manus_logger/    # MANUS Core Integrated logger for WSL/Linux
└── recordings/
    └── session_<UTC>/
        ├── manifest.json
        ├── episode_NNNN/
        │   ├── vive_poses.csv
        │   ├── manus_raw_skeleton.csv
        │   ├── manus_ergonomics.csv
        │   ├── manus_raw_devices.csv
        │   └── session.rrd
        └── calibration/
            └── tracker_to_wrist_<side>.json
```

## Calibration

Before recording for the first time (or after remounting the tracker on a glove):

```
python -m calibration.collect --side right
python -m calibration.solve <session-dir>
```

Two-phase capture: ~30s full rotation, then ~10s wrist stationary. Solver outputs
`R_X` (rotation) and `t_X` (translation) from Vive tracker frame to Manus wrist frame.
