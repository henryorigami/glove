"""
Placeholder for the camera ingest path.

When the Raspberry Pi CM5 + 2 Arducam fisheye rig comes online, this module
will spawn a TCP listener on the laptop, accept JPEG frames from the Pi, and
write them into the active session under camera0/ and camera1/.

## Planned wire protocol (Pi → laptop)

The Pi opens one TCP connection per camera to `ws://<laptop>:<port>/cam<N>`.
Each frame is sent as a length-prefixed binary message:

    struct frame_header {
        uint64_t unix_ns;          // little-endian; Pi wall clock (NTP'd
                                   // against laptop, target skew < 5 ms)
        uint32_t frame_seq;        // monotonically increasing per camera
        uint32_t jpeg_byte_len;    // size of the JPEG payload to follow
    };
    // followed by jpeg_byte_len bytes of JPEG-encoded image

The Pi should also send a one-line JSON metadata blob as the very first
message after the TCP handshake, telling us the resolution, fps target,
fisheye parameters, and the camera's USB device path.

## On-disk layout in the session

    session_<UTC>/
        cameras/
            cam0/
                index.csv          # wall_time_ns, frame_seq, filename
                manifest.json      # resolution, intrinsics, etc.
                frame_<frame_seq>.jpg
                ...
            cam1/
                ...

`index.csv` is the master alignment table — for each camera frame, the
wall_time_ns is the same clock as the Vive/Manus CSVs, so a single SQL
join (or pandas merge) gives synchronized data at training time.

## Status

Not yet implemented. The Pi side is not built. This file exists so the
directory layout and protocol expectations are documented in code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class CameraReceiver:
    """Stub. To be implemented when the Pi side is ready."""
    session_dir: Path
    port: int = 9876
    num_cameras: int = 2

    def start(self) -> None:
        raise NotImplementedError("Pi camera bridge is not yet implemented.")

    def stop(self) -> None:
        pass

    def stats(self) -> dict:
        return {
            "implemented": False,
            "expected_cameras": self.num_cameras,
        }
