"""
Apply tracker_to_wrist calibration profiles at runtime, and log a fused
"world/hand_<side>/wrist" transform whenever a Vive tracker pose arrives.

Loaded by the orchestrator. The Manus C++ logger writes its own skeleton to
`manus/glove_<id>/...` paths in the same Rerun viewer; we add the world-frame
wrist anchor here so post-processing or downstream consumers know exactly
where the wrist sat in the room at any time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation


@dataclass
class TrackerToWrist:
    """Calibration result: rigid transform from Vive tracker frame to Manus
    wrist frame. World wrist pose is:

        wrist_pos_world = tracker_pos_world + R_world_tracker @ t_meters
        wrist_R_world   = R_world_tracker @ R_quat_wxyz
    """
    side: str                      # "left" / "right"
    R_quat_wxyz: np.ndarray        # (4,)
    t_meters: np.ndarray           # (3,)
    source_file: str

    @classmethod
    def from_json(cls, path: Path) -> "TrackerToWrist":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            side=data["side"],
            R_quat_wxyz=np.array(data["R_quat_wxyz"], dtype=np.float64),
            t_meters=np.array(data["t_meters"], dtype=np.float64),
            source_file=str(path),
        )

    def to_scipy(self) -> Rotation:
        # scipy wants xyzw
        q = self.R_quat_wxyz
        return Rotation.from_quat([q[1], q[2], q[3], q[0]])


class FusionBridge:
    """Holds zero or more calibration profiles and emits world-frame
    wrist transforms.

    Each Vive WM tracker is mapped to a specific side based on the active
    calibration. When more than one calibration exists, the bridge will only
    fuse for the side whose tracker pose is currently being received; the
    mapping from tracker serial → side is established once at startup.
    """

    def __init__(self, profiles_dir: Path):
        self.profiles_dir = profiles_dir
        self._by_side: Dict[str, TrackerToWrist] = {}
        self._tracker_to_side: Dict[str, str] = {}  # serial -> side
        self._side_to_tracker_assigned: Dict[str, str] = {}
        self.load_latest()

    def load_latest(self) -> None:
        """Pick the most recent JSON for each side."""
        if not self.profiles_dir.exists():
            return
        by_side: Dict[str, Path] = {}
        for f in self.profiles_dir.glob("tracker_to_wrist_*.json"):
            data = json.loads(f.read_text(encoding="utf-8"))
            side = data.get("side")
            if side not in {"left", "right"}:
                continue
            if side not in by_side or f.stat().st_mtime > by_side[side].stat().st_mtime:
                by_side[side] = f
        for side, path in by_side.items():
            self._by_side[side] = TrackerToWrist.from_json(path)

    def has_any(self) -> bool:
        return bool(self._by_side)

    def loaded_profiles(self) -> list[str]:
        return [Path(c.source_file).name for c in self._by_side.values()]

    def _assign_tracker(self, serial: str) -> Optional[str]:
        """Best-effort: first tracker we see goes to whichever calibration
        we have. With two trackers we expect the user to label or to set
        the mapping explicitly via set_tracker_side."""
        if serial in self._tracker_to_side:
            return self._tracker_to_side[serial]
        for side in ("right", "left"):
            if side in self._by_side and side not in self._side_to_tracker_assigned:
                self._tracker_to_side[serial] = side
                self._side_to_tracker_assigned[side] = serial
                return side
        return None

    def set_tracker_side(self, serial: str, side: str) -> None:
        assert side in {"left", "right"}
        self._tracker_to_side[serial] = side
        self._side_to_tracker_assigned[side] = serial

    def log_world_wrist(
        self,
        serial: str,
        tracker_pos: np.ndarray,
        tracker_quat_xyzw: np.ndarray,
    ) -> Optional[str]:
        """Compute and log the world-frame wrist transform for the side
        this tracker is assigned to. Returns the side ("left"/"right") if
        a calibration was applied, None otherwise.
        """
        side = self._assign_tracker(serial)
        if side is None:
            return None
        cal = self._by_side[side]
        R_world_tracker = Rotation.from_quat(tracker_quat_xyzw)
        # World wrist position
        wrist_pos = tracker_pos + R_world_tracker.apply(cal.t_meters)
        wrist_R = R_world_tracker * cal.to_scipy()
        q_xyzw = wrist_R.as_quat()
        entity = f"world/hand_{side}/wrist"
        rr.log(
            entity,
            rr.Transform3D(
                translation=wrist_pos.tolist(),
                rotation=rr.Quaternion(xyzw=q_xyzw.tolist()),
            ),
        )
        rr.log(
            f"{entity}/marker",
            rr.Points3D([wrist_pos.tolist()], radii=0.015,
                        colors=[(255, 200, 80)]),
        )
        return side
