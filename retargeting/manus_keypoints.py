from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from scipy.spatial.transform import Rotation


# Observed in MANUS Core 3.1 raw skeleton output.
CHAIN_TO_FINGER = {
    5: "thumb",
    6: "index",
    7: "middle",
    8: "ring",
    9: "pinky",
}

FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]


@dataclass
class ManusFrame:
    frame_seq: int
    t_wall_ns: int
    glove_id: int
    points: dict[str, list[np.ndarray]]
    wrist_pos: np.ndarray
    wrist_quat_wxyz: np.ndarray


def iter_manus_frames(csv_path: str | Path) -> Iterator[ManusFrame]:
    csv_path = Path(csv_path)
    current_seq: str | None = None
    rows: list[dict[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq = row["frame_seq"]
            if current_seq is None:
                current_seq = seq
            if seq != current_seq:
                frame = _rows_to_frame(rows)
                if frame is not None:
                    yield frame
                rows = []
                current_seq = seq
            rows.append(row)
    if rows:
        frame = _rows_to_frame(rows)
        if frame is not None:
            yield frame


def _rows_to_frame(rows: list[dict[str, str]]) -> ManusFrame | None:
    if not rows:
        return None
    by_finger: dict[str, list[tuple[int, np.ndarray]]] = {f: [] for f in FINGER_ORDER}
    wrist_pos = np.zeros(3)
    wrist_quat = np.array([1.0, 0.0, 0.0, 0.0])
    frame_seq = int(rows[0]["frame_seq"])
    t_wall_ns = int(rows[0]["t_wall_ns"])
    glove_id = int(rows[0]["glove_id"])
    for row in rows:
        chain = int(row["chain_type"])
        node_id = int(row["node_id"])
        pos = np.array([float(row["px"]), float(row["py"]), float(row["pz"])], dtype=float)
        quat = np.array([float(row["qw"]), float(row["qx"]), float(row["qy"]), float(row["qz"])], dtype=float)
        if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(quat)):
            return None
        if chain == 13:
            wrist_pos = pos
            wrist_quat = quat
            continue
        finger = CHAIN_TO_FINGER.get(chain)
        if finger is not None:
            by_finger[finger].append((node_id, pos))
    points: dict[str, list[np.ndarray]] = {}
    for finger, vals in by_finger.items():
        vals.sort(key=lambda x: x[0])
        if len(vals) >= 4:
            points[finger] = [v[1] for v in vals]
    if not points:
        return None
    return ManusFrame(frame_seq, t_wall_ns, glove_id, points, wrist_pos, wrist_quat)


def frame_from_jsonl(obj: dict) -> ManusFrame | None:
    """Convert one `--stream-jsonl` skeleton object from manus_integrated_logger."""
    if obj.get("type") != "skeleton":
        return None
    nodes = obj.get("nodes") or []
    if not nodes:
        return None
    rows = []
    for node in nodes:
        rows.append({
            "frame_seq": str(obj["frame_seq"]),
            "t_wall_ns": str(obj["t_wall_ns"]),
            "glove_id": str(obj["glove_id"]),
            "chain_type": str(node.get("chain_type", -1)),
            "node_id": str(node.get("node_id", 0)),
            "px": str(node.get("px", 0.0)),
            "py": str(node.get("py", 0.0)),
            "pz": str(node.get("pz", 0.0)),
            "qw": str(node.get("qw", 1.0)),
            "qx": str(node.get("qx", 0.0)),
            "qy": str(node.get("qy", 0.0)),
            "qz": str(node.get("qz", 0.0)),
        })
    return _rows_to_frame(rows)


def frame_points_in_wrist(frame: ManusFrame) -> dict[str, list[np.ndarray]]:
    """Return points expressed in the wrist/root frame.

    MANUS raw skeleton positions are already usually root-relative, but applying
    the inverse wrist transform makes this robust if world coordinates are used.
    """
    q = frame.wrist_quat_wxyz
    n = np.linalg.norm(q)
    if n < 1e-9:
        rot = Rotation.identity()
    else:
        rot = Rotation.from_quat([q[1] / n, q[2] / n, q[3] / n, q[0] / n])
    out: dict[str, list[np.ndarray]] = {}
    for finger, pts in frame.points.items():
        out[finger] = [rot.inv().apply(p - frame.wrist_pos) for p in pts]
    return out


def frame_points_local(frame: ManusFrame) -> dict[str, list[np.ndarray]]:
    """Return raw MANUS points relative to the wrist position only.

    Use this when MANUS raw skeleton positions are already expressed in a
    wrist/root-oriented frame. This avoids double-applying the IMU wrist
    rotation, which makes stationary fingers appear to rotate when the hand
    rotates.
    """
    return {
        finger: [p - frame.wrist_pos for p in pts]
        for finger, pts in frame.points.items()
    }


def frame_points(frame: ManusFrame, wrist_mode: str = "local") -> dict[str, list[np.ndarray]]:
    if wrist_mode == "local":
        return frame_points_local(frame)
    if wrist_mode == "world":
        return frame_points_in_wrist(frame)
    raise ValueError(f"unknown wrist_mode: {wrist_mode}")
