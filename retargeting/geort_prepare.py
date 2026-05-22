from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from retargeting.manus_keypoints import FINGER_ORDER, frame_points, iter_manus_frames
from retargeting.retarget_hand_v8 import DEFAULT_URDF


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GEORT_ROOT = PROJECT_ROOT / "external" / "GeoRT"
DEFAULT_HAND_NAME = "hand_v9_right"

GEORT_JOINT_ORDER = [
    "i3", "i2", "i1",
    "m3", "m2", "m1",
    "r3", "r2", "r1",
    "p3", "p2", "p1",
    "t4", "t3", "t2", "t1", "t0",
]

FINGERTIP_CONFIG = [
    {"name": "thumb", "link": "024200000001_5", "joint": ["t4", "t3", "t2", "t1", "t0"], "human_hand_id": 4},
    {"name": "index", "link": "024200000001", "joint": ["i3", "i2", "i1"], "human_hand_id": 8},
    {"name": "middle", "link": "024200000001_2", "joint": ["m3", "m2", "m1"], "human_hand_id": 12},
    {"name": "ring", "link": "024200000001_3", "joint": ["r3", "r2", "r1"], "human_hand_id": 16},
    {"name": "pinky", "link": "024200000001_4", "joint": ["p3", "p2", "p1"], "human_hand_id": 20},
]


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.zeros(3, dtype=float)
    return v / n


def _select_four(points: list[np.ndarray]) -> list[np.ndarray] | None:
    if len(points) < 4:
        return None
    if len(points) == 4:
        return points
    return [points[0], points[1], points[-2], points[-1]]


def manus_points_to_geort21(points: dict[str, list[np.ndarray]]) -> np.ndarray | None:
    """Convert MANUS chains to GeoRT/MediaPipe-style 21 hand keypoints.

    Output frame follows GeoRT's right-hand convention:
    +Y palm center -> thumb, +Z palm center -> middle fingertip, +X palm normal.
    """
    selected: dict[str, list[np.ndarray]] = {}
    for finger in FINGER_ORDER:
        pts = _select_four(points.get(finger, []))
        if pts is None:
            return None
        selected[finger] = pts

    mcp_points = np.vstack([selected[f][0] for f in ["index", "middle", "ring", "pinky"]])
    palm = np.mean(mcp_points, axis=0)
    wrist = palm - 0.45 * (selected["middle"][-1] - selected["middle"][0])

    y_axis = _unit(selected["thumb"][-1] - palm)
    z_axis = _unit(selected["middle"][-1] - palm)
    x_axis = _unit(np.cross(y_axis, z_axis))
    if np.linalg.norm(x_axis) < 1e-9:
        return None
    y_axis = _unit(np.cross(z_axis, x_axis))
    basis = np.vstack([x_axis, y_axis, z_axis]).T

    world_points = [wrist]
    for finger in ["thumb", "index", "middle", "ring", "pinky"]:
        world_points.extend(selected[finger])
    arr = np.vstack(world_points)
    return (arr - palm) @ basis


def export_human_data(
    manus_csv: Path,
    out: Path,
    wrist_mode: str,
    sample_every: int,
    max_frames: int,
) -> dict:
    frames_written = 0
    skipped = 0
    samples: list[np.ndarray] = []
    for i, frame in enumerate(iter_manus_frames(manus_csv)):
        if i % sample_every:
            continue
        pts = frame_points(frame, wrist_mode)
        arr = manus_points_to_geort21(pts)
        if arr is None or not np.isfinite(arr).all():
            skipped += 1
            continue
        samples.append(arr.astype(np.float32))
        frames_written += 1
        if max_frames and frames_written >= max_frames:
            break
    if not samples:
        raise RuntimeError(f"No usable MANUS frames exported from {manus_csv}")
    out.parent.mkdir(parents=True, exist_ok=True)
    data = np.stack(samples, axis=0)
    np.save(out, data)
    return {"out": str(out), "shape": list(data.shape), "skipped": skipped}


def _limit_for_joint(name: str) -> tuple[float, float]:
    if name[0] in "imrp":
        if name.endswith("0"):
            return -0.05, 0.05
        return -2.75, 0.0
    if name == "t0":
        return -2.45, 0.0
    if name == "t1":
        return -2.45, 0.0
    if name in {"t2", "t3", "t4"}:
        return 0.0, 2.45
    return -2.0, 2.0


def _load_limit_overrides(path: Path | None) -> dict[str, tuple[float, float]]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, tuple[float, float]] = {}
    for name, spec in data.get("joints", {}).items():
        out[name] = (float(spec["lower"]), float(spec["upper"]))
    return out


def prepare_hand_asset(urdf: Path, geort_root: Path, hand_name: str, limits_json: Path | None = None) -> dict:
    limit_overrides = _load_limit_overrides(limits_json)
    asset_dir = geort_root / "assets" / hand_name
    asset_dir.mkdir(parents=True, exist_ok=True)
    for child in urdf.parent.iterdir():
        if child.is_file():
            shutil.copy2(child, asset_dir / child.name)

    tree = ET.parse(asset_dir / urdf.name)
    root = tree.getroot()
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename", "")
        filename = filename.removeprefix("package:///").removeprefix("package://")
        mesh.attrib["filename"] = filename
    for joint in root.findall("joint"):
        if joint.attrib.get("type") != "revolute":
            continue
        limit = joint.find("limit")
        if limit is None:
            limit = ET.SubElement(joint, "limit")
        lo, hi = limit_overrides.get(joint.attrib["name"], _limit_for_joint(joint.attrib["name"]))
        limit.attrib["lower"] = f"{lo:.6f}"
        limit.attrib["upper"] = f"{hi:.6f}"
        limit.attrib.setdefault("effort", "1")
        limit.attrib.setdefault("velocity", "20")
    sanitized = asset_dir / "robot_geort.urdf"
    tree.write(sanitized, encoding="utf-8", xml_declaration=True)

    config = {
        "name": hand_name,
        "urdf_path": f"./assets/{hand_name}/{sanitized.name}",
        "base_link": "palm",
        "joint_order": GEORT_JOINT_ORDER,
        "fingertip_link": [
            {**entry, "center_offset": [0.0, 0.0, 0.0]}
            for entry in FINGERTIP_CONFIG
        ],
    }
    config_path = geort_root / "geort" / "config" / f"{hand_name}.json"
    config_path.write_text(json.dumps(config, indent=4), encoding="utf-8", newline="\n")
    return {"asset_dir": str(asset_dir), "urdf": str(sanitized), "config": str(config_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare MANUS/V9 assets for GeoRT.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_asset = sub.add_parser("prepare-hand")
    p_asset.add_argument("--geort-root", type=Path, default=DEFAULT_GEORT_ROOT)
    p_asset.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    p_asset.add_argument("--hand-name", default=DEFAULT_HAND_NAME)
    p_asset.add_argument("--limits-json", type=Path, default=PROJECT_ROOT / "config" / "hand_v9_discovered_limits.json")

    p_data = sub.add_parser("export-human")
    p_data.add_argument("--manus-csv", type=Path, required=True)
    p_data.add_argument("--out", type=Path, default=DEFAULT_GEORT_ROOT / "data" / "manus_v9.npy")
    p_data.add_argument("--wrist-mode", choices=["world", "local"], default="world")
    p_data.add_argument("--sample-every", type=int, default=3)
    p_data.add_argument("--max-frames", type=int, default=5000)

    args = parser.parse_args()
    if args.cmd == "prepare-hand":
        print(json.dumps(prepare_hand_asset(args.urdf, args.geort_root, args.hand_name, args.limits_json), indent=2))
    elif args.cmd == "export-human":
        print(json.dumps(export_human_data(
            args.manus_csv,
            args.out,
            args.wrist_mode,
            max(1, args.sample_every),
            max(0, args.max_frames),
        ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
