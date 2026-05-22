from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from retargeting.retarget_hand_v8 import DEFAULT_URDF
from retargeting.urdf_kinematics import URDFKinematics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "config" / "hand_v9_discovered_limits.json"


def _vec(text: str | None, default=(0.0, 0.0, 0.0)) -> np.ndarray:
    if not text:
        return np.array(default, dtype=float)
    return np.array([float(x) for x in text.split()], dtype=float)


def _origin_matrix(elem: ET.Element | None) -> np.ndarray:
    out = np.eye(4)
    if elem is None:
        return out
    out[:3, 3] = _vec(elem.attrib.get("xyz"))
    out[:3, :3] = Rotation.from_euler("xyz", _vec(elem.attrib.get("rpy"))).as_matrix()
    return out


@dataclass
class CollisionBox:
    link: str
    local_from_link: np.ndarray
    center_local: np.ndarray
    half_extents: np.ndarray


@dataclass
class WorldBox:
    link: str
    center: np.ndarray
    axes: np.ndarray
    half_extents: np.ndarray


def _mesh_bounds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import trimesh

    mesh = trimesh.load_mesh(path, force="mesh")
    if mesh.is_empty:
        raise RuntimeError(f"empty mesh: {path}")
    bounds = np.asarray(mesh.bounds, dtype=float)
    return bounds[0], bounds[1]


def load_collision_boxes(urdf: Path, pad_m: float = 0.001) -> list[CollisionBox]:
    root = ET.parse(urdf).getroot()
    boxes: list[CollisionBox] = []
    for link in root.findall("link"):
        link_name = link.attrib["name"]
        for collision in link.findall("collision"):
            origin = _origin_matrix(collision.find("origin"))
            geom = collision.find("geometry")
            if geom is None:
                continue
            box = geom.find("box")
            mesh = geom.find("mesh")
            if box is not None:
                size = _vec(box.attrib.get("size"))
                center = np.zeros(3)
                half = np.maximum(size / 2.0 + pad_m, pad_m)
            elif mesh is not None:
                filename = mesh.attrib.get("filename", "")
                filename = filename.removeprefix("package:///").removeprefix("package://")
                lo, hi = _mesh_bounds(urdf.parent / filename)
                center = (lo + hi) / 2.0
                half = np.maximum((hi - lo) / 2.0 + pad_m, pad_m)
            else:
                continue
            boxes.append(CollisionBox(link_name, origin, center, half))
    if not boxes:
        raise RuntimeError("No collision boxes/meshes found in URDF")
    return boxes


def transform_boxes(kin: URDFKinematics, boxes: list[CollisionBox], q: dict[str, float]) -> list[WorldBox]:
    poses = kin.forward(q)
    out: list[WorldBox] = []
    for box in boxes:
        if box.link not in poses:
            continue
        t = poses[box.link] @ box.local_from_link
        center_h = np.ones(4)
        center_h[:3] = box.center_local
        out.append(WorldBox(
            link=box.link,
            center=(t @ center_h)[:3],
            axes=t[:3, :3],
            half_extents=box.half_extents,
        ))
    return out


def obb_intersects(a: WorldBox, b: WorldBox, eps: float = 1e-7) -> bool:
    # SAT for two oriented boxes. Axes columns are box local unit axes in world.
    ra = a.half_extents
    rb = b.half_extents
    r = a.axes.T @ b.axes
    abs_r = np.abs(r) + eps
    t = a.axes.T @ (b.center - a.center)

    for i in range(3):
        if abs(t[i]) > ra[i] + float(np.dot(rb, abs_r[i, :])):
            return False
    for j in range(3):
        if abs(float(np.dot(t, r[:, j]))) > float(np.dot(ra, abs_r[:, j])) + rb[j]:
            return False
    for i in range(3):
        for j in range(3):
            lhs = abs(t[(i + 2) % 3] * r[(i + 1) % 3, j] - t[(i + 1) % 3] * r[(i + 2) % 3, j])
            rhs = (
                ra[(i + 1) % 3] * abs_r[(i + 2) % 3, j]
                + ra[(i + 2) % 3] * abs_r[(i + 1) % 3, j]
                + rb[(j + 1) % 3] * abs_r[i, (j + 2) % 3]
                + rb[(j + 2) % 3] * abs_r[i, (j + 1) % 3]
            )
            if lhs > rhs:
                return False
    return True


def link_distances(kin: URDFKinematics) -> dict[tuple[str, str], int]:
    graph: dict[str, set[str]] = {}
    for joint in kin.joints:
        graph.setdefault(joint.parent, set()).add(joint.child)
        graph.setdefault(joint.child, set()).add(joint.parent)
    out: dict[tuple[str, str], int] = {}
    for start in graph:
        seen = {start: 0}
        q = deque([start])
        while q:
            cur = q.popleft()
            for nxt in graph[cur]:
                if nxt in seen:
                    continue
                seen[nxt] = seen[cur] + 1
                q.append(nxt)
        for end, dist in seen.items():
            out[(start, end)] = dist
    return out


def colliding_pairs(
    world_boxes: list[WorldBox],
    distances: dict[tuple[str, str], int],
    ignore_depth: int,
) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for i in range(len(world_boxes)):
        for j in range(i + 1, len(world_boxes)):
            a = world_boxes[i]
            b = world_boxes[j]
            if a.link == b.link:
                continue
            if distances.get((a.link, b.link), math.inf) <= ignore_depth:
                continue
            if obb_intersects(a, b):
                pairs.add(tuple(sorted((a.link, b.link))))
    return pairs


def discover_limits(
    urdf: Path,
    joints: list[str],
    max_abs: float,
    step: float,
    margin: float,
    ignore_depth: int,
) -> dict:
    kin = URDFKinematics(urdf)
    boxes = load_collision_boxes(urdf)
    distances = link_distances(kin)
    zero_q = {name: 0.0 for name in kin.actuated_joints}
    baseline = colliding_pairs(transform_boxes(kin, boxes, zero_q), distances, ignore_depth)
    discovered: dict[str, dict] = {}

    def sweep(joint: str, direction: float) -> tuple[float, list[tuple[str, str]]]:
        last_ok = 0.0
        first_new: set[tuple[str, str]] = set()
        values = np.arange(step, max_abs + step * 0.5, step)
        for mag in values:
            val = float(direction * mag)
            q = dict(zero_q)
            q[joint] = val
            pairs = colliding_pairs(transform_boxes(kin, boxes, q), distances, ignore_depth)
            new_pairs = pairs - baseline
            if new_pairs:
                first_new = new_pairs
                break
            last_ok = val
        if last_ok > 0:
            last_ok = max(0.0, last_ok - margin)
        elif last_ok < 0:
            last_ok = min(0.0, last_ok + margin)
        return last_ok, sorted(first_new)

    for joint in joints:
        if joint not in kin.actuated_joints:
            continue
        lo, lo_hit = sweep(joint, -1.0)
        hi, hi_hit = sweep(joint, 1.0)
        discovered[joint] = {
            "lower": float(lo),
            "upper": float(hi),
            "negative_collision_at": lo_hit,
            "positive_collision_at": hi_hit,
        }

    return {
        "kind": "hand_v9_collision_proxy_limits",
        "urdf": str(urdf),
        "method": "collision mesh AABB boxes transformed as OBBs; single-joint sweeps from neutral",
        "max_abs": max_abs,
        "step": step,
        "margin": margin,
        "ignore_depth": ignore_depth,
        "baseline_collision_pairs": sorted(baseline),
        "joints": discovered,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Discover V9 joint limits from collision-proxy sweeps.")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-abs", type=float, default=3.0)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--ignore-depth", type=int, default=2)
    parser.add_argument("--joints", nargs="*", default=[
        "i3", "i2", "i1", "i0",
        "m3", "m2", "m1", "m0",
        "r3", "r2", "r1", "r0",
        "p3", "p2", "p1", "p0",
        "t4", "t3", "t2", "t1", "t0",
    ])
    args = parser.parse_args()

    result = discover_limits(args.urdf, args.joints, args.max_abs, args.step, args.margin, args.ignore_depth)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
