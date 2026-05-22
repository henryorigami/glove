from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from retargeting.hand_v8_mapping import FINGER_ROBOT, robot_keypoints


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JOINT_CALIBRATION = PROJECT_ROOT / "config" / "hand_v9_joint_calibration.json"


def load_joint_calibration(path: str | Path | None = DEFAULT_JOINT_CALIBRATION) -> dict[str, Any] | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_joint_calibration(path: str | Path, calibration: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calibration, indent=2), encoding="utf-8", newline="\n")


def apply_joint_calibration(q: np.ndarray, joint_names: list[str], calibration: dict[str, Any] | None) -> np.ndarray:
    if calibration is None:
        return q
    joints = calibration.get("joints", {})
    out = np.asarray(q, dtype=float).copy()
    for i, name in enumerate(joint_names):
        spec = joints.get(name)
        if not spec or not spec.get("enabled", True):
            continue
        neutral = float(spec.get("neutral_raw", 0.0))
        display_neutral = float(spec.get("display_neutral", 0.0))
        sign = float(spec.get("sign", 1.0))
        scale = float(spec.get("scale", 1.0))
        value = display_neutral + sign * scale * (out[i] - neutral)
        lower = spec.get("lower")
        upper = spec.get("upper")
        if lower is not None:
            value = max(float(lower), value)
        if upper is not None:
            value = min(float(upper), value)
        out[i] = value
    return out


def model_forward_signs(retargeter, eps: float = 0.35) -> dict[str, float]:
    """Infer which MuJoCo/URDF joint direction curls each finger toward the palm."""
    signs: dict[str, float] = {}
    base = np.zeros(len(retargeter.joint_names), dtype=float)
    name_to_idx = {name: i for i, name in enumerate(retargeter.joint_names)}
    rest = robot_keypoints(retargeter.kin, base)

    for finger, spec in FINGER_ROBOT.items():
        if finger not in rest or not rest[finger]:
            continue
        palm_point = rest[finger][0]
        tip_idx = len(rest[finger]) - 1
        for joint in spec["joints"]:
            idx = name_to_idx.get(joint)
            if idx is None:
                continue
            q_pos = base.copy()
            q_neg = base.copy()
            q_pos[idx] = eps
            q_neg[idx] = -eps
            pos_pts = robot_keypoints(retargeter.kin, q_pos).get(finger, [])
            neg_pts = robot_keypoints(retargeter.kin, q_neg).get(finger, [])
            if len(pos_pts) <= tip_idx or len(neg_pts) <= tip_idx:
                signs[joint] = 1.0
                continue
            d_pos = float(np.linalg.norm(pos_pts[tip_idx] - palm_point))
            d_neg = float(np.linalg.norm(neg_pts[tip_idx] - palm_point))
            signs[joint] = 1.0 if d_pos <= d_neg else -1.0
    return signs


def summarize_joint_samples(samples: list[np.ndarray]) -> dict[str, float]:
    if not samples:
        return {"median": 0.0, "p05": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(samples, dtype=float)
    return {
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }
