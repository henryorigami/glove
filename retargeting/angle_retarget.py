from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from retargeting.hand_v8_mapping import FINGER_ROBOT
from retargeting.joint_calibration import model_forward_signs
from retargeting.manus_keypoints import FINGER_ORDER
from retargeting.retarget_hand_v8 import DEFAULT_URDF, HandV8Retargeter


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.zeros(3, dtype=float)
    return v / n


def _bend_angles(points: list[np.ndarray]) -> list[float]:
    """Return unsigned bend angles between consecutive MANUS bone vectors.

    Straight finger segments produce values close to 0. Curling increases the
    angle. This intentionally ignores twist around each bone, because that is
    the part that was making the V9 model look chaotic.
    """
    if len(points) < 3:
        return []
    dirs = [_unit(points[i + 1] - points[i]) for i in range(len(points) - 1)]
    out: list[float] = []
    for a, b in zip(dirs, dirs[1:]):
        if np.linalg.norm(a) < 1e-9 or np.linalg.norm(b) < 1e-9:
            out.append(0.0)
            continue
        dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
        out.append(float(np.arccos(dot)))
    return out


@dataclass
class AngleRetargeter:
    """Fast, constrained MANUS -> Hand V9 retargeter.

    This is deliberately simpler than full IK: it measures per-finger curl from
    MANUS bone vectors, drives only the bend joints, and leaves base twist/spread
    joints quiet. It is a stable live preview mode, not the final high-fidelity
    offline retargeter.
    """

    urdf_path: object = DEFAULT_URDF
    max_curl_rad: float = 1.65
    thumb_max_curl_rad: float = 1.35
    smoothness: float = 0.35
    neutral_bends: dict[str, list[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._model_retargeter = HandV8Retargeter(self.urdf_path, max_nfev=1)
        self.joint_names = list(self._model_retargeter.joint_names)
        self.forward_signs = model_forward_signs(self._model_retargeter)
        self.prev_q = np.zeros(len(self.joint_names), dtype=float)
        self.name_to_idx = {name: i for i, name in enumerate(self.joint_names)}

    def calibrate(self, manus_points: dict[str, list[np.ndarray]]) -> dict:
        self.neutral_bends = {
            finger: _bend_angles(pts)
            for finger, pts in manus_points.items()
            if finger in FINGER_ORDER
        }
        return {
            "mode": "angle",
            "fingers": sorted(self.neutral_bends.keys()),
            "neutral_bends": {k: [float(v) for v in vals] for k, vals in self.neutral_bends.items()},
        }

    def solve(self, manus_points: dict[str, list[np.ndarray]]) -> tuple[np.ndarray, dict]:
        q = np.zeros(len(self.joint_names), dtype=float)
        curls: list[float] = []

        for finger in FINGER_ORDER:
            pts = manus_points.get(finger)
            spec = FINGER_ROBOT.get(finger)
            if not pts or not spec:
                continue
            bends = _bend_angles(pts)
            neutral = self.neutral_bends.get(finger, [])
            bends = [
                max(0.0, bend - (neutral[i] if i < len(neutral) else 0.0))
                for i, bend in enumerate(bends)
            ]
            if finger == "thumb":
                self._apply_thumb(q, bends)
            else:
                self._apply_finger(q, spec["joints"], bends)
            curls.extend(bends)

        q = (1.0 - self.smoothness) * q + self.smoothness * self.prev_q
        self.prev_q = q.copy()
        return q, {
            "mode": "angle",
            "mean_tip_error_m": float("nan"),
            "mean_curl_rad": float(np.mean(curls)) if curls else 0.0,
            "max_curl_rad": float(np.max(curls)) if curls else 0.0,
        }

    def _set_joint(self, q: np.ndarray, joint: str, value: float, max_abs: float) -> None:
        idx = self.name_to_idx.get(joint)
        if idx is None:
            return
        sign = float(self.forward_signs.get(joint, 1.0))
        q[idx] = sign * float(np.clip(value, 0.0, max_abs))

    def _apply_finger(self, q: np.ndarray, joints: list[str], bends: list[float]) -> None:
        # V9 names are distal-to-base in the URDF export: x3, x2, x1, x0.
        # Keep x0 at zero for now; it was the biggest source of side twist.
        distal, middle, proximal, base = joints
        del base
        b0 = bends[0] if len(bends) > 0 else 0.0
        b1 = bends[1] if len(bends) > 1 else b0 * 0.6
        b2 = bends[2] if len(bends) > 2 else b1 * 0.6
        self._set_joint(q, proximal, 0.95 * b0, self.max_curl_rad)
        self._set_joint(q, middle, 0.90 * b1, self.max_curl_rad)
        self._set_joint(q, distal, 0.65 * b2, self.max_curl_rad)

    def _apply_thumb(self, q: np.ndarray, bends: list[float]) -> None:
        # Thumb has one extra V9 joint. Hold the root/opposition-ish joints calm
        # until we have a proper hand-specific calibration.
        b0 = bends[0] if len(bends) > 0 else 0.0
        b1 = bends[1] if len(bends) > 1 else b0 * 0.6
        b2 = bends[2] if len(bends) > 2 else b1 * 0.6
        self._set_joint(q, "t2", 0.85 * b0, self.thumb_max_curl_rad)
        self._set_joint(q, "t3", 0.80 * b1, self.thumb_max_curl_rad)
        self._set_joint(q, "t4", 0.60 * b2, self.thumb_max_curl_rad)
