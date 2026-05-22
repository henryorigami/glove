from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from retargeting.hand_v8_mapping import (
    FINGER_ROBOT,
    robot_joint_regularization_weights,
    robot_keypoints,
    select_manus_points,
)
from retargeting.manus_keypoints import FINGER_ORDER, frame_points_in_wrist, iter_manus_frames
from retargeting.urdf_kinematics import URDFKinematics


DEFAULT_URDF = Path(r"C:\Users\henry\Downloads\Hand_V8_add_tip\Hand_V8_add_tip\robot_right_identified.urdf")


def _stack_points(points: dict[str, list[np.ndarray]]) -> tuple[list[tuple[str, int]], np.ndarray]:
    keys = []
    vals = []
    for finger in FINGER_ORDER:
        if finger not in points:
            continue
        for i, p in enumerate(points[finger]):
            keys.append((finger, i))
            vals.append(p)
    return keys, np.vstack(vals)


def fit_affine(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Fit target ~= A @ [source, 1]. Returns 3x4 affine matrix."""
    x = np.hstack([source, np.ones((source.shape[0], 1))])
    mat, *_ = np.linalg.lstsq(x, target, rcond=None)
    return mat.T


def apply_affine(a: np.ndarray, p: np.ndarray) -> np.ndarray:
    return (a @ np.array([p[0], p[1], p[2], 1.0]))[:3]


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.zeros_like(v)
    return v / n


def _fit_rotation(source_dirs: list[np.ndarray], target_dirs: list[np.ndarray]) -> Rotation:
    src = np.vstack([_unit(v) for v in source_dirs if np.linalg.norm(v) > 1e-9])
    tgt = np.vstack([_unit(v) for v in target_dirs if np.linalg.norm(v) > 1e-9])
    n = min(len(src), len(tgt))
    if n < 2:
        return Rotation.identity()
    rot, _ = Rotation.align_vectors(tgt[:n], src[:n])
    return rot


class HandV8Retargeter:
    def __init__(
        self,
        urdf_path: str | Path = DEFAULT_URDF,
        pos_weight: float = 1.0,
        segment_weight: float = 0.55,
        regularization: float = 0.02,
        smoothness: float = 0.08,
    ):
        self.kin = URDFKinematics(urdf_path)
        self.joint_names = list(self.kin.actuated_joints)
        self.lower = self.kin.lower
        self.upper = self.kin.upper
        self.pos_weight = pos_weight
        self.segment_weight = segment_weight
        self.regularization = regularization
        self.smoothness = smoothness
        self.reg_weights = robot_joint_regularization_weights(self.joint_names)
        self.affine: np.ndarray | None = None
        self.finger_rot: dict[str, Rotation] = {}
        self.robot_rest: dict[str, list[np.ndarray]] = {}
        self.robot_lengths: dict[str, list[float]] = {}
        self.keys: list[tuple[str, int]] = []
        self.q_neutral = np.zeros(len(self.joint_names))
        self.prev_q = self.q_neutral.copy()

    def calibrate(self, manus_points: dict[str, list[np.ndarray]]) -> dict:
        manus_points = select_manus_points(manus_points)
        robot_points = robot_keypoints(self.kin, self.q_neutral)
        self.robot_rest = robot_points
        self.finger_rot = {}
        self.robot_lengths = {}
        segment_errors = []
        for finger in FINGER_ROBOT:
            if finger not in manus_points or finger not in robot_points:
                continue
            h = manus_points[finger]
            r = robot_points[finger]
            n = min(len(h), len(r))
            if n < 2:
                continue
            h_segments = [h[i + 1] - h[i] for i in range(n - 1)]
            r_segments = [r[i + 1] - r[i] for i in range(n - 1)]
            rot = _fit_rotation(h_segments, r_segments)
            self.finger_rot[finger] = rot
            self.robot_lengths[finger] = [float(np.linalg.norm(v)) for v in r_segments]
            for hs, rs in zip(h_segments, r_segments):
                segment_errors.append(np.linalg.norm(rot.apply(_unit(hs)) - _unit(rs)))

        m_keys, m_stack = _stack_points(manus_points)
        r_lookup = {(f, i): p for f, pts in robot_points.items() for i, p in enumerate(pts)}
        common_keys = [k for k in m_keys if k in r_lookup]
        source = np.vstack([m_stack[m_keys.index(k)] for k in common_keys])
        target = np.vstack([r_lookup[k] for k in common_keys])
        self.affine = fit_affine(source, target)
        self.keys = common_keys
        err = target - np.vstack([apply_affine(self.affine, p) for p in source])
        return {
            "keypoints": len(common_keys),
            "mean_affine_error_m": float(np.linalg.norm(err, axis=1).mean()),
            "max_affine_error_m": float(np.linalg.norm(err, axis=1).max()),
            "mean_segment_dir_error": float(np.mean(segment_errors)) if segment_errors else None,
        }

    def target_points(self, manus_points: dict[str, list[np.ndarray]]) -> dict[str, list[np.ndarray]]:
        manus_points = select_manus_points(manus_points)
        if not self.robot_rest:
            raise RuntimeError("Retargeter must be calibrated first")
        out: dict[str, list[np.ndarray]] = {}
        for finger, pts in manus_points.items():
            if finger not in self.robot_rest or finger not in self.finger_rot:
                continue
            rest = self.robot_rest[finger]
            n = min(len(pts), len(rest))
            if n < 2:
                continue
            rot = self.finger_rot[finger]
            lengths = self.robot_lengths[finger]
            target = [rest[0]]
            for i in range(n - 1):
                seg = pts[i + 1] - pts[i]
                direction = rot.apply(_unit(seg))
                target.append(target[-1] + direction * lengths[i])
            out[finger] = target
        return out

    def solve(self, manus_points: dict[str, list[np.ndarray]]) -> tuple[np.ndarray, dict]:
        target = self.target_points(manus_points)
        q0 = np.clip(self.prev_q, self.lower, self.upper)

        def residual(q: np.ndarray) -> np.ndarray:
            robot = robot_keypoints(self.kin, q)
            res: list[np.ndarray] = []
            for finger in FINGER_ORDER:
                if finger not in target or finger not in robot:
                    continue
                n = min(len(target[finger]), len(robot[finger]))
                for i in range(n):
                    res.append(self.pos_weight * (robot[finger][i] - target[finger][i]))
                for i in range(n - 1):
                    rt = robot[finger][i + 1] - robot[finger][i]
                    ht = target[finger][i + 1] - target[finger][i]
                    res.append(self.segment_weight * (rt - ht))
            res.append(self.regularization * self.reg_weights * (q - self.q_neutral))
            res.append(self.smoothness * (q - self.prev_q))
            return np.concatenate(res)

        ans = least_squares(
            residual,
            q0,
            bounds=(self.lower, self.upper),
            max_nfev=80,
            xtol=1e-5,
            ftol=1e-5,
            gtol=1e-5,
        )
        self.prev_q = ans.x.copy()
        mean_tip_error = self._mean_tip_error(ans.x, target)
        return ans.x, {
            "cost": float(ans.cost),
            "success": bool(ans.success),
            "nfev": int(ans.nfev),
            "mean_tip_error_m": float(mean_tip_error),
        }

    def _mean_tip_error(self, q: np.ndarray, target: dict[str, list[np.ndarray]]) -> float:
        robot = robot_keypoints(self.kin, q)
        errs = []
        for finger, pts in target.items():
            if finger in robot and pts and robot[finger]:
                errs.append(np.linalg.norm(robot[finger][-1] - pts[-1]))
        return float(np.mean(errs)) if errs else float("nan")


def retarget_csv(
    manus_csv: Path,
    output_csv: Path,
    urdf_path: Path = DEFAULT_URDF,
    max_frames: int = 0,
    calibration_frame: int = 0,
) -> dict:
    retargeter = HandV8Retargeter(urdf_path)
    frames = iter_manus_frames(manus_csv)
    skipped = 0
    for i, frame in enumerate(frames):
        points = frame_points_in_wrist(frame)
        if i < calibration_frame:
            continue
        calib = retargeter.calibrate(points)
        first_frame = frame
        break
    else:
        raise RuntimeError("No MANUS frames found")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    errors = []
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "t_wall_ns",
            "manus_frame_seq",
            *retargeter.joint_names,
            "ik_cost",
            "ik_success",
            "ik_nfev",
            "mean_tip_error_m",
        ])
        # Include calibration frame in output.
        pending = [first_frame]
        for frame in frames:
            pending.append(frame)
            if max_frames and len(pending) >= max_frames:
                break
        for frame in pending:
            points = frame_points_in_wrist(frame)
            try:
                q, stats = retargeter.solve(points)
            except Exception:
                skipped += 1
                continue
            errors.append(stats["mean_tip_error_m"])
            writer.writerow([
                frame.t_wall_ns,
                frame.frame_seq,
                *[f"{v:.9f}" for v in q],
                f"{stats['cost']:.9g}",
                int(stats["success"]),
                stats["nfev"],
                f"{stats['mean_tip_error_m']:.9f}",
            ])
            rows_written += 1

    summary = {
        "manus_csv": str(manus_csv),
        "output_csv": str(output_csv),
        "urdf": str(urdf_path),
        "joint_names": retargeter.joint_names,
        "frames_written": rows_written,
        "frames_skipped": skipped,
        "calibration": calib,
        "mean_tip_error_m": float(np.mean(errors)) if errors else None,
        "p95_tip_error_m": float(np.percentile(errors, 95)) if errors else None,
    }
    output_csv.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Retarget MANUS raw skeleton CSV to Hand V8 URDF joint angles.")
    parser.add_argument("--manus-csv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--max-frames", type=int, default=1000, help="0 = all frames")
    parser.add_argument("--calibration-frame", type=int, default=0, help="Frame index to use as neutral/open-hand alignment.")
    args = parser.parse_args()
    summary = retarget_csv(args.manus_csv, args.out, args.urdf, args.max_frames, args.calibration_frame)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
