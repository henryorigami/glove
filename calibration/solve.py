"""
Solve the rigid transform (R_X, t_X) from Vive tracker frame to Manus
wrist frame, using data captured by calibration.collect.

Inputs:
    A calibration session directory containing:
        vive_poses.csv
        manus_raw_skeleton.csv
        calibration_meta.json     (phase boundaries, side)

Outputs:
    profiles/tracker_to_wrist_<side>_<timestamp>.json
        {
            "side": "right",
            "R_quat_wxyz": [w, x, y, z],
            "t_meters": [x, y, z],
            "phase1_rmsd_deg": float,
            "phase2_sphere_residual_mm": float,
            "phase2_t_x_std_mm": float,
            "num_rotation_pairs": int,
            "num_translation_points": int,
            "source_session": "<path>",
            "solved_at_utc": "..."
        }

Math:
    Phase 1: hand-eye AX=XB on rotations.
        For each pair (i,j) of timestamps, compute relative rotations of both
        sensors. The constant rotation R_X satisfies dR_a · R_X = R_X · dR_b
        where dR_a is Manus IMU delta, dR_b is Vive tracker delta. We use the
        Tsai-Lenz Modified-Rodrigues-Parameter formulation.

    Phase 2: sphere fit on tracker positions while wrist is stationary.
        Tracker positions trace a sphere around the wrist joint. Sphere center
        is the wrist position in Vive world. For each sample i,
        t_X_in_tracker_i = R_v(t_i)^T · (p_i - center).
        Mean is the final translation vector; std is the residual.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp


CHAIN_TYPE_HAND = 13  # from ManusSDKTypes.h:ChainType enum


# --- IO --------------------------------------------------------------------


def load_vive_tracker(csv_path: Path,
                      t_start_ns: int, t_end_ns: int,
                      name_prefix: str = "WM"):
    """Load Vive tracker poses (position + quaternion) within time range."""
    ts, quats_wxyz, positions = [], [], []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = int(row["wall_time_ns"])
            if t < t_start_ns or t > t_end_ns:
                continue
            if not row["name"].startswith(name_prefix):
                continue
            quats_wxyz.append([
                float(row["qw"]), float(row["qx"]),
                float(row["qy"]), float(row["qz"]),
            ])
            positions.append([
                float(row["x"]), float(row["y"]), float(row["z"]),
            ])
            ts.append(t)
    return np.array(ts, dtype=np.int64), \
           np.array(quats_wxyz, dtype=np.float64), \
           np.array(positions, dtype=np.float64)


def load_manus_wrist(csv_path: Path,
                     t_start_ns: int, t_end_ns: int,
                     side: str):
    """Load Manus wrist node orientation (raw_skeleton, chain_type == Hand)."""
    side_code = 1 if side == "left" else 2  # Side_Left=1, Side_Right=2
    ts, quats_wxyz, positions = [], [], []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = int(row["t_wall_ns"])
            if t < t_start_ns or t > t_end_ns:
                continue
            if int(row["chain_type"]) != CHAIN_TYPE_HAND:
                continue
            if int(row["side"]) != side_code:
                continue
            quats_wxyz.append([
                float(row["qw"]), float(row["qx"]),
                float(row["qy"]), float(row["qz"]),
            ])
            positions.append([
                float(row["px"]), float(row["py"]), float(row["pz"]),
            ])
            ts.append(t)
    return np.array(ts, dtype=np.int64), \
           np.array(quats_wxyz, dtype=np.float64), \
           np.array(positions, dtype=np.float64)


# --- Time alignment --------------------------------------------------------


def slerp_to_times(src_ts: np.ndarray, src_quats_wxyz: np.ndarray,
                   tgt_ts: np.ndarray) -> np.ndarray:
    """Interpolate src quaternions to target timestamps via SLERP."""
    # scipy expects xyzw order
    quats_xyzw = src_quats_wxyz[:, [1, 2, 3, 0]]
    rot = R.from_quat(quats_xyzw)
    slerp = Slerp(src_ts.astype(np.float64), rot)
    in_range = (tgt_ts >= src_ts.min()) & (tgt_ts <= src_ts.max())
    out = np.zeros((len(tgt_ts), 4), dtype=np.float64)
    interp = slerp(tgt_ts[in_range].astype(np.float64)).as_quat()  # xyzw
    out[in_range] = np.concatenate(
        [interp[:, 3:4], interp[:, :3]], axis=1)  # back to wxyz
    return out, in_range


# --- Tsai-Lenz hand-eye rotation solver ------------------------------------


def _quat_wxyz_to_rotvec(q):
    """quaternion (wxyz) -> rotation vector (axis * angle)."""
    return R.from_quat(np.array([q[1], q[2], q[3], q[0]])).as_rotvec()


def tsai_lenz_rotation(R_a_list, R_b_list):
    """
    Given lists of relative rotations from two rigidly-connected sensors,
    solve R_X such that R_a · R_X = R_X · R_b for all pairs.

    Parameters
    ----------
    R_a_list, R_b_list : list of scipy.spatial.transform.Rotation
        Same length.

    Returns
    -------
    R_X : scipy.spatial.transform.Rotation
    residual_deg : float
        Mean angular residual after solve.
    """
    M_rows = []
    rhs = []
    for Ra, Rb in zip(R_a_list, R_b_list):
        # Modified Rodrigues Parameters: 2 * tan(theta/2) * axis
        rv_a = Ra.as_rotvec()
        rv_b = Rb.as_rotvec()
        ang_a = np.linalg.norm(rv_a)
        ang_b = np.linalg.norm(rv_b)
        if ang_a < 1e-6 or ang_b < 1e-6:
            continue
        pa = 2 * np.tan(ang_a / 2) * (rv_a / ang_a)
        pb = 2 * np.tan(ang_b / 2) * (rv_b / ang_b)
        M_rows.append(_skew(pa + pb))
        rhs.append(pb - pa)
    if not M_rows:
        raise ValueError("Not enough usable rotation pairs.")
    A = np.vstack(M_rows)
    b = np.concatenate(rhs)
    p_x_prime, *_ = np.linalg.lstsq(A, b, rcond=None)
    norm_sq = float(p_x_prime @ p_x_prime)
    p_x = 2 * p_x_prime / np.sqrt(1.0 + norm_sq)
    px_norm = np.linalg.norm(p_x)
    # MRP -> rotation matrix (Tsai-Lenz formula)
    eye = np.eye(3)
    skew_px = _skew(p_x)
    R_X_mat = ((1 - 0.5 * px_norm * px_norm) * eye
               + 0.5 * (np.outer(p_x, p_x) + np.sqrt(4 - px_norm * px_norm)
                        * skew_px))
    R_X = R.from_matrix(R_X_mat)

    # Residual: how well does R_a @ R_X ≈ R_X @ R_b ?
    res_angles = []
    for Ra, Rb in zip(R_a_list, R_b_list):
        lhs = Ra * R_X
        rhs_r = R_X * Rb
        err = (lhs * rhs_r.inv()).magnitude()
        res_angles.append(err)
    return R_X, float(np.degrees(np.mean(res_angles)))


def _skew(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])


# --- Sphere fit ------------------------------------------------------------


def fit_sphere(points: np.ndarray):
    """Algebraic least-squares sphere fit. Returns (center, radius, rmsd_mm)."""
    A = np.hstack([2 * points, np.ones((len(points), 1))])
    b = np.sum(points ** 2, axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    center = sol[:3]
    radius = float(np.sqrt(sol[3] + center @ center))
    dists = np.linalg.norm(points - center, axis=1)
    rmsd_m = float(np.sqrt(np.mean((dists - radius) ** 2)))
    return center, radius, rmsd_m


# --- Top-level solve -------------------------------------------------------


def solve(calib_dir: Path) -> dict:
    meta = json.loads((calib_dir / "calibration_meta.json").read_text())
    side = meta["side"]
    phase1 = next(p for p in meta["phases"] if p["name"] == "rotation")
    phase2 = next(p for p in meta["phases"] if p["name"] == "translation")

    # --- Phase 1: rotation -------------------------------------------------
    vive_ts, vive_q, _ = load_vive_tracker(
        calib_dir / "vive_poses.csv",
        phase1["start_wall_ns"], phase1["end_wall_ns"],
    )
    manus_ts, manus_q, _ = load_manus_wrist(
        calib_dir / "manus_raw_skeleton.csv",
        phase1["start_wall_ns"], phase1["end_wall_ns"],
        side=side,
    )
    if len(vive_ts) < 20 or len(manus_ts) < 20:
        raise ValueError(
            f"Not enough samples in phase 1: vive={len(vive_ts)} "
            f"manus={len(manus_ts)}. Re-record with more motion.")

    # Resample Vive onto Manus timeline via SLERP.
    vive_q_at_manus, valid = slerp_to_times(vive_ts, vive_q, manus_ts)
    valid_idx = np.where(valid)[0]
    if len(valid_idx) < 20:
        raise ValueError("Too little overlap between Vive and Manus times.")
    manus_q = manus_q[valid_idx]
    vive_q_at_manus = vive_q_at_manus[valid_idx]

    # Convert to Rotation objects.
    manus_R = R.from_quat(manus_q[:, [1, 2, 3, 0]])
    vive_R = R.from_quat(vive_q_at_manus[:, [1, 2, 3, 0]])

    # Relative motion in body frame: dR_i = R(t_i)^-1 * R(t_{i+stride})
    # Use a stride to ensure each pair has meaningful rotation.
    stride = max(1, len(manus_R) // 200)
    dR_m, dR_v = [], []
    for i in range(0, len(manus_R) - stride):
        dRa = manus_R[i].inv() * manus_R[i + stride]
        dRb = vive_R[i].inv() * vive_R[i + stride]
        if dRa.magnitude() < np.deg2rad(5):  # skip tiny motions
            continue
        dR_m.append(dRa)
        dR_v.append(dRb)
    if len(dR_m) < 10:
        raise ValueError(
            "Not enough significant rotations in phase 1; rotate harder.")

    R_X, phase1_rmsd_deg = tsai_lenz_rotation(dR_m, dR_v)
    print(f"Phase 1: {len(dR_m)} pairs, R_X residual = {phase1_rmsd_deg:.2f}°")

    # --- Phase 2: translation ---------------------------------------------
    v2_ts, v2_q, v2_p = load_vive_tracker(
        calib_dir / "vive_poses.csv",
        phase2["start_wall_ns"], phase2["end_wall_ns"],
    )
    if len(v2_p) < 50:
        raise ValueError("Not enough phase-2 tracker positions for sphere fit.")
    center, radius, sphere_rmsd_m = fit_sphere(v2_p)
    print(f"Phase 2: sphere center={center}, radius={radius*1000:.1f}mm, "
          f"rmsd={sphere_rmsd_m*1000:.2f}mm")

    # Recover t_X direction: for each frame, t_X_tracker = R_v(t)^-1 · (p - c)
    v2_R = R.from_quat(v2_q[:, [1, 2, 3, 0]])
    rel = v2_p - center
    t_X_per_sample = np.array([v2_R[i].inv().apply(rel[i])
                               for i in range(len(rel))])
    t_X = t_X_per_sample.mean(axis=0)
    t_X_std = t_X_per_sample.std(axis=0)
    print(f"Phase 2: t_X = {t_X*1000} mm  (per-axis std = {t_X_std*1000} mm)")

    # --- Pack result -------------------------------------------------------
    q_xyzw = R_X.as_quat()
    result = {
        "side": side,
        "R_quat_wxyz": [float(q_xyzw[3]), float(q_xyzw[0]),
                        float(q_xyzw[1]), float(q_xyzw[2])],
        "t_meters": [float(t_X[0]), float(t_X[1]), float(t_X[2])],
        "phase1_rmsd_deg": phase1_rmsd_deg,
        "phase2_sphere_residual_mm": sphere_rmsd_m * 1000,
        "phase2_t_x_std_mm": [float(v) for v in t_X_std * 1000],
        "num_rotation_pairs": len(dR_m),
        "num_translation_points": int(len(v2_p)),
        "source_session": str(calib_dir),
        "solved_at_utc": datetime.datetime.now(
            tz=datetime.timezone.utc).isoformat(timespec="seconds"),
        "notes": (
            "R_quat_wxyz is the rotation from Vive tracker frame to Manus "
            "wrist frame. t_meters is the wrist origin expressed in the "
            "tracker frame (i.e. wrist = tracker_pos + R_world_tracker · t_X)."
        ),
    }
    return result


def _cli() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("calib_dir", type=Path, help="Calibration session dir")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output JSON path. Defaults to calibration/profiles/.")
    args = ap.parse_args()

    result = solve(args.calib_dir)

    if args.out is None:
        out_dir = Path(__file__).resolve().parent / "profiles"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%SZ")
        args.out = out_dir / f"tracker_to_wrist_{result['side']}_{stamp}.json"

    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")
    print(f"  rotation residual: {result['phase1_rmsd_deg']:.2f}°")
    print(f"  sphere residual:   {result['phase2_sphere_residual_mm']:.2f} mm")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
