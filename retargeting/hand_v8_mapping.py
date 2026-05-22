from __future__ import annotations

import numpy as np


FINGER_ROBOT = {
    "index": {
        "joints": ["i3", "i2", "i1", "i0"],
        "links": ["i_mc", "i_pp", "i_mp", "024200000001"],
        "tip_link": "024200000001",
    },
    "middle": {
        "joints": ["m3", "m2", "m1", "m0"],
        "links": ["m_mc", "m_pp", "m_mp", "024200000001_2"],
        "tip_link": "024200000001_2",
    },
    "ring": {
        "joints": ["r3", "r2", "r1", "r0"],
        "links": ["r_mc", "r_pp", "r_mp", "024200000001_3"],
        "tip_link": "024200000001_3",
    },
    "pinky": {
        "joints": ["p3", "p2", "p1", "p0"],
        "links": ["p_mc", "p_pp", "p_mp", "024200000001_4"],
        "tip_link": "024200000001_4",
    },
    "thumb": {
        "joints": ["t4", "t3", "t2", "t1", "t0"],
        "links": ["t_mc1", "t_mc2", "t_pp", "t_mp", "024200000001_5"],
        "tip_link": "024200000001_5",
    },
}


# MANUS raw skeleton often includes one more anatomical point than this robot
# exposes as useful IK keypoints. These indices select comparable points.
MANUS_POINT_INDICES = {
    "thumb": [0, 1, 2, 3],
    "index": [0, 1, 3, 4],
    "middle": [0, 1, 3, 4],
    "ring": [0, 1, 3, 4],
    "pinky": [0, 1, 3, 4],
}


def select_manus_points(points: dict[str, list[np.ndarray]]) -> dict[str, list[np.ndarray]]:
    out: dict[str, list[np.ndarray]] = {}
    for finger, idxs in MANUS_POINT_INDICES.items():
        pts = points.get(finger)
        if not pts:
            continue
        selected = []
        for i in idxs:
            if i < len(pts):
                selected.append(pts[i])
        if len(selected) >= 2:
            out[finger] = selected
    return out


def robot_keypoints(kin, q: np.ndarray) -> dict[str, list[np.ndarray]]:
    poses = kin.forward(q)
    out: dict[str, list[np.ndarray]] = {}
    for finger, spec in FINGER_ROBOT.items():
        pts = []
        for link in spec["links"][:-1]:
            pts.append(kin.point(poses, link))
        tip_link = spec["tip_link"]
        pts.append(kin.point(poses, tip_link, kin.link_visual_point(tip_link)))
        out[finger] = pts
    return out


def robot_joint_regularization_weights(joint_names: list[str]) -> np.ndarray:
    weights = np.ones(len(joint_names), dtype=float)
    for i, name in enumerate(joint_names):
        if name.endswith("3") or name in {"t4", "t3"}:
            weights[i] = 0.5
        if name.endswith("0"):
            weights[i] = 0.7
    return weights
