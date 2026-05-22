from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def _vec(text: str | None, default=(0.0, 0.0, 0.0)) -> np.ndarray:
    if not text:
        return np.array(default, dtype=float)
    return np.array([float(x) for x in text.split()], dtype=float)


def _origin_matrix(elem: ET.Element | None) -> np.ndarray:
    t = np.eye(4)
    if elem is None:
        return t
    xyz = _vec(elem.attrib.get("xyz"))
    rpy = _vec(elem.attrib.get("rpy"))
    t[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    t[:3, 3] = xyz
    return t


def _axis_angle(axis: np.ndarray, q: float) -> np.ndarray:
    t = np.eye(4)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return t
    t[:3, :3] = Rotation.from_rotvec(axis / n * q).as_matrix()
    return t


@dataclass
class Joint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


class URDFKinematics:
    def __init__(self, urdf_path: str | Path):
        self.urdf_path = Path(urdf_path)
        self.root = ET.parse(self.urdf_path).getroot()
        self.joints: list[Joint] = []
        self.children: dict[str, list[Joint]] = {}
        self.parent_links: set[str] = set()
        self.child_links: set[str] = set()
        self.link_visual_points = self._parse_visual_points()
        self._parse_joints()
        roots = sorted(self.parent_links - self.child_links)
        self.root_link = roots[0] if roots else "palm"
        self.actuated_joints = [j.name for j in self.joints if j.joint_type != "fixed"]
        self.lower = np.array([self.joint_by_name(n).lower for n in self.actuated_joints])
        self.upper = np.array([self.joint_by_name(n).upper for n in self.actuated_joints])

    def _parse_joints(self) -> None:
        for elem in self.root.findall("joint"):
            name = elem.attrib["name"]
            joint_type = elem.attrib.get("type", "fixed")
            parent = elem.find("parent").attrib["link"]
            child = elem.find("child").attrib["link"]
            axis_elem = elem.find("axis")
            limit_elem = elem.find("limit")
            lower, upper = -math.inf, math.inf
            if limit_elem is not None:
                lower = float(limit_elem.attrib.get("lower", lower))
                upper = float(limit_elem.attrib.get("upper", upper))
            if joint_type == "fixed":
                lower = upper = 0.0
            joint = Joint(
                name=name,
                joint_type=joint_type,
                parent=parent,
                child=child,
                origin=_origin_matrix(elem.find("origin")),
                axis=_vec(axis_elem.attrib.get("xyz") if axis_elem is not None else None, (0, 0, 1)),
                lower=lower,
                upper=upper,
            )
            self.joints.append(joint)
            self.children.setdefault(parent, []).append(joint)
            self.parent_links.add(parent)
            self.child_links.add(child)

    def _parse_visual_points(self) -> dict[str, np.ndarray]:
        points: dict[str, np.ndarray] = {}
        for link in self.root.findall("link"):
            origins = []
            for visual in link.findall("visual"):
                origin = visual.find("origin")
                if origin is not None:
                    origins.append(_vec(origin.attrib.get("xyz")))
            if origins:
                points[link.attrib["name"]] = np.mean(np.vstack(origins), axis=0)
        return points

    def joint_by_name(self, name: str) -> Joint:
        for joint in self.joints:
            if joint.name == name:
                return joint
        raise KeyError(name)

    def forward(self, q: np.ndarray | dict[str, float]) -> dict[str, np.ndarray]:
        if isinstance(q, dict):
            q_map = q
        else:
            q_map = {name: float(q[i]) for i, name in enumerate(self.actuated_joints)}
        poses = {self.root_link: np.eye(4)}

        def visit(link: str) -> None:
            parent_t = poses[link]
            for joint in self.children.get(link, []):
                value = q_map.get(joint.name, 0.0)
                child_t = parent_t @ joint.origin
                if joint.joint_type != "fixed":
                    child_t = child_t @ _axis_angle(joint.axis, value)
                poses[joint.child] = child_t
                visit(joint.child)

        visit(self.root_link)
        return poses

    def point(self, poses: dict[str, np.ndarray], link: str, local: np.ndarray | None = None) -> np.ndarray:
        if local is None:
            local = np.zeros(3)
        p = np.ones(4)
        p[:3] = local
        return (poses[link] @ p)[:3]

    def link_visual_point(self, link: str) -> np.ndarray:
        return self.link_visual_points.get(link, np.zeros(3))
