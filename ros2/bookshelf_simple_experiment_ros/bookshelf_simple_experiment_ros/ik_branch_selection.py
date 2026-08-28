"""Reusable scoring utilities for singularity-aware xArm7 IK branch selection."""

from __future__ import annotations

import math
from pathlib import Path
import subprocess

from ament_index_python import get_package_share_directory
import numpy as np
import pinocchio as pin


def wrapped_joint_delta(first, second):
    """Return shortest signed revolute-joint deltas in [-pi, pi)."""
    difference = np.asarray(first, dtype=np.float64) - np.asarray(
        second, dtype=np.float64
    )
    return (difference + math.pi) % (2.0 * math.pi) - math.pi


def is_duplicate(candidate, existing, tolerance_rad):
    return any(
        float(np.max(np.abs(wrapped_joint_delta(candidate, previous))))
        < float(tolerance_rad)
        for previous in existing
    )


def joint_limit_margin(joints, lower, upper):
    joints = np.asarray(joints, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    return float(np.min(np.minimum(joints - lower, upper - joints)))


def trajectory_joint_path_length(trajectory, expected_joint_names):
    names = list(trajectory.joint_trajectory.joint_names)
    indices = [names.index(name) for name in expected_joint_names]
    points = trajectory.joint_trajectory.points
    if len(points) < 2:
        return math.inf
    length = 0.0
    previous = np.asarray([points[0].positions[index] for index in indices])
    for point in points[1:]:
        current = np.asarray([point.positions[index] for index in indices])
        length += float(np.linalg.norm(wrapped_joint_delta(current, previous)))
        previous = current
    return length


def select_candidate(candidates, similar_condition_band):
    """Select transition-efficient candidate among similarly nonsingular branches."""
    planned = [candidate for candidate in candidates if candidate.get("plan") is not None]
    if not planned:
        return None
    best_condition = min(float(candidate["max_condition"]) for candidate in planned)
    contenders = [
        candidate
        for candidate in planned
        if float(candidate["max_condition"])
        <= best_condition + float(similar_condition_band)
    ]
    return min(
        contenders,
        key=lambda candidate: (
            float(candidate["transition_cost"]),
            float(candidate["max_condition"]),
            int(candidate["candidate_id"]),
        ),
    )


def diverse_seeds(current, lower, upper, count, random_seed):
    """Return the current branch first, followed by deterministic broad seeds."""
    current = np.asarray(current, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    requested = max(int(count), 1)
    rng = np.random.default_rng(int(random_seed))
    return [current.copy()] + [
        rng.uniform(lower, upper) for _ in range(requested - 1)
    ]


class XArm7Kinematics:
    """Pinocchio xArm7 model used only for offline prediction during planning."""

    def __init__(self, joint_names, tcp_frame):
        xacro = (
            Path(get_package_share_directory("xarm_description"))
            / "urdf"
            / "xarm_device.urdf.xacro"
        )
        xml = subprocess.run(
            [
                "xacro",
                str(xacro),
                "dof:=7",
                "robot_type:=xarm",
                "limited:=false",
                "add_gripper:=true",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        self.model = pin.buildModelFromXML(xml)
        self.data = self.model.createData()
        self.tcp_frame_id = self.model.getFrameId(str(tcp_frame))
        if self.tcp_frame_id >= len(self.model.frames):
            raise ValueError(f"Pinocchio model has no frame {tcp_frame}")
        self.joint_names = [str(name) for name in joint_names]
        self.q_indices = []
        self.v_indices = []
        for name in self.joint_names:
            joint_id = self.model.getJointId(name)
            if joint_id == 0:
                raise ValueError(f"Pinocchio model has no joint {name}")
            joint = self.model.joints[joint_id]
            if joint.nq != 1 or joint.nv != 1:
                raise ValueError(f"expected one-DoF revolute joint {name}")
            self.q_indices.append(joint.idx_q)
            self.v_indices.append(joint.idx_v)
        self.lower = np.asarray(
            [self.model.lowerPositionLimit[index] for index in self.q_indices],
            dtype=np.float64,
        )
        self.upper = np.asarray(
            [self.model.upperPositionLimit[index] for index in self.q_indices],
            dtype=np.float64,
        )

    def condition_number(self, joints):
        q = pin.neutral(self.model)
        q[self.q_indices] = np.asarray(joints, dtype=np.float64)
        jacobian = pin.computeFrameJacobian(
            self.model,
            self.data,
            q,
            self.tcp_frame_id,
            pin.ReferenceFrame.LOCAL,
        )[:, self.v_indices]
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        smallest = float(singular_values[-1])
        if smallest <= np.finfo(np.float64).eps:
            return math.inf
        return float(singular_values[0] / smallest)

    def joint_limit_margin(self, joints):
        return joint_limit_margin(joints, self.lower, self.upper)
