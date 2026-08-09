"""Pure geometry and fail-closed gates for physical policy-tool deployment."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import threading

import numpy as np


MOTION_LABELS = ("dx", "dy", "dz", "dyaw", "dpitch")


class OneShotExecutionGuard:
    """Atomically consume the process's single trajectory submission slot."""

    def __init__(self):
        self._lock = threading.Lock()
        self._consumed = False

    @property
    def consumed(self) -> bool:
        with self._lock:
            return self._consumed

    @property
    def execution_count(self) -> int:
        return int(self.consumed)

    def try_consume(self) -> bool:
        """Return true exactly once for the lifetime of this guard."""

        with self._lock:
            if self._consumed:
                return False
            self._consumed = True
            return True


@dataclass(frozen=True)
class TargetSafetyLimits:
    maximum_delta: tuple[float, float, float, float, float] = (
        0.008,
        0.003,
        0.007,
        math.radians(0.8),
        math.radians(0.6),
    )
    maximum_tcp_translation_step_m: float = 0.010
    maximum_tcp_rotation_step_rad: float = math.radians(1.5)
    workspace_min_xyz: tuple[float, float, float] = (0.20, -0.60, 0.05)
    workspace_max_xyz: tuple[float, float, float] = (1.00, 0.60, 1.00)


@dataclass(frozen=True)
class PolicyToolTarget:
    scaled_delta: np.ndarray
    transform_base_policy_tool_current: np.ndarray
    transform_slot_policy_tool_current: np.ndarray
    transform_slot_policy_tool_target: np.ndarray
    transform_base_policy_tool_target: np.ndarray
    transform_base_tcp_target: np.ndarray
    tcp_translation_step_m: float
    tcp_rotation_step_rad: float
    target_id: str


def make_transform(translation, quaternion_xyzw=None) -> np.ndarray:
    translation = _vector(translation, 3, "translation")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = translation
    if quaternion_xyzw is not None:
        transform[:3, :3] = quaternion_xyzw_to_matrix(quaternion_xyzw)
    return transform


def invert_transform(transform) -> np.ndarray:
    transform = validated_transform(transform)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -(result[:3, :3] @ transform[:3, 3])
    return result


def quaternion_xyzw_to_matrix(quaternion) -> np.ndarray:
    x, y, z, w = _normalised_quaternion(quaternion)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion_xyzw(matrix) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("Rotation matrix must have shape (3, 3).")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ],
            dtype=np.float64,
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 0.0)) * 2.0
            quaternion = np.array(
                [
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                ]
            )
        elif index == 1:
            scale = math.sqrt(max(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2], 0.0)) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                ]
            )
        else:
            scale = math.sqrt(max(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1], 0.0)) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ]
            )
    return _normalised_quaternion(quaternion)


def euler_xyz_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return Rz(yaw) Ry(pitch) Rx(roll), matching standard ROS RPY."""

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def matrix_to_euler_xyz(matrix) -> tuple[float, float, float]:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("Rotation matrix must have shape (3, 3).")
    sine_pitch = float(np.clip(-matrix[2, 0], -1.0, 1.0))
    pitch = math.asin(sine_pitch)
    cosine_pitch = math.cos(pitch)
    if abs(cosine_pitch) > 1.0e-8:
        roll = math.atan2(float(matrix[2, 1]), float(matrix[2, 2]))
        yaw = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
    else:
        roll = 0.0
        yaw = math.atan2(float(-matrix[0, 1]), float(matrix[1, 1]))
    return roll, pitch, yaw


def compute_policy_tool_target(
    transform_base_slot,
    transform_base_tcp,
    transform_tcp_policy_tool,
    motion_delta,
    *,
    command_scale=0.10,
) -> PolicyToolTarget:
    """Apply the trained delta in slot coordinates and solve the TCP target.

    The simulator applies translation directly in its world frame and adds
    pitch/yaw to the policy tool's world-frame RPY. The simulator slot is
    aligned with that world frame. For deployment, the same operation is
    performed in the measured slot frame before converting back to ``link_tcp``.
    """

    transform_base_slot = validated_transform(transform_base_slot)
    transform_base_tcp = validated_transform(transform_base_tcp)
    transform_tcp_policy_tool = validated_transform(transform_tcp_policy_tool)
    delta = _vector(motion_delta, 5, "motion_delta")
    command_scale = float(command_scale)
    if not math.isfinite(command_scale) or not 0.0 < command_scale <= 1.0:
        raise ValueError("command_scale must be finite and in (0, 1].")
    scaled_delta = delta * command_scale

    transform_base_policy_tool_current = transform_base_tcp @ transform_tcp_policy_tool
    transform_slot_policy_tool_current = (
        invert_transform(transform_base_slot) @ transform_base_policy_tool_current
    )
    transform_slot_policy_tool_target = np.array(
        transform_slot_policy_tool_current, copy=True
    )
    transform_slot_policy_tool_target[:3, 3] += scaled_delta[:3]

    roll, pitch, yaw = matrix_to_euler_xyz(
        transform_slot_policy_tool_current[:3, :3]
    )
    transform_slot_policy_tool_target[:3, :3] = euler_xyz_to_matrix(
        roll,
        _wrap_to_pi(pitch + float(scaled_delta[4])),
        _wrap_to_pi(yaw + float(scaled_delta[3])),
    )

    transform_base_policy_tool_target = (
        transform_base_slot @ transform_slot_policy_tool_target
    )
    transform_base_tcp_target = (
        transform_base_policy_tool_target @ invert_transform(transform_tcp_policy_tool)
    )
    tcp_step = invert_transform(transform_base_tcp) @ transform_base_tcp_target
    translation_step = float(np.linalg.norm(tcp_step[:3, 3]))
    rotation_step = rotation_angle_rad(tcp_step[:3, :3])
    target_id = target_digest(
        transform_base_tcp_target,
        scaled_delta,
        transform_base_slot,
    )
    return PolicyToolTarget(
        scaled_delta=scaled_delta,
        transform_base_policy_tool_current=transform_base_policy_tool_current,
        transform_slot_policy_tool_current=transform_slot_policy_tool_current,
        transform_slot_policy_tool_target=transform_slot_policy_tool_target,
        transform_base_policy_tool_target=transform_base_policy_tool_target,
        transform_base_tcp_target=transform_base_tcp_target,
        tcp_translation_step_m=translation_step,
        tcp_rotation_step_rad=rotation_step,
        target_id=target_id,
    )


def target_safety_error(
    target: PolicyToolTarget,
    motion_delta,
    limits: TargetSafetyLimits = TargetSafetyLimits(),
) -> str | None:
    delta = _vector(motion_delta, 5, "motion_delta")
    maximum_delta = _vector(limits.maximum_delta, 5, "maximum_delta")
    if np.any(maximum_delta <= 0.0):
        return "configured motion limits must be positive"
    exceeded = [
        label
        for label, value, maximum in zip(MOTION_LABELS, np.abs(delta), maximum_delta)
        if float(value) > float(maximum) + 1.0e-12
    ]
    if exceeded:
        return f"unscaled policy delta exceeds configured limits: {exceeded}"
    if target.tcp_translation_step_m > limits.maximum_tcp_translation_step_m:
        return (
            f"TCP translation step {target.tcp_translation_step_m:.6f} m exceeds "
            f"{limits.maximum_tcp_translation_step_m:.6f} m"
        )
    if target.tcp_rotation_step_rad > limits.maximum_tcp_rotation_step_rad:
        return (
            f"TCP rotation step {math.degrees(target.tcp_rotation_step_rad):.3f} deg "
            f"exceeds {math.degrees(limits.maximum_tcp_rotation_step_rad):.3f} deg"
        )
    workspace_min = _vector(limits.workspace_min_xyz, 3, "workspace_min_xyz")
    workspace_max = _vector(limits.workspace_max_xyz, 3, "workspace_max_xyz")
    if np.any(workspace_min >= workspace_max):
        return "workspace bounds are invalid"
    position = target.transform_base_tcp_target[:3, 3]
    if np.any(position < workspace_min) or np.any(position > workspace_max):
        return (
            "target TCP position is outside workspace bounds: "
            f"{position.tolist()}"
        )
    return None


def provenance_error(
    adapter_debug,
    policy_debug,
    *,
    expected_policy_tool_status: str,
    expected_slot_status: str,
    expected_book_status: str,
    expected_bundle_sha256: str,
    allow_unverified_policy_tool: bool,
) -> str | None:
    if not isinstance(adapter_debug, dict) or not adapter_debug.get("valid", False):
        return "adapter debug is missing or invalid"
    if not isinstance(policy_debug, dict) or not policy_debug.get("valid", False):
        return "policy debug is missing or invalid"
    actual_tool_status = str(adapter_debug.get("policy_tool_transform_status", ""))
    if actual_tool_status != expected_policy_tool_status:
        return (
            "policy-tool status mismatch: "
            f"expected {expected_policy_tool_status}, got {actual_tool_status}"
        )
    if (
        actual_tool_status.lower().startswith("derived_unverified_")
        and not allow_unverified_policy_tool
    ):
        return f"policy-tool transform is unverified: {actual_tool_status}"
    if str(adapter_debug.get("static_slot_transform_status", "")) != expected_slot_status:
        return "static slot transform status mismatch"
    if str(adapter_debug.get("eef_book_transform_status", "")) != expected_book_status:
        return "EEF-to-book transform status mismatch"
    if str(policy_debug.get("bundle_sha256", "")) != expected_bundle_sha256:
        return "policy bundle SHA-256 mismatch"
    if bool(policy_debug.get("release_requested_diagnostic", False)):
        return "policy requested release; physical release is disabled"
    if bool(policy_debug.get("release_executed", False)):
        return "upstream reported a release execution"
    return None


def execution_authorization_error(
    *,
    dry_run: bool,
    allow_execution: bool,
    planning_scene_complete: bool,
    approval_token: str,
    configured_token: str,
    plan_age_s,
    maximum_plan_age_s: float,
    plan_valid: bool,
    busy: bool,
    execution_consumed: bool,
) -> str | None:
    if dry_run:
        return "dry_run is true"
    if not allow_execution:
        return "allow_execution is false"
    if not planning_scene_complete:
        return "planning_scene_complete is false"
    configured_token = str(configured_token)
    if not configured_token or configured_token in ("CHANGE_ME", "DISABLED"):
        return "approval token is not explicitly configured"
    if str(approval_token) != configured_token:
        return "approval token does not match"
    if busy:
        return "an execution is already active"
    if execution_consumed:
        return "the one-execution-per-process allowance has already been consumed"
    if not plan_valid:
        return "no valid collision-checked plan is available"
    if plan_age_s is None or not math.isfinite(float(plan_age_s)):
        return "plan age is unavailable"
    if float(plan_age_s) > float(maximum_plan_age_s):
        return f"plan is stale ({float(plan_age_s):.3f} s)"
    return None


def maximum_named_joint_difference(
    current_names,
    current_positions,
    planned_names,
    planned_positions,
) -> float:
    current = dict(zip(current_names, current_positions))
    planned = dict(zip(planned_names, planned_positions))
    common = sorted(set(current).intersection(planned))
    if not common:
        raise ValueError("Current and planned joint states have no common names.")
    return max(abs(float(current[name]) - float(planned[name])) for name in common)


def rotation_angle_rad(rotation) -> float:
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("Rotation matrix must have shape (3, 3).")
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    if cosine >= 1.0 - 1.0e-12:
        return 0.0
    return math.acos(cosine)


def transform_to_dict(transform) -> dict:
    transform = validated_transform(transform)
    return {
        "translation_xyz_m": [float(value) for value in transform[:3, 3]],
        "quaternion_xyzw": [
            float(value) for value in matrix_to_quaternion_xyzw(transform[:3, :3])
        ],
    }


def target_digest(*arrays) -> str:
    hasher = hashlib.sha256()
    for value in arrays:
        array = np.asarray(value, dtype=np.float64)
        hasher.update(array.shape.__repr__().encode("ascii"))
        hasher.update(array.tobytes(order="C"))
    return hasher.hexdigest()


def validated_transform(transform) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("Transform must be a finite 4x4 matrix.")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-8):
        raise ValueError("Transform has an invalid homogeneous final row.")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6):
        raise ValueError("Transform rotation is not orthonormal.")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-6):
        raise ValueError("Transform rotation determinant is not +1.")
    return transform


def _normalised_quaternion(quaternion) -> np.ndarray:
    quaternion = _vector(quaternion, 4, "quaternion")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1.0e-12:
        raise ValueError("Cannot normalize a zero quaternion.")
    return quaternion / norm


def _vector(value, size: int, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    if value.shape != (size,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite vector with shape ({size},).")
    return value


def _wrap_to_pi(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def json_dumps(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
