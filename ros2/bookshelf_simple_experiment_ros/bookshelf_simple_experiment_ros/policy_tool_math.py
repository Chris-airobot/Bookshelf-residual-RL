"""Verified current-relative policy-tool to xArm TCP command math.

The target convention is copied from guarded_control's pure
policy_tool_control_math and direct_policy_servo_math modules.
"""

from dataclasses import dataclass
import hashlib
import math
import threading

import numpy as np


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


class OneShotExecutionGuard:
    """Allow exactly one policy-command execution per node process."""

    def __init__(self):
        self._lock = threading.Lock()
        self._consumed = False

    def try_consume(self) -> bool:
        with self._lock:
            if self._consumed:
                return False
            self._consumed = True
            return True


def _vector(value, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {size} finite values")
    return array


def validated_transform(transform) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("transform must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-8):
        raise ValueError("transform has an invalid homogeneous bottom row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6):
        raise ValueError("transform rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-6):
        raise ValueError("transform rotation determinant is not +1")
    return transform


def invert_transform(transform) -> np.ndarray:
    transform = validated_transform(transform)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -(result[:3, :3] @ transform[:3, 3])
    return result


def quaternion_xyzw_to_matrix(quaternion) -> np.ndarray:
    quaternion = _vector(quaternion, 4, "quaternion")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1.0e-12:
        raise ValueError("quaternion must be non-zero")
    x, y, z, w = quaternion / norm
    return np.array(
        [
            [1.0 - 2.0 * (y*y + z*z), 2.0 * (x*y - z*w), 2.0 * (x*z + y*w)],
            [2.0 * (x*y + z*w), 1.0 - 2.0 * (x*x + z*z), 2.0 * (y*z - x*w)],
            [2.0 * (x*z - y*w), 2.0 * (y*z + x*w), 1.0 - 2.0 * (x*x + y*y)],
        ],
        dtype=np.float64,
    )


def make_transform(translation, quaternion_xyzw=None) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = _vector(translation, 3, "translation")
    if quaternion_xyzw is not None:
        transform[:3, :3] = quaternion_xyzw_to_matrix(quaternion_xyzw)
    return transform


def euler_xyz_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr],
        ],
        dtype=np.float64,
    )


def matrix_to_euler_xyz(matrix) -> tuple[float, float, float]:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("Rotation matrix must have shape (3, 3).")
    pitch = math.asin(float(np.clip(-matrix[2, 0], -1.0, 1.0)))
    if abs(math.cos(pitch)) > 1.0e-8:
        roll = math.atan2(float(matrix[2, 1]), float(matrix[2, 2]))
        yaw = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
    else:
        roll = 0.0
        yaw = math.atan2(float(-matrix[0, 1]), float(matrix[1, 1]))
    return roll, pitch, yaw


def _wrap_to_pi(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def rotation_angle_rad(matrix) -> float:
    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    if cosine >= 1.0 - 1.0e-12:
        return 0.0
    return math.acos(cosine)


def _target_digest(transform, delta, slot) -> str:
    digest = hashlib.sha256()
    for value in (transform, delta, slot):
        array = np.asarray(value, dtype=np.float64)
        digest.update(array.shape.__repr__().encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def compute_policy_tool_target(
    transform_base_slot,
    transform_base_tcp,
    transform_tcp_policy_tool,
    motion_delta,
    *,
    command_scale=0.10,
) -> PolicyToolTarget:
    """Apply the trained delta in slot coordinates and solve the TCP target."""

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
    return PolicyToolTarget(
        scaled_delta=scaled_delta,
        transform_base_policy_tool_current=transform_base_policy_tool_current,
        transform_slot_policy_tool_current=transform_slot_policy_tool_current,
        transform_slot_policy_tool_target=transform_slot_policy_tool_target,
        transform_base_policy_tool_target=transform_base_policy_tool_target,
        transform_base_tcp_target=transform_base_tcp_target,
        tcp_translation_step_m=float(np.linalg.norm(tcp_step[:3, 3])),
        tcp_rotation_step_rad=rotation_angle_rad(tcp_step[:3, :3]),
        target_id=_target_digest(transform_base_tcp_target, scaled_delta, transform_base_slot),
    )


def eef_target_from_tcp_target(target_base_tcp, transform_eef_tcp) -> np.ndarray:
    return validated_transform(target_base_tcp) @ invert_transform(transform_eef_tcp)


def matrix_to_quaternion_xyzw(matrix) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array([
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
            0.25 * scale,
        ])
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 0.0)) * 2.0
            quaternion = np.array([0.25*scale, (matrix[0,1]+matrix[1,0])/scale, (matrix[0,2]+matrix[2,0])/scale, (matrix[2,1]-matrix[1,2])/scale])
        elif index == 1:
            scale = math.sqrt(max(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2], 0.0)) * 2.0
            quaternion = np.array([(matrix[0,1]+matrix[1,0])/scale, 0.25*scale, (matrix[1,2]+matrix[2,1])/scale, (matrix[0,2]-matrix[2,0])/scale])
        else:
            scale = math.sqrt(max(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1], 0.0)) * 2.0
            quaternion = np.array([(matrix[0,2]+matrix[2,0])/scale, (matrix[1,2]+matrix[2,1])/scale, 0.25*scale, (matrix[1,0]-matrix[0,1])/scale])
    return quaternion / np.linalg.norm(quaternion)


def matrix_to_axis_angle_vector(matrix) -> np.ndarray:
    quaternion = matrix_to_quaternion_xyzw(matrix)
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    vector_norm = float(np.linalg.norm(quaternion[:3]))
    if vector_norm < 1.0e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(vector_norm, float(quaternion[3]))
    return quaternion[:3] * (angle / vector_norm)


def _bounded_vector(vector, maximum_norm: float, label: str) -> np.ndarray:
    vector = _vector(vector, 3, "velocity vector")
    maximum_norm = float(maximum_norm)
    if not math.isfinite(maximum_norm) or maximum_norm <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    norm = float(np.linalg.norm(vector))
    return vector if norm <= maximum_norm else vector * (maximum_norm / norm)


def bounded_error_twist(
    current,
    target,
    *,
    duration_s: float,
    maximum_linear_speed_m_s: float,
    maximum_angular_speed_rad_s: float,
    translation_tolerance_m: float,
    rotation_tolerance_rad: float,
) -> np.ndarray:
    current = validated_transform(current)
    target = validated_transform(target)
    duration_s = float(duration_s)
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("duration_s must be finite and positive")
    translation_error = target[:3, 3] - current[:3, 3]
    rotation_error = matrix_to_axis_angle_vector(target[:3, :3] @ current[:3, :3].T)
    if float(np.linalg.norm(translation_error)) <= float(translation_tolerance_m):
        translation_error = np.zeros(3)
    if float(np.linalg.norm(rotation_error)) <= float(rotation_tolerance_rad):
        rotation_error = np.zeros(3)
    if not np.any(translation_error) and not np.any(rotation_error):
        return np.zeros(6, dtype=np.float64)
    return np.concatenate((
        _bounded_vector(translation_error / duration_s, maximum_linear_speed_m_s, "maximum_linear_speed_m_s"),
        _bounded_vector(rotation_error / duration_s, maximum_angular_speed_rad_s, "maximum_angular_speed_rad_s"),
    ))
