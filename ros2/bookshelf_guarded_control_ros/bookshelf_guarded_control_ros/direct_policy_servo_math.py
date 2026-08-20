"""Pure conversion helpers for bounded MoveIt Servo commands."""

from __future__ import annotations

import math

import numpy as np

from .policy_tool_control_math import (
    invert_transform,
    matrix_to_quaternion_xyzw,
    validated_transform,
)


def eef_target_from_tcp_target(target_base_tcp, transform_eef_tcp) -> np.ndarray:
    """Convert a calibrated TCP target into the link_eef target Servo controls."""

    target_base_tcp = validated_transform(target_base_tcp)
    transform_eef_tcp = validated_transform(transform_eef_tcp)
    return target_base_tcp @ invert_transform(transform_eef_tcp)


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
    """Return a base-frame velocity command that approaches a fixed target."""

    current = validated_transform(current)
    target = validated_transform(target)
    duration_s = float(duration_s)
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("duration_s must be finite and positive")
    translation_tolerance_m = float(translation_tolerance_m)
    rotation_tolerance_rad = float(rotation_tolerance_rad)
    if not math.isfinite(translation_tolerance_m) or translation_tolerance_m < 0.0:
        raise ValueError("translation_tolerance_m must be finite and non-negative")
    if not math.isfinite(rotation_tolerance_rad) or rotation_tolerance_rad < 0.0:
        raise ValueError("rotation_tolerance_rad must be finite and non-negative")

    translation_error = target[:3, 3] - current[:3, 3]
    rotation_error = matrix_to_axis_angle_vector(
        target[:3, :3] @ current[:3, :3].T
    )
    if float(np.linalg.norm(translation_error)) <= translation_tolerance_m:
        translation_error = np.zeros(3, dtype=np.float64)
    if float(np.linalg.norm(rotation_error)) <= rotation_tolerance_rad:
        rotation_error = np.zeros(3, dtype=np.float64)
    if not np.any(translation_error) and not np.any(rotation_error):
        return np.zeros(6, dtype=np.float64)

    linear = _bounded_vector(
        translation_error / duration_s,
        maximum_linear_speed_m_s,
        "maximum_linear_speed_m_s",
    )
    angular = _bounded_vector(
        rotation_error / duration_s,
        maximum_angular_speed_rad_s,
        "maximum_angular_speed_rad_s",
    )
    return np.concatenate((linear, angular))


def _bounded_vector(vector, maximum_norm: float, label: str) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    maximum_norm = float(maximum_norm)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("velocity vector must contain three finite values")
    if not math.isfinite(maximum_norm) or maximum_norm <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    norm = float(np.linalg.norm(vector))
    if norm <= maximum_norm:
        return vector
    return vector * (maximum_norm / norm)


def matrix_to_axis_angle_vector(matrix) -> np.ndarray:
    """Return the principal axis-angle rotation vector for a rotation matrix."""

    quaternion = matrix_to_quaternion_xyzw(matrix)
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    vector_norm = float(np.linalg.norm(quaternion[:3]))
    if vector_norm < 1.0e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(vector_norm, float(quaternion[3]))
    return quaternion[:3] * (angle / vector_norm)
