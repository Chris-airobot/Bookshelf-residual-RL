"""Pure conversion helpers for direct xArm Cartesian servo commands."""

from __future__ import annotations

import math

import numpy as np

from .policy_tool_control_math import (
    make_transform,
    matrix_to_quaternion_xyzw,
    validated_transform,
)


def interpolate_transform(start, target, fraction: float) -> np.ndarray:
    """Interpolate translation and orientation along the shortest quaternion arc."""

    start = validated_transform(start)
    target = validated_transform(target)
    fraction = float(fraction)
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be finite and in [0, 1].")
    if fraction == 0.0:
        return np.array(start, copy=True)
    if fraction == 1.0:
        return np.array(target, copy=True)

    start_quaternion = matrix_to_quaternion_xyzw(start[:3, :3])
    target_quaternion = matrix_to_quaternion_xyzw(target[:3, :3])
    if float(np.dot(start_quaternion, target_quaternion)) < 0.0:
        target_quaternion = -target_quaternion

    dot = float(np.clip(np.dot(start_quaternion, target_quaternion), -1.0, 1.0))
    if dot > 0.9995:
        quaternion = start_quaternion + fraction * (
            target_quaternion - start_quaternion
        )
        quaternion /= np.linalg.norm(quaternion)
    else:
        angle = math.acos(dot)
        sine_angle = math.sin(angle)
        quaternion = (
            math.sin((1.0 - fraction) * angle) / sine_angle * start_quaternion
            + math.sin(fraction * angle) / sine_angle * target_quaternion
        )

    translation = (
        start[:3, 3] + fraction * (target[:3, 3] - start[:3, 3])
    )
    return make_transform(translation, quaternion)


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


def transform_to_xarm_axis_angle_pose(transform) -> list[float]:
    """Convert a base-frame TCP transform to xArm mm plus axis-angle format."""

    transform = validated_transform(transform)
    rotation_vector = matrix_to_axis_angle_vector(transform[:3, :3])
    return [
        float(transform[0, 3] * 1000.0),
        float(transform[1, 3] * 1000.0),
        float(transform[2, 3] * 1000.0),
        float(rotation_vector[0]),
        float(rotation_vector[1]),
        float(rotation_vector[2]),
    ]
