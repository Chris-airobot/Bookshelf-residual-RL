"""Pure transform helpers for the software-only policy Servo rehearsal."""

from __future__ import annotations

import math

import numpy as np

from .policy_tool_control_math import validated_transform


def integrate_base_frame_twist(transform, twist, duration_s: float) -> np.ndarray:
    """Integrate one base-frame spatial velocity step."""

    transform = validated_transform(transform)
    twist = np.asarray(twist, dtype=np.float64)
    duration_s = float(duration_s)
    if twist.shape != (6,) or not np.all(np.isfinite(twist)):
        raise ValueError("twist must contain six finite values")
    if not math.isfinite(duration_s) or duration_s < 0.0:
        raise ValueError("duration_s must be finite and non-negative")

    result = np.array(transform, copy=True)
    result[:3, 3] += twist[:3] * duration_s
    result[:3, :3] = _axis_angle_matrix(twist[3:] * duration_s) @ result[:3, :3]
    return result


def initial_eef_from_slot_book(
    transform_base_slot,
    transform_slot_book,
    transform_eef_book,
) -> np.ndarray:
    """Place the simulated EEF so its rigid book has the requested slot pose."""

    transform_base_slot = validated_transform(transform_base_slot)
    transform_slot_book = validated_transform(transform_slot_book)
    transform_eef_book = validated_transform(transform_eef_book)
    transform_book_eef = np.eye(4, dtype=np.float64)
    transform_book_eef[:3, :3] = transform_eef_book[:3, :3].T
    transform_book_eef[:3, 3] = -(
        transform_book_eef[:3, :3] @ transform_eef_book[:3, 3]
    )
    return transform_base_slot @ transform_slot_book @ transform_book_eef


def _axis_angle_matrix(rotation_vector) -> np.ndarray:
    rotation_vector = np.asarray(rotation_vector, dtype=np.float64)
    angle = float(np.linalg.norm(rotation_vector))
    if angle < 1.0e-12:
        return np.eye(3, dtype=np.float64)
    axis = rotation_vector / angle
    x, y, z = axis
    skew = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + math.sin(angle) * skew + (
        1.0 - math.cos(angle)
    ) * (skew @ skew)
