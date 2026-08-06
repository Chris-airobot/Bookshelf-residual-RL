"""Pure geometry for transferring the simulator policy tool to xArm TCP."""

from __future__ import annotations

import numpy as np

from .marker_book_calibration import (
    average_quaternions_xyzw,
    quaternion_angle_deg,
)
from .policy_observation_math import (
    invert_transform,
    make_transform,
    matrix_to_quaternion_xyzw,
)


def representative_transform(transforms) -> tuple[np.ndarray, dict]:
    """Return a robust translation/orientation representative and dispersion."""

    values = np.asarray(transforms, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (4, 4) or values.shape[0] == 0:
        raise ValueError("Expected one or more transforms with shape (N, 4, 4).")
    if not np.all(np.isfinite(values)):
        raise ValueError("Transforms must be finite.")

    translations = values[:, :3, 3]
    quaternions = np.asarray(
        [matrix_to_quaternion_xyzw(value[:3, :3]) for value in values]
    )
    translation = np.median(translations, axis=0)
    quaternion = average_quaternions_xyzw(quaternions)
    representative = make_transform(translation, quaternion)

    translation_residual = np.linalg.norm(
        translations - translation[None, :], axis=1
    )
    rotation_residual = np.asarray(
        [quaternion_angle_deg(value, quaternion) for value in quaternions]
    )
    norms = np.linalg.norm(translations, axis=1)
    return representative, {
        "samples": int(values.shape[0]),
        "translation_axis_min_m": translations.min(axis=0).tolist(),
        "translation_axis_median_m": translation.tolist(),
        "translation_axis_max_m": translations.max(axis=0).tolist(),
        "translation_norm_m": _statistics(norms),
        "translation_residual_m": _statistics(translation_residual),
        "rotation_residual_deg": _statistics(rotation_residual),
    }


def derive_xarm_policy_tool_transform(
    transform_policy_book_policy_tool_sim,
    transform_eef_book_real,
    transform_eef_tcp_real,
) -> dict:
    """Derive a virtual xArm policy tool that recreates simulator semantics.

    All transforms use ``T_A_B`` notation: the pose of frame B expressed in A.
    """

    transform_policy_book_policy_tool_sim = _validated_transform(
        transform_policy_book_policy_tool_sim
    )
    transform_eef_book_real = _validated_transform(transform_eef_book_real)
    transform_eef_tcp_real = _validated_transform(transform_eef_tcp_real)

    transform_eef_policy_tool = (
        transform_eef_book_real @ transform_policy_book_policy_tool_sim
    )
    transform_tcp_policy_tool = (
        invert_transform(transform_eef_tcp_real) @ transform_eef_policy_tool
    )
    reconstructed = (
        invert_transform(transform_eef_book_real)
        @ transform_eef_tcp_real
        @ transform_tcp_policy_tool
    )
    error = invert_transform(transform_policy_book_policy_tool_sim) @ reconstructed
    translation_error = float(np.linalg.norm(error[:3, 3]))
    rotation_error = quaternion_angle_deg(
        matrix_to_quaternion_xyzw(error[:3, :3]),
        [0.0, 0.0, 0.0, 1.0],
    )
    return {
        "transform_eef_policy_tool": transform_eef_policy_tool,
        "transform_tcp_policy_tool": transform_tcp_policy_tool,
        "reconstructed_policy_book_policy_tool": reconstructed,
        "round_trip_translation_error_m": translation_error,
        "round_trip_rotation_error_deg": rotation_error,
    }


def _statistics(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95.0)),
        "max": float(np.max(values)),
    }


def _validated_transform(transform) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("Transform must be a finite 4x4 matrix.")
    return transform
