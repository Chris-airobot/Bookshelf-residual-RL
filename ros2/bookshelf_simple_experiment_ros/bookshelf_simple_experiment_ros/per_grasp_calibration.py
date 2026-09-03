"""Robust episode-fixed EEF-to-book calibration for the real experiment."""

from __future__ import annotations

import math

import numpy as np


def select_eef_book_transform(per_grasp, fixed):
    """Select the frozen transform, or the explicit reviewed fallback."""
    return np.asarray(fixed if per_grasp is None else per_grasp, dtype=np.float64)


def semantic_held_gripper_observation(measured_value, held_value):
    """Map physical aperture to the simulation held semantic for INSERT only."""
    measured = float(measured_value)
    held = float(held_value)
    if not math.isfinite(measured):
        raise ValueError("measured gripper aperture must be finite")
    if not math.isfinite(held) or not 0.0 <= held <= 1.0:
        raise ValueError("held gripper semantic must lie in [0, 1]")
    return held


def rotation_angle_rad(rotation):
    value = (float(np.trace(rotation)) - 1.0) * 0.5
    return math.acos(float(np.clip(value, -1.0, 1.0)))


def robust_average_transforms(
    transforms, translation_outlier_m=0.005, orientation_outlier_rad=math.radians(5.0)
):
    """Return a robust rigid-transform mean and calibration diagnostics."""

    values = np.asarray(transforms, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (4, 4) or len(values) == 0:
        raise ValueError("calibration requires one or more 4x4 transforms")
    if not np.all(np.isfinite(values)):
        raise ValueError("calibration transforms must be finite")

    center_t = np.median(values[:, :3, 3], axis=0)
    rotation_sum = np.sum(values[:, :3, :3], axis=0)
    u, _, vt = np.linalg.svd(rotation_sum)
    center_r = u @ vt
    if np.linalg.det(center_r) < 0.0:
        u[:, -1] *= -1.0
        center_r = u @ vt

    translation_error = np.linalg.norm(values[:, :3, 3] - center_t, axis=1)
    orientation_error = np.asarray([
        rotation_angle_rad(center_r.T @ value[:3, :3]) for value in values
    ])
    keep = ((translation_error <= float(translation_outlier_m))
            & (orientation_error <= float(orientation_outlier_rad)))
    accepted = values[keep]
    if len(accepted) == 0:
        raise ValueError("all per-grasp calibration samples were rejected")

    mean_t = np.mean(accepted[:, :3, 3], axis=0)
    u, _, vt = np.linalg.svd(np.sum(accepted[:, :3, :3], axis=0))
    mean_r = u @ vt
    if np.linalg.det(mean_r) < 0.0:
        u[:, -1] *= -1.0
        mean_r = u @ vt
    translation_residual = accepted[:, :3, 3] - mean_t
    orientation_residual = np.asarray([
        rotation_angle_rad(mean_r.T @ value[:3, :3]) for value in accepted
    ])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = mean_r
    transform[:3, 3] = mean_t
    return transform, {
        "sample_count": int(len(values)),
        "accepted_count": int(len(accepted)),
        "rejected_count": int(len(values) - len(accepted)),
        "translation_std_m": np.std(translation_residual, axis=0).tolist(),
        "translation_rms_m": float(np.sqrt(np.mean(np.sum(translation_residual**2, axis=1)))),
        "orientation_std_rad": float(np.std(orientation_residual)),
        "orientation_rms_rad": float(np.sqrt(np.mean(orientation_residual**2))),
        "orientation_max_rad": float(np.max(orientation_residual)),
    }
