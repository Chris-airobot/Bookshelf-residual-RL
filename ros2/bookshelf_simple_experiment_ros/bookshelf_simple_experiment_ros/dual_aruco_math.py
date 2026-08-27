"""Small, ROS-free geometry helpers for dual-ArUco book calibration."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml

from .geometry import invert_transform, make_transform, matrix_to_quaternion_xyzw


def marker_object_points(marker_size_m: float) -> np.ndarray:
    half = 0.5 * float(marker_size_m)
    if not math.isfinite(half) or half <= 0.0:
        raise ValueError("Marker black-square size must be positive and finite.")
    return np.asarray(
        [[-half, half, 0.0], [half, half, 0.0],
         [half, -half, 0.0], [-half, -half, 0.0]], dtype=np.float64)


def load_reference_book_transform(path) -> tuple[dict, np.ndarray]:
    """Load the reviewed legacy mount and return ``T_reference_book``."""
    path = Path(path).expanduser()
    with path.open("r", encoding="utf-8") as stream:
        mount = yaml.safe_load(stream)
    center = mount["marker_center_in_book_m"]
    rotation = np.asarray(mount["rotation_book_marker"], dtype=np.float64)
    transform_book_reference = np.eye(4, dtype=np.float64)
    transform_book_reference[:3, :3] = rotation
    transform_book_reference[:3, 3] = [center["x"], center["y"], center["z"]]
    return mount, invert_transform(transform_book_reference)


def load_secondary_book_transform(path) -> tuple[dict, np.ndarray]:
    path = Path(path).expanduser()
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    transform = data["transform_secondary_book"]
    return data, make_transform(
        transform["translation_xyz_m"], transform["quaternion_xyzw"])


def derive_secondary_book(transform_reference_secondary, transform_reference_book):
    """Return ``T_secondary_book = inv(T_reference_secondary) T_reference_book``."""
    return invert_transform(transform_reference_secondary) @ transform_reference_book


def quaternion_angle_deg(first, second) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first = first / np.linalg.norm(first)
    second = second / np.linalg.norm(second)
    cosine = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
    return math.degrees(2.0 * math.acos(cosine))


def _mean_quaternion(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    reference = values[0] / np.linalg.norm(values[0])
    values = values / np.linalg.norm(values, axis=1)[:, None]
    values[np.sum(values * reference, axis=1) < 0.0] *= -1.0
    eigenvalues, eigenvectors = np.linalg.eigh(values.T @ values)
    result = eigenvectors[:, int(np.argmax(eigenvalues))]
    if np.dot(result, reference) < 0.0:
        result *= -1.0
    return result / np.linalg.norm(result)


def _stats(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {key: float(function(values)) for key, function in (
        ("mean", np.mean), ("std", np.std), ("median", np.median), ("max", np.max))}


class RobustTransformAccumulator:
    """Median/medoid gate followed by a rigid-transform mean."""

    def __init__(self, maximum_translation_deviation_m=0.010,
                 maximum_rotation_deviation_deg=5.0):
        self.transforms = []
        self.maximum_translation_deviation_m = float(maximum_translation_deviation_m)
        self.maximum_rotation_deviation_deg = float(maximum_rotation_deviation_deg)

    def add(self, transform):
        value = np.asarray(transform, dtype=np.float64)
        if value.shape != (4, 4) or not np.all(np.isfinite(value)):
            raise ValueError("Transform must be finite and 4x4.")
        self.transforms.append(value.copy())

    def result(self) -> dict:
        if not self.transforms:
            raise ValueError("No simultaneous-marker samples were collected.")
        translations = np.asarray([value[:3, 3] for value in self.transforms])
        quaternions = np.asarray(
            [matrix_to_quaternion_xyzw(value[:3, :3]) for value in self.transforms])
        center_t = np.median(translations, axis=0)
        distances = np.zeros((len(quaternions), len(quaternions)))
        for i in range(len(quaternions)):
            for j in range(i + 1, len(quaternions)):
                distances[i, j] = distances[j, i] = quaternion_angle_deg(
                    quaternions[i], quaternions[j])
        center_q = quaternions[int(np.argmin(np.sum(distances, axis=1)))]
        translation_errors = np.linalg.norm(translations - center_t, axis=1)
        rotation_errors = np.asarray(
            [quaternion_angle_deg(value, center_q) for value in quaternions])
        inliers = (
            (translation_errors <= self.maximum_translation_deviation_m)
            & (rotation_errors <= self.maximum_rotation_deviation_deg))
        if not np.any(inliers):
            raise ValueError("Outlier rejection removed every sample.")
        mean_t = np.mean(translations[inliers], axis=0)
        mean_q = _mean_quaternion(quaternions[inliers])
        t_residual = np.linalg.norm(translations[inliers] - mean_t, axis=1)
        r_residual = np.asarray(
            [quaternion_angle_deg(value, mean_q) for value in quaternions[inliers]])
        return {
            "transform": make_transform(mean_t, mean_q),
            "input_samples": len(self.transforms),
            "accepted_samples": int(np.count_nonzero(inliers)),
            "translation_variation_m": _stats(t_residual),
            "rotation_variation_deg": _stats(r_residual),
        }
