"""Pure geometry and statistics for marker-to-book grasp calibration."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .policy_observation_math import (
    invert_transform,
    make_transform,
    matrix_to_quaternion_xyzw,
    quaternion_xyzw_to_matrix,
)


# The marker is on the left cover. Its printed right direction follows book
# depth, its printed top follows book height, and its outward normal is -book Y.
DEFAULT_BOOK_FROM_MARKER_ROTATION = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class CalibrationSample:
    frame_index: int
    stamp_ns: int
    reprojection_error_px: float
    marker_depth_m: float
    depth_error_m: float
    transform_camera_marker: np.ndarray
    transform_eef_book: np.ndarray


def validate_rotation(rotation) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("Rotation must be a finite 3x3 matrix.")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-7):
        raise ValueError("Rotation is not orthonormal.")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-7):
        raise ValueError("Rotation must have determinant +1.")
    return rotation


def make_book_marker_transform(
    marker_center_in_book,
    rotation_book_marker=DEFAULT_BOOK_FROM_MARKER_ROTATION,
) -> np.ndarray:
    """Return ``T_book_marker`` from the measured rigid marker mounting."""

    marker_center = np.asarray(marker_center_in_book, dtype=np.float64)
    if marker_center.shape != (3,) or not np.all(np.isfinite(marker_center)):
        raise ValueError("Marker centre must be a finite 3-vector.")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = validate_rotation(rotation_book_marker)
    transform[:3, 3] = marker_center
    return transform


def compose_eef_book_transform(
    transform_eef_camera,
    transform_camera_marker,
    transform_book_marker,
) -> np.ndarray:
    """Compose the recorded hand-eye pose, marker pose, and rigid mounting."""

    return (
        _validated_transform(transform_eef_camera)
        @ _validated_transform(transform_camera_marker)
        @ invert_transform(_validated_transform(transform_book_marker))
    )


def average_quaternions_xyzw(quaternions) -> np.ndarray:
    """Return the Markley mean of unit quaternions in XYZW order."""

    values = np.asarray(quaternions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4 or values.shape[0] == 0:
        raise ValueError("Expected one or more quaternions with shape (N, 4).")
    norms = np.linalg.norm(values, axis=1)
    if np.any(~np.isfinite(values)) or np.any(norms < 1.0e-12):
        raise ValueError("Quaternions must be finite and nonzero.")
    values = values / norms[:, None]
    reference = values[0]
    values[np.sum(values * reference, axis=1) < 0.0] *= -1.0
    accumulator = values.T @ values
    eigenvalues, eigenvectors = np.linalg.eigh(accumulator)
    result = eigenvectors[:, int(np.argmax(eigenvalues))]
    if float(np.dot(result, reference)) < 0.0:
        result *= -1.0
    return result / np.linalg.norm(result)


def quaternion_angle_deg(first, second) -> float:
    first = np.asarray(first, dtype=np.float64).copy()
    second = np.asarray(second, dtype=np.float64).copy()
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    cosine = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
    return math.degrees(2.0 * math.acos(cosine))


def quaternion_medoid_xyzw(quaternions) -> np.ndarray:
    """Return the observed quaternion nearest to the orientation consensus."""

    values = np.asarray(quaternions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4 or values.shape[0] == 0:
        raise ValueError("Expected one or more quaternions with shape (N, 4).")
    distances = np.zeros((len(values), len(values)), dtype=np.float64)
    for first_index in range(len(values)):
        for second_index in range(first_index + 1, len(values)):
            distance = quaternion_angle_deg(
                values[first_index], values[second_index]
            )
            distances[first_index, second_index] = distance
            distances[second_index, first_index] = distance
    return values[int(np.argmin(np.sum(distances, axis=1)))].copy()


class MarkerBookCalibrationAccumulator:
    """Collect static-grasp transforms and reject inconsistent pose estimates."""

    def __init__(
        self,
        *,
        maximum_translation_deviation_m=0.010,
        maximum_rotation_deviation_deg=5.0,
    ):
        self.maximum_translation_deviation_m = float(maximum_translation_deviation_m)
        self.maximum_rotation_deviation_deg = float(maximum_rotation_deviation_deg)
        self.samples: list[CalibrationSample] = []

    def add(self, sample: CalibrationSample):
        _validated_transform(sample.transform_camera_marker)
        _validated_transform(sample.transform_eef_book)
        self.samples.append(sample)

    def result(self) -> dict:
        if not self.samples:
            raise ValueError("No valid marker calibration samples were collected.")

        transforms = np.asarray(
            [sample.transform_eef_book for sample in self.samples], dtype=np.float64
        )
        translations = transforms[:, :3, 3]
        quaternions = np.asarray(
            [matrix_to_quaternion_xyzw(value[:3, :3]) for value in transforms]
        )

        preliminary_translation = np.median(translations, axis=0)
        # Use an observed medoid for the first gate. A direct mean can itself be
        # pulled outside the acceptance gate by one large planar-pose outlier.
        preliminary_quaternion = quaternion_medoid_xyzw(quaternions)
        translation_error = np.linalg.norm(
            translations - preliminary_translation[None, :], axis=1
        )
        rotation_error = np.asarray(
            [
                quaternion_angle_deg(value, preliminary_quaternion)
                for value in quaternions
            ]
        )

        translation_limit = min(
            self.maximum_translation_deviation_m,
            _robust_limit(translation_error, minimum=0.0010),
        )
        rotation_limit = min(
            self.maximum_rotation_deviation_deg,
            _robust_limit(rotation_error, minimum=0.50),
        )
        inliers = (translation_error <= translation_limit) & (
            rotation_error <= rotation_limit
        )
        if not np.any(inliers):
            raise ValueError("Robust filtering rejected every calibration sample.")

        inlier_translations = translations[inliers]
        inlier_quaternions = quaternions[inliers]
        mean_translation = np.mean(inlier_translations, axis=0)
        mean_quaternion = average_quaternions_xyzw(inlier_quaternions)
        mean_transform = make_transform(mean_translation, mean_quaternion)

        translation_residuals = np.linalg.norm(
            inlier_translations - mean_translation[None, :], axis=1
        )
        rotation_residuals = np.asarray(
            [quaternion_angle_deg(value, mean_quaternion) for value in inlier_quaternions]
        )
        reprojection = np.asarray(
            [sample.reprojection_error_px for sample in self.samples], dtype=np.float64
        )[inliers]
        depth_error = np.asarray(
            [sample.depth_error_m for sample in self.samples], dtype=np.float64
        )[inliers]
        finite_depth_error = depth_error[np.isfinite(depth_error)]

        return {
            "transform_eef_book": mean_transform,
            "translation_xyz_m": mean_translation,
            "quaternion_xyzw": mean_quaternion,
            "input_samples": len(self.samples),
            "inlier_samples": int(np.count_nonzero(inliers)),
            "inlier_fraction": float(np.mean(inliers)),
            "inlier_mask": inliers,
            "translation_filter_limit_m": float(translation_limit),
            "rotation_filter_limit_deg": float(rotation_limit),
            "translation_residual_m": _statistics(translation_residuals),
            "rotation_residual_deg": _statistics(rotation_residuals),
            "reprojection_error_px": _statistics(reprojection),
            "depth_error_m": (
                _statistics(finite_depth_error) if finite_depth_error.size else None
            ),
        }


def _robust_limit(values: np.ndarray, *, minimum: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return max(float(minimum), median + 4.0 * 1.4826 * mad)


def _statistics(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def _validated_transform(transform) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("Transform must be a finite 4x4 matrix.")
    validate_rotation(transform[:3, :3])
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-9):
        raise ValueError("Transform has an invalid homogeneous final row.")
    return transform
