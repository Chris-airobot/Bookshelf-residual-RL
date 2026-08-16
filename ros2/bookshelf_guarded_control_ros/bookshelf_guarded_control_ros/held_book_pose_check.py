"""Pure live-versus-configured held-book transform checks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import yaml

from .policy_tool_control_math import make_transform, matrix_to_quaternion_xyzw


@dataclass(frozen=True)
class TransformComparison:
    translation_error_m: float
    rotation_error_deg: float


def load_configured_transform(path) -> np.ndarray:
    """Load the frozen ``T_link_tcp_book`` from a scene-manager YAML."""

    scene_path = Path(path)
    try:
        document = yaml.safe_load(scene_path.read_text(encoding="utf-8"))
        parameters = document["bookshelf_scene_manager"]["ros__parameters"]
        translation = parameters["held_book_center_tcp_xyz"]
        quaternion = parameters["held_book_quaternion_tcp_xyzw"]
    except yaml.YAMLError as error:
        raise ValueError(f"invalid scene YAML: {error}") from error
    except (KeyError, TypeError) as error:
        raise ValueError(
            "scene YAML must define bookshelf_scene_manager.ros__parameters "
            "held_book_center_tcp_xyz and held_book_quaternion_tcp_xyzw"
        ) from error
    return make_transform(translation, quaternion)


def rotation_error_deg(left_rotation, right_rotation) -> float:
    """Return the shortest angular distance between two rotation matrices."""

    left = np.asarray(left_rotation, dtype=np.float64)
    right = np.asarray(right_rotation, dtype=np.float64)
    if left.shape != (3, 3) or right.shape != (3, 3):
        raise ValueError("rotations must be 3x3 matrices")
    cosine = 0.5 * (float(np.trace(left.T @ right)) - 1.0)
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def compare_transforms(configured_transform, live_transform) -> TransformComparison:
    """Compare two transforms expressed in the same parent and child frames."""

    configured = np.asarray(configured_transform, dtype=np.float64)
    live = np.asarray(live_transform, dtype=np.float64)
    if configured.shape != (4, 4) or live.shape != (4, 4):
        raise ValueError("transforms must be 4x4 matrices")
    return TransformComparison(
        translation_error_m=float(
            np.linalg.norm(configured[:3, 3] - live[:3, 3])
        ),
        rotation_error_deg=rotation_error_deg(
            configured[:3, :3], live[:3, :3]
        ),
    )


def mean_transform(transforms) -> np.ndarray:
    """Average a small cluster of rigid transforms without changing its frame."""

    values = [np.asarray(value, dtype=np.float64) for value in transforms]
    if not values or any(value.shape != (4, 4) for value in values):
        raise ValueError("at least one 4x4 transform is required")

    translation = np.mean([value[:3, 3] for value in values], axis=0)
    quaternions = np.asarray(
        [matrix_to_quaternion_xyzw(value[:3, :3]) for value in values],
        dtype=np.float64,
    )
    reference = quaternions[0]
    for index in range(1, len(quaternions)):
        if float(np.dot(reference, quaternions[index])) < 0.0:
            quaternions[index] *= -1.0
    quaternion = np.mean(quaternions, axis=0)
    norm = float(np.linalg.norm(quaternion))
    if norm < 1.0e-12:
        raise ValueError("mean quaternion is degenerate")
    return make_transform(translation, quaternion / norm)


def transform_spread(transforms, center_transform) -> TransformComparison:
    """Return maximum translation and rotation deviations from a center."""

    comparisons = [
        compare_transforms(center_transform, transform) for transform in transforms
    ]
    if not comparisons:
        raise ValueError("at least one transform is required")
    return TransformComparison(
        translation_error_m=max(value.translation_error_m for value in comparisons),
        rotation_error_deg=max(value.rotation_error_deg for value in comparisons),
    )
