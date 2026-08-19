"""Pure geometry for auditing real-book and simulator book-frame conventions."""

from __future__ import annotations

import math

import numpy as np

from .policy_observation_math import (
    make_transform,
    matrix_to_quaternion_xyzw,
)


# Candidate T_old_book_policy_book. The recorded calibration approximately maps
# old (+X, +Y, +Z) to EEF (+Y, +Z, +X), while the same-grasp hypothesis expects
# policy (+X, +Y, +Z) to map to EEF (+Z, -Y, +X). This signed axis permutation
# is a +90 degree rotation about the old book +Z axis.
DEFAULT_OLD_BOOK_POLICY_BOOK_QUATERNION_XYZW = (
    0.0,
    0.0,
    math.sqrt(0.5),
    math.sqrt(0.5),
)

# Expected T_eef_policy_book rotation under the explicitly stated hypothesis:
# xArm link_eef axes represent the same grasp semantics as the simulator hand,
# with policy book +X=depth/approach, +Y=thickness, and +Z=up.
DEFAULT_EXPECTED_POLICY_BOOK_IN_EEF_QUATERNION_XYZW = (
    math.sqrt(0.5),
    0.0,
    math.sqrt(0.5),
    0.0,
)


def book_axis_correction_transform(
    quaternion_xyzw=DEFAULT_OLD_BOOK_POLICY_BOOK_QUATERNION_XYZW,
) -> np.ndarray:
    """Return the candidate frame-only transform ``T_old_book_policy_book``."""

    return make_transform([0.0, 0.0, 0.0], quaternion_xyzw)


def expected_policy_book_rotation_in_eef(
    quaternion_xyzw=DEFAULT_EXPECTED_POLICY_BOOK_IN_EEF_QUATERNION_XYZW,
) -> np.ndarray:
    """Return the same-grasp hypothesis rotation ``R_eef_policy_book``."""

    return make_transform([0.0, 0.0, 0.0], quaternion_xyzw)[:3, :3]


def apply_book_axis_correction(
    transform_eef_old_book,
    transform_old_book_policy_book=None,
) -> np.ndarray:
    """Compose a candidate policy-book frame without changing its centre."""

    current = _validated_transform(transform_eef_old_book)
    correction = (
        book_axis_correction_transform()
        if transform_old_book_policy_book is None
        else _validated_transform(transform_old_book_policy_book)
    )
    if not np.allclose(correction[:3, 3], 0.0, atol=1.0e-12):
        raise ValueError("Book-axis correction must not translate the book centre.")
    return current @ correction


def book_frame_audit_report(
    transform_eef_old_book,
    *,
    transform_old_book_policy_book=None,
    expected_rotation_eef_policy_book=None,
) -> dict:
    """Compare the saved and candidate book frames with no automatic selection."""

    current = _validated_transform(transform_eef_old_book)
    correction = (
        book_axis_correction_transform()
        if transform_old_book_policy_book is None
        else _validated_transform(transform_old_book_policy_book)
    )
    candidate = apply_book_axis_correction(current, correction)
    expected = (
        expected_policy_book_rotation_in_eef()
        if expected_rotation_eef_policy_book is None
        else _validated_rotation(expected_rotation_eef_policy_book)
    )
    current_error = _rotation_error_deg(expected, current[:3, :3])
    candidate_error = _rotation_error_deg(expected, candidate[:3, :3])
    candidate_preferred = candidate_error + 1.0e-6 < current_error
    return {
        "schema_version": 1,
        "kind": "bookshelf_book_frame_axis_audit",
        "hardware_commanded": False,
        "read_only": True,
        "selection_authorized": False,
        "active_configuration_modified": False,
        "same_grasp_hypothesis": {
            "policy_book_axes": "+X depth/approach, +Y thickness, +Z up",
            "expected_axes_in_link_eef": {
                "+X_policy_book": "+Z_link_eef",
                "+Y_policy_book": "-Y_link_eef",
                "+Z_policy_book": "+X_link_eef",
            },
            "expected_rotation_eef_policy_book": expected.tolist(),
        },
        "candidate_axis_conversion": {
            "mapping": {
                "+X_policy_book": "+Y_saved_book",
                "+Y_policy_book": "-X_saved_book",
                "+Z_policy_book": "+Z_saved_book",
            },
            "transform_old_book_policy_book": _transform_to_dict(correction),
        },
        "saved_transform_eef_book": _transform_to_dict(current),
        "candidate_transform_eef_policy_book": _transform_to_dict(candidate),
        "saved_rotation_error_to_same_grasp_hypothesis_deg": current_error,
        "candidate_rotation_error_to_same_grasp_hypothesis_deg": candidate_error,
        "candidate_improvement_deg": current_error - candidate_error,
        "diagnostic_preferred_frame": (
            "candidate" if candidate_preferred else "saved"
        ),
        "candidate_preferred": candidate_preferred,
        "human_visual_review_required": True,
        "limitations": [
            "The expected xArm-to-simulator hand-axis correspondence is a hypothesis.",
            "Only the recorded RGB-D image and physical book geometry can select a frame.",
            "This audit never updates a calibration or authorizes robot motion.",
        ],
    }


def _rotation_error_deg(reference, value) -> float:
    relative = _validated_rotation(reference).T @ _validated_rotation(value)
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _transform_to_dict(transform) -> dict:
    transform = _validated_transform(transform)
    quaternion = matrix_to_quaternion_xyzw(transform[:3, :3])
    return {
        "translation_xyz_m": [float(value) for value in transform[:3, 3]],
        "quaternion_xyzw": [float(value) for value in quaternion],
        "matrix": transform.tolist(),
    }


def _validated_rotation(rotation) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("Rotation must be a finite 3x3 matrix.")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-7):
        raise ValueError("Rotation must be orthonormal.")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-7):
        raise ValueError("Rotation must have determinant +1.")
    return rotation


def _validated_transform(transform) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("Transform must be a finite 4x4 matrix.")
    _validated_rotation(transform[:3, :3])
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-10):
        raise ValueError("Transform must have homogeneous final row [0, 0, 0, 1].")
    return transform
