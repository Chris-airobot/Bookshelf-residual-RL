import math

import numpy as np
import pytest

from bookshelf_shadow_ros.book_frame_audit import (
    apply_book_axis_correction,
    book_axis_correction_transform,
    book_frame_audit_report,
    expected_policy_book_rotation_in_eef,
)
from bookshelf_shadow_ros.policy_observation_math import make_transform


def test_default_axis_correction_maps_ideal_saved_axes_to_policy_axes():
    saved_rotation = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    transform = np.eye(4)
    transform[:3, :3] = saved_rotation
    transform[:3, 3] = [0.008, -0.010, 0.124]

    corrected = apply_book_axis_correction(transform)

    np.testing.assert_allclose(
        corrected[:3, :3], expected_policy_book_rotation_in_eef(), atol=1.0e-12
    )
    np.testing.assert_allclose(corrected[:3, 3], transform[:3, 3], atol=1.0e-12)


def test_stale_cover_rotation_candidate_reduces_same_grasp_axis_error():
    recorded = make_transform(
        [0.008124357683356356, -0.010156856549182705, 0.12425871757162561],
        [
            0.47569361268145194,
            0.46802866719711034,
            0.5317004002617697,
            0.5214973038254745,
        ],
    )

    report = book_frame_audit_report(recorded)

    assert report["saved_rotation_error_to_same_grasp_hypothesis_deg"] == pytest.approx(
        89.1495448587
    )
    assert report[
        "candidate_rotation_error_to_same_grasp_hypothesis_deg"
    ] == pytest.approx(6.3604064198)
    assert report["candidate_improvement_deg"] > 80.0
    assert report["selection_authorized"] is False
    assert report["active_configuration_modified"] is False
    assert report["hardware_commanded"] is False


def test_axis_correction_rejects_translation():
    correction = book_axis_correction_transform()
    correction[0, 3] = 0.001

    with pytest.raises(ValueError, match="must not translate"):
        apply_book_axis_correction(np.eye(4), correction)


def test_expected_rotation_and_correction_are_proper_rotations():
    for rotation in (
        book_axis_correction_transform()[:3, :3],
        expected_policy_book_rotation_in_eef(),
    ):
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-12)
        assert math.isclose(np.linalg.det(rotation), 1.0, abs_tol=1.0e-12)
