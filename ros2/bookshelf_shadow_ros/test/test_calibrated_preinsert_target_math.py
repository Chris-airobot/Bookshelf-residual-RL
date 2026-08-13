import numpy as np
import pytest

from bookshelf_shadow_ros.calibrated_preinsert_target_math import (
    PreinsertTargetSpec,
    calibration_sensitivity,
    compare_current_eef_to_target,
    compute_calibrated_preinsert_target,
    compute_preserved_tcp_orientation_preinsert_target,
)
from bookshelf_shadow_ros.policy_observation_math import make_transform


def test_aligned_target_places_front_face_at_standoff():
    spec = PreinsertTargetSpec(standoff=0.030, vertical_offset=0.006)
    target = compute_calibrated_preinsert_target(
        np.eye(4), np.eye(4), spec=spec
    )

    np.testing.assert_allclose(
        target.transform_slot_book_target[:3, 3],
        [-0.108, 0.0, 0.006],
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        target.transform_base_eef_target,
        target.transform_slot_book_target,
    )
    assert target.raw_metrics[1] == pytest.approx(-0.186)
    assert target.raw_metrics[2] == pytest.approx(0.230)
    assert target.raw_metrics[3] == pytest.approx(0.0)
    assert target.raw_metrics[4] == pytest.approx(0.006)
    assert target.expected_clipped_labels == ("rear_to_mouth", "front_to_back")
    assert target.unexpected_clipped_labels == ()


def test_target_eef_reconstructs_target_book_with_nontrivial_calibration():
    transform_base_slot = make_transform(
        [0.8, 0.1, 0.3], [0.01, -0.02, 0.03, 0.99]
    )
    transform_eef_book = make_transform(
        [0.02, -0.01, 0.12], [0.4, 0.5, 0.5, 0.58]
    )
    target = compute_calibrated_preinsert_target(
        transform_base_slot, transform_eef_book
    )

    np.testing.assert_allclose(
        target.transform_base_eef_target @ transform_eef_book,
        target.transform_base_book_target,
        atol=1.0e-10,
    )


def test_preserved_tcp_orientation_keeps_rotation_and_places_book_center():
    transform_base_eef_current = make_transform(
        [0.5, 0.1, 0.2], [0.0, 0.0, 0.2588190451, 0.9659258263]
    )
    transform_base_tcp_current = (
        transform_base_eef_current @ make_transform([0.0, 0.0, 0.172])
    )
    transform_eef_book = make_transform(
        [0.008, -0.010, 0.124], [0.4757, 0.4680, 0.5317, 0.5215]
    )

    target, diagnostics = compute_preserved_tcp_orientation_preinsert_target(
        np.eye(4),
        transform_eef_book,
        transform_base_eef_current,
        transform_base_tcp_current,
        spec=PreinsertTargetSpec(standoff=0.030, vertical_offset=0.006),
    )

    np.testing.assert_allclose(
        diagnostics["transform_base_tcp_target"][:3, :3],
        transform_base_tcp_current[:3, :3],
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        target.transform_base_book_target[:3, 3],
        [-0.108, 0.0, 0.006],
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        target.transform_base_eef_target @ transform_eef_book,
        target.transform_base_book_target,
        atol=1.0e-10,
    )
    assert diagnostics["tcp_orientation_change_deg"] == pytest.approx(0.0)
    assert diagnostics["book_center_error_m"] == pytest.approx(0.0, abs=1.0e-12)


def test_preserved_tcp_orientation_reports_book_slot_orientation_error():
    quarter_turn = [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]
    transform_base_eef_current = make_transform([0.0, 0.0, 0.0], quarter_turn)
    transform_base_tcp_current = (
        transform_base_eef_current @ make_transform([0.0, 0.0, 0.172])
    )
    target, diagnostics = compute_preserved_tcp_orientation_preinsert_target(
        np.eye(4),
        np.eye(4),
        transform_base_eef_current,
        transform_base_tcp_current,
    )

    assert diagnostics["book_orientation_error_deg"] == pytest.approx(90.0)
    np.testing.assert_allclose(
        target.transform_base_book_target[:3, 3],
        [-0.108, 0.0, 0.006],
        atol=1.0e-12,
    )


def test_policy_observation_uses_explicit_policy_tool_not_eef_origin():
    transform_eef_policy_tool = make_transform([0.032, 0.0, 0.0])
    target = compute_calibrated_preinsert_target(
        np.eye(4),
        np.eye(4),
        transform_eef_policy_tool=transform_eef_policy_tool,
    )

    np.testing.assert_allclose(
        target.transform_base_policy_tool_target,
        target.transform_base_eef_target @ transform_eef_policy_tool,
    )
    assert target.raw_metrics[6] == pytest.approx(0.032)
    assert target.raw_metrics[7] == pytest.approx(0.0)
    assert target.raw_metrics[8] == pytest.approx(0.0)


def test_current_pose_equal_to_target_has_zero_pose_delta():
    transform_base_slot = make_transform(
        [0.8, 0.1, 0.3], [0.01, -0.02, 0.03, 0.99]
    )
    transform_eef_book = make_transform(
        [0.02, -0.01, 0.12], [0.4, 0.5, 0.5, 0.58]
    )
    target = compute_calibrated_preinsert_target(
        transform_base_slot, transform_eef_book
    )
    comparison = compare_current_eef_to_target(
        target.transform_base_eef_target,
        transform_base_slot,
        transform_eef_book,
        target,
    )

    assert comparison["target_minus_current_translation_norm_m"] == pytest.approx(
        0.0
    )
    assert comparison["target_minus_current_rotation_deg"] == pytest.approx(0.0)
    np.testing.assert_allclose(
        comparison["observation_12d"], target.observation_12d, atol=1.0e-6
    )


def test_zero_calibration_uncertainty_has_zero_target_error():
    result = calibration_sensitivity(
        np.eye(4),
        make_transform([0.02, -0.01, 0.12], [0.4, 0.5, 0.5, 0.58]),
        samples=20,
        translation_uncertainty_m=0.0,
        rotation_uncertainty_deg=0.0,
    )

    assert result["translation_error_norm_m"]["max"] == pytest.approx(0.0)
    assert result["rotation_error_deg"]["max"] == pytest.approx(0.0)


def test_bounded_sensitivity_is_deterministic_and_finite():
    arguments = dict(
        samples=100,
        translation_uncertainty_m=0.002,
        rotation_uncertainty_deg=2.0,
        seed=123,
    )
    first = calibration_sensitivity(np.eye(4), np.eye(4), **arguments)
    second = calibration_sensitivity(np.eye(4), np.eye(4), **arguments)

    assert first == second
    assert (
        first["translation_error_norm_m"]["max"]
        <= np.sqrt(3.0) * 0.002 + 1.0e-12
    )
    assert first["rotation_error_deg"]["max"] <= 2.0 + 1.0e-9
