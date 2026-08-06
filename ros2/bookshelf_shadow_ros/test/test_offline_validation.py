from pathlib import Path

import numpy as np

from bookshelf_shadow_ros.offline_validation import (
    PolicyStreamAuditAccumulator,
    SlotAuditAccumulator,
    audit_shadow_source_tree,
    controller_config_parity,
    make_pose_transform,
    perturb_transform,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_SOURCE = (
    REPOSITORY_ROOT
    / "ros2"
    / "bookshelf_shadow_ros"
)
ENV_CFG = (
    REPOSITORY_ROOT
    / "source"
    / "bookshelf"
    / "bookshelf"
    / "tasks"
    / "direct"
    / "bookshelf"
    / "bookshelf_residual_env_cfg.py"
)


def test_shadow_source_tree_has_no_robot_command_path():
    assert audit_shadow_source_tree(PACKAGE_SOURCE) == []


def test_portable_controller_constants_match_simulator_source():
    result = controller_config_parity(ENV_CFG)
    assert result["passed"], result["mismatches"]
    assert result["checked_values"] >= 25


def test_pose_perturbation_is_deterministic_and_finite():
    base = make_pose_transform([0.078, 0.0, 0.006])
    first = perturb_transform(
        base,
        translation_xyz=[0.001, -0.002, 0.003],
        rotation_rpy=[0.01, -0.02, 0.03],
    )
    second = perturb_transform(
        base,
        translation_xyz=[0.001, -0.002, 0.003],
        rotation_rpy=[0.01, -0.02, 0.03],
    )
    np.testing.assert_array_equal(first, second)
    assert np.all(np.isfinite(first))


def test_slot_audit_accumulator_reports_repeatability_and_invalid_frames():
    audit = SlotAuditAccumulator(minimum_confidence=0.60)
    audit.add(0.0)
    for index in range(10):
        audit.add(
            0.8 + 0.001 * index,
            width=0.037 + index * 1.0e-5,
            position=[0.01, -0.02 + index * 1.0e-5, 0.34],
            quaternion_xyzw=[0.0, 0.0, 0.0, 1.0],
        )
    summary = audit.summary()
    assert summary["samples"] == 11
    assert summary["valid_samples"] == 10
    assert summary["valid_fraction"] == 10 / 11
    assert summary["width_m"]["std"] > 0.0
    assert summary["orientation_error_deg"]["max"] == 0.0


def _add_complete_policy_sample(audit, *, width=0.0384, lateral=0.001):
    return audit.add(
        confidence=0.83,
        slot_width=width,
        slot_position=[0.45, -0.02, 0.30],
        slot_quaternion_xyzw=[0.0, 0.0, 0.0, 1.0],
        book_position=[0.37, -0.02 + lateral, 0.30],
        book_quaternion_xyzw=[0.0, 0.0, 0.0, 1.0],
        raw_metrics=[
            0.0,
            -0.01,
            0.04,
            -lateral,
            0.0,
            0.0,
            -0.06,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        observation=[
            0.0,
            -0.125,
            0.5,
            -lateral / 0.05,
            0.0,
            0.0,
            -0.24,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        policy_action=[0.1, -0.2, 0.3, -0.4, 0.5, -1.0],
        nominal_delta=[0.001, 0.00025, 0.00108, 0.0, 0.0],
        residual_delta=[0.0002, -0.0002, 0.00045, -0.002, 0.002],
        final_delta=[0.0012, 0.00005, 0.00153, -0.002, 0.002],
        book_pose_source="configured_eef_book",
        eef_book_transform_status="approximate_smoke_only",
        slot_pose_source="configured_static",
        static_slot_transform_status="measured_rgbd_static_no_absolute_ground_truth",
    )


def test_policy_stream_audit_reports_base_axes_and_reference_width_error():
    audit = PolicyStreamAuditAccumulator(reference_slot_width_m=0.038)
    for index in range(5):
        assert _add_complete_policy_sample(
            audit,
            width=0.0384 + index * 1.0e-5,
            lateral=0.001 + index * 1.0e-5,
        )
    audit.add_invalid("policy invalid: stale observation")

    summary = audit.summary()
    assert summary["samples"] == 6
    assert summary["complete_samples"] == 5
    assert summary["invalid_samples"] == 1
    assert summary["slot_width_error_m"]["mean_absolute"] > 0.0004
    np.testing.assert_allclose(
        summary["slot_axes_in_base"]["x"]["mean_direction"],
        [1.0, 0.0, 0.0],
        atol=1.0e-12,
    )
    assert summary["book_pose_sources"] == {"configured_eef_book": 5}
    assert summary["slot_pose_sources"] == {"configured_static": 5}
    assert summary["static_slot_transform_statuses"] == {
        "measured_rgbd_static_no_absolute_ground_truth": 5
    }
    assert summary["observation_clip_fraction"] == 0.0
    assert len(list(audit.csv_rows())) == 5


def test_policy_stream_audit_rejects_malformed_complete_sample():
    audit = PolicyStreamAuditAccumulator()
    accepted = audit.add(
        confidence=0.83,
        slot_width=0.038,
        slot_position=[0.45, -0.02, 0.30],
        slot_quaternion_xyzw=[0.0, 0.0, 0.0, 1.0],
        book_position=[0.37, -0.02, 0.30],
        book_quaternion_xyzw=[0.0, 0.0, 0.0, 1.0],
        raw_metrics=[0.0] * 11,
        observation=[0.0] * 12,
        policy_action=[0.0] * 6,
        nominal_delta=[0.0] * 5,
        residual_delta=[0.0] * 5,
        final_delta=[0.0] * 5,
        book_pose_source="configured_eef_book",
        eef_book_transform_status="approximate_smoke_only",
    )
    assert not accepted
    summary = audit.summary()
    assert summary["complete_samples"] == 0
    assert summary["invalid_samples"] == 1
    assert "raw_metrics must have shape" in next(iter(summary["invalid_reasons"]))
