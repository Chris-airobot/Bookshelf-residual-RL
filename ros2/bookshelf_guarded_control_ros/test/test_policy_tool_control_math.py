import math

import numpy as np

from bookshelf_guarded_control_ros.policy_tool_control_math import (
    TargetSafetyLimits,
    compute_policy_tool_target,
    euler_xyz_to_matrix,
    execution_authorization_error,
    make_transform,
    maximum_named_joint_difference,
    provenance_error,
    target_safety_error,
)


def test_translation_delta_is_applied_in_slot_frame():
    transform_base_slot = np.eye(4)
    transform_base_slot[:3, :3] = euler_xyz_to_matrix(0.0, 0.0, math.pi / 2.0)
    transform_base_tcp = make_transform([0.5, 0.0, 0.4])
    transform_tcp_tool = make_transform([0.02, -0.03, 0.04])

    target = compute_policy_tool_target(
        transform_base_slot,
        transform_base_tcp,
        transform_tcp_tool,
        [0.002, 0.0, 0.0, 0.0, 0.0],
        command_scale=0.5,
    )

    expected_base_step = np.array([0.0, 0.001, 0.0])
    actual_base_step = (
        target.transform_base_policy_tool_target[:3, 3]
        - target.transform_base_policy_tool_current[:3, 3]
    )
    np.testing.assert_allclose(actual_base_step, expected_base_step, atol=1.0e-12)
    np.testing.assert_allclose(
        target.transform_base_tcp_target @ transform_tcp_tool,
        target.transform_base_policy_tool_target,
        atol=1.0e-12,
    )


def test_pitch_and_yaw_are_applied_to_tool_rpy_in_slot_frame():
    current_rpy = (0.3, -0.2, 0.4)
    transform_base_tcp = np.eye(4)
    transform_base_tcp[:3, :3] = euler_xyz_to_matrix(*current_rpy)
    target = compute_policy_tool_target(
        np.eye(4),
        transform_base_tcp,
        np.eye(4),
        [0.0, 0.0, 0.0, 0.06, -0.04],
        command_scale=0.5,
    )
    expected = euler_xyz_to_matrix(0.3, -0.22, 0.43)
    np.testing.assert_allclose(
        target.transform_base_tcp_target[:3, :3], expected, atol=1.0e-12
    )


def test_target_safety_rejects_unscaled_delta_and_workspace_violation():
    target = compute_policy_tool_target(
        np.eye(4),
        make_transform([0.5, 0.0, 0.4]),
        np.eye(4),
        [0.009, 0.0, 0.0, 0.0, 0.0],
        command_scale=0.1,
    )
    assert "exceeds configured limits" in target_safety_error(
        target, [0.009, 0.0, 0.0, 0.0, 0.0]
    )

    outside = compute_policy_tool_target(
        np.eye(4),
        make_transform([1.1, 0.0, 0.4]),
        np.eye(4),
        [0.001, 0.0, 0.0, 0.0, 0.0],
        command_scale=0.1,
    )
    assert "outside workspace" in target_safety_error(
        outside,
        [0.001, 0.0, 0.0, 0.0, 0.0],
        TargetSafetyLimits(),
    )


def test_provenance_rejects_unverified_candidate_and_release():
    adapter = {
        "valid": True,
        "policy_tool_transform_status": "derived_unverified_candidate",
        "static_slot_transform_status": "measured_slot",
        "eef_book_transform_status": "measured_book",
    }
    policy = {
        "valid": True,
        "bundle_sha256": "abc",
        "release_requested_diagnostic": False,
        "release_executed": False,
    }
    common = {
        "expected_policy_tool_status": "derived_unverified_candidate",
        "expected_slot_status": "measured_slot",
        "expected_book_status": "measured_book",
        "expected_bundle_sha256": "abc",
    }
    assert "unverified" in provenance_error(
        adapter, policy, allow_unverified_policy_tool=False, **common
    )
    assert provenance_error(
        adapter, policy, allow_unverified_policy_tool=True, **common
    ) is None
    policy["release_requested_diagnostic"] = True
    assert "release" in provenance_error(
        adapter, policy, allow_unverified_policy_tool=True, **common
    )


def test_execution_requires_every_gate_and_consumable_token():
    values = {
        "dry_run": True,
        "allow_execution": False,
        "planning_scene_complete": False,
        "approval_token": "wrong",
        "configured_token": "trial-01",
        "plan_age_s": 0.1,
        "maximum_plan_age_s": 1.0,
        "plan_valid": True,
        "busy": False,
    }
    assert execution_authorization_error(**values) == "dry_run is true"
    values.update(
        dry_run=False,
        allow_execution=True,
        planning_scene_complete=True,
        approval_token="trial-01",
    )
    assert execution_authorization_error(**values) is None
    values["plan_age_s"] = 2.0
    assert "stale" in execution_authorization_error(**values)


def test_named_joint_drift_uses_names_not_message_order():
    difference = maximum_named_joint_difference(
        ["joint2", "joint1"],
        [0.2, 0.1],
        ["joint1", "joint2"],
        [0.11, 0.19],
    )
    assert math.isclose(difference, 0.01, abs_tol=1.0e-12)

