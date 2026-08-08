import math
from dataclasses import replace

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


def _recorded_riot_transforms():
    """Return the static slot, live TCP, and candidate tool transforms from the audit."""

    transform_base_slot = make_transform(
        [0.8741794456118123, 0.0807494671987291, 0.2473493846631505],
        [
            0.002228045026061973,
            0.0499056226703669,
            0.05101853094791474,
            0.9974475295177995,
        ],
    )
    transform_base_tcp = make_transform(
        [0.509, 0.064, 0.170],
        [0.719, 0.024, 0.694, -0.026],
    )
    transform_tcp_policy_tool = make_transform(
        [0.009119326451322136, -0.04695895701970321, -0.04939917878569483],
        [
            -0.7018277662283674,
            -0.03279647137253109,
            -0.00294897656373922,
            0.7115851892455586,
        ],
    )
    return transform_base_slot, transform_base_tcp, transform_tcp_policy_tool


def _recorded_riot_limits():
    return TargetSafetyLimits(
        maximum_delta=(
            0.008,
            0.003,
            0.007,
            math.radians(0.8),
            math.radians(0.6),
        ),
        maximum_tcp_translation_step_m=0.003,
        maximum_tcp_rotation_step_rad=math.radians(0.25),
        workspace_min_xyz=(0.20, -0.60, 0.05),
        workspace_max_xyz=(1.00, 0.60, 1.00),
    )


def test_recorded_riot_zero_delta_preserves_current_tcp_exactly():
    transform_base_slot, transform_base_tcp, transform_tcp_tool = (
        _recorded_riot_transforms()
    )

    target = compute_policy_tool_target(
        transform_base_slot,
        transform_base_tcp,
        transform_tcp_tool,
        [0.0] * 5,
        command_scale=0.1,
    )

    np.testing.assert_allclose(
        target.transform_base_tcp_target,
        transform_base_tcp,
        atol=1.0e-12,
    )
    assert target.tcp_translation_step_m < 1.0e-12
    assert target.tcp_rotation_step_rad < 1.0e-12
    assert target_safety_error(target, [0.0] * 5, _recorded_riot_limits()) is None


def test_recorded_riot_policy_delta_stays_near_current_tcp():
    transform_base_slot, transform_base_tcp, transform_tcp_tool = (
        _recorded_riot_transforms()
    )
    # Representative saturated shadow output from the static pre-insertion audit.
    motion_delta = [-0.0020, 0.0004, 0.0033, 0.01222, -0.00960]

    target = compute_policy_tool_target(
        transform_base_slot,
        transform_base_tcp,
        transform_tcp_tool,
        motion_delta,
        command_scale=0.1,
    )

    assert target.tcp_translation_step_m < 0.001
    assert target.tcp_rotation_step_rad < math.radians(0.15)
    assert target_safety_error(target, motion_delta, _recorded_riot_limits()) is None
    assert np.linalg.norm(
        target.transform_base_tcp_target[:3, 3] - transform_base_tcp[:3, 3]
    ) < 0.001

    historical_bad_target = np.array(
        [0.20633152100867308, 0.000057675413723541624, -0.05120369947506111]
    )
    assert np.linalg.norm(
        target.transform_base_tcp_target[:3, 3] - historical_bad_target
    ) > 0.30


def test_historical_absolute_target_is_rejected_as_a_large_tcp_step():
    transform_base_slot, transform_base_tcp, transform_tcp_tool = (
        _recorded_riot_transforms()
    )
    target = compute_policy_tool_target(
        transform_base_slot,
        transform_base_tcp,
        transform_tcp_tool,
        [0.0] * 5,
        command_scale=0.1,
    )
    historical_bad_transform = np.array(target.transform_base_tcp_target, copy=True)
    historical_bad_transform[:3, 3] = [
        0.20633152100867308,
        0.000057675413723541624,
        -0.05120369947506111,
    ]
    historical_step = np.linalg.inv(transform_base_tcp) @ historical_bad_transform
    bad_target = replace(
        target,
        transform_base_tcp_target=historical_bad_transform,
        tcp_translation_step_m=float(np.linalg.norm(historical_step[:3, 3])),
    )

    error = target_safety_error(bad_target, [0.0] * 5, _recorded_riot_limits())
    assert error is not None
    assert error.startswith("TCP translation step")


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
