import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from bookshelf_simple_experiment_ros.policy_observation_math import (
    compute_policy_observation,
)
from bookshelf_simple_experiment_ros.policy_tool_math import (
    OneShotExecutionGuard,
    compute_policy_tool_target,
    make_transform,
)
from bookshelf_simple_experiment_ros.residual_policy_math import (
    NumpyActorBundle,
    combine_motion_delta,
    compute_insert_nominal_delta,
    compute_policy_nominal_delta,
    release_requested_for_mode,
    scale_residual_action,
)
from bookshelf_simple_experiment_ros.simple_policy_control_node import (
    load_reviewed_policy_geometry,
    named_joint_positions,
)


PACKAGE = Path(__file__).resolve().parents[1]
REPOSITORY = PACKAGE.parents[1]


def _reference_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, REPOSITORY / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_observation_math_matches_verified_shadow_implementation():
    reference = _reference_module(
        "_reference_policy_observation_math",
        "ros2/bookshelf_shadow_ros/bookshelf_shadow_ros/policy_observation_math.py",
    )
    slot_book = make_transform([0.021, -0.004, 0.007], [0.01, -0.02, 0.03, 0.999])
    slot_tool = make_transform([0.005, 0.002, 0.015], [0.0, 0.0, 0.0, 1.0])
    actual = compute_policy_observation(slot_book, slot_tool, gripper_open=0.17)
    expected = reference.compute_policy_observation(
        slot_book, slot_tool, gripper_open=0.17
    )
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def test_insert_nominal_residual_fusion_and_release_match_reference():
    reference = _reference_module(
        "_reference_policy_shadow_math",
        "ros2/bookshelf_shadow_ros/bookshelf_shadow_ros/policy_shadow_math.py",
    )
    raw = np.array([0.0, -0.02, 0.12, 0.002, 0.004, 0.03, 0.1, 0.0, 0.0, 0.0, 0.02, 0.01])
    action = np.array([0.5, -0.3, 0.8, -0.2, 0.1, 0.7])
    np.testing.assert_array_equal(
        compute_insert_nominal_delta(raw), reference.compute_insert_nominal_delta(raw)
    )
    np.testing.assert_array_equal(
        compute_policy_nominal_delta(raw), reference.compute_policy_nominal_delta(raw)
    )
    residual = scale_residual_action(action)
    expected_residual = reference.scale_residual_action(action)
    np.testing.assert_array_equal(residual, expected_residual)
    np.testing.assert_array_equal(
        combine_motion_delta(compute_insert_nominal_delta(raw), residual),
        reference.combine_motion_delta(
            reference.compute_insert_nominal_delta(raw), expected_residual
        ),
    )
    assert release_requested_for_mode(0.7, 0.0, 0.5) is reference.release_requested_for_mode(0.7, 0.0, 0.5)
    assert release_requested_for_mode(0.7, 1.0, 0.5) is False


def test_policy_tool_tcp_target_matches_guarded_pure_math():
    reference = _reference_module(
        "_reference_policy_tool_control_math",
        "ros2/bookshelf_guarded_control_ros/bookshelf_guarded_control_ros/policy_tool_control_math.py",
    )
    base_slot = make_transform([0.855, 0.084, 0.171], [0.001, -0.021, 0.039, 0.999])
    base_tcp = make_transform([0.70, 0.09, 0.18], [0.01, -0.02, 0.03, 0.999])
    tcp_tool = make_transform([0.0058, 0.0032, -0.0281], [-0.032, -0.004, 0.017, 0.999])
    delta = np.array([0.001, -0.0004, 0.0007, 0.003, -0.002])
    actual = compute_policy_tool_target(base_slot, base_tcp, tcp_tool, delta, command_scale=0.1)
    expected = reference.compute_policy_tool_target(base_slot, base_tcp, tcp_tool, delta, command_scale=0.1)
    for name in (
        "scaled_delta",
        "transform_base_policy_tool_current",
        "transform_slot_policy_tool_current",
        "transform_slot_policy_tool_target",
        "transform_base_policy_tool_target",
        "transform_base_tcp_target",
    ):
        np.testing.assert_allclose(getattr(actual, name), getattr(expected, name), atol=1.0e-15)
    assert actual.tcp_translation_step_m == pytest.approx(expected.tcp_translation_step_m)
    assert actual.tcp_rotation_step_rad == pytest.approx(expected.tcp_rotation_step_rad)
    assert actual.target_id == expected.target_id


def test_actor_and_vecnormalize_match_reference_when_bundle_is_available():
    actor_path = Path(
        "/home/chris/BookshelfFiles/trained_models/"
        "bookshelf_residual_2026-07-08_shadow_actor.npz"
    )
    if not actor_path.is_file():
        pytest.skip("Alienware actor bundle is unavailable")
    reference = _reference_module(
        "_reference_actor_policy_shadow_math",
        "ros2/bookshelf_shadow_ros/bookshelf_shadow_ros/policy_shadow_math.py",
    )
    observation = np.linspace(-0.8, 0.8, 12, dtype=np.float32)
    actual = NumpyActorBundle(actor_path).predict(observation)
    expected = reference.NumpyActorBundle(actor_path).predict(observation)
    for actual_value, expected_value in zip(actual, expected):
        np.testing.assert_array_equal(actual_value, expected_value)


def test_approved_geometry_derives_verified_tcp_policy_tool_transform():
    path = Path(
        "/home/chris/BookshelfFiles/experiment_configs/"
        "stationary_approved_53e7fe80d56d_20260819_142355/trial_static_slot.yaml"
    )
    if not path.is_file():
        pytest.skip("Alienware approved configuration is unavailable")
    geometry = load_reviewed_policy_geometry(path)
    np.testing.assert_allclose(
        geometry.transform_eef_tcp[:3, 3],
        [0.0, 0.0, 0.172],
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        geometry.transform_tcp_policy_tool[:3, 3],
        [0.005846315351888094, 0.003170522180675528, -0.02806561588472603],
        atol=1.0e-15,
    )


def test_named_joint_positions_uses_explicit_arm_order():
    message = SimpleNamespace(
        name=["joint3", "drive_joint", "joint1", "joint2"],
        position=[0.3, 0.85, 0.1, 0.2],
    )
    assert named_joint_positions(message, ["joint1", "joint2", "joint3"]) == [
        0.1,
        0.2,
        0.3,
    ]


def test_one_shot_guard_allows_exactly_one_execution():
    guard = OneShotExecutionGuard()
    assert guard.try_consume() is True
    assert guard.try_consume() is False
