"""Contracts for the live-marker physical episode coordinator."""

from pathlib import Path

import pytest

from bookshelf_guarded_control_ros.physical_episode_coordinator_math import (
    HARDWARE_AUTHORIZATION_TOKEN,
    trailing_depth_target_reached,
    validate_episode_operation,
)


ROOT = Path(__file__).resolve().parents[1]
NODE = ROOT / "bookshelf_guarded_control_ros" / "physical_episode_coordinator_node.py"


def test_control_requires_exact_explicit_authorization():
    assert validate_episode_operation("calculate", "") == "calculate"
    assert (
        validate_episode_operation("control", HARDWARE_AUTHORIZATION_TOKEN)
        == "control"
    )
    with pytest.raises(ValueError, match="authorization token"):
        validate_episode_operation("control", "yes")


def test_live_trailing_depth_stops_at_target_with_tolerance():
    assert trailing_depth_target_reached(-0.0115, -0.012, 0.001)
    assert trailing_depth_target_reached(-0.0125, -0.012, 0.001)
    assert not trailing_depth_target_reached(-0.014, -0.012, 0.001)


def test_live_trailing_depth_rejects_invalid_values():
    with pytest.raises(ValueError, match="finite"):
        trailing_depth_target_reached(float("nan"), -0.012, 0.001)
    with pytest.raises(ValueError, match="nonnegative"):
        trailing_depth_target_reached(-0.012, -0.012, -0.001)


def test_calculate_mode_constructs_no_robot_command_interfaces():
    source = NODE.read_text(encoding="utf-8")
    assert 'self.phase = "calculate" if not self.control_mode' in source
    assert "if self.control_mode:" in source
    assert "self.gripper_client = None" in source
    assert "self.twist_publisher = None" in source
    assert "self.control_enable_publisher = None" in source
    assert "Calculate mode deliberately creates none of these command interfaces" in source


def test_episode_uses_live_marker_for_release_and_push_completion():
    source = NODE.read_text(encoding="utf-8")
    required = (
        '"book_frame", "target_book_center"',
        "physical_release_guard_state(",
        "required_book_push_distance(",
        "trailing_depth_target_reached(",
        '"book_pose_source": "live_marker_tf"',
        'self._fail(geometry_error)',
        'self._publish_mode(0)',
        'self._publish_mode(1)',
        'self._publish_mode(2)',
        '"live book reached target trailing depth; full episode complete"',
    )
    for token in required:
        assert token in source
    assert "TransformBroadcaster" not in source
    assert "sendTransform" not in source


def test_episode_hands_policy_control_around_straight_retreat():
    source = NODE.read_text(encoding="utf-8")
    release = source.index("def _begin_release")
    disable = source.index("self._publish_control(False)", release)
    retreat = source.index("def _run_retreat", disable)
    direct_twist = source.index("self._publish_direction_twist", retreat)
    push = source.index("def _run_waiting_for_push_policy", direct_twist)
    enable = source.index("self._publish_control(True)", push)

    assert release < disable < retreat < direct_twist < push < enable


def test_episode_requires_gripper_action_before_policy_insertion():
    source = NODE.read_text(encoding="utf-8")
    waiting = source.index('if self.phase == "waiting_for_start"')
    gripper_check = source.index(
        "if not self.gripper_client.server_is_ready()", waiting
    )
    insertion_enable = source.index("self._publish_control(True)", gripper_check)

    assert waiting < gripper_check < insertion_enable
    assert "insertion was not started" in source
