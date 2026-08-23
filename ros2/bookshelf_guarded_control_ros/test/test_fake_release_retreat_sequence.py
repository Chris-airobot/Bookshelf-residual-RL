"""Contracts for the simulation-only release and retreat sequence."""

from pathlib import Path

import numpy as np
import pytest

from bookshelf_guarded_control_ros.fake_release_retreat_sequence_node import (
    grasp_alignment_target_eef,
    oriented_box_contact_gap,
    physical_release_guard_state,
    required_book_push_distance,
    retreat_progress,
    simulated_book_push_distance,
)
from bookshelf_guarded_control_ros.policy_tool_control_math import make_transform


ROOT = Path(__file__).resolve().parents[1]
NODE = (
    ROOT
    / "bookshelf_guarded_control_ros"
    / "fake_release_retreat_sequence_node.py"
)


def test_retreat_progress_projects_motion_onto_retreat_direction():
    assert retreat_progress(
        [1.0, 2.0, 3.0], [0.91, 2.02, 3.0], [-1, 0, 0]
    ) == pytest.approx(0.09)
    assert retreat_progress(
        [1.0, 2.0, 3.0], [1.0, 2.1, 3.0], [-1, 0, 0]
    ) == pytest.approx(0.0)


def test_retreat_progress_rejects_invalid_vectors():
    with pytest.raises(ValueError, match="nonzero"):
        retreat_progress(np.zeros(3), np.ones(3), np.zeros(3))
    with pytest.raises(ValueError, match="finite"):
        retreat_progress(
            np.zeros(3), [np.nan, 0.0, 0.0], [1.0, 0.0, 0.0]
        )


def test_grasp_alignment_moves_eef_while_preserving_book_world_pose():
    current_eef = make_transform([0.5, -0.1, 0.2])
    nominal_eef_book = make_transform([0.0, 0.0, 0.18])
    adjusted_eef_book = nominal_eef_book @ make_transform([0.028, 0.0, 0.0])

    target_eef = grasp_alignment_target_eef(
        current_eef, nominal_eef_book, adjusted_eef_book
    )

    assert target_eef @ adjusted_eef_book == pytest.approx(
        current_eef @ nominal_eef_book
    )
    assert target_eef[0, 3] == pytest.approx(current_eef[0, 3] - 0.028)


def test_simulated_book_only_moves_after_push_reaches_contact():
    assert simulated_book_push_distance(0.08, 0.09, 0.03) == pytest.approx(0.0)
    assert simulated_book_push_distance(0.10, 0.09, 0.03) == pytest.approx(0.01)
    assert simulated_book_push_distance(0.20, 0.09, 0.03) == pytest.approx(0.03)


def test_simulated_book_push_rejects_invalid_distances():
    with pytest.raises(ValueError, match="finite and nonnegative"):
        simulated_book_push_distance(-0.01, 0.09, 0.03)


def test_required_book_push_distance_reaches_task_success_depth():
    assert required_book_push_distance(-0.086, -0.012) == pytest.approx(0.074)
    assert required_book_push_distance(-0.010, -0.012) == pytest.approx(0.0)


def test_required_book_push_distance_rejects_nonfinite_depths():
    with pytest.raises(ValueError, match="finite"):
        required_book_push_distance(float("nan"), -0.012)


def test_oriented_box_contact_gap_uses_near_face_instead_of_box_center():
    box = make_transform([1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
    assert oriented_box_contact_gap(
        [0.7, 0.0, 0.0], box, [0.2, 0.4, 0.6], [1.0, 0.0, 0.0]
    ) == pytest.approx(0.2)
    assert oriented_box_contact_gap(
        [0.9, 0.0, 0.0], box, [0.2, 0.4, 0.6], [1.0, 0.0, 0.0]
    ) == pytest.approx(0.0)


def test_oriented_box_contact_gap_respects_box_rotation():
    box = make_transform(
        [1.0, 0.0, 0.0],
        [0.0, 0.0, np.sin(np.pi / 4.0), np.cos(np.pi / 4.0)],
    )
    assert oriented_box_contact_gap(
        [0.75, 0.0, 0.0], box, [0.2, 0.4, 0.6], [1.0, 0.0, 0.0]
    ) == pytest.approx(0.05)


def test_oriented_box_contact_gap_rejects_invalid_geometry():
    box = np.eye(4)
    with pytest.raises(ValueError, match="positive"):
        oriented_box_contact_gap([0, 0, 0], box, [1, 0, 1], [1, 0, 0])
    with pytest.raises(ValueError, match="nonzero"):
        oriented_box_contact_gap([0, 0, 0], box, [1, 1, 1], [0, 0, 0])


def test_physical_release_guard_allows_release_at_mouth_with_supported_book():
    state = physical_release_guard_state(
        np.eye(4),
        make_transform([-0.006, 0.0, 0.0]),
        make_transform([0.04, 0.0, 0.0]),
        [0.156, 0.034, 0.236],
        -0.006,
        0.08,
    )

    assert state["physical_boundary_reached"] is True
    assert state["book_supported"] is True
    assert state["release_allowed"] is True
    assert state["book_leading_penetration_m"] == pytest.approx(0.118)


def test_physical_release_guard_rejects_too_deep_grasp_at_mouth():
    state = physical_release_guard_state(
        np.eye(4),
        make_transform([-0.006, 0.0, 0.0]),
        make_transform([-0.05, 0.0, 0.0]),
        [0.156, 0.034, 0.236],
        -0.006,
        0.08,
    )

    assert state["physical_boundary_reached"] is True
    assert state["book_supported"] is False
    assert state["release_allowed"] is False


def test_physical_release_guard_waits_before_tcp_reaches_mouth_boundary():
    state = physical_release_guard_state(
        np.eye(4),
        make_transform([-0.02, 0.0, 0.0]),
        make_transform([0.04, 0.0, 0.0]),
        [0.156, 0.034, 0.236],
        -0.006,
        0.08,
    )

    assert state["physical_boundary_reached"] is False
    assert state["book_supported"] is True
    assert state["release_allowed"] is False


def test_sequence_connects_release_gripper_mode_book_tf_retreat_and_push():
    source = NODE.read_text(encoding="utf-8")
    required = (
        '"release_action"',
        '"/xarm_gripper_traj_controller/follow_joint_trajectory"',
        '"/bookshelf_policy/mode"',
        '"/bookshelf_sim/policy_control_enabled"',
        "self.book_attached = False",
        "self.tf_broadcaster.sendTransform(message)",
        "self._publish_mode(1)",
        "self._publish_mode(2)",
        'self.phase = "push"',
        'self._publish_control(True)',
        'self._publish_status("release, retreat, and policy push complete")',
        "simulated_book_push_distance(",
        "oriented_box_contact_gap(",
        "self.push_contact_distance_m",
        '"tcp_frame", "link_tcp"',
        '"book_size_xyz", [0.156, 0.034, 0.236]',
        '"gripper_open_position", 0.0',
        '"gripper_closed_position", 0.85',
        '"physical_release_guard_enabled", False',
        '"physical_release_tcp_x_limit_m", -0.006',
        '"minimum_book_leading_penetration_m", 0.08',
        '"push_to_target_trailing_depth_enabled", False',
        '"push_target_trailing_depth_m", -0.012',
        "self.requested_book_push_distance_m",
        'self._begin_release("physical_gripper_boundary")',
        "simulated grasp is too deep",
    )
    for token in required:
        assert token in source
    assert "contact_distance = self.retreat_distance_m" not in source


def test_xarm_gripper_uses_official_open_and_closed_positions():
    source = NODE.read_text(encoding="utf-8")
    assert '"gripper_open_position", 0.0' in source
    assert '"gripper_closed_position", 0.85' in source


def test_sequence_closes_gripper_before_enabling_insert_control():
    source = NODE.read_text(encoding="utf-8")
    pretarget = source.index('self.phase = "closing_for_insert"')
    close_goal = source.index('"close_for_insert",', pretarget)
    close_result = source.index('if kind == "close_for_insert":', close_goal)
    insert = source.index('self.phase = "insert"', close_result)
    enable = source.index("self._publish_control(True)", insert)

    assert pretarget < close_goal < close_result < insert < enable


def test_sequence_aligns_setback_before_closing_gripper():
    source = NODE.read_text(encoding="utf-8")
    align = source.index('self.phase = "aligning_grasp"')
    align_complete = source.index('self.phase = "closing_for_insert"', align)
    close_goal = source.index('"close_for_insert",', align_complete)

    assert align < align_complete < close_goal
    assert "self.alignment_book_transform.copy()" in source
    assert "grasp_alignment_target_eef(" in source
    assert '"start_servo_service", "/servo_server/start_servo"' in source
    assert "self.start_servo_client.call_async(Trigger.Request())" in source


def test_sequence_retries_transient_gripper_startup_rejections():
    source = NODE.read_text(encoding="utf-8")
    assert '"gripper_goal_retry_timeout_s", 15.0' in source
    assert '"gripper_goal_retry_period_s", 0.25' in source
    assert "if goal_handle is None or not goal_handle.accepted:" in source
    assert 'self.gripper_goal_pending = False' in source
    assert '"fake gripper controller is starting; retrying "' in source
    assert '"fake gripper goal remained rejected for "' in source


def test_sequence_is_fake_only_and_has_no_physical_xarm_api():
    source = NODE.read_text(encoding="utf-8")
    assert '"simulation_only": True' in source
    assert '"hardware_commanded": False' in source
    for token in ("xarm_msgs", "/xarm/set_mode", "/xarm/set_state"):
        assert token not in source
