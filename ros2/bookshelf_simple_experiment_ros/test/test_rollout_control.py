from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from action_msgs.msg import GoalStatus
from bookshelf_simple_experiment_ros.simple_policy_control_node import (
    SimplePolicyControlNode,
)


def _geometry():
    return SimpleNamespace(
        transform_base_slot=np.eye(4),
        transform_eef_book=np.eye(4),
        transform_eef_policy_tool=np.eye(4),
    )


def _completion_harness(*, rollout=True, total_steps=1, max_steps=150):
    return SimpleNamespace(
        eef_frame="link_eef",
        tcp_frame="link_tcp",
        _lookup=Mock(side_effect=[np.eye(4), np.eye(4)]),
        record={"servo_result": "target_reached", "release_requested": False},
        geometry=_geometry(),
        _publish_visualization=Mock(),
        _active_eef_book_transform=lambda: np.eye(4),
        _arm_joint_positions=Mock(
            return_value=([f"joint{index}" for index in range(1, 8)], [0.0] * 7)
        ),
        latest_servo_status=0,
        nonzero_command_count=3,
        _write_record=Mock(),
        rollout=rollout,
        total_steps=total_steps,
        max_steps=max_steps,
        step_index=total_steps - 1,
        target=object(),
        target_eef=object(),
        phase="settling",
        _publish_status=Mock(),
        _begin_visualization_hold=Mock(),
        _terminate_rollout=Mock(),
    )


def test_rollout_reenters_live_state_calculation_for_each_new_step():
    harness = _completion_harness(total_steps=1, max_steps=3)
    SimplePolicyControlNode._complete_execution(harness)
    assert harness.phase == "waiting_for_live_state"
    assert harness.step_index == 1
    assert harness.record is None
    assert harness.target is None

    harness._try_calculate = Mock()
    SimplePolicyControlNode._timer_callback(harness)
    harness.phase = "waiting_for_live_state"
    SimplePolicyControlNode._timer_callback(harness)
    assert harness._try_calculate.call_count == 2


def test_rollout_stops_before_motion_when_release_is_requested():
    harness = SimpleNamespace(
        rollout=True,
        record={"ppo_residual_action": [0.0] * 5 + [0.6], "step_index": 12},
        twist_publisher=object(),
        _publish_twist=Mock(),
        geometry=_geometry(),
        _arm_joint_positions=Mock(
            return_value=([f"joint{index}" for index in range(1, 8)], [0.0] * 7)
        ),
        _write_record=Mock(),
        _log_phase_event=Mock(),
        _now_ns=lambda: 123,
        _publish_status=Mock(),
        phase="continuous_rollout",
        gripper_goal_pending=True,
        gripper_goal_kind="stale",
        gripper_retry_start_ns=1,
        gripper_next_attempt_ns=1,
        get_logger=lambda: SimpleNamespace(warning=Mock()),
        _active_eef_book_transform=lambda: np.eye(4),
    )
    stopped = SimplePolicyControlNode._stop_rollout_for_release(
        harness, True, np.eye(4), np.eye(4)
    )
    assert stopped is True
    assert harness.record["servo_result"] == "release_requested_no_motion"
    assert harness.record["servo_nonzero_command_count"] == 0
    np.testing.assert_array_equal(
        harness._publish_twist.call_args.args[0], np.zeros(6)
    )
    assert harness.phase == "opening_gripper"
    np.testing.assert_array_equal(harness.released_book_transform, np.eye(4))
    assert [call.args[0] for call in harness._log_phase_event.call_args_list] == [
        "release_requested",
        "release_started",
    ]


def test_continuous_rollout_recomputes_policy_without_waiting_for_target():
    harness = SimpleNamespace(
        phase="continuous_rollout",
        _continuous_rollout_timed_out=lambda: False,
        _try_calculate=Mock(),
    )
    SimplePolicyControlNode._continuous_policy_tick(harness)
    harness._try_calculate.assert_called_once_with()


def test_continuous_rollout_uses_max_steps_as_overall_timeout():
    harness = SimpleNamespace(
        rollout_start_ns=1_000_000_000,
        max_steps=150,
        _now_ns=lambda: 151_000_000_000,
    )
    assert SimplePolicyControlNode._continuous_rollout_timed_out(harness)


def test_singularity_deceleration_status_is_nonfatal():
    harness = SimpleNamespace(
        latest_servo_status=1,
        observed_servo_statuses=set(),
        target_eef=np.eye(4),
        eef_frame="link_eef",
        _lookup=lambda _frame: np.eye(4),
        get_parameter=lambda name: SimpleNamespace(value={
            "policy_command_duration_s": 0.2,
            "maximum_linear_speed_m_s": 0.025,
            "maximum_angular_speed_rad_s": 0.10,
            "translation_tolerance_m": 0.0005,
            "rotation_tolerance_rad": 0.004363323129985824,
        }[name]),
        _halt_and_fail=Mock(),
        _publish_twist=Mock(),
        _now_ns=lambda: 2_000_000_000,
        first_servo_command_ns=None,
        last_servo_command_ns=None,
        continuous_servo_command_count=0,
    )
    SimplePolicyControlNode._continuous_servo_tick(harness)
    harness._halt_and_fail.assert_not_called()
    harness._publish_twist.assert_called_once()
    assert harness.observed_servo_statuses == {1}
    assert harness.continuous_servo_command_count == 1


def test_rollout_stops_on_execution_error():
    harness = SimpleNamespace(
        record={"step_index": 2},
        step_index=2,
        execute=True,
        rollout=True,
        latest_servo_status=-1,
        _write_record=Mock(),
        _publish_status=Mock(),
        _terminate_rollout=Mock(),
        get_logger=lambda: SimpleNamespace(error=Mock()),
    )
    SimplePolicyControlNode._fail(harness, "Servo unavailable")
    harness._terminate_rollout.assert_called_once_with(
        "error", error="Servo unavailable"
    )


def test_existing_one_step_completion_still_holds_and_stops():
    harness = _completion_harness(rollout=False)
    SimplePolicyControlNode._complete_execution(harness)
    harness._begin_visualization_hold.assert_called_once_with()
    harness._terminate_rollout.assert_not_called()


def test_open_then_retreat_then_empty_close_then_push_phase_order():
    harness = SimpleNamespace(
        gripper_goal_kind="open",
        gripper_goal_pending=True,
        gripper_retry_start_ns=1,
        gripper_next_attempt_ns=1,
        _lookup=Mock(side_effect=[np.eye(4), np.eye(4)]),
        eef_frame="link_eef",
        tcp_frame="link_tcp",
        _now_ns=lambda: 10,
        _log_phase_event=Mock(),
        _publish_status=Mock(),
        _halt_and_fail=Mock(),
        get_logger=lambda: SimpleNamespace(warning=Mock()),
    )
    result = SimpleNamespace(
        status=GoalStatus.STATUS_SUCCEEDED,
    )
    future = Mock()
    future.result.return_value = result
    SimplePolicyControlNode._gripper_goal_result(harness, future)
    assert harness.phase == "retreat"
    assert harness.retreat_distance_m == 0.0
    assert [call.args[0] for call in harness._publish_status.call_args_list] == [
        "release_complete",
        "retreat_started",
    ]

    harness.gripper_goal_kind = "close_empty"
    harness.gripper_goal_pending = True
    harness.released_book_transform = np.eye(4)
    harness._lookup = Mock(side_effect=[np.eye(4), np.eye(4)])
    SimplePolicyControlNode._gripper_goal_result(harness, future)
    assert harness.phase == "push"
    assert harness.push_book_origin is not harness.released_book_transform
    assert harness.push_book_transform is not harness.released_book_transform
    release_pose = harness.released_book_transform.copy()
    harness.push_book_transform[0, 3] += 0.03
    np.testing.assert_array_equal(harness.released_book_transform, release_pose)
    events = [call.args[0] for call in harness._log_phase_event.call_args_list]
    assert events == [
        "release_complete",
        "retreat_started",
        "empty_gripper_closed",
        "push_started",
    ]


def test_push_uses_thirty_mm_from_geometric_contact_without_uncertainty_offset():
    release_pose = np.eye(4)
    release_pose[0, 3] = 0.168  # Near face is X=0.09 for a 156 mm-deep book.
    current = np.eye(4)
    current[0, 3] = 0.121
    harness = SimpleNamespace(
        _post_servo_status_is_fatal=lambda: False,
        _lookup=Mock(side_effect=[current, current]),
        eef_frame="link_eef",
        tcp_frame="link_tcp",
        retreat_direction=np.array([-1.0, 0.0, 0.0]),
        push_start_xyz=np.zeros(3),
        push_book_origin=release_pose.copy(),
        push_book_transform=release_pose.copy(),
        released_book_transform=release_pose.copy(),
        push_geometric_contact_distance_m=None,
        push_contact_distance_m=None,
        book_contact_gap_m=None,
        push_distance_m=0.0,
        book_push_distance_m=0.0,
        geometry=SimpleNamespace(book_size=(0.156, 0.034, 0.236)),
        get_parameter=lambda name: SimpleNamespace(
            value={
                "contact_tolerance_m": 0.001,
                "push_book_distance_m": 0.03,
                "push_x_uncertainty_m": 0.005,
            }[name]
        ),
        _publish_post_visualization=Mock(),
        _publish_twist=Mock(),
        _log_phase_event=Mock(),
        _complete_episode=Mock(),
    )

    SimplePolicyControlNode._push_servo_tick(harness)

    assert harness.push_geometric_contact_distance_m == pytest.approx(0.09)
    assert harness.push_contact_distance_m == pytest.approx(0.09)
    assert harness.book_push_distance_m == pytest.approx(0.03)
    np.testing.assert_array_equal(harness.released_book_transform, release_pose)
    assert harness.push_book_transform[0, 3] == pytest.approx(0.198)
    harness._complete_episode.assert_called_once()
