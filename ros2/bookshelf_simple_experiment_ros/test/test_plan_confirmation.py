from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from moveit_msgs.msg import MoveItErrorCodes, RobotState, RobotTrajectory
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectoryPoint

from bookshelf_simple_experiment_ros.joint_pose import JOINT_NAMES
from bookshelf_simple_experiment_ros.preinsert_node import (
    SimplePreinsertNode,
    build_direct_joint_trajectory,
)
from bookshelf_simple_experiment_ros.simple_policy_control_node import (
    SimplePolicyControlNode,
)


def _successful_plan_result():
    trajectory = RobotTrajectory()
    trajectory.joint_trajectory.points = [
        JointTrajectoryPoint(),
        JointTrajectoryPoint(),
    ]
    result = SimpleNamespace(
        error_code=MoveItErrorCodes(val=MoveItErrorCodes.SUCCESS),
        trajectory=trajectory,
        trajectory_start=RobotState(),
    )
    return SimpleNamespace(result=lambda: SimpleNamespace(motion_plan_response=result))


def test_successful_separate_plan_is_stored_and_not_automatically_executed():
    logger = SimpleNamespace(warning=Mock())
    harness = SimpleNamespace(
        group_name="xarm7",
        robot_model_id="UF_ROBOT",
        display_publisher=SimpleNamespace(publish=Mock()),
        planned_trajectory=None,
        planned_kind=None,
        planned_type=None,
        pending={"request": "active", "kind": "preinsert"},
        get_parameter=lambda _name: SimpleNamespace(value=True),
        get_logger=lambda: logger,
        _publish_status=Mock(),
        _send_execution=Mock(),
        _fail=Mock(),
    )

    SimplePreinsertNode._plan_response_callback(harness, _successful_plan_result())

    assert harness.planned_trajectory is not None
    assert harness.planned_type == "moveit"
    assert harness.pending is None
    harness.display_publisher.publish.assert_called_once()
    display = harness.display_publisher.publish.call_args.args[0]
    assert display.model_id == "UF_ROBOT"
    assert display.model_id != harness.group_name
    harness._publish_status.assert_called_once_with("awaiting_execute_confirmation")
    harness._send_execution.assert_not_called()
    logger.warning.assert_called_once_with(
        "PREINSERT PLAN READY - waiting for execution confirmation"
    )


def test_execute_service_rejects_when_no_stored_plan_is_waiting():
    harness = SimpleNamespace(
        allow_execution=True,
        execution_client=SimpleNamespace(wait_for_server=Mock(return_value=True)),
        phase="planned",
        planned_trajectory=None,
        planned_kind=None,
        planned_type=None,
        executing_kind=None,
        executing_type=None,
        get_parameter=lambda _name: SimpleNamespace(value=True),
        _send_execution=Mock(),
    )
    response = SimpleNamespace(success=None, message="")

    returned = SimplePreinsertNode._execute_trigger_callback(harness, None, response)

    assert returned is response
    assert response.success is False
    assert response.message == "no confirmed trajectory is awaiting execution"
    harness._send_execution.assert_not_called()


def test_execute_service_submits_the_exact_stored_plan_without_replanning():
    stored_trajectory = RobotTrajectory()
    plan_client = Mock()
    ik_client = Mock()
    harness = SimpleNamespace(
        allow_execution=True,
        execution_client=SimpleNamespace(wait_for_server=Mock(return_value=True)),
        phase="awaiting_execute_confirmation",
        planned_trajectory=stored_trajectory,
        planned_kind="scan",
        planned_type="moveit",
        executing_kind=None,
        executing_type=None,
        direct_execution_client=None,
        get_parameter=lambda _name: SimpleNamespace(value=True),
        _send_execution=Mock(),
        plan_client=plan_client,
        ik_client=ik_client,
    )
    response = SimpleNamespace(success=None, message="")

    returned = SimplePreinsertNode._execute_trigger_callback(harness, None, response)

    assert returned is response
    assert response.success is True
    harness._send_execution.assert_called_once_with(stored_trajectory)
    assert harness._send_execution.call_args.args[0] is stored_trajectory
    assert harness.planned_trajectory is None
    assert harness.executing_kind == "scan"
    plan_client.assert_not_called()
    ik_client.assert_not_called()


def test_direct_trajectory_preview_and_execution_use_identical_points():
    trajectory = build_direct_joint_trajectory(
        JOINT_NAMES, [0.0] * 7, [0.7] * 7, duration_s=2.0, sample_count=7
    )
    check = {
        "kind": "scan",
        "trajectory": trajectory,
        "trajectory_start": RobotState(joint_state=JointState()),
    }
    harness = SimpleNamespace(
        direct_check=check,
        robot_model_id="UF_ROBOT",
        display_publisher=SimpleNamespace(publish=Mock()),
        planned_trajectory=None,
        planned_kind=None,
        planned_type=None,
        pending={"kind": "scan"},
        get_logger=lambda: SimpleNamespace(warning=Mock()),
        _publish_status=Mock(),
    )
    SimplePreinsertNode._accept_direct_trajectory(harness)
    display = harness.display_publisher.publish.call_args.args[0]
    assert display.trajectory[0] is trajectory
    assert harness.planned_trajectory is trajectory
    assert harness.planned_type == "direct_joint"

    send = Mock()
    harness.allow_execution = True
    harness.shadow_full_sequence = False
    harness.phase = "awaiting_execute_confirmation"
    harness.executing_kind = None
    harness.executing_type = None
    harness.execution_client = None
    harness.direct_execution_client = SimpleNamespace(
        wait_for_server=Mock(return_value=True)
    )
    harness.get_parameter = lambda _name: SimpleNamespace(value=True)
    harness._send_direct_execution = send
    harness._fail = Mock()
    response = SimpleNamespace(success=None, message="")
    SimplePreinsertNode._execute_trigger_callback(harness, None, response)
    assert response.success is True
    assert send.call_args.args[0] is trajectory


def test_direct_action_goal_contains_the_exact_reviewed_joint_trajectory():
    trajectory = build_direct_joint_trajectory(
        JOINT_NAMES, [0.0] * 7, [0.4] * 7, duration_s=2.0, sample_count=4
    )
    submitted = []

    class Client:
        def send_goal_async(self, goal):
            submitted.append(goal)
            return SimpleNamespace(add_done_callback=Mock())

    harness = SimpleNamespace(
        direct_execution_client=Client(),
        _publish_status=Mock(),
        _direct_execution_goal_callback=Mock(),
    )
    SimplePreinsertNode._send_direct_execution(harness, trajectory)
    assert submitted[0].trajectory is trajectory.joint_trajectory
    assert [
        list(point.positions) for point in submitted[0].trajectory.points
    ] == [list(point.positions) for point in trajectory.joint_trajectory.points]


def test_direct_collision_verification_checks_each_exact_sample():
    trajectory = build_direct_joint_trajectory(
        JOINT_NAMES, [0.0] * 7, [0.2] * 7, duration_s=2.0, sample_count=2
    )
    calls = []

    class Client:
        def call_async(self, request):
            calls.append(list(request.robot_state.joint_state.position))
            return SimpleNamespace(add_done_callback=Mock())

    states = [
        RobotState(joint_state=JointState(name=JOINT_NAMES, position=[0.0] * 7))
    ] + [
        RobotState(
            joint_state=JointState(name=JOINT_NAMES, position=point.positions)
        )
        for point in trajectory.joint_trajectory.points
    ]
    harness = SimpleNamespace(
        direct_check={"states": states, "index": 0},
        group_name="xarm7",
        state_validity_client=Client(),
        _accept_direct_trajectory=Mock(),
        _direct_state_check_callback=Mock(),
    )
    for index, state in enumerate(states):
        harness.direct_check["index"] = index
        SimplePreinsertNode._request_next_direct_state_check(harness)
        assert calls[-1] == list(state.joint_state.position)


def test_fixed_joint_pose_does_not_call_moveit_planner():
    scene_future = SimpleNamespace(add_done_callback=Mock())
    harness = SimpleNamespace(
        phase="waiting_for_slot",
        latest_joint_state=JointState(name=JOINT_NAMES, position=[0.0] * 7),
        latest_joint_state_ns=1,
        state_validity_client=SimpleNamespace(wait_for_service=Mock(return_value=True)),
        planning_scene_client=SimpleNamespace(
            wait_for_service=Mock(return_value=True),
            call_async=Mock(return_value=scene_future),
        ),
        scene_client=SimpleNamespace(wait_for_service=Mock(return_value=False)),
        book_detach_pending=False,
        plan_client=Mock(),
        planned_trajectory=None,
        planned_kind=None,
        planned_type=None,
        executing_kind=None,
        executing_type=None,
        pending=None,
        _fresh=Mock(return_value=True),
        _planning_scene_request=Mock(return_value=object()),
        _publish_status=Mock(),
        _direct_scene_state_response_callback=Mock(),
        _prepare_direct_joint_trajectory=Mock(),
    )
    response = SimpleNamespace(success=None, message="")
    SimplePreinsertNode._plan_joint_pose(harness, "loading", [0.1] * 7, response)
    assert response.success is True
    harness.planning_scene_client.call_async.assert_called_once_with(
        harness._planning_scene_request.return_value
    )
    scene_future.add_done_callback.assert_called_once_with(
        harness._direct_scene_state_response_callback
    )
    harness.plan_client.assert_not_called()


def test_direct_failure_publishes_kind_before_clearing_stale_trajectory():
    observed = []
    harness = SimpleNamespace(
        pending={"kind": "loading"},
        branch_search=None,
        direct_check={"kind": "loading"},
        planned_trajectory=object(),
        planned_kind=None,
        planned_type=None,
        executing_kind=None,
        executing_type=None,
    )

    def publish(phase, reason=None):
        observed.append((phase, reason, harness.executing_kind, harness.executing_type))

    harness._publish_status = publish
    SimplePreinsertNode._fail(harness, "collision")
    assert observed == [("failed", "collision", "loading", "direct_joint")]
    assert harness.planned_trajectory is None
    assert harness.executing_kind is None


def _contact(first, second):
    return SimpleNamespace(contact_body_1=first, contact_body_2=second)


def _return_contact_harness():
    logger = SimpleNamespace(warning=Mock())
    harness = SimpleNamespace(
        direct_check={
            "kind": "return_loading",
            "index": 0,
            "states": [object()] * 5,
        },
        get_parameter=lambda name: SimpleNamespace(value={
            "held_book_collision_id": "bookshelf_simple_held_book",
        }[name]),
        get_logger=lambda: logger,
    )
    harness.logger = logger
    harness._collision_failure_message = lambda prefix, sample, pairs: (
        f"{prefix} at sample {sample}; "
        + ",".join("<->".join(sorted(pair)) for pair in pairs)
    )
    harness._pair_labels = SimplePreinsertNode._pair_labels
    return harness


def test_return_ignores_missing_or_stale_released_book_scene_contacts():
    harness = _return_contact_harness()
    contacts = [
        _contact("left_finger", "bookshelf_simple_held_book"),
        _contact("bookshelf_simple_held_book", "bookshelf_shelf"),
    ]
    from bookshelf_simple_experiment_ros.preinsert_node import collision_pairs
    result = SimpleNamespace(valid=False, contacts=contacts)
    assert SimplePreinsertNode._return_contact_error(
        harness, result, collision_pairs(result)
    ) is None
    harness.logger.warning.assert_called_once()


def test_return_rejects_new_shelf_collision():
    harness = _return_contact_harness()
    from bookshelf_simple_experiment_ros.preinsert_node import collision_pairs
    shelf = SimpleNamespace(
        valid=False, contacts=[_contact("left_finger", "bookshelf_shelf")]
    )
    error = SimplePreinsertNode._return_contact_error(
        harness, shelf, collision_pairs(shelf)
    )
    assert "invalid/colliding" in error
    assert "bookshelf_shelf" in error


def test_return_rejects_self_collision():
    harness = _return_contact_harness()
    from bookshelf_simple_experiment_ros.preinsert_node import collision_pairs
    collision = SimpleNamespace(
        valid=False, contacts=[_contact("link3", "link5")]
    )
    error = SimplePreinsertNode._return_contact_error(
        harness, collision, collision_pairs(collision)
    )
    assert "invalid/colliding" in error
    assert "link3" in error


def test_h_does_not_require_released_book_scene_state():
    logger = SimpleNamespace(warning=Mock())
    harness = SimpleNamespace(
        phase="waiting_for_slot",
        latest_joint_state=JointState(name=JOINT_NAMES, position=[0.0] * 7),
        latest_joint_state_ns=1,
        state_validity_client=SimpleNamespace(wait_for_service=Mock(return_value=True)),
        planning_scene_client=SimpleNamespace(wait_for_service=Mock(), call_async=Mock()),
        book_detach_pending=True,
        book_scene_transition_state="warning",
        planned_trajectory=None,
        planned_kind=None,
        planned_type=None,
        executing_kind=None,
        executing_type=None,
        pending=None,
        _fresh=Mock(return_value=True),
        get_logger=lambda: logger,
        _prepare_direct_joint_trajectory=Mock(),
    )
    response = SimpleNamespace(success=None, message="")
    SimplePreinsertNode._plan_joint_pose(
        harness, "return_loading", [0.1] * 7, response
    )
    assert response.success is True
    harness._prepare_direct_joint_trajectory.assert_called_once_with()
    harness.planning_scene_client.wait_for_service.assert_not_called()
    harness.planning_scene_client.call_async.assert_not_called()
    logger.warning.assert_called_once()


def test_e_executes_exact_reviewed_h_trajectory():
    trajectory = build_direct_joint_trajectory(
        JOINT_NAMES, [0.0] * 7, [0.3] * 7, duration_s=2.0, sample_count=4
    )
    send = Mock()
    harness = SimpleNamespace(
        shadow_full_sequence=False,
        allow_execution=True,
        phase="awaiting_execute_confirmation",
        planned_trajectory=trajectory,
        planned_kind="return_loading",
        planned_type="direct_joint",
        executing_kind=None,
        executing_type=None,
        direct_execution_client=SimpleNamespace(
            wait_for_server=Mock(return_value=True)
        ),
        execution_client=None,
        get_parameter=lambda _name: SimpleNamespace(value=True),
        _send_direct_execution=send,
        _fail=Mock(),
    )
    response = SimpleNamespace(success=None, message="")
    SimplePreinsertNode._execute_trigger_callback(harness, None, response)
    assert response.success is True
    assert send.call_args.args[0] is trajectory


def test_release_scene_update_failure_is_warning_only():
    transition = Mock()
    preinsert_event_harness = SimpleNamespace(
        get_parameter=lambda name: SimpleNamespace(value={
            "attach_book_collision": True,
        }[name]),
        _begin_release_detach=transition,
    )
    SimplePreinsertNode._policy_status_callback(
        preinsert_event_harness, String(data='{"phase":"release_complete"}')
    )
    transition.assert_called_once_with()

    statuses = []
    logger = SimpleNamespace(warning=Mock())
    warning_harness = SimpleNamespace(
        book_detach_pending=True,
        book_scene_transition_state="transitioning",
        get_logger=lambda: logger,
        _publish_status=lambda phase, reason=None: statuses.append((phase, reason)),
    )
    SimplePreinsertNode._book_scene_transition_warning(
        warning_harness, "world add missing"
    )
    assert warning_harness.book_detach_pending is False
    assert warning_harness.book_scene_transition_state == "warning"
    assert statuses == [("book_scene_transition_warning", "world add missing")]
    logger.warning.assert_called_once()
