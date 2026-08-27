from types import SimpleNamespace
from unittest.mock import Mock

from moveit_msgs.msg import MoveItErrorCodes, RobotState, RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

from bookshelf_simple_experiment_ros.preinsert_node import SimplePreinsertNode


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
        pending={"request": "active"},
        get_parameter=lambda _name: SimpleNamespace(value=True),
        get_logger=lambda: logger,
        _publish_status=Mock(),
        _send_execution=Mock(),
        _fail=Mock(),
    )

    SimplePreinsertNode._plan_response_callback(harness, _successful_plan_result())

    assert harness.planned_trajectory is not None
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
        get_parameter=lambda _name: SimpleNamespace(value=True),
        _send_execution=Mock(),
    )
    response = SimpleNamespace(success=None, message="")

    returned = SimplePreinsertNode._execute_trigger_callback(harness, None, response)

    assert returned is response
    assert response.success is False
    assert response.message == "no confirmed MoveIt plan is awaiting execution"
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
    plan_client.assert_not_called()
    ik_client.assert_not_called()
