from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from bookshelf_simple_experiment_ros.execution_gate import hardware_commands_allowed
from bookshelf_simple_experiment_ros.operator_action_node import (
    GRIPPER_COMMAND,
    GRIPPER_TRAJECTORY,
    OperatorActionNode,
    log_operator_message,
    make_gripper_goal,
)
from bookshelf_simple_experiment_ros.preinsert_node import SimplePreinsertNode


PACKAGE = Path(__file__).resolve().parents[1]


def test_shadow_overrides_every_execution_request():
    assert hardware_commands_allowed(True, True) is False
    assert hardware_commands_allowed(False, True) is False
    assert hardware_commands_allowed(True, False) is True


def test_fake_and_real_gripper_goal_types_are_preserved():
    assert make_gripper_goal(GRIPPER_COMMAND, 0.85, 0.0, 2.0).command.position == 0.85
    fake = make_gripper_goal(GRIPPER_TRAJECTORY, 0.0, 0.0, 2.0)
    assert fake.trajectory.joint_names == ["drive_joint"]


def test_operator_logging_has_fixed_call_sites():
    logger = SimpleNamespace(info=Mock(), error=Mock())
    log_operator_message(logger, True, "ok")
    log_operator_message(logger, False, "bad")
    logger.info.assert_called_once_with("ok")
    logger.error.assert_called_once_with("bad")


def test_operator_helper_has_no_arm_command_interface():
    source = (PACKAGE / "bookshelf_simple_experiment_ros" / "operator_action_node.py").read_text()
    assert "ExecuteTrajectory" not in source
    assert "trajectory_action" not in source
    assert "/bookshelf_simple/finish_return" in source


def test_failed_post_return_open_never_marks_ready():
    harness = SimpleNamespace(busy="return_open", _publish=Mock())
    OperatorActionNode._gripper_failed(harness, "return_open", "failed")
    actions = [call.args[0] for call in harness._publish.call_args_list]
    assert actions == ["return_open", "return_failed"]
    assert "ready" not in actions


def test_shadow_execute_suppresses_stored_trajectory():
    harness = SimpleNamespace(
        shadow_full_sequence=True,
        phase="awaiting_execute_confirmation",
        planned_trajectory=object(),
        planned_kind="scan",
        planned_type="direct_joint",
        executing_kind=None,
        executing_type=None,
        direct_execution_client=None,
        execution_client=None,
        allow_execution=False,
        get_parameter=lambda name: SimpleNamespace(
            value={"separate_execution_confirmation": True}[name]
        ),
        get_logger=lambda: SimpleNamespace(warning=Mock()),
        _publish_status=Mock(),
    )
    response = SimpleNamespace(success=None, message="")
    SimplePreinsertNode._execute_trigger_callback(harness, None, response)
    assert response.success is True
    assert harness.planned_trajectory is None
    assert harness.execution_client is None
    assert [call.args[0] for call in harness._publish_status.call_args_list] == [
        "executing", "done"
    ]


def test_real_shadow_launch_has_no_fake_state_and_gate_reaches_all_nodes():
    launch = (PACKAGE / "launch" / "real_experiment_operator.launch.py").read_text()
    assert "fake_policy_start" not in launch
    assert "virtual_trigger" not in launch
    assert "static_transform_publisher" not in launch
    assert launch.count('"shadow_full_sequence"') >= 4


def test_offline_rehearsal_executes_fake_interfaces_not_shadow():
    launch = (PACKAGE / "launch" / "offline_full_sequence_rehearsal.launch.py").read_text()
    assert '"allow_execution": "true"' in launch
    assert '"execute": "true"' in launch
    assert '"shadow_full_sequence": "false"' in launch
    assert '"command_scale": "1.0"' in launch


def test_offline_rehearsal_feeds_production_per_grasp_tf_path_only():
    offline = (
        PACKAGE / "launch" / "offline_full_sequence_rehearsal.launch.py"
    ).read_text()
    real = (PACKAGE / "launch" / "real_experiment_operator.launch.py").read_text()

    assert 'name="dry_run_mock_held_book_tf"' in offline
    assert '"link_eef"' in offline
    assert '"target_book_center"' in offline
    assert "per_grasp_eef_book" not in offline
    assert "dry_run_mock_held_book_tf" not in real
