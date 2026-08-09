from types import SimpleNamespace

from moveit_msgs.msg import RobotTrajectory
from std_msgs.msg import String

from bookshelf_guarded_control_ros.guarded_policy_tool_executor_node import (
    GuardedPolicyToolExecutorNode,
)
from bookshelf_guarded_control_ros.policy_tool_control_math import OneShotExecutionGuard


class _SubmissionFailureClient:
    def wait_for_server(self, timeout_sec):
        return True

    def send_goal_async(self, goal):
        raise RuntimeError("submission failed")


def _plan(generated_ns=1_000_000_000):
    return SimpleNamespace(
        generated_ns=generated_ns,
        target=SimpleNamespace(target_id="target-01"),
        trajectory=RobotTrajectory(),
    )


def _executor_without_ros_runtime():
    node = GuardedPolicyToolExecutorNode.__new__(GuardedPolicyToolExecutorNode)
    parameter_values = {
        "dry_run": False,
        "allow_execution": True,
        "planning_scene_complete": True,
        "approval_token": "trial-01",
        "maximum_plan_age_s": 1.0,
    }
    node.get_parameter = lambda name: SimpleNamespace(value=parameter_values[name])
    node._now_ns = lambda: 1_100_000_000
    node._input_error = lambda: None
    node._start_state_error = lambda plan: None
    node.execution_client = _SubmissionFailureClient()
    node.execution_busy = False
    node.execution_guard = OneShotExecutionGuard()
    node.latest_plan = _plan()
    node.reports = []
    node._publish_execution_report = (
        lambda value, warning=False: node.reports.append((value, warning))
    )
    return node


def test_submission_failure_still_permanently_consumes_process_allowance():
    node = _executor_without_ros_runtime()

    node._approval_callback(String(data="trial-01"))

    assert node.execution_guard.consumed is True
    assert node.execution_guard.execution_count == 1
    assert node.reports[-1][0]["accepted"] is True
    assert node.reports[-1][0]["approval_consumed"] is True
    assert "submission failed" in node.reports[-1][0]["reason"]

    node.latest_plan = _plan()
    node._approval_callback(String(data="trial-01"))

    assert node.execution_guard.execution_count == 1
    assert node.reports[-1][0]["accepted"] is False
    assert "one-execution-per-process" in node.reports[-1][0]["reason"]
