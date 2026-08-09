#!/usr/bin/env python3
"""Execute one recent collision-checked policy-tool plan after explicit approval."""

import json
import math

from moveit_msgs.action import ExecuteTrajectory
from rclpy.action import ActionClient
import rclpy
from std_msgs.msg import String

from .policy_tool_control_math import (
    OneShotExecutionGuard,
    execution_authorization_error,
    maximum_named_joint_difference,
)
from .policy_tool_planner_base import PlanSnapshot, PolicyToolPlannerBase


class GuardedPolicyToolExecutorNode(PolicyToolPlannerBase):
    """Fail-closed, token-gated, one-plan-at-a-time trajectory executor."""

    def __init__(self):
        super().__init__("guarded_policy_tool_executor")
        self.declare_parameter("dry_run", True)
        self.declare_parameter("allow_execution", False)
        self.declare_parameter("approval_token", "DISABLED")
        self.declare_parameter("approval_topic", "/bookshelf_guarded/approve_once")
        self.declare_parameter("execution_action", "/execute_trajectory")
        self.declare_parameter("maximum_plan_age_s", 1.0)
        self.declare_parameter("maximum_start_joint_drift_rad", 0.02)
        self.declare_parameter("result_topic", "/bookshelf_guarded/execution_report")

        self.execution_client = None
        if self._execution_interface_allowed_at_startup():
            self.execution_client = ActionClient(
                self,
                ExecuteTrajectory,
                str(self.get_parameter("execution_action").value),
            )
        self.execution_report_publisher = self.create_publisher(
            String,
            str(self.get_parameter("result_topic").value),
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("approval_topic").value),
            self._approval_callback,
            10,
        )
        self.execution_busy = False
        self.execution_guard = OneShotExecutionGuard()

        if self.execution_client is None:
            self.get_logger().warning(
                "GUARDED EXECUTOR started without an execution action client. "
                "Startup gates are closed; this process cannot command motion."
            )
        else:
            self.get_logger().warning(
                "GUARDED EXECUTOR command interface is active. Every accepted "
                "token is consumed after exactly one recent checked plan."
            )
        self.get_logger().warning(
            "No gripper interface exists in this process. Release requests always fail closed."
        )

    def _execution_interface_allowed_at_startup(self) -> bool:
        status = str(self.get_parameter("expected_policy_tool_status").value).lower()
        transform_allowed = status.startswith(("verified_", "validated_")) or bool(
            self.get_parameter("allow_unverified_policy_tool").value
        )
        token = str(self.get_parameter("approval_token").value)
        token_configured = bool(token) and token not in ("CHANGE_ME", "DISABLED")
        return (
            not bool(self.get_parameter("dry_run").value)
            and bool(self.get_parameter("allow_execution").value)
            and bool(self.get_parameter("planning_scene_complete").value)
            and transform_allowed
            and token_configured
        )

    def _approval_callback(self, message: String):
        now_ns = self._now_ns()
        plan_age = (
            None
            if self.latest_plan is None
            else (now_ns - self.latest_plan.generated_ns) * 1.0e-9
        )
        error = execution_authorization_error(
            dry_run=bool(self.get_parameter("dry_run").value),
            allow_execution=bool(self.get_parameter("allow_execution").value),
            planning_scene_complete=bool(
                self.get_parameter("planning_scene_complete").value
            ),
            approval_token=message.data,
            configured_token=str(self.get_parameter("approval_token").value),
            plan_age_s=plan_age,
            maximum_plan_age_s=float(
                self.get_parameter("maximum_plan_age_s").value
            ),
            plan_valid=self.latest_plan is not None,
            busy=self.execution_busy,
            execution_consumed=self.execution_guard.consumed,
        )
        if error is None:
            error = self._input_error()
        if error is None:
            error = self._start_state_error(self.latest_plan)
        if error:
            self._publish_execution_report(
                {
                    "accepted": False,
                    "hardware_commanded": False,
                    "reason": error,
                },
                warning=True,
            )
            return
        if self.execution_client is None:
            self._publish_execution_report(
                {
                    "accepted": False,
                    "hardware_commanded": False,
                    "reason": (
                        "execution action client was not created at startup; "
                        "restart with a separately reviewed configuration"
                    ),
                },
                warning=True,
            )
            return
        if not self.execution_client.wait_for_server(timeout_sec=0.25):
            self._publish_execution_report(
                {
                    "accepted": False,
                    "hardware_commanded": False,
                    "reason": "MoveIt execute_trajectory action is unavailable",
                },
                warning=True,
            )
            return

        # Consume the process-lifetime allowance before submitting the goal. It
        # is deliberately never restored, including when submission or
        # execution fails. try_consume() is atomic so this remains one-shot if
        # callbacks are ever dispatched by a multi-threaded executor.
        if not self.execution_guard.try_consume():
            self._publish_execution_report(
                {
                    "accepted": False,
                    "hardware_commanded": False,
                    "approval_consumed": True,
                    "execution_count": self.execution_guard.execution_count,
                    "reason": (
                        "the one-execution-per-process allowance has already "
                        "been consumed"
                    ),
                },
                warning=True,
            )
            return

        plan = self.latest_plan
        self.latest_plan = None
        self.execution_busy = True
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = plan.trajectory
        try:
            future = self.execution_client.send_goal_async(goal)
        except Exception as error:
            self.execution_busy = False
            self._publish_execution_report(
                {
                    "accepted": True,
                    "hardware_commanded": False,
                    "target_id": plan.target.target_id,
                    "approval_consumed": True,
                    "execution_count": self.execution_guard.execution_count,
                    "reason": f"trajectory goal submission failed: {error}",
                },
                warning=True,
            )
            return
        future.add_done_callback(
            lambda result_future: self._goal_response_callback(result_future, plan)
        )
        self._publish_execution_report(
            {
                "accepted": True,
                "hardware_commanded": True,
                "target_id": plan.target.target_id,
                "approval_consumed": True,
                "execution_count": self.execution_guard.execution_count,
                "reason": "one approved MoveIt trajectory submitted",
            }
        )

    def _start_state_error(self, plan: PlanSnapshot) -> str | None:
        if plan is None or self.latest_joint_state is None:
            return "plan or current joint state is unavailable"
        try:
            difference = maximum_named_joint_difference(
                self.latest_joint_state.name,
                self.latest_joint_state.position,
                plan.trajectory_start.joint_state.name,
                plan.trajectory_start.joint_state.position,
            )
        except ValueError as error:
            return str(error)
        maximum = float(self.get_parameter("maximum_start_joint_drift_rad").value)
        if difference > maximum:
            return (
                f"current joints drifted {difference:.6f} rad from the plan start; "
                f"limit is {maximum:.6f} rad"
            )
        return None

    def _goal_response_callback(self, future, plan: PlanSnapshot):
        try:
            goal_handle = future.result()
        except Exception as error:
            self.execution_busy = False
            self._publish_execution_report(
                {
                    "accepted": False,
                    "hardware_commanded": False,
                    "target_id": plan.target.target_id,
                    "approval_consumed": True,
                    "execution_count": self.execution_guard.execution_count,
                    "reason": f"trajectory goal submission failed: {error}",
                },
                warning=True,
            )
            return
        if not goal_handle.accepted:
            self.execution_busy = False
            self._publish_execution_report(
                {
                    "accepted": False,
                    "hardware_commanded": False,
                    "target_id": plan.target.target_id,
                    "approval_consumed": True,
                    "execution_count": self.execution_guard.execution_count,
                    "reason": "MoveIt rejected the trajectory goal",
                },
                warning=True,
            )
            return
        try:
            result_future = goal_handle.get_result_async()
        except Exception as error:
            self.execution_busy = False
            self._publish_execution_report(
                {
                    "accepted": True,
                    "hardware_commanded": True,
                    "target_id": plan.target.target_id,
                    "approval_consumed": True,
                    "execution_count": self.execution_guard.execution_count,
                    "reason": f"trajectory result request failed: {error}",
                },
                warning=True,
            )
            return
        result_future.add_done_callback(
            lambda value: self._execution_result_callback(value, plan)
        )

    def _execution_result_callback(self, future, plan: PlanSnapshot):
        self.execution_busy = False
        try:
            wrapped = future.result()
            error_code = int(wrapped.result.error_code.val)
            status = int(wrapped.status)
        except Exception as error:
            self._publish_execution_report(
                {
                    "accepted": True,
                    "hardware_commanded": True,
                    "target_id": plan.target.target_id,
                    "approval_consumed": True,
                    "execution_count": self.execution_guard.execution_count,
                    "reason": f"trajectory result failed: {error}",
                },
                warning=True,
            )
            return
        self._publish_execution_report(
            {
                "accepted": True,
                "hardware_commanded": True,
                "target_id": plan.target.target_id,
                "moveit_error_code": error_code,
                "action_status": status,
                "execution_count": self.execution_guard.execution_count,
                "approval_consumed": True,
                "reason": "trajectory execution completed",
            },
            warning=error_code != 1,
        )

    def _on_invalid_plan(self, reason: str):
        if not self.execution_busy:
            self.latest_plan = None

    def _publish_execution_report(self, value: dict, warning=False):
        value = {
            "dry_run": bool(self.get_parameter("dry_run").value),
            "allow_execution": bool(self.get_parameter("allow_execution").value),
            "planning_scene_complete": bool(
                self.get_parameter("planning_scene_complete").value
            ),
            "gripper_command_interface": False,
            "release_executed": False,
            "execution_action_client_created": self.execution_client is not None,
            **value,
        }
        message = json.dumps(value, sort_keys=True)
        self.execution_report_publisher.publish(String(data=message))
        if warning:
            self.get_logger().warning(message)
        else:
            self.get_logger().info(message)


def main(args=None):
    rclpy.init(args=args)
    node = GuardedPolicyToolExecutorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
