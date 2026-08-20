#!/usr/bin/env python3
"""Execute one reviewed global pre-insertion trajectory after explicit approval."""

from __future__ import annotations

import json
import math
import re

from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import RobotTrajectory
from rclpy.action import ActionClient
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String

from .policy_tool_control_math import (
    OneShotExecutionGuard,
    TRAJECTORY_FINGERPRINT_KIND,
    canonical_ros_message_sha256,
    maximum_named_joint_difference,
)


class GuardedPreinsertExecutorNode(Node):
    """Token-gated, one-shot consumer of a validated pre-insertion plan."""

    def __init__(self):
        super().__init__("guarded_preinsert_executor")
        self._declare_parameters()
        self.execution_client = None
        if self._startup_gates_open():
            self.execution_client = ActionClient(
                self,
                ExecuteTrajectory,
                str(self.get_parameter("execution_action").value),
            )

        self.latest_plan_valid = False
        self.latest_report = None
        self.latest_trajectory = None
        self.latest_trajectory_sha256 = None
        self.latest_joint_state = None
        self.latest_joint_state_ns = None
        self.latest_scene_status = None
        self.latest_scene_status_ns = None
        self.plan_first_seen_ns = None
        self.plan_target_id = None
        self.execution_busy = False
        self.execution_guard = OneShotExecutionGuard()

        latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("plan_valid_topic").value),
            self._plan_valid_callback,
            latched,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("plan_report_topic").value),
            self._plan_report_callback,
            latched,
        )
        self.create_subscription(
            RobotTrajectory,
            str(self.get_parameter("planned_trajectory_topic").value),
            self._trajectory_callback,
            latched,
        )
        self.create_subscription(
            JointState,
            str(self.get_parameter("joint_states_topic").value),
            self._joint_state_callback,
            20,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("scene_status_topic").value),
            self._scene_status_callback,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("approval_topic").value),
            self._approval_callback,
            10,
        )
        self.result_publisher = self.create_publisher(
            String,
            str(self.get_parameter("result_topic").value),
            latched,
        )

        if self.execution_client is None:
            self.get_logger().warning(
                "GLOBAL PRE-INSERT EXECUTOR has no action client because startup "
                "gates are closed. This process cannot command motion."
            )
        else:
            self.get_logger().warning(
                "GLOBAL PRE-INSERT EXECUTOR action client is active. Exactly one "
                "fresh reviewed trajectory may be submitted after token approval."
            )
        self.get_logger().warning("This process has no gripper interface.")

    def _declare_parameters(self):
        self.declare_parameter("dry_run", True)
        self.declare_parameter("allow_execution", False)
        self.declare_parameter("planning_scene_complete", False)
        self.declare_parameter("human_trajectory_review_complete", False)
        self.declare_parameter("target_transform_physically_validated", False)
        self.declare_parameter("approval_token", "DISABLED")
        self.declare_parameter("expected_scene_config_sha256", "DISABLED")
        self.declare_parameter("expected_target_transform_status", "DISABLED")
        self.declare_parameter("required_scene_mode", "global_approach")
        self.declare_parameter(
            "required_planning_sequence",
            "seeded_collision_aware_ik_then_joint_goal_plan",
        )
        self.declare_parameter("maximum_plan_age_s", 120.0)
        self.declare_parameter("maximum_joint_state_age_s", 0.50)
        self.declare_parameter("maximum_scene_status_age_s", 1.0)
        self.declare_parameter("maximum_start_joint_drift_rad", 0.02)
        self.declare_parameter("minimum_trajectory_point_count", 2)
        self.declare_parameter("execution_action", "/execute_trajectory")
        self.declare_parameter("approval_topic", "/bookshelf_preinsert/approve_once")
        self.declare_parameter("result_topic", "/bookshelf_preinsert/execution_report")
        self.declare_parameter("plan_valid_topic", "/bookshelf_preinsert/plan_valid")
        self.declare_parameter("plan_report_topic", "/bookshelf_preinsert/plan_report")
        self.declare_parameter(
            "planned_trajectory_topic", "/bookshelf_preinsert/planned_trajectory"
        )
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("scene_status_topic", "/bookshelf_scene/status")

    @staticmethod
    def _configured(value: str) -> bool:
        return bool(value) and value not in ("CHANGE_ME", "DISABLED")

    @staticmethod
    def _sha256_configured(value: str) -> bool:
        return re.fullmatch(r"[0-9a-f]{64}", value) is not None

    def _startup_gates_open(self) -> bool:
        token = str(self.get_parameter("approval_token").value)
        scene_sha = str(self.get_parameter("expected_scene_config_sha256").value)
        status = str(self.get_parameter("expected_target_transform_status").value)
        return (
            not bool(self.get_parameter("dry_run").value)
            and bool(self.get_parameter("allow_execution").value)
            and bool(self.get_parameter("planning_scene_complete").value)
            and bool(self.get_parameter("human_trajectory_review_complete").value)
            and bool(
                self.get_parameter("target_transform_physically_validated").value
            )
            and self._configured(token)
            and self._sha256_configured(scene_sha)
            and self._configured(status)
        )

    def _now_ns(self) -> int:
        return int(self.get_clock().now().nanoseconds)

    def _plan_valid_callback(self, message: Bool):
        self.latest_plan_valid = bool(message.data)

    def _plan_report_callback(self, message: String):
        try:
            report = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            self.latest_report = None
            return
        if not isinstance(report, dict):
            self.latest_report = None
            return
        target_id = report.get("target_id")
        if bool(report.get("valid")) and target_id != self.plan_target_id:
            self.plan_target_id = target_id
            self.plan_first_seen_ns = self._now_ns()
        self.latest_report = report

    def _trajectory_callback(self, message: RobotTrajectory):
        self.latest_trajectory = message
        try:
            self.latest_trajectory_sha256 = canonical_ros_message_sha256(message)
        except (TypeError, ValueError) as error:
            self.latest_trajectory_sha256 = None
            self.get_logger().warning(
                f"Received trajectory cannot be fingerprinted safely: {error}"
            )

    def _joint_state_callback(self, message: JointState):
        self.latest_joint_state = message
        self.latest_joint_state_ns = self._now_ns()

    def _scene_status_callback(self, message: String):
        try:
            status = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            self.latest_scene_status = None
            self.latest_scene_status_ns = None
            return
        if not isinstance(status, dict):
            self.latest_scene_status = None
            self.latest_scene_status_ns = None
            return
        self.latest_scene_status = status
        self.latest_scene_status_ns = self._now_ns()

    def _approval_error(self, supplied_token: str) -> str | None:
        if not self._startup_gates_open():
            return "executor startup gates are closed"
        if supplied_token != str(self.get_parameter("approval_token").value):
            return "approval token mismatch"
        if self.execution_busy:
            return "trajectory execution is already in progress"
        if self.execution_guard.consumed:
            return "the one-execution-per-process allowance has already been consumed"
        if not self.latest_plan_valid or self.latest_report is None:
            return "no valid pre-insertion plan report is available"
        if self.latest_trajectory is None or self.latest_joint_state is None:
            return "trajectory or current joint state is unavailable"
        if self.latest_joint_state_ns is None:
            return "current joint state age is unavailable"
        joint_age_s = (self._now_ns() - self.latest_joint_state_ns) * 1.0e-9
        if joint_age_s > float(
            self.get_parameter("maximum_joint_state_age_s").value
        ):
            return f"current joint state is stale ({joint_age_s:.3f} s)"
        if self.latest_scene_status is None or self.latest_scene_status_ns is None:
            return "live planning scene status is unavailable"
        scene_age_s = (self._now_ns() - self.latest_scene_status_ns) * 1.0e-9
        if scene_age_s > float(
            self.get_parameter("maximum_scene_status_age_s").value
        ):
            return f"live planning scene status is stale ({scene_age_s:.3f} s)"
        if self.plan_first_seen_ns is None:
            return "plan age is unavailable"
        age_s = (self._now_ns() - self.plan_first_seen_ns) * 1.0e-9
        if age_s > float(self.get_parameter("maximum_plan_age_s").value):
            return f"reviewed plan is stale ({age_s:.3f} s)"

        report = self.latest_report
        required_values = (
            (report.get("valid") is True, "plan report is not valid"),
            (report.get("path_planned") is True, "MoveIt path was not planned"),
            (report.get("collision_checked") is True, "path is not collision checked"),
            (report.get("hardware_commanded") is False, "plan-only provenance is invalid"),
            ((report.get("ik_joint_branch") or {}).get("passed") is True, "IK branch check did not pass"),
            ((report.get("trajectory_sanity") or {}).get("passed") is True, "trajectory sanity did not pass"),
        )
        for passed, reason in required_values:
            if not passed:
                return reason
        required_sequence = str(
            self.get_parameter("required_planning_sequence").value
        )
        if report.get("planning_sequence") != required_sequence:
            return "planning sequence does not match the reviewed executor configuration"
        if report.get("trajectory_fingerprint_kind") != TRAJECTORY_FINGERPRINT_KIND:
            return "trajectory fingerprint scheme does not match the executor"
        if report.get("trajectory_sha256") != self.latest_trajectory_sha256:
            return "trajectory does not match the reviewed plan report"

        scene = report.get("scene_status") or {}
        if scene.get("scene_applied") is not True:
            return "global planning scene was not applied"
        if scene.get("mode") != str(self.get_parameter("required_scene_mode").value):
            return "planning scene mode is not global_approach"
        objects = scene.get("objects") or {}
        if not all(objects.get(name) is True for name in ("bookshelf_keepout", "held_book", "table")):
            return "required global collision objects are incomplete"
        expected_scene_sha = str(
            self.get_parameter("expected_scene_config_sha256").value
        )
        if (scene.get("scene_config") or {}).get("sha256") != expected_scene_sha:
            return "planning scene configuration hash mismatch"
        live_scene = self.latest_scene_status
        if live_scene.get("scene_applied") is not True:
            return "live global planning scene is not applied"
        if live_scene.get("mode") != str(
            self.get_parameter("required_scene_mode").value
        ):
            return "live planning scene mode is not global_approach"
        live_objects = live_scene.get("objects") or {}
        if not all(
            live_objects.get(name) is True
            for name in ("bookshelf_keepout", "held_book", "table")
        ):
            return "live global collision objects are incomplete"
        if (live_scene.get("scene_config") or {}).get("sha256") != expected_scene_sha:
            return "live planning scene configuration hash mismatch"
        debug = report.get("target_calculator_debug") or {}
        expected_status = str(
            self.get_parameter("expected_target_transform_status").value
        )
        if debug.get("policy_tool_transform_status") != expected_status:
            return "target transform status does not match reviewed configuration"

        trajectory = self.latest_trajectory.joint_trajectory
        if len(trajectory.points) < int(
            self.get_parameter("minimum_trajectory_point_count").value
        ):
            return "trajectory has too few points"
        try:
            drift = maximum_named_joint_difference(
                self.latest_joint_state.name,
                self.latest_joint_state.position,
                trajectory.joint_names,
                trajectory.points[0].positions,
            )
        except ValueError as error:
            return str(error)
        maximum_drift = float(
            self.get_parameter("maximum_start_joint_drift_rad").value
        )
        if not math.isfinite(drift) or drift > maximum_drift:
            return (
                f"current joints drifted {drift:.6f} rad from the plan start; "
                f"limit is {maximum_drift:.6f} rad"
            )
        return None

    def _approval_callback(self, message: String):
        error = self._approval_error(message.data)
        if error:
            self._publish_result(False, False, error, warning=True)
            return
        if self.execution_client is None:
            self._publish_result(
                False,
                False,
                "execution action client was not created at startup",
                warning=True,
            )
            return
        if not self.execution_client.wait_for_server(timeout_sec=0.25):
            self._publish_result(
                False, False, "MoveIt execute_trajectory action is unavailable", True
            )
            return
        if not self.execution_guard.try_consume():
            self._publish_result(
                False, False, "one-shot execution allowance is already consumed", True
            )
            return

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = self.latest_trajectory
        self.execution_busy = True
        try:
            future = self.execution_client.send_goal_async(goal)
        except Exception as error:
            self.execution_busy = False
            self._publish_result(
                True, False, f"trajectory goal submission failed: {error}", True
            )
            return
        future.add_done_callback(self._goal_response_callback)
        self._publish_result(True, True, "one reviewed global trajectory submitted")

    def _goal_response_callback(self, future):
        try:
            goal_handle = future.result()
        except Exception as error:
            self.execution_busy = False
            self._publish_result(
                True, False, f"trajectory goal response failed: {error}", True
            )
            return
        if not goal_handle.accepted:
            self.execution_busy = False
            self._publish_result(False, False, "MoveIt rejected the trajectory goal", True)
            return
        goal_handle.get_result_async().add_done_callback(self._result_callback)

    def _result_callback(self, future):
        self.execution_busy = False
        try:
            wrapped = future.result()
            error_code = int(wrapped.result.error_code.val)
            status = int(wrapped.status)
        except Exception as error:
            self._publish_result(
                True, True, f"trajectory result failed: {error}", True
            )
            return
        self._publish_result(
            True,
            True,
            "global pre-insertion trajectory execution completed",
            warning=error_code != 1,
            extra={"moveit_error_code": error_code, "action_status": status},
        )

    def _publish_result(
        self, accepted, hardware_commanded, reason, warning=False, extra=None
    ):
        value = {
            "accepted": bool(accepted),
            "hardware_commanded": bool(hardware_commanded),
            "reason": str(reason),
            "dry_run": bool(self.get_parameter("dry_run").value),
            "allow_execution": bool(self.get_parameter("allow_execution").value),
            "execution_action_client_created": self.execution_client is not None,
            "execution_count": self.execution_guard.execution_count,
            "approval_consumed": self.execution_guard.consumed,
            "gripper_command_interface": False,
            "target_id": self.plan_target_id,
            **(extra or {}),
        }
        payload = json.dumps(value, sort_keys=True)
        self.result_publisher.publish(String(data=payload))
        (self.get_logger().warning if warning else self.get_logger().info)(payload)


def main(args=None):
    rclpy.init(args=args)
    node = GuardedPreinsertExecutorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
