#!/usr/bin/env python3
"""Operator-authorized gripper actions protected by the shared shadow gate."""

from __future__ import annotations

import json

from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory, GripperCommand
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectoryPoint

from .execution_gate import hardware_commands_allowed


GRIPPER_COMMAND = "gripper_command"
GRIPPER_TRAJECTORY = "follow_joint_trajectory"


def make_gripper_goal(action_type, position, max_effort, duration_s):
    if action_type == GRIPPER_COMMAND:
        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = float(max_effort)
        return goal
    if action_type == GRIPPER_TRAJECTORY:
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ["drive_joint"]
        point = JointTrajectoryPoint()
        point.positions = [float(position)]
        duration_ns = int(max(float(duration_s), 0.1) * 1e9)
        point.time_from_start.sec = duration_ns // 1_000_000_000
        point.time_from_start.nanosec = duration_ns % 1_000_000_000
        goal.trajectory.points = [point]
        return goal
    raise ValueError(f"unsupported gripper action type: {action_type!r}")


def log_operator_message(logger, success, message):
    """Keep rclpy severity fixed at each logging call site."""

    if success:
        logger.info(str(message))
    else:
        logger.error(str(message))


class OperatorActionNode(Node):
    """Execute gripper actions only; all arm plans belong to SimplePreinsertNode."""

    def __init__(self):
        super().__init__("bookshelf_operator_actions")
        self.declare_parameter("allow_execution", False)
        self.declare_parameter("shadow_full_sequence", False)
        self.declare_parameter("gripper_action", "/xarm_gripper/gripper_action")
        self.declare_parameter("gripper_action_type", GRIPPER_COMMAND)
        self.declare_parameter("joint_move_duration_s", 5.0)
        self.declare_parameter("gripper_open_position", 0.0)
        self.declare_parameter("gripper_closed_position", 0.85)
        self.declare_parameter("gripper_max_effort", 0.0)

        self.shadow = bool(self.get_parameter("shadow_full_sequence").value)
        requested = bool(self.get_parameter("allow_execution").value)
        self.commands_enabled = hardware_commands_allowed(requested, self.shadow)
        self.gripper_action_type = str(
            self.get_parameter("gripper_action_type").value
        )
        if self.gripper_action_type not in (GRIPPER_COMMAND, GRIPPER_TRAJECTORY):
            raise ValueError(
                f"unsupported gripper_action_type: {self.gripper_action_type!r}"
            )

        self.status_publisher = self.create_publisher(
            String, "/bookshelf_simple/operator_action_status", 10
        )
        self.gripper_client = None
        if self.commands_enabled:
            action_type = (
                GripperCommand
                if self.gripper_action_type == GRIPPER_COMMAND
                else FollowJointTrajectory
            )
            self.gripper_client = ActionClient(
                self, action_type, str(self.get_parameter("gripper_action").value)
            )

        self.busy = None
        self._one_shot_timers = []
        self.create_service(Trigger, "/bookshelf_simple/open_gripper", self._open)
        self.create_service(Trigger, "/bookshelf_simple/close_gripper", self._close)
        self.create_service(
            Trigger, "/bookshelf_simple/finish_return", self._finish_return
        )

        mode = "SHADOW" if self.shadow else (
            "EXECUTION ENABLED" if self.commands_enabled else "EXECUTION DISABLED"
        )
        self.get_logger().warning(
            f"Operator gripper helper ready: {mode}; no arm motion interface exists"
        )

    def _publish(self, action, success, message):
        payload = {
            "action": str(action),
            "success": bool(success),
            "message": str(message),
            "shadow_full_sequence": self.shadow,
        }
        self.status_publisher.publish(String(data=json.dumps(payload, sort_keys=True)))
        log_operator_message(self.get_logger(), success, message)

    @staticmethod
    def _accepted(response, message):
        response.success = True
        response.message = str(message)
        return response

    @staticmethod
    def _rejected(response, message):
        response.success = False
        response.message = str(message)
        return response

    def _schedule_once(self, callback):
        holder = {}

        def invoke():
            callback()
            holder["timer"].cancel()
            self._one_shot_timers.remove(holder["timer"])

        holder["timer"] = self.create_timer(0.05, invoke)
        self._one_shot_timers.append(holder["timer"])

    def _open(self, _request, response):
        return self._request_gripper("open", response)

    def _close(self, _request, response):
        return self._request_gripper("close", response)

    def _finish_return(self, _request, response):
        if self.busy:
            return self._rejected(response, f"operator action busy: {self.busy}")
        if self.shadow:
            self._schedule_once(self._shadow_return_open)
            return self._accepted(response, "shadow return-open action accepted")
        if not self.commands_enabled:
            return self._rejected(response, "gripper execution is disabled")
        self._send_gripper("return_open")
        return self._accepted(response, "post-return gripper opening submitted")

    def _request_gripper(self, kind, response):
        if self.busy:
            return self._rejected(response, f"operator action busy: {self.busy}")
        if self.shadow:
            self._schedule_once(
                lambda: self._publish(kind, True, f"SHADOW: would {kind} gripper")
            )
            return self._accepted(response, f"shadow {kind} action accepted")
        if not self.commands_enabled:
            return self._rejected(response, "gripper execution is disabled")
        self._send_gripper(kind)
        return self._accepted(response, f"gripper {kind} submitted")

    def _shadow_return_open(self):
        self._publish("return_open", True, "SHADOW: would open gripper")
        self._publish("ready", True, "SHADOW: READY FOR NEXT BOOK")

    def _send_gripper(self, kind):
        if self.gripper_client is None or not self.gripper_client.wait_for_server(
            timeout_sec=1.0
        ):
            self._gripper_failed(kind, "gripper action server is unavailable")
            return
        position = float(
            self.get_parameter(
                "gripper_open_position"
                if kind in ("open", "return_open")
                else "gripper_closed_position"
            ).value
        )
        goal = make_gripper_goal(
            self.gripper_action_type,
            position,
            self.get_parameter("gripper_max_effort").value,
            self.get_parameter("joint_move_duration_s").value,
        )
        self.busy = kind
        self.gripper_client.send_goal_async(goal).add_done_callback(
            self._gripper_goal_response
        )

    def _gripper_goal_response(self, future):
        kind = self.busy
        try:
            goal_handle = future.result()
        except Exception as error:
            self._gripper_failed(kind, str(error))
            return
        if goal_handle is None or not goal_handle.accepted:
            self._gripper_failed(kind, "gripper goal rejected")
            return
        goal_handle.get_result_async().add_done_callback(self._gripper_result)

    def _gripper_result(self, future):
        kind = self.busy
        try:
            wrapped = future.result()
            success = int(wrapped.status) == int(GoalStatus.STATUS_SUCCEEDED)
            if self.gripper_action_type == GRIPPER_TRAJECTORY:
                success = success and int(wrapped.result.error_code) == int(
                    FollowJointTrajectory.Result.SUCCESSFUL
                )
        except Exception as error:
            self._gripper_failed(kind, str(error))
            return
        self.busy = None
        if not success:
            self._gripper_failed(kind, "gripper action failed")
            return
        self._publish(kind, True, f"gripper {kind} complete")
        if kind == "return_open":
            self._publish("ready", True, "READY FOR NEXT BOOK")

    def _gripper_failed(self, kind, reason):
        self.busy = None
        self._publish(kind, False, reason)
        if kind == "return_open":
            self._publish(
                "return_failed", False, "return reached but gripper did not open"
            )


def main(args=None):
    rclpy.init(args=args)
    node = OperatorActionNode()
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
