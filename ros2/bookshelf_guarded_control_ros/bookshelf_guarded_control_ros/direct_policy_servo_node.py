#!/usr/bin/env python3
"""Apply fresh policy deltas through the existing MoveIt Servo server."""

from __future__ import annotations

import json
import math

from geometry_msgs.msg import Pose, PoseStamped, TwistStamped
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Bool, Float32MultiArray, String
from std_srvs.srv import Trigger
import tf2_ros

from .direct_policy_servo_math import (
    SupervisedTranslationBudget,
    bounded_error_twist,
    eef_target_from_tcp_target,
)
from .policy_tool_control_math import (
    TargetSafetyLimits,
    compute_policy_tool_target,
    make_transform,
    matrix_to_quaternion_xyzw,
    provenance_error,
    target_safety_error,
    transform_to_dict,
)


def _pose_to_transform(pose) -> np.ndarray:
    return make_transform(
        [pose.position.x, pose.position.y, pose.position.z],
        [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
    )


def _transform_message_to_matrix(message) -> np.ndarray:
    transform = message.transform
    return make_transform(
        [transform.translation.x, transform.translation.y, transform.translation.z],
        [transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w],
    )


def _transform_to_pose(transform: np.ndarray) -> Pose:
    message = Pose()
    message.position.x = float(transform[0, 3])
    message.position.y = float(transform[1, 3])
    message.position.z = float(transform[2, 3])
    quaternion = matrix_to_quaternion_xyzw(transform[:3, :3])
    message.orientation.x = float(quaternion[0])
    message.orientation.y = float(quaternion[1])
    message.orientation.z = float(quaternion[2])
    message.orientation.w = float(quaternion[3])
    return message


class DirectPolicyServo(Node):
    """Convert policy-tool deltas into bounded link_eef velocity commands."""

    def __init__(self):
        super().__init__("direct_policy_servo")
        self._declare_parameters()
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.eef_frame = str(self.get_parameter("eef_frame").value)
        self.tcp_frame = str(self.get_parameter("tcp_frame").value)
        self.translation_budget = SupervisedTranslationBudget(
            self.get_parameter("maximum_total_translation_m").value
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.start_servo_client = self.create_client(
            Trigger, str(self.get_parameter("start_servo_service").value)
        )
        self.twist_publisher = self.create_publisher(
            TwistStamped, str(self.get_parameter("twist_command_topic").value), 10
        )

        self.latest_observation_valid = False
        self.latest_observation_valid_ns = None
        self.latest_inference_valid = False
        self.latest_inference_valid_ns = None
        self.latest_delta = None
        self.latest_delta_ns = None
        self.latest_delta_generation = 0
        self.latest_slot_pose = None
        self.latest_slot_pose_ns = None
        self.latest_adapter_debug = None
        self.latest_adapter_debug_ns = None
        self.latest_policy_debug = None
        self.latest_policy_debug_ns = None

        self.servo_state = "start_needed"
        self.start_request_pending = False
        self.last_prepared_generation = 0
        self.active_target = None
        self.active_target_eef = None
        self.hardware_commanded = False
        self.command_count = 0
        self.zero_command_count = 0
        self.last_status_key = None

        self.command_valid_publisher = self.create_publisher(
            Bool, str(self.get_parameter("command_valid_topic").value), 10
        )
        self.status_publisher = self.create_publisher(
            String, str(self.get_parameter("status_topic").value), 10
        )
        self.target_publisher = self.create_publisher(
            PoseStamped, str(self.get_parameter("target_tcp_topic").value), 10
        )

        self.create_subscription(
            Bool,
            str(self.get_parameter("observation_valid_topic").value),
            self._observation_valid_callback,
            10,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("inference_valid_topic").value),
            self._inference_valid_callback,
            10,
        )
        self.create_subscription(
            Float32MultiArray,
            str(self.get_parameter("final_delta_topic").value),
            self._delta_callback,
            10,
        )
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("slot_pose_base_topic").value),
            self._slot_pose_callback,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("adapter_debug_topic").value),
            self._adapter_debug_callback,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("policy_debug_topic").value),
            self._policy_debug_callback,
            10,
        )

        rate = max(float(self.get_parameter("control_rate_hz").value), 1.0)
        self.timer = self.create_timer(1.0 / rate, self._timer_callback)
        self.get_logger().warning(
            "POLICY MOVEIT SERVO is motion-capable. It publishes bounded "
            "twists to the existing xArm trajectory controller and has no "
            "gripper interface."
        )

    def _declare_parameters(self):
        self.declare_parameter("base_frame", "link_base")
        self.declare_parameter("eef_frame", "link_eef")
        self.declare_parameter("tcp_frame", "link_tcp")
        self.declare_parameter(
            "start_servo_service", "/servo_server/start_servo"
        )
        self.declare_parameter(
            "twist_command_topic", "/servo_server/delta_twist_cmds"
        )
        self.declare_parameter("control_rate_hz", 30.0)
        self.declare_parameter("policy_command_duration_s", 0.20)
        self.declare_parameter("maximum_linear_speed_m_s", 0.025)
        self.declare_parameter("maximum_angular_speed_rad_s", 0.10)
        self.declare_parameter("translation_tolerance_m", 0.0005)
        self.declare_parameter("rotation_tolerance_rad", math.radians(0.25))
        self.declare_parameter("message_max_age_s", 0.50)
        self.declare_parameter("tf_max_age_s", 0.50)
        self.declare_parameter("tf_lookup_timeout_s", 0.02)
        self.declare_parameter("command_scale", 1.0)
        self.declare_parameter("command_target_is_hardware", True)
        self.declare_parameter("maximum_total_translation_m", 0.0)

        self.declare_parameter("eef_tcp_translation_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter(
            "eef_tcp_quaternion_xyzw", [0.0, 0.0, 0.0, 1.0]
        )

        self.declare_parameter("tcp_policy_tool_translation_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter(
            "tcp_policy_tool_quaternion_xyzw", [0.0, 0.0, 0.0, 1.0]
        )
        self.declare_parameter("expected_policy_tool_status", "")
        self.declare_parameter("expected_slot_status", "")
        self.declare_parameter("expected_book_status", "")
        self.declare_parameter("expected_bundle_sha256", "")
        self.declare_parameter("allow_unverified_policy_tool", False)

        self.declare_parameter(
            "maximum_policy_delta",
            [0.008, 0.003, 0.007, math.radians(0.8), math.radians(0.6)],
        )
        self.declare_parameter("maximum_tcp_translation_step_m", 0.010)
        self.declare_parameter("maximum_tcp_rotation_step_rad", math.radians(1.5))
        self.declare_parameter("workspace_min_xyz", [0.20, -0.60, 0.05])
        self.declare_parameter("workspace_max_xyz", [1.00, 0.60, 1.00])
        self.declare_parameter(
            "blocked_node_names",
            [
                "guarded_policy_tool_executor",
                "policy_tool_plan_checker",
                "policy_to_robot_node",
                "cartesian_action_executor_node",
                "action_executor_node",
            ],
        )

        self.declare_parameter(
            "observation_valid_topic", "/bookshelf_policy/observation_valid"
        )
        self.declare_parameter(
            "inference_valid_topic", "/bookshelf_shadow/inference_valid"
        )
        self.declare_parameter("final_delta_topic", "/bookshelf_shadow/final_delta")
        self.declare_parameter(
            "slot_pose_base_topic", "/bookshelf_policy/slot_pose_base"
        )
        self.declare_parameter("adapter_debug_topic", "/bookshelf_policy/adapter_debug")
        self.declare_parameter("policy_debug_topic", "/bookshelf_shadow/policy_debug")
        self.declare_parameter("command_valid_topic", "/bookshelf_control/command_valid")
        self.declare_parameter("status_topic", "/bookshelf_control/status")
        self.declare_parameter("target_tcp_topic", "/bookshelf_control/target_tcp")

    def _now_ns(self) -> int:
        return int(self.get_clock().now().nanoseconds)

    def _observation_valid_callback(self, message: Bool):
        self.latest_observation_valid = bool(message.data)
        self.latest_observation_valid_ns = self._now_ns()

    def _inference_valid_callback(self, message: Bool):
        self.latest_inference_valid = bool(message.data)
        self.latest_inference_valid_ns = self._now_ns()

    def _delta_callback(self, message: Float32MultiArray):
        self.latest_delta = np.asarray(message.data, dtype=np.float64)
        self.latest_delta_ns = self._now_ns()
        self.latest_delta_generation += 1

    def _slot_pose_callback(self, message: PoseStamped):
        self.latest_slot_pose = message
        self.latest_slot_pose_ns = self._now_ns()

    def _adapter_debug_callback(self, message: String):
        self.latest_adapter_debug = self._parse_debug(message.data)
        self.latest_adapter_debug_ns = self._now_ns()

    def _policy_debug_callback(self, message: String):
        self.latest_policy_debug = self._parse_debug(message.data)
        self.latest_policy_debug_ns = self._now_ns()

    @staticmethod
    def _parse_debug(value: str):
        try:
            parsed = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return None
        return parsed if isinstance(parsed, dict) else None

    def _fresh(self, timestamp_ns) -> bool:
        if timestamp_ns is None:
            return False
        maximum_age_s = float(self.get_parameter("message_max_age_s").value)
        if maximum_age_s <= 0.0:
            return True
        return (self._now_ns() - timestamp_ns) * 1.0e-9 <= maximum_age_s

    def _input_error(self) -> str | None:
        if self.translation_budget.terminal_reason is not None:
            return self.translation_budget.terminal_reason
        if not self.latest_observation_valid:
            return "observation_valid is false"
        if not self._fresh(self.latest_observation_valid_ns):
            return "observation_valid is missing or stale"
        if not self.latest_inference_valid:
            return "inference_valid is false"
        if not self._fresh(self.latest_inference_valid_ns):
            return "inference_valid is missing or stale"
        required = (
            (self.latest_delta, self.latest_delta_ns, "final delta"),
            (self.latest_slot_pose, self.latest_slot_pose_ns, "slot pose"),
            (self.latest_adapter_debug, self.latest_adapter_debug_ns, "adapter debug"),
            (self.latest_policy_debug, self.latest_policy_debug_ns, "policy debug"),
        )
        for value, timestamp, label in required:
            if value is None or not self._fresh(timestamp):
                return f"{label} is missing or stale"
        if self.latest_delta.shape != (5,) or not np.all(np.isfinite(self.latest_delta)):
            return "final delta must be a finite 5D vector"
        if self.latest_slot_pose.header.frame_id != self.base_frame:
            return (
                f"slot pose frame is {self.latest_slot_pose.header.frame_id}, "
                f"expected {self.base_frame}"
            )
        error = provenance_error(
            self.latest_adapter_debug,
            self.latest_policy_debug,
            expected_policy_tool_status=str(
                self.get_parameter("expected_policy_tool_status").value
            ),
            expected_slot_status=str(self.get_parameter("expected_slot_status").value),
            expected_book_status=str(self.get_parameter("expected_book_status").value),
            expected_bundle_sha256=str(self.get_parameter("expected_bundle_sha256").value),
            allow_unverified_policy_tool=bool(
                self.get_parameter("allow_unverified_policy_tool").value
            ),
        )
        if error:
            return error
        blocked = self._blocked_nodes_present()
        if blocked:
            return f"competing local-control nodes are active: {blocked}"
        return None

    def _blocked_nodes_present(self) -> list[str]:
        blocked = {
            str(value).strip().lstrip("/")
            for value in self.get_parameter("blocked_node_names").value
        }
        active = {str(value).strip().lstrip("/") for value in self.get_node_names()}
        active.discard(self.get_name().lstrip("/"))
        return sorted(blocked.intersection(active))

    def _lookup_base_eef(self):
        try:
            message = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.eef_frame,
                Time(),
                timeout=Duration(
                    seconds=float(self.get_parameter("tf_lookup_timeout_s").value)
                ),
            )
        except Exception as error:
            return None, f"TF {self.base_frame} <- {self.eef_frame} unavailable: {error}"
        maximum_age = float(self.get_parameter("tf_max_age_s").value)
        stamp_ns = int(message.header.stamp.sec) * 1_000_000_000 + int(
            message.header.stamp.nanosec
        )
        if maximum_age > 0.0 and stamp_ns > 0:
            age = (self._now_ns() - stamp_ns) * 1.0e-9
            if age > maximum_age:
                return None, f"TF {self.base_frame} <- {self.eef_frame} is stale"
        return _transform_message_to_matrix(message), None

    def _eef_tcp_transform(self) -> np.ndarray:
        return make_transform(
            self.get_parameter("eef_tcp_translation_xyz").value,
            self.get_parameter("eef_tcp_quaternion_xyzw").value,
        )

    def _tool_transform(self) -> np.ndarray:
        return make_transform(
            self.get_parameter("tcp_policy_tool_translation_xyz").value,
            self.get_parameter("tcp_policy_tool_quaternion_xyzw").value,
        )

    def _safety_limits(self) -> TargetSafetyLimits:
        return TargetSafetyLimits(
            maximum_delta=tuple(
                float(value)
                for value in self.get_parameter("maximum_policy_delta").value
            ),
            maximum_tcp_translation_step_m=float(
                self.get_parameter("maximum_tcp_translation_step_m").value
            ),
            maximum_tcp_rotation_step_rad=float(
                self.get_parameter("maximum_tcp_rotation_step_rad").value
            ),
            workspace_min_xyz=tuple(
                float(value) for value in self.get_parameter("workspace_min_xyz").value
            ),
            workspace_max_xyz=tuple(
                float(value) for value in self.get_parameter("workspace_max_xyz").value
            ),
        )

    def _start_servo(self) -> str | None:
        if self.servo_state == "ready":
            return None
        if self.start_request_pending:
            return "MoveIt Servo start request is pending"
        if self.servo_state != "start_needed":
            return f"MoveIt Servo failed to start: {self.servo_state}"
        if not self.start_servo_client.service_is_ready():
            return "MoveIt Servo start service is unavailable"
        self.start_request_pending = True
        future = self.start_servo_client.call_async(Trigger.Request())
        future.add_done_callback(self._start_servo_response)
        return "MoveIt Servo start request sent"

    def _start_servo_response(self, future):
        self.start_request_pending = False
        try:
            response = future.result()
        except Exception as error:
            self.servo_state = f"start exception: {error}"
            return
        if response is None or not bool(response.success):
            message = "no response" if response is None else response.message
            self.servo_state = f"start rejected: {message}"
            return
        self.servo_state = "ready"

    def _timer_callback(self):
        error = self._input_error()
        if error:
            self._clear_target_and_halt()
            self._publish_status(False, error)
            return

        transform_base_eef, error = self._lookup_base_eef()
        if error:
            self._clear_target_and_halt()
            self._publish_status(False, error)
            return
        transform_eef_tcp = self._eef_tcp_transform()
        transform_base_tcp = transform_base_eef @ transform_eef_tcp
        target = self.active_target
        prepared_new_target = False
        if (
            self.latest_delta_generation != self.last_prepared_generation
            and not self.translation_budget.exhausted
        ):
            try:
                target = compute_policy_tool_target(
                    _pose_to_transform(self.latest_slot_pose.pose),
                    transform_base_tcp,
                    self._tool_transform(),
                    self.latest_delta,
                    command_scale=float(
                        self.get_parameter("command_scale").value
                    ),
                )
            except (TypeError, ValueError, np.linalg.LinAlgError) as error:
                self._clear_target_and_halt()
                self._publish_status(False, f"target calculation failed: {error}")
                return
            error = target_safety_error(
                target, self.latest_delta, self._safety_limits()
            )
            if error:
                self._clear_target_and_halt()
                self._publish_status(False, error, target=target)
                return
            prepared_new_target = True

        start_reason = self._start_servo()
        if start_reason:
            self._clear_target_and_halt()
            self._publish_status(False, start_reason, target=target)
            return

        if prepared_new_target:
            try:
                terminal_reason = self.translation_budget.accept_target(
                    target.tcp_translation_step_m
                )
            except ValueError as error:
                self._clear_target_and_halt()
                self._publish_status(False, str(error), target=target)
                return
            if terminal_reason is not None:
                self._clear_target_and_halt()
                self._publish_status(False, terminal_reason, target=target)
                return
            self.active_target = target
            self.active_target_eef = eef_target_from_tcp_target(
                target.transform_base_tcp_target,
                transform_eef_tcp,
            )
            self.last_prepared_generation = self.latest_delta_generation
        if self.active_target_eef is None:
            self._publish_zero_twist()
            self._publish_status(True, None, target=target)
            return

        self._publish_target(self.active_target.transform_base_tcp_target)

        try:
            twist = bounded_error_twist(
                transform_base_eef,
                self.active_target_eef,
                duration_s=float(
                    self.get_parameter("policy_command_duration_s").value
                ),
                maximum_linear_speed_m_s=float(
                    self.get_parameter("maximum_linear_speed_m_s").value
                ),
                maximum_angular_speed_rad_s=float(
                    self.get_parameter("maximum_angular_speed_rad_s").value
                ),
                translation_tolerance_m=float(
                    self.get_parameter("translation_tolerance_m").value
                ),
                rotation_tolerance_rad=float(
                    self.get_parameter("rotation_tolerance_rad").value
                ),
            )
        except (TypeError, ValueError, np.linalg.LinAlgError) as error:
            self._clear_target_and_halt()
            self._publish_status(False, f"twist calculation failed: {error}")
            return

        if float(np.linalg.norm(twist)) == 0.0:
            self.active_target_eef = None
            self._publish_zero_twist()
            terminal_reason = self.translation_budget.finish_at_limit()
            if terminal_reason is not None:
                self._publish_status(False, terminal_reason, target=self.active_target)
                return
        else:
            self._publish_twist(twist)
            if bool(self.get_parameter("command_target_is_hardware").value):
                self.hardware_commanded = True
            self.command_count += 1
        self._publish_status(True, None, target=self.active_target)

    def _publish_twist(self, values):
        message = TwistStamped()
        message.header.frame_id = self.base_frame
        message.header.stamp = self.get_clock().now().to_msg()
        message.twist.linear.x = float(values[0])
        message.twist.linear.y = float(values[1])
        message.twist.linear.z = float(values[2])
        message.twist.angular.x = float(values[3])
        message.twist.angular.y = float(values[4])
        message.twist.angular.z = float(values[5])
        self.twist_publisher.publish(message)

    def _publish_zero_twist(self):
        self._publish_twist(np.zeros(6, dtype=np.float64))
        self.zero_command_count += 1

    def _clear_target_and_halt(self):
        self.active_target = None
        self.active_target_eef = None
        if self.servo_state == "ready":
            self._publish_zero_twist()

    def _publish_target(self, transform):
        message = PoseStamped()
        message.header.frame_id = self.base_frame
        message.header.stamp = self.get_clock().now().to_msg()
        message.pose = _transform_to_pose(transform)
        self.target_publisher.publish(message)

    def _publish_status(self, valid: bool, reason, *, target=None):
        self.command_valid_publisher.publish(Bool(data=bool(valid)))
        report = {
            "valid": bool(valid),
            "reason": None if reason is None else str(reason),
            "base_frame": self.base_frame,
            "eef_frame": self.eef_frame,
            "tcp_frame": self.tcp_frame,
            "servo_state": self.servo_state,
            "command_scale": float(self.get_parameter("command_scale").value),
            "maximum_total_translation_m": self.translation_budget.maximum_m,
            "total_commanded_translation_m": self.translation_budget.total_m,
            "terminal_reason": self.translation_budget.terminal_reason,
            "target_active": self.active_target_eef is not None,
            "start_request_pending": self.start_request_pending,
            "command_count": self.command_count,
            "zero_command_count": self.zero_command_count,
            "hardware_commanded": self.hardware_commanded,
            "command_target_is_hardware": bool(
                self.get_parameter("command_target_is_hardware").value
            ),
            "gripper_command_interface": False,
            "moveit_planning_interface": False,
            "moveit_servo_interface": True,
        }
        if target is not None:
            report.update(
                {
                    "target_id": target.target_id,
                    "unscaled_delta": [float(value) for value in self.latest_delta],
                    "scaled_delta": [float(value) for value in target.scaled_delta],
                    "tcp_translation_step_m": target.tcp_translation_step_m,
                    "tcp_rotation_step_deg": math.degrees(
                        target.tcp_rotation_step_rad
                    ),
                    "target_tcp_base": transform_to_dict(
                        target.transform_base_tcp_target
                    ),
                }
            )
        self.status_publisher.publish(String(data=json.dumps(report, sort_keys=True)))
        key = (bool(valid), report["reason"], self.servo_state)
        if key == self.last_status_key:
            return
        self.last_status_key = key
        if valid:
            self.get_logger().info("Policy MoveIt Servo inputs are valid.")
        else:
            self.get_logger().warning(f"Policy MoveIt Servo blocked: {reason}")


def main(args=None):
    rclpy.init(args=args)
    node = DirectPolicyServo()
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
