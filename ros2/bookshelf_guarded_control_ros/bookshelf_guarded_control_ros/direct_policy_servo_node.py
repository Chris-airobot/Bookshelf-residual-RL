#!/usr/bin/env python3
"""Apply fresh policy deltas through the xArm Cartesian servo service."""

from __future__ import annotations

from collections import deque
import json
import math

from geometry_msgs.msg import Pose, PoseStamped
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Bool, Float32MultiArray, String
import tf2_ros
from xarm_msgs.srv import MoveCartesian, SetInt16

from .direct_policy_servo_math import (
    interpolate_transform,
    transform_to_xarm_axis_angle_pose,
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
    """Convert policy-tool deltas into smooth absolute xArm TCP commands."""

    def __init__(self):
        super().__init__("direct_policy_servo")
        self._declare_parameters()
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.tcp_frame = str(self.get_parameter("tcp_frame").value)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.servo_client = self.create_client(
            MoveCartesian, str(self.get_parameter("servo_service").value)
        )
        self.mode_client = self.create_client(
            SetInt16, str(self.get_parameter("set_mode_service").value)
        )
        self.state_client = self.create_client(
            SetInt16, str(self.get_parameter("set_state_service").value)
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

        self.mode_state = (
            "mode_needed"
            if bool(self.get_parameter("configure_servo_mode").value)
            else "ready"
        )
        self.mode_request_pending = False
        self.servo_request_pending = False
        self.last_prepared_generation = 0
        self.command_queue = deque()
        self.hardware_commanded = False
        self.command_count = 0
        self.last_response_ret = None
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
            "DIRECT POLICY SERVO is motion-capable. It uses xArm Cartesian "
            "servo commands and has no MoveIt or gripper interface."
        )

    def _declare_parameters(self):
        self.declare_parameter("base_frame", "link_base")
        self.declare_parameter("tcp_frame", "link_tcp")
        self.declare_parameter("servo_service", "/xarm/set_servo_cartesian_aa")
        self.declare_parameter("set_mode_service", "/xarm/set_mode")
        self.declare_parameter("set_state_service", "/xarm/set_state")
        self.declare_parameter("configure_servo_mode", True)
        self.declare_parameter("servo_mode", 1)
        self.declare_parameter("ready_state", 0)
        self.declare_parameter("control_rate_hz", 100.0)
        self.declare_parameter("policy_command_duration_s", 0.05)
        self.declare_parameter("message_max_age_s", 0.50)
        self.declare_parameter("tf_max_age_s", 0.50)
        self.declare_parameter("tf_lookup_timeout_s", 0.02)
        self.declare_parameter("command_scale", 1.0)

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

    def _lookup_base_tcp(self):
        try:
            message = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.tcp_frame,
                Time(),
                timeout=Duration(
                    seconds=float(self.get_parameter("tf_lookup_timeout_s").value)
                ),
            )
        except Exception as error:
            return None, f"TF {self.base_frame} <- {self.tcp_frame} unavailable: {error}"
        maximum_age = float(self.get_parameter("tf_max_age_s").value)
        stamp_ns = int(message.header.stamp.sec) * 1_000_000_000 + int(
            message.header.stamp.nanosec
        )
        if maximum_age > 0.0 and stamp_ns > 0:
            age = (self._now_ns() - stamp_ns) * 1.0e-9
            if age > maximum_age:
                return None, f"TF {self.base_frame} <- {self.tcp_frame} is stale"
        return _transform_message_to_matrix(message), None

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

    def _configure_mode(self) -> str | None:
        if self.mode_state == "ready":
            return None
        if self.mode_request_pending:
            return f"xArm servo configuration is pending ({self.mode_state})"
        if self.mode_state == "mode_needed":
            if not self.mode_client.service_is_ready():
                return "xArm set_mode service is unavailable"
            request = SetInt16.Request()
            request.data = int(self.get_parameter("servo_mode").value)
            self.mode_request_pending = True
            future = self.mode_client.call_async(request)
            future.add_done_callback(self._mode_response_callback)
            return "xArm servo mode request sent"
        if self.mode_state == "state_needed":
            if not self.state_client.service_is_ready():
                return "xArm set_state service is unavailable"
            request = SetInt16.Request()
            request.data = int(self.get_parameter("ready_state").value)
            self.mode_request_pending = True
            future = self.state_client.call_async(request)
            future.add_done_callback(self._state_response_callback)
            return "xArm ready-state request sent"
        return f"xArm servo configuration failed: {self.mode_state}"

    def _mode_response_callback(self, future):
        self.mode_request_pending = False
        try:
            response = future.result()
        except Exception as error:
            self.mode_state = f"set_mode exception: {error}"
            return
        if response is None or int(response.ret) != 0:
            ret = None if response is None else int(response.ret)
            self.mode_state = f"set_mode returned {ret}"
            return
        self.mode_state = "state_needed"

    def _state_response_callback(self, future):
        self.mode_request_pending = False
        try:
            response = future.result()
        except Exception as error:
            self.mode_state = f"set_state exception: {error}"
            return
        if response is None or int(response.ret) != 0:
            ret = None if response is None else int(response.ret)
            self.mode_state = f"set_state returned {ret}"
            return
        self.mode_state = "ready"

    def _timer_callback(self):
        error = self._input_error()
        if error:
            self.command_queue.clear()
            self._publish_status(False, error)
            return
        if not self.servo_client.service_is_ready():
            self.command_queue.clear()
            self._publish_status(False, "xArm Cartesian servo service is unavailable")
            return

        transform_base_tcp, error = self._lookup_base_tcp()
        if error:
            self.command_queue.clear()
            self._publish_status(False, error)
            return
        try:
            target = compute_policy_tool_target(
                _pose_to_transform(self.latest_slot_pose.pose),
                transform_base_tcp,
                self._tool_transform(),
                self.latest_delta,
                command_scale=float(self.get_parameter("command_scale").value),
            )
        except (TypeError, ValueError, np.linalg.LinAlgError) as error:
            self.command_queue.clear()
            self._publish_status(False, f"target calculation failed: {error}")
            return
        error = target_safety_error(target, self.latest_delta, self._safety_limits())
        if error:
            self.command_queue.clear()
            self._publish_status(False, error, target=target)
            return

        self._publish_target(target.transform_base_tcp_target)
        mode_reason = self._configure_mode()
        if mode_reason:
            self.command_queue.clear()
            self._publish_status(False, mode_reason, target=target)
            return

        if self.latest_delta_generation != self.last_prepared_generation:
            self._prepare_interpolation(transform_base_tcp, target.transform_base_tcp_target)
            self.last_prepared_generation = self.latest_delta_generation
        if self.servo_request_pending or not self.command_queue:
            self._publish_status(True, None, target=target)
            return

        command_transform = self.command_queue.popleft()
        request = MoveCartesian.Request()
        request.pose = transform_to_xarm_axis_angle_pose(command_transform)
        request.speed = 0.0
        request.acc = 0.0
        request.mvtime = 0.0
        request.wait = False
        request.timeout = -1.0
        request.radius = -1.0
        request.is_tool_coord = False
        request.relative = False
        request.motion_type = 0
        self.servo_request_pending = True
        future = self.servo_client.call_async(request)
        future.add_done_callback(self._servo_response_callback)
        self._publish_status(True, None, target=target)

    def _prepare_interpolation(self, start, target):
        rate = max(float(self.get_parameter("control_rate_hz").value), 1.0)
        duration = max(
            float(self.get_parameter("policy_command_duration_s").value), 0.0
        )
        point_count = max(int(math.ceil(rate * duration)), 1)
        self.command_queue = deque(
            interpolate_transform(start, target, index / point_count)
            for index in range(1, point_count + 1)
        )

    def _servo_response_callback(self, future):
        self.servo_request_pending = False
        try:
            response = future.result()
        except Exception as error:
            self.command_queue.clear()
            self.last_response_ret = None
            self._publish_status(False, f"xArm servo service exception: {error}")
            return
        self.last_response_ret = None if response is None else int(response.ret)
        if response is None or self.last_response_ret != 0:
            self.command_queue.clear()
            self._publish_status(
                False, f"xArm servo command returned {self.last_response_ret}"
            )
            return
        self.hardware_commanded = True
        self.command_count += 1

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
            "tcp_frame": self.tcp_frame,
            "mode_state": self.mode_state,
            "command_scale": float(self.get_parameter("command_scale").value),
            "queued_points": len(self.command_queue),
            "command_pending": self.servo_request_pending,
            "command_count": self.command_count,
            "last_response_ret": self.last_response_ret,
            "hardware_commanded": self.hardware_commanded,
            "gripper_command_interface": False,
            "moveit_interface": False,
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
        key = (bool(valid), report["reason"], self.mode_state)
        if key == self.last_status_key:
            return
        self.last_status_key = key
        if valid:
            self.get_logger().info("Direct policy servo inputs are valid.")
        else:
            self.get_logger().warning(f"Direct policy servo blocked: {reason}")


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

