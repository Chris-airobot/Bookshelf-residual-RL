#!/usr/bin/env python3
"""Integrate MoveIt Servo twists against a software-only bookshelf state."""

from __future__ import annotations

import json
from pathlib import Path

from geometry_msgs.msg import Point, PoseStamped, TransformStamped, TwistStamped
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32MultiArray, String
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from .policy_servo_simulation_math import integrate_base_frame_twist
from .planning_scene_math import configured_box, shelf_box_from_slot
from .direct_policy_servo_math import MAXIMUM_SUPERVISED_TRANSLATION_REASON
from .policy_tool_control_math import (
    invert_transform,
    make_transform,
    matrix_to_quaternion_xyzw,
    transform_to_dict,
)


def _finite_vector(values, size: int, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain {size} finite values")
    return result


class PolicyServoSimulator(Node):
    """Provide the tiny Servo/TF surface needed for closed-loop rehearsal."""

    def __init__(self):
        super().__init__("policy_servo_simulator")
        self._declare_parameters()
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.eef_frame = str(self.get_parameter("eef_frame").value)
        self.tcp_frame = str(self.get_parameter("tcp_frame").value)
        self.book_frame = str(self.get_parameter("book_frame").value)
        self.transform_base_eef = make_transform(
            self.get_parameter("initial_eef_translation_xyz").value,
            self.get_parameter("initial_eef_quaternion_xyzw").value,
        )
        self.initial_transform_base_eef = np.array(
            self.transform_base_eef, copy=True
        )
        self.transform_eef_tcp = make_transform(
            self.get_parameter("eef_tcp_translation_xyz").value,
            self.get_parameter("eef_tcp_quaternion_xyzw").value,
        )
        self.transform_eef_book = make_transform(
            self.get_parameter("eef_book_translation_xyz").value,
            self.get_parameter("eef_book_quaternion_xyzw").value,
        )
        self.transform_base_slot = make_transform(
            self.get_parameter("slot_translation_xyz").value,
            self.get_parameter("slot_quaternion_xyzw").value,
        )
        self.transform_book_marker = make_transform(
            self.get_parameter("book_marker_translation_xyz").value,
            self.get_parameter("book_marker_quaternion_xyzw").value,
        )

        self.started = False
        self.blocked_reason = None
        self.latest_twist = np.zeros(6, dtype=np.float64)
        self.latest_twist_ns = None
        self.twist_messages = 0
        self.nonzero_twist_messages = 0
        self.observation_valid_messages = 0
        self.inference_valid_messages = 0
        self.controller_valid_messages = 0
        self.latest_controller_status = None
        self.latest_policy_delta = None
        self.policy_delta_messages = 0
        self.path_length_m = 0.0
        self.maximum_linear_speed_m_s = 0.0
        self.maximum_angular_speed_rad_s = 0.0
        self.last_update_ns = self._now_ns()
        self.last_report_ns = 0
        self.path_points = [self.transform_base_eef[:3, 3].copy()]

        self.tf_broadcaster = TransformBroadcaster(self)
        self.joint_publisher = self.create_publisher(
            JointState, str(self.get_parameter("joint_states_topic").value), 10
        )
        self.eef_pose_publisher = self.create_publisher(
            PoseStamped, "/bookshelf_sim/eef_pose", 10
        )
        self.book_pose_publisher = self.create_publisher(
            PoseStamped, "/bookshelf_sim/book_pose", 10
        )
        self.status_publisher = self.create_publisher(
            String, "/bookshelf_sim/status", 10
        )
        self.marker_publisher = self.create_publisher(
            MarkerArray, "/bookshelf_sim/markers", 10
        )
        self.create_service(
            Trigger,
            str(self.get_parameter("start_servo_service").value),
            self._start_servo,
        )
        self.create_subscription(
            TwistStamped,
            str(self.get_parameter("twist_command_topic").value),
            self._twist_callback,
            10,
        )
        self.create_subscription(
            Bool,
            "/bookshelf_policy/observation_valid",
            self._observation_callback,
            10,
        )
        self.create_subscription(
            Bool,
            "/bookshelf_shadow/inference_valid",
            self._inference_callback,
            10,
        )
        self.create_subscription(
            String,
            "/bookshelf_control/status",
            self._controller_status_callback,
            10,
        )
        self.create_subscription(
            Float32MultiArray,
            "/bookshelf_shadow/final_delta",
            self._policy_delta_callback,
            10,
        )

        rate = max(float(self.get_parameter("update_rate_hz").value), 1.0)
        self.timer = self.create_timer(1.0 / rate, self._timer_callback)
        self.get_logger().warning(
            "SOFTWARE-ONLY POLICY SERVO SIMULATION. No xArm, controller, "
            "planning, gripper, or hardware service is created."
        )

    def _declare_parameters(self):
        self.declare_parameter("base_frame", "sim_link_base")
        self.declare_parameter("eef_frame", "sim_link_eef")
        self.declare_parameter("tcp_frame", "sim_link_tcp")
        self.declare_parameter("book_frame", "sim_target_book_center")
        self.declare_parameter(
            "start_servo_service", "/bookshelf_sim/servo/start"
        )
        self.declare_parameter(
            "twist_command_topic", "/bookshelf_sim/servo/delta_twist_cmds"
        )
        self.declare_parameter("joint_states_topic", "/bookshelf_sim/joint_states")
        self.declare_parameter("update_rate_hz", 100.0)
        self.declare_parameter("command_timeout_s", 0.20)
        self.declare_parameter("gripper_position", 0.85)
        self.declare_parameter("maximum_linear_speed_m_s", 0.025)
        self.declare_parameter("maximum_angular_speed_rad_s", 0.10)
        self.declare_parameter("maximum_path_length_m", 0.25)
        self.declare_parameter("minimum_forward_progress_m", 0.001)
        self.declare_parameter("report_write_period_s", 0.5)
        self.declare_parameter("output_dir", "/tmp/bookshelf_policy_servo_sim")
        self.declare_parameter("candidate_id", "unknown")
        self.declare_parameter("initial_eef_translation_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter(
            "initial_eef_quaternion_xyzw", [0.0, 0.0, 0.0, 1.0]
        )
        self.declare_parameter("eef_tcp_translation_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter(
            "eef_tcp_quaternion_xyzw", [0.0, 0.0, 0.0, 1.0]
        )
        self.declare_parameter("eef_book_translation_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter(
            "eef_book_quaternion_xyzw", [0.0, 0.0, 0.0, 1.0]
        )
        self.declare_parameter("slot_translation_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter(
            "slot_quaternion_xyzw", [0.0, 0.0, 0.0, 1.0]
        )
        self.declare_parameter("slot_width_m", 0.04)
        self.declare_parameter("slot_depth_m", 0.20)
        self.declare_parameter("book_size_xyz", [0.156, 0.034, 0.236])
        self.declare_parameter("marker_size_m", 0.039)
        self.declare_parameter("marker_thickness_m", 0.002)
        self.declare_parameter("book_marker_translation_xyz", [-0.08, 0.0375, 0.0665])
        self.declare_parameter(
            "book_marker_quaternion_xyzw", [0.5, -0.5, -0.5, 0.5]
        )
        self.declare_parameter("shelf_size_xyz", [0.30, 0.95, 0.40])
        self.declare_parameter("shelf_center_offset_slot_xyz", [0.15, 0.0, 0.0])
        self.declare_parameter("shelf_bottom_height_base_m", 0.015)
        self.declare_parameter("table_size_xyz", [1.50, 0.60, 0.05])
        self.declare_parameter("table_center_base_xyz", [0.75, 0.0, -0.025])
        self.declare_parameter(
            "table_quaternion_base_xyzw", [0.0, 0.0, 0.0, 1.0]
        )

    def _now_ns(self) -> int:
        return int(self.get_clock().now().nanoseconds)

    def _start_servo(self, _request, response):
        self.started = True
        response.success = True
        response.message = "software-only Servo simulation started"
        return response

    def _twist_callback(self, message: TwistStamped):
        self.twist_messages += 1
        if message.header.frame_id != self.base_frame:
            self.blocked_reason = (
                f"twist frame is {message.header.frame_id}, expected {self.base_frame}"
            )
            return
        values = np.array(
            [
                message.twist.linear.x,
                message.twist.linear.y,
                message.twist.linear.z,
                message.twist.angular.x,
                message.twist.angular.y,
                message.twist.angular.z,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            self.blocked_reason = "received a non-finite twist"
            return
        linear_speed = float(np.linalg.norm(values[:3]))
        angular_speed = float(np.linalg.norm(values[3:]))
        self.maximum_linear_speed_m_s = max(
            self.maximum_linear_speed_m_s, linear_speed
        )
        self.maximum_angular_speed_rad_s = max(
            self.maximum_angular_speed_rad_s, angular_speed
        )
        linear_limit = float(self.get_parameter("maximum_linear_speed_m_s").value)
        angular_limit = float(
            self.get_parameter("maximum_angular_speed_rad_s").value
        )
        if linear_speed > linear_limit + 1.0e-9:
            self.blocked_reason = "linear velocity exceeded the simulation limit"
            return
        if angular_speed > angular_limit + 1.0e-9:
            self.blocked_reason = "angular velocity exceeded the simulation limit"
            return
        self.latest_twist = values
        self.latest_twist_ns = self._now_ns()
        if linear_speed > 1.0e-9 or angular_speed > 1.0e-9:
            self.nonzero_twist_messages += 1

    def _observation_callback(self, message: Bool):
        if message.data:
            self.observation_valid_messages += 1

    def _inference_callback(self, message: Bool):
        if message.data:
            self.inference_valid_messages += 1

    def _controller_status_callback(self, message: String):
        try:
            value = json.loads(message.data)
        except json.JSONDecodeError:
            return
        if isinstance(value, dict):
            self.latest_controller_status = value
            if value.get("valid") is True:
                self.controller_valid_messages += 1

    def _policy_delta_callback(self, message: Float32MultiArray):
        try:
            value = _finite_vector(message.data, 5, "final_delta")
        except ValueError:
            return
        self.latest_policy_delta = value
        self.policy_delta_messages += 1

    def _timer_callback(self):
        now_ns = self._now_ns()
        duration_s = min(max((now_ns - self.last_update_ns) * 1.0e-9, 0.0), 0.05)
        self.last_update_ns = now_ns
        twist = np.zeros(6, dtype=np.float64)
        if self.started and self.blocked_reason is None and self.latest_twist_ns:
            age_s = (now_ns - self.latest_twist_ns) * 1.0e-9
            if age_s <= float(self.get_parameter("command_timeout_s").value):
                twist = self.latest_twist
        if np.any(twist):
            previous = self.transform_base_eef[:3, 3].copy()
            self.transform_base_eef = integrate_base_frame_twist(
                self.transform_base_eef, twist, duration_s
            )
            self.path_length_m += float(
                np.linalg.norm(self.transform_base_eef[:3, 3] - previous)
            )
            self.path_points.append(self.transform_base_eef[:3, 3].copy())
            if self.path_length_m > float(
                self.get_parameter("maximum_path_length_m").value
            ):
                self.blocked_reason = "simulated path length exceeded its limit"
                self.latest_twist = np.zeros(6, dtype=np.float64)
        self._publish_state()
        period_ns = int(
            max(float(self.get_parameter("report_write_period_s").value), 0.1)
            * 1.0e9
        )
        if now_ns - self.last_report_ns >= period_ns:
            self._write_report()
            self.last_report_ns = now_ns

    def _publish_state(self):
        stamp = self.get_clock().now().to_msg()
        self._broadcast_transform(
            self.base_frame, self.eef_frame, self.transform_base_eef, stamp
        )
        self._broadcast_transform(
            self.eef_frame, self.tcp_frame, self.transform_eef_tcp, stamp
        )
        self._broadcast_transform(
            self.eef_frame, self.book_frame, self.transform_eef_book, stamp
        )
        transform_base_book = self.transform_base_eef @ self.transform_eef_book
        self.eef_pose_publisher.publish(
            self._pose_message(self.transform_base_eef, stamp)
        )
        self.book_pose_publisher.publish(
            self._pose_message(transform_base_book, stamp)
        )
        joints = JointState()
        joints.header.stamp = stamp
        joints.name = ["drive_joint"]
        joints.position = [float(self.get_parameter("gripper_position").value)]
        self.joint_publisher.publish(joints)
        self.marker_publisher.publish(
            self._markers(transform_base_book, stamp)
        )
        self.status_publisher.publish(
            String(data=json.dumps(self._report(), sort_keys=True))
        )

    def _broadcast_transform(self, parent, child, transform, stamp):
        message = TransformStamped()
        message.header.stamp = stamp
        message.header.frame_id = parent
        message.child_frame_id = child
        message.transform.translation.x = float(transform[0, 3])
        message.transform.translation.y = float(transform[1, 3])
        message.transform.translation.z = float(transform[2, 3])
        quaternion = matrix_to_quaternion_xyzw(transform[:3, :3])
        message.transform.rotation.x = float(quaternion[0])
        message.transform.rotation.y = float(quaternion[1])
        message.transform.rotation.z = float(quaternion[2])
        message.transform.rotation.w = float(quaternion[3])
        self.tf_broadcaster.sendTransform(message)

    def _pose_message(self, transform, stamp):
        message = PoseStamped()
        message.header.stamp = stamp
        message.header.frame_id = self.base_frame
        message.pose.position.x = float(transform[0, 3])
        message.pose.position.y = float(transform[1, 3])
        message.pose.position.z = float(transform[2, 3])
        quaternion = matrix_to_quaternion_xyzw(transform[:3, :3])
        message.pose.orientation.x = float(quaternion[0])
        message.pose.orientation.y = float(quaternion[1])
        message.pose.orientation.z = float(quaternion[2])
        message.pose.orientation.w = float(quaternion[3])
        return message

    def _markers(self, transform_base_book, stamp):
        book = Marker()
        book.header.frame_id = self.base_frame
        book.header.stamp = stamp
        book.ns = "policy_servo_sim"
        book.id = 1
        book.type = Marker.CUBE
        book.action = Marker.ADD
        book.pose = self._pose_message(transform_base_book, stamp).pose
        size = _finite_vector(
            self.get_parameter("book_size_xyz").value, 3, "book_size_xyz"
        )
        book.scale.x, book.scale.y, book.scale.z = map(float, size)
        book.color.r = 0.10
        book.color.g = 0.85
        book.color.b = 0.95
        book.color.a = 0.55

        slot = Marker()
        slot.header.frame_id = self.base_frame
        slot.header.stamp = stamp
        slot.ns = "policy_servo_sim"
        slot.id = 2
        slot.type = Marker.CUBE
        slot.action = Marker.ADD
        slot_depth = float(self.get_parameter("slot_depth_m").value)
        transform_base_slot_volume = self.transform_base_slot @ make_transform(
            [0.5 * slot_depth, 0.0, 0.0]
        )
        slot.pose = self._pose_message(transform_base_slot_volume, stamp).pose
        slot.scale.x = slot_depth
        slot.scale.y = float(self.get_parameter("slot_width_m").value)
        slot.scale.z = float(size[2])
        slot.color.r = 0.20
        slot.color.g = 0.95
        slot.color.b = 0.25
        slot.color.a = 0.18

        path = Marker()
        path.header.frame_id = self.base_frame
        path.header.stamp = stamp
        path.ns = "policy_servo_sim"
        path.id = 3
        path.type = Marker.LINE_STRIP
        path.action = Marker.ADD
        path.scale.x = 0.003
        path.color.r = 1.0
        path.color.g = 0.85
        path.color.b = 0.10
        path.color.a = 1.0
        for value in self.path_points[-2000:]:
            point = Point()
            point.x, point.y, point.z = map(float, value)
            path.points.append(point)

        marker_size = float(self.get_parameter("marker_size_m").value)
        marker_thickness = float(
            self.get_parameter("marker_thickness_m").value
        )
        transform_base_marker = transform_base_book @ self.transform_book_marker
        aruco = self._cube_marker(
            marker_id=4,
            name="aruco_marker_0",
            transform=transform_base_marker,
            size_xyz=[marker_size, marker_size, marker_thickness],
            color=[0.03, 0.03, 0.03, 1.0],
            stamp=stamp,
        )

        shelf = shelf_box_from_slot(
            self.transform_base_slot,
            base_frame=self.base_frame,
            size_xyz=self.get_parameter("shelf_size_xyz").value,
            center_offset_slot_xyz=self.get_parameter(
                "shelf_center_offset_slot_xyz"
            ).value,
            level_with_base=True,
            bottom_height_base_m=float(
                self.get_parameter("shelf_bottom_height_base_m").value
            ),
        )
        shelf_marker = self._cube_marker(
            marker_id=5,
            name="bookshelf",
            transform=shelf.transform_frame_box,
            size_xyz=shelf.size_xyz,
            color=[0.72, 0.22, 0.12, 0.20],
            stamp=stamp,
        )
        table = configured_box(
            frame_id=self.base_frame,
            size_xyz=self.get_parameter("table_size_xyz").value,
            center_xyz=self.get_parameter("table_center_base_xyz").value,
            quaternion_xyzw=self.get_parameter(
                "table_quaternion_base_xyzw"
            ).value,
            label="table",
        )
        table_marker = self._cube_marker(
            marker_id=6,
            name="table",
            transform=table.transform_frame_box,
            size_xyz=table.size_xyz,
            color=[0.30, 0.34, 0.40, 0.75],
            stamp=stamp,
        )

        transform_base_tcp = self.transform_base_eef @ self.transform_eef_tcp
        gripper = self._cube_marker(
            marker_id=7,
            name="gripper",
            transform=transform_base_tcp,
            size_xyz=[0.10, 0.09, 0.04],
            color=[0.88, 0.88, 0.90, 0.75],
            stamp=stamp,
        )
        robot_base = self._cylinder_marker(
            marker_id=8,
            name="xarm_base",
            center_xyz=[0.0, 0.0, 0.06],
            size_xyz=[0.16, 0.16, 0.12],
            color=[0.82, 0.84, 0.88, 0.90],
            stamp=stamp,
        )
        robot_arm = self._robot_arm_marker(stamp)
        return MarkerArray(
            markers=[
                table_marker,
                shelf_marker,
                slot,
                robot_base,
                robot_arm,
                gripper,
                book,
                aruco,
                path,
            ]
        )

    def _cube_marker(
        self, *, marker_id, name, transform, size_xyz, color, stamp
    ):
        marker = Marker()
        marker.header.frame_id = self.base_frame
        marker.header.stamp = stamp
        marker.ns = "policy_servo_sim_scene"
        marker.id = int(marker_id)
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose = self._pose_message(transform, stamp).pose
        marker.scale.x, marker.scale.y, marker.scale.z = map(
            float, _finite_vector(size_xyz, 3, f"{name}_size_xyz")
        )
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = map(
            float, color
        )
        marker.text = str(name)
        return marker

    def _cylinder_marker(
        self, *, marker_id, name, center_xyz, size_xyz, color, stamp
    ):
        marker = self._cube_marker(
            marker_id=marker_id,
            name=name,
            transform=make_transform(center_xyz),
            size_xyz=size_xyz,
            color=color,
            stamp=stamp,
        )
        marker.type = Marker.CYLINDER
        return marker

    def _robot_arm_marker(self, stamp):
        marker = Marker()
        marker.header.frame_id = self.base_frame
        marker.header.stamp = stamp
        marker.ns = "policy_servo_sim_scene"
        marker.id = 9
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.055
        marker.color.r = 0.82
        marker.color.g = 0.84
        marker.color.b = 0.88
        marker.color.a = 0.90
        eef = self.transform_base_eef[:3, 3]
        points = (
            [0.0, 0.0, 0.12],
            [0.12, 0.0, 0.28],
            [0.35, 0.5 * float(eef[1]), max(float(eef[2]) + 0.12, 0.30)],
            eef,
        )
        for value in points:
            point = Point()
            point.x, point.y, point.z = map(float, value)
            marker.points.append(point)
        return marker

    def _slot_book_transform(self):
        transform_base_book = self.transform_base_eef @ self.transform_eef_book
        return invert_transform(self.transform_base_slot) @ transform_base_book

    def _report(self) -> dict:
        transform_slot_book = self._slot_book_transform()
        initial_base_book = (
            self.initial_transform_base_eef @ self.transform_eef_book
        )
        initial_slot_book = (
            invert_transform(self.transform_base_slot) @ initial_base_book
        )
        forward_progress = float(
            transform_slot_book[0, 3] - initial_slot_book[0, 3]
        )
        minimum_progress = float(
            self.get_parameter("minimum_forward_progress_m").value
        )
        terminal_reason = None
        if isinstance(self.latest_controller_status, dict):
            terminal_reason = self.latest_controller_status.get("terminal_reason")
        bounded_stop_reached = (
            terminal_reason == MAXIMUM_SUPERVISED_TRANSLATION_REASON
        )
        forward_progress_check_passed = forward_progress >= minimum_progress
        passed = bool(
            self.started
            and self.nonzero_twist_messages > 0
            and self.observation_valid_messages > 0
            and self.inference_valid_messages > 0
            and self.controller_valid_messages > 0
            and self.policy_delta_messages > 0
            and self.blocked_reason is None
            and bounded_stop_reached
        )
        return {
            "kind": "bookshelf_policy_servo_software_simulation",
            "candidate_id": str(self.get_parameter("candidate_id").value),
            "passed": passed,
            "reason": self.blocked_reason,
            "simulation_only": True,
            "execution_authorized": False,
            "hardware_commanded": False,
            "servo_started": self.started,
            "twist_messages": self.twist_messages,
            "nonzero_twist_messages": self.nonzero_twist_messages,
            "observation_valid_messages": self.observation_valid_messages,
            "inference_valid_messages": self.inference_valid_messages,
            "controller_valid_messages": self.controller_valid_messages,
            "policy_delta_messages": self.policy_delta_messages,
            "latest_policy_delta": (
                None
                if self.latest_policy_delta is None
                else self.latest_policy_delta.astype(float).tolist()
            ),
            "bounded_stop_reached": bounded_stop_reached,
            "path_length_m": self.path_length_m,
            "forward_progress_m": forward_progress,
            "minimum_forward_progress_m": minimum_progress,
            "forward_progress_check_passed": forward_progress_check_passed,
            "maximum_linear_speed_m_s": self.maximum_linear_speed_m_s,
            "maximum_angular_speed_rad_s": self.maximum_angular_speed_rad_s,
            "initial_book_pose_slot": transform_to_dict(initial_slot_book),
            "final_book_pose_slot": transform_to_dict(transform_slot_book),
            "final_eef_pose_base": transform_to_dict(self.transform_base_eef),
            "latest_controller_status": self.latest_controller_status,
        }

    def _write_report(self):
        output_dir = Path(
            str(self.get_parameter("output_dir").value)
        ).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "policy_servo_simulation_report.json"
        path.write_text(
            json.dumps(self._report(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def close(self):
        self._write_report()


def main(args=None):
    rclpy.init(args=args)
    node = PolicyServoSimulator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
