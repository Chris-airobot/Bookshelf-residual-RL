#!/usr/bin/env python3
"""Run the trained task's scripted release and retreat on fake xArm hardware."""

from __future__ import annotations

import json
import math

from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import TransformStamped, TwistStamped
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import Bool, Int32, String
from std_srvs.srv import Trigger
import tf2_ros
from trajectory_msgs.msg import JointTrajectoryPoint

from .policy_tool_control_math import (
    invert_transform,
    make_transform,
    matrix_to_quaternion_xyzw,
)


def _transform_message_to_matrix(message) -> np.ndarray:
    transform = message.transform
    return make_transform(
        [transform.translation.x, transform.translation.y, transform.translation.z],
        [
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
            transform.rotation.w,
        ],
    )


def retreat_progress(start_xyz, current_xyz, direction_xyz) -> float:
    """Return signed travel from start along a normalized retreat direction."""
    start = np.asarray(start_xyz, dtype=np.float64)
    current = np.asarray(current_xyz, dtype=np.float64)
    direction = np.asarray(direction_xyz, dtype=np.float64)
    if start.shape != (3,) or current.shape != (3,) or direction.shape != (3,):
        raise ValueError("retreat vectors must have three elements")
    if not np.all(np.isfinite(np.concatenate((start, current, direction)))):
        raise ValueError("retreat vectors must be finite")
    magnitude = float(np.linalg.norm(direction))
    if magnitude <= 0.0:
        raise ValueError("retreat direction must be nonzero")
    return float(np.dot(current - start, direction / magnitude))


def grasp_alignment_target_eef(
    current_eef_transform,
    nominal_eef_book_transform,
    adjusted_eef_book_transform,
) -> np.ndarray:
    """Keep the book fixed while changing its rigid offset from the EEF."""

    transforms = [
        np.asarray(value, dtype=np.float64)
        for value in (
            current_eef_transform,
            nominal_eef_book_transform,
            adjusted_eef_book_transform,
        )
    ]
    if any(value.shape != (4, 4) for value in transforms):
        raise ValueError("grasp alignment transforms must be 4x4")
    if not all(np.all(np.isfinite(value)) for value in transforms):
        raise ValueError("grasp alignment transforms must be finite")
    current_eef, nominal_eef_book, adjusted_eef_book = transforms
    return current_eef @ nominal_eef_book @ invert_transform(adjusted_eef_book)


def simulated_book_push_distance(
    push_progress_m: float,
    contact_distance_m: float,
    requested_book_distance_m: float,
) -> float:
    """Return fake book travel after the closed gripper reaches contact."""

    values = np.asarray(
        [push_progress_m, contact_distance_m, requested_book_distance_m],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("push distances must be finite and nonnegative")
    return float(
        np.clip(
            float(push_progress_m) - float(contact_distance_m),
            0.0,
            float(requested_book_distance_m),
        )
    )


def required_book_push_distance(
    current_trailing_depth_m: float,
    target_trailing_depth_m: float,
) -> float:
    """Return insertion travel needed to reach the task's trailing-edge target."""

    values = np.asarray(
        [current_trailing_depth_m, target_trailing_depth_m], dtype=np.float64
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("trailing-edge depths must be finite")
    return max(0.0, float(target_trailing_depth_m - current_trailing_depth_m))


def oriented_box_contact_gap(
    contact_point_xyz,
    box_transform,
    box_size_xyz,
    approach_direction_xyz,
) -> float:
    """Return the signed gap from a point to an oriented box's near face."""

    point = np.asarray(contact_point_xyz, dtype=np.float64)
    transform = np.asarray(box_transform, dtype=np.float64)
    size = np.asarray(box_size_xyz, dtype=np.float64)
    direction = np.asarray(approach_direction_xyz, dtype=np.float64)
    if point.shape != (3,) or transform.shape != (4, 4):
        raise ValueError("contact point and box transform have invalid shapes")
    if size.shape != (3,) or direction.shape != (3,):
        raise ValueError("box size and approach direction must have three elements")
    values = np.concatenate((point, transform.reshape(-1), size, direction))
    if not np.all(np.isfinite(values)):
        raise ValueError("contact geometry must be finite")
    if np.any(size <= 0.0):
        raise ValueError("box dimensions must be positive")
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 0.0:
        raise ValueError("approach direction must be nonzero")

    direction = direction / direction_norm
    local_direction = transform[:3, :3].T @ direction
    support_radius = float(np.dot(np.abs(local_direction), size * 0.5))
    near_face_position = float(np.dot(transform[:3, 3], direction)) - support_radius
    return near_face_position - float(np.dot(point, direction))


def physical_release_guard_state(
    transform_base_slot,
    transform_base_tcp,
    transform_base_book,
    book_size_xyz,
    tcp_x_limit_m: float,
    minimum_book_leading_penetration_m: float,
) -> dict:
    """Measure whether the simulated xArm must release at the shelf mouth."""

    transforms = [
        np.asarray(value, dtype=np.float64)
        for value in (transform_base_slot, transform_base_tcp, transform_base_book)
    ]
    if any(value.shape != (4, 4) for value in transforms):
        raise ValueError("release-guard transforms must be 4x4")
    size = np.asarray(book_size_xyz, dtype=np.float64)
    values = np.concatenate(
        [value.reshape(-1) for value in transforms]
        + [
            size.reshape(-1),
            np.asarray(
                [tcp_x_limit_m, minimum_book_leading_penetration_m],
                dtype=np.float64,
            ),
        ]
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("release-guard geometry must be finite")
    if size.shape != (3,) or np.any(size <= 0.0):
        raise ValueError("release-guard book dimensions must be positive")
    if minimum_book_leading_penetration_m < 0.0:
        raise ValueError("minimum book penetration must be nonnegative")

    transform_slot_base = invert_transform(transforms[0])
    transform_slot_tcp = transform_slot_base @ transforms[1]
    transform_slot_book = transform_slot_base @ transforms[2]
    half = size * 0.5
    corners_book = np.asarray(
        [
            [x, y, z, 1.0]
            for x in (-half[0], half[0])
            for y in (-half[1], half[1])
            for z in (-half[2], half[2])
        ],
        dtype=np.float64,
    )
    corners_slot = (transform_slot_book @ corners_book.T).T[:, :3]
    tcp_slot_x_m = float(transform_slot_tcp[0, 3])
    leading_penetration_m = float(np.max(corners_slot[:, 0]))
    trailing_depth_m = float(np.min(corners_slot[:, 0]))
    boundary_reached = tcp_slot_x_m >= float(tcp_x_limit_m)
    book_supported = (
        leading_penetration_m >= float(minimum_book_leading_penetration_m)
    )
    return {
        "tcp_slot_x_m": tcp_slot_x_m,
        "tcp_x_limit_m": float(tcp_x_limit_m),
        "book_leading_penetration_m": leading_penetration_m,
        "book_trailing_depth_m": trailing_depth_m,
        "minimum_book_leading_penetration_m": float(
            minimum_book_leading_penetration_m
        ),
        "physical_boundary_reached": bool(boundary_reached),
        "book_supported": bool(book_supported),
        "release_allowed": bool(boundary_reached and book_supported),
    }


class FakeReleaseRetreatSequence(Node):
    """Coordinate fake release/retreat and the policy-controlled push stage."""

    def __init__(self):
        super().__init__("fake_release_retreat_sequence")
        self._declare_parameters()
        if not bool(self.get_parameter("simulation_only").value):
            raise ValueError("fake release/retreat requires simulation_only=true")

        self.base_frame = str(self.get_parameter("base_frame").value)
        self.eef_frame = str(self.get_parameter("eef_frame").value)
        self.tcp_frame = str(self.get_parameter("tcp_frame").value)
        self.book_frame = str(self.get_parameter("book_frame").value)
        self.book_size_xyz = np.asarray(
            self.get_parameter("book_size_xyz").value,
            dtype=np.float64,
        )
        if (
            self.book_size_xyz.shape != (3,)
            or not np.all(np.isfinite(self.book_size_xyz))
            or np.any(self.book_size_xyz <= 0.0)
        ):
            raise ValueError("book_size_xyz must contain three positive values")
        self.transform_eef_book = make_transform(
            self.get_parameter("eef_book_translation_xyz").value,
            self.get_parameter("eef_book_quaternion_xyzw").value,
        )
        self.nominal_transform_eef_book = make_transform(
            self.get_parameter("nominal_eef_book_translation_xyz").value,
            self.get_parameter("nominal_eef_book_quaternion_xyzw").value,
        )
        self.initial_grasp_alignment_enabled = bool(
            self.get_parameter("initial_grasp_alignment_enabled").value
        )
        self.physical_release_guard_enabled = bool(
            self.get_parameter("physical_release_guard_enabled").value
        )
        self.transform_base_slot = make_transform(
            self.get_parameter("slot_translation_base_xyz").value,
            self.get_parameter("slot_quaternion_base_xyzw").value,
        )
        direction = np.asarray(
            self.get_parameter("retreat_direction_base_xyz").value,
            dtype=np.float64,
        )
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError("retreat_direction_base_xyz must be a finite 3D vector")
        magnitude = float(np.linalg.norm(direction))
        if magnitude <= 0.0:
            raise ValueError("retreat_direction_base_xyz must be nonzero")
        self.retreat_direction = direction / magnitude

        latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.control_enable_publisher = self.create_publisher(
            Bool,
            str(self.get_parameter("policy_control_enable_topic").value),
            latched,
        )
        self.mode_publisher = self.create_publisher(
            Int32, str(self.get_parameter("mode_topic").value), latched
        )
        self.status_publisher = self.create_publisher(
            String, str(self.get_parameter("status_topic").value), latched
        )
        self.complete_publisher = self.create_publisher(
            Bool, str(self.get_parameter("complete_topic").value), latched
        )
        self.twist_publisher = self.create_publisher(
            TwistStamped,
            str(self.get_parameter("twist_command_topic").value),
            10,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("pretarget_ready_topic").value),
            self._pretarget_ready_callback,
            latched,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("policy_debug_topic").value),
            self._policy_debug_callback,
            10,
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.gripper_client = ActionClient(
            self,
            FollowJointTrajectory,
            str(self.get_parameter("gripper_action").value),
        )
        self.start_servo_client = self.create_client(
            Trigger, str(self.get_parameter("start_servo_service").value)
        )

        self.phase = "waiting_for_pretarget"
        self.pretarget_ready = False
        self.policy_release_action = None
        self.release_pending = False
        self.release_trigger_source = None
        self.physical_release_state = None
        self.book_attached = True
        self.transform_base_book = None
        self.alignment_target_eef = None
        self.alignment_book_transform = None
        self.alignment_distance_m = 0.0
        self.servo_start_pending = False
        self.servo_started_for_alignment = False
        self.retreat_start_xyz = None
        self.retreat_distance_m = 0.0
        self.push_start_xyz = None
        self.push_distance_m = 0.0
        self.book_push_distance_m = 0.0
        self.requested_book_push_distance_m = float(
            self.get_parameter("push_book_distance_m").value
        )
        self.push_book_origin = None
        self.push_contact_distance_m = None
        self.book_contact_gap_m = None
        self.gripper_goal_pending = False
        self.gripper_goal_kind = None
        self.gripper_goal_retry_started_ns = None
        self.gripper_goal_next_attempt_ns = 0
        self.phase_start_ns = self._now_ns()
        self.last_status_key = None
        self.timer = self.create_timer(0.02, self._timer_callback)
        self._publish_mode(0)
        self._publish_control(False)
        self.complete_publisher.publish(Bool(data=False))
        self._publish_status("waiting for fake xArm pre-target")

    def _declare_parameters(self):
        self.declare_parameter("simulation_only", True)
        self.declare_parameter("base_frame", "link_base")
        self.declare_parameter("eef_frame", "link_eef")
        self.declare_parameter("tcp_frame", "link_tcp")
        self.declare_parameter("book_frame", "target_book_center")
        self.declare_parameter("book_size_xyz", [0.156, 0.034, 0.236])
        self.declare_parameter("eef_book_translation_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter(
            "eef_book_quaternion_xyzw", [0.0, 0.0, 0.0, 1.0]
        )
        self.declare_parameter(
            "nominal_eef_book_translation_xyz", [0.0, 0.0, 0.0]
        )
        self.declare_parameter(
            "nominal_eef_book_quaternion_xyzw", [0.0, 0.0, 0.0, 1.0]
        )
        self.declare_parameter("initial_grasp_alignment_enabled", False)
        self.declare_parameter("initial_grasp_alignment_speed_m_s", 0.04)
        self.declare_parameter("initial_grasp_alignment_tolerance_m", 0.0005)
        self.declare_parameter("initial_grasp_alignment_timeout_s", 4.0)
        self.declare_parameter("physical_release_guard_enabled", False)
        self.declare_parameter("slot_translation_base_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter(
            "slot_quaternion_base_xyzw", [0.0, 0.0, 0.0, 1.0]
        )
        self.declare_parameter("physical_release_tcp_x_limit_m", -0.006)
        self.declare_parameter("minimum_book_leading_penetration_m", 0.08)
        self.declare_parameter("start_servo_service", "/servo_server/start_servo")
        self.declare_parameter(
            "pretarget_ready_topic", "/bookshelf_sim/pretarget_ready"
        )
        self.declare_parameter(
            "policy_control_enable_topic", "/bookshelf_sim/policy_control_enabled"
        )
        self.declare_parameter("mode_topic", "/bookshelf_policy/mode")
        self.declare_parameter("policy_debug_topic", "/bookshelf_shadow/policy_debug")
        self.declare_parameter("status_topic", "/bookshelf_sim/task_status")
        self.declare_parameter("complete_topic", "/bookshelf_sim/task_complete")
        self.declare_parameter(
            "twist_command_topic", "/servo_server/delta_twist_cmds"
        )
        self.declare_parameter(
            "gripper_action",
            "/xarm_gripper_traj_controller/follow_joint_trajectory",
        )
        self.declare_parameter("gripper_joint_name", "drive_joint")
        # Official xArm gripper convention: 0.0 is open, about 0.85 is closed.
        self.declare_parameter("gripper_open_position", 0.0)
        self.declare_parameter("gripper_closed_position", 0.85)
        self.declare_parameter("gripper_move_duration_s", 0.6)
        self.declare_parameter("gripper_goal_retry_timeout_s", 15.0)
        self.declare_parameter("gripper_goal_retry_period_s", 0.25)
        self.declare_parameter("release_threshold", 0.5)
        self.declare_parameter("retreat_direction_base_xyz", [-1.0, 0.0, 0.0])
        self.declare_parameter("retreat_distance_m", 0.09)
        self.declare_parameter("retreat_speed_m_s", 0.05)
        self.declare_parameter("retreat_timeout_s", 6.0)
        self.declare_parameter("push_book_distance_m", 0.03)
        self.declare_parameter("push_to_target_trailing_depth_enabled", False)
        self.declare_parameter("push_target_trailing_depth_m", -0.012)
        self.declare_parameter("push_timeout_s", 90.0)
        self.declare_parameter("contact_tolerance_m", 0.001)
        self.declare_parameter("tf_lookup_timeout_s", 0.02)

    def _now_ns(self) -> int:
        return int(self.get_clock().now().nanoseconds)

    def _pretarget_ready_callback(self, message: Bool):
        self.pretarget_ready = bool(message.data)

    def _policy_debug_callback(self, message: String):
        try:
            debug = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            return
        if not isinstance(debug, dict) or not bool(debug.get("valid", False)):
            return
        if self.phase == "waiting_for_push_policy":
            observation = debug.get("observation_12d")
            if (
                isinstance(observation, list)
                and observation
                and math.isclose(float(observation[0]), 1.0, abs_tol=1.0e-4)
            ):
                self.phase = "push"
                self.phase_start_ns = self._now_ns()
                self._publish_control(True)
                self._publish_status("policy push running")
            return
        if self.phase != "insert":
            return
        release_action = float(debug.get("release_action", float("nan")))
        if not math.isfinite(release_action):
            return
        self.policy_release_action = release_action
        if release_action > float(self.get_parameter("release_threshold").value):
            self.release_pending = True

    def _lookup_base_frame(self, frame):
        try:
            message = self.tf_buffer.lookup_transform(
                self.base_frame,
                frame,
                Time(),
                timeout=Duration(
                    seconds=float(self.get_parameter("tf_lookup_timeout_s").value)
                ),
            )
        except Exception:
            return None
        return _transform_message_to_matrix(message)

    def _timer_callback(self):
        transform_base_eef = self._lookup_base_frame(self.eef_frame)
        transform_base_tcp = self._lookup_base_frame(self.tcp_frame)
        if transform_base_eef is not None and self.book_attached:
            if self.phase == "aligning_grasp" and self.alignment_book_transform is not None:
                self.transform_base_book = self.alignment_book_transform.copy()
            elif self.phase == "waiting_for_pretarget" and self.initial_grasp_alignment_enabled:
                self.transform_base_book = (
                    transform_base_eef @ self.nominal_transform_eef_book
                )
            else:
                self.transform_base_book = transform_base_eef @ self.transform_eef_book
        self._broadcast_book_transform()

        if self.phase == "waiting_for_pretarget":
            if self.pretarget_ready and transform_base_eef is not None:
                self.phase_start_ns = self._now_ns()
                self._publish_mode(0)
                self._publish_control(False)
                self.gripper_goal_pending = False
                if self.initial_grasp_alignment_enabled:
                    self.alignment_book_transform = (
                        transform_base_eef @ self.nominal_transform_eef_book
                    )
                    self.alignment_target_eef = grasp_alignment_target_eef(
                        transform_base_eef,
                        self.nominal_transform_eef_book,
                        self.transform_eef_book,
                    )
                    self.phase = "aligning_grasp"
                    self._publish_status(
                        "aligning fake xArm for simulation-only grasp setback"
                    )
                else:
                    self.phase = "closing_for_insert"
                    self._publish_status(
                        "fake xArm pre-target ready; closing gripper for insertion"
                    )
            return

        if self.phase == "aligning_grasp":
            self._run_initial_grasp_alignment(transform_base_eef)
            return

        if self.phase == "closing_for_insert":
            if not self.gripper_goal_pending:
                self._send_gripper_goal(
                    float(self.get_parameter("gripper_closed_position").value),
                    "close_for_insert",
                )
            return

        if self.phase == "insert":
            self._run_insert_release_logic(transform_base_tcp)
            return

        if self.phase == "opening":
            if not self.gripper_goal_pending:
                self._send_gripper_goal(
                    float(self.get_parameter("gripper_open_position").value),
                    "open",
                )
            return

        if self.phase == "retreat":
            self._run_retreat(transform_base_eef)
            return

        if self.phase == "push":
            self._run_push(transform_base_eef, transform_base_tcp)
            return

        if self.phase == "closing":
            if not self.gripper_goal_pending:
                self._send_gripper_goal(
                    float(self.get_parameter("gripper_closed_position").value),
                    "close",
                )

    def _run_initial_grasp_alignment(self, transform_base_eef):
        if transform_base_eef is None or self.alignment_target_eef is None:
            self._publish_zero_twist()
            self._publish_status("waiting for EEF TF during grasp alignment")
            return
        elapsed_s = (self._now_ns() - self.phase_start_ns) * 1.0e-9
        if elapsed_s > float(
            self.get_parameter("initial_grasp_alignment_timeout_s").value
        ):
            self._fail("simulation-only grasp alignment timed out")
            return
        if not self._ensure_servo_started_for_alignment():
            return
        delta = self.alignment_target_eef[:3, 3] - transform_base_eef[:3, 3]
        distance = float(np.linalg.norm(delta))
        self.alignment_distance_m = distance
        tolerance = float(
            self.get_parameter("initial_grasp_alignment_tolerance_m").value
        )
        if distance <= tolerance:
            self._publish_zero_twist()
            self.phase = "closing_for_insert"
            self.phase_start_ns = self._now_ns()
            self.gripper_goal_pending = False
            self._publish_status(
                "simulation-only grasp alignment complete; closing gripper"
            )
            return
        speed = float(self.get_parameter("initial_grasp_alignment_speed_m_s").value)
        speed = min(speed, distance * 50.0)
        message = TwistStamped()
        message.header.frame_id = self.base_frame
        message.header.stamp = self.get_clock().now().to_msg()
        message.twist.linear.x = float(delta[0] / distance * speed)
        message.twist.linear.y = float(delta[1] / distance * speed)
        message.twist.linear.z = float(delta[2] / distance * speed)
        self.twist_publisher.publish(message)
        self._publish_status("aligning fake xArm for simulation-only grasp setback")

    def _run_insert_release_logic(self, transform_base_tcp):
        if not self.physical_release_guard_enabled:
            if self.release_pending and self.transform_base_book is not None:
                self._begin_release("policy")
            return
        if transform_base_tcp is None or self.transform_base_book is None:
            return

        self.physical_release_state = physical_release_guard_state(
            self.transform_base_slot,
            transform_base_tcp,
            self.transform_base_book,
            self.book_size_xyz,
            float(self.get_parameter("physical_release_tcp_x_limit_m").value),
            float(
                self.get_parameter(
                    "minimum_book_leading_penetration_m"
                ).value
            ),
        )
        supported = bool(self.physical_release_state["book_supported"])
        boundary_reached = bool(
            self.physical_release_state["physical_boundary_reached"]
        )
        if self.release_pending and supported:
            self._begin_release("policy")
            return
        if boundary_reached and supported:
            self._begin_release("physical_gripper_boundary")
            return
        if boundary_reached:
            self._fail(
                "physical gripper boundary reached before the book had enough "
                "shelf support; simulated grasp is too deep"
            )

    def _begin_release(self, source: str):
        self.release_pending = False
        self.release_trigger_source = str(source)
        self.requested_book_push_distance_m = float(
            self.get_parameter("push_book_distance_m").value
        )
        if (
            bool(
                self.get_parameter(
                    "push_to_target_trailing_depth_enabled"
                ).value
            )
            and self.physical_release_state is not None
        ):
            self.requested_book_push_distance_m = required_book_push_distance(
                float(self.physical_release_state["book_trailing_depth_m"]),
                float(
                    self.get_parameter("push_target_trailing_depth_m").value
                ),
            )
        self.book_attached = False
        self.phase = "opening"
        self.phase_start_ns = self._now_ns()
        self._publish_control(False)
        self._publish_mode(1)
        self._publish_zero_twist()
        if source == "policy":
            reason = "policy release accepted; opening fake gripper"
        else:
            reason = "physical gripper boundary reached; opening fake gripper"
        self._publish_status(reason)

    def _ensure_servo_started_for_alignment(self) -> bool:
        if self.servo_started_for_alignment:
            return True
        if self.servo_start_pending:
            self._publish_status("starting MoveIt Servo for grasp alignment")
            return False
        if not self.start_servo_client.service_is_ready():
            self._publish_status("waiting for MoveIt Servo start service")
            return False
        self.servo_start_pending = True
        future = self.start_servo_client.call_async(Trigger.Request())
        future.add_done_callback(self._servo_start_response)
        self._publish_status("starting MoveIt Servo for grasp alignment")
        return False

    def _servo_start_response(self, future):
        self.servo_start_pending = False
        try:
            response = future.result()
        except Exception as error:
            self._fail(f"MoveIt Servo start failed during grasp alignment: {error}")
            return
        if response is None or not bool(response.success):
            message = "no response" if response is None else response.message
            self._fail(f"MoveIt Servo rejected grasp alignment start: {message}")
            return
        self.servo_started_for_alignment = True
        self._publish_status("MoveIt Servo ready for grasp alignment")

    def _run_retreat(self, transform_base_eef):
        if transform_base_eef is None:
            self._publish_zero_twist()
            self._publish_status("waiting for EEF TF during scripted retreat")
            return
        current_xyz = transform_base_eef[:3, 3]
        if self.retreat_start_xyz is None:
            self.retreat_start_xyz = current_xyz.copy()
        self.retreat_distance_m = max(
            0.0,
            retreat_progress(
                self.retreat_start_xyz,
                current_xyz,
                self.retreat_direction,
            ),
        )
        requested_distance = float(self.get_parameter("retreat_distance_m").value)
        if self.retreat_distance_m >= requested_distance:
            self._publish_zero_twist()
            self.phase = "closing"
            self.phase_start_ns = self._now_ns()
            self.gripper_goal_pending = False
            self._publish_status("scripted retreat complete; closing fake gripper")
            return
        elapsed_s = (self._now_ns() - self.phase_start_ns) * 1.0e-9
        if elapsed_s > float(self.get_parameter("retreat_timeout_s").value):
            self._fail("scripted retreat timed out")
            return
        speed = float(self.get_parameter("retreat_speed_m_s").value)
        remaining = max(requested_distance - self.retreat_distance_m, 0.0)
        speed = min(speed, remaining * 50.0)
        message = TwistStamped()
        message.header.frame_id = self.base_frame
        message.header.stamp = self.get_clock().now().to_msg()
        message.twist.linear.x = float(self.retreat_direction[0] * speed)
        message.twist.linear.y = float(self.retreat_direction[1] * speed)
        message.twist.linear.z = float(self.retreat_direction[2] * speed)
        self.twist_publisher.publish(message)
        self._publish_status("scripted retreat running")

    def _run_push(self, transform_base_eef, transform_base_tcp):
        if transform_base_eef is None or transform_base_tcp is None:
            self._publish_status("waiting for EEF/TCP TF during policy push")
            return
        current_xyz = transform_base_eef[:3, 3]
        insertion_direction = -self.retreat_direction
        if self.push_start_xyz is None:
            self.push_start_xyz = current_xyz.copy()
            self.push_book_origin = self.transform_base_book.copy()
        self.push_distance_m = max(
            0.0,
            retreat_progress(
                self.push_start_xyz,
                current_xyz,
                insertion_direction,
            ),
        )
        requested_book_distance = self.requested_book_push_distance_m
        self.book_contact_gap_m = oriented_box_contact_gap(
            transform_base_tcp[:3, 3],
            self.push_book_origin,
            self.book_size_xyz,
            insertion_direction,
        )
        if self.push_contact_distance_m is None and self.book_contact_gap_m <= float(
            self.get_parameter("contact_tolerance_m").value
        ):
            self.push_contact_distance_m = max(
                0.0,
                self.push_distance_m + self.book_contact_gap_m,
            )
            self.get_logger().info(
                "closed gripper reached the simulated book at "
                f"{self.push_contact_distance_m * 1000.0:.1f} mm of return travel"
            )

        if self.push_contact_distance_m is None:
            self.book_push_distance_m = 0.0
        else:
            self.book_push_distance_m = simulated_book_push_distance(
                self.push_distance_m,
                self.push_contact_distance_m,
                requested_book_distance,
            )
        self.transform_base_book = self.push_book_origin.copy()
        self.transform_base_book[:3, 3] += (
            insertion_direction * self.book_push_distance_m
        )

        if self.book_push_distance_m >= requested_book_distance:
            self.phase = "complete"
            self.phase_start_ns = self._now_ns()
            self._publish_control(False)
            self._publish_zero_twist()
            self.complete_publisher.publish(Bool(data=True))
            self._publish_status("release, retreat, and policy push complete")
            return
        elapsed_s = (self._now_ns() - self.phase_start_ns) * 1.0e-9
        if elapsed_s > float(self.get_parameter("push_timeout_s").value):
            self._fail("policy push timed out")
            return
        self._publish_status("policy push running")

    def _send_gripper_goal(self, position: float, kind: str):
        now_ns = self._now_ns()
        if now_ns < self.gripper_goal_next_attempt_ns:
            return
        if not self.gripper_client.server_is_ready():
            self._publish_status(f"waiting for fake gripper before {kind}")
            return
        if self.gripper_goal_retry_started_ns is None:
            self.gripper_goal_retry_started_ns = now_ns
        duration_s = float(self.get_parameter("gripper_move_duration_s").value)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [
            str(self.get_parameter("gripper_joint_name").value)
        ]
        point = JointTrajectoryPoint()
        point.positions = [float(position)]
        duration_ns = int(round(duration_s * 1.0e9))
        point.time_from_start.sec = duration_ns // 1_000_000_000
        point.time_from_start.nanosec = duration_ns % 1_000_000_000
        goal.trajectory.points = [point]
        self.gripper_goal_pending = True
        self.gripper_goal_kind = kind
        future = self.gripper_client.send_goal_async(goal)
        future.add_done_callback(self._gripper_goal_response)

    def _gripper_goal_response(self, future):
        try:
            goal_handle = future.result()
        except Exception as error:
            self._fail(f"fake gripper goal failed: {error}")
            return
        if goal_handle is None or not goal_handle.accepted:
            elapsed_s = (
                self._now_ns() - self.gripper_goal_retry_started_ns
            ) * 1.0e-9
            timeout_s = float(
                self.get_parameter("gripper_goal_retry_timeout_s").value
            )
            if elapsed_s >= timeout_s:
                self._fail(
                    "fake gripper goal remained rejected for "
                    f"{elapsed_s:.1f} s"
                )
                return
            retry_period_s = float(
                self.get_parameter("gripper_goal_retry_period_s").value
            )
            self.gripper_goal_pending = False
            retry_kind = self.gripper_goal_kind
            self.gripper_goal_kind = None
            self.gripper_goal_next_attempt_ns = self._now_ns() + int(
                max(retry_period_s, 0.02) * 1.0e9
            )
            self._publish_status(
                "fake gripper controller is starting; retrying "
                f"{retry_kind}"
            )
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._gripper_goal_result)

    def _gripper_goal_result(self, future):
        try:
            wrapped_result = future.result()
            error_code = int(wrapped_result.result.error_code)
        except Exception as error:
            self._fail(f"fake gripper result failed: {error}")
            return
        if error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            self._fail(f"fake gripper trajectory error {error_code}")
            return
        kind = self.gripper_goal_kind
        self.gripper_goal_pending = False
        self.gripper_goal_kind = None
        self.gripper_goal_retry_started_ns = None
        self.gripper_goal_next_attempt_ns = 0
        if kind == "close_for_insert":
            self.phase = "insert"
            self.phase_start_ns = self._now_ns()
            self._publish_mode(0)
            self._publish_control(True)
            self._publish_status("fake gripper closed; policy insertion running")
            return
        if kind == "open":
            self.phase = "retreat"
            self.phase_start_ns = self._now_ns()
            self.retreat_start_xyz = None
            self._publish_status("fake gripper open; starting scripted retreat")
            return
        if kind == "close":
            self.phase = "waiting_for_push_policy"
            self.phase_start_ns = self._now_ns()
            self.push_start_xyz = None
            self.push_book_origin = None
            self.push_contact_distance_m = None
            self.book_contact_gap_m = None
            self._publish_mode(2)
            self._publish_control(False)
            self.complete_publisher.publish(Bool(data=False))
            self._publish_status("fake gripper closed; waiting for PUSH-mode policy")

    def _broadcast_book_transform(self):
        if self.transform_base_book is None:
            return
        transform = self.transform_base_book
        quaternion = matrix_to_quaternion_xyzw(transform[:3, :3])
        message = TransformStamped()
        message.header.frame_id = self.base_frame
        message.header.stamp = self.get_clock().now().to_msg()
        message.child_frame_id = self.book_frame
        message.transform.translation.x = float(transform[0, 3])
        message.transform.translation.y = float(transform[1, 3])
        message.transform.translation.z = float(transform[2, 3])
        message.transform.rotation.x = float(quaternion[0])
        message.transform.rotation.y = float(quaternion[1])
        message.transform.rotation.z = float(quaternion[2])
        message.transform.rotation.w = float(quaternion[3])
        self.tf_broadcaster.sendTransform(message)

    def _publish_control(self, enabled: bool):
        self.control_enable_publisher.publish(Bool(data=bool(enabled)))

    def _publish_mode(self, mode: int):
        self.mode_publisher.publish(Int32(data=int(mode)))

    def _publish_zero_twist(self):
        message = TwistStamped()
        message.header.frame_id = self.base_frame
        message.header.stamp = self.get_clock().now().to_msg()
        self.twist_publisher.publish(message)

    def _fail(self, reason: str):
        self.phase = "failed"
        self._publish_control(False)
        self._publish_zero_twist()
        self.complete_publisher.publish(Bool(data=False))
        self._publish_status(reason)

    def _publish_status(self, reason: str):
        report = {
            "valid": self.phase != "failed",
            "phase": self.phase,
            "reason": str(reason),
            "simulation_only": True,
            "hardware_commanded": False,
            "policy_release_action": self.policy_release_action,
            "release_trigger_source": self.release_trigger_source,
            "physical_release_guard_enabled": self.physical_release_guard_enabled,
            "physical_release_guard": self.physical_release_state,
            "gripper_positions": {
                "open": float(self.get_parameter("gripper_open_position").value),
                "closed": float(
                    self.get_parameter("gripper_closed_position").value
                ),
            },
            "book_attached": self.book_attached,
            "initial_grasp_alignment_enabled": self.initial_grasp_alignment_enabled,
            "grasp_alignment_remaining_m": self.alignment_distance_m,
            "retreat_distance_m": self.retreat_distance_m,
            "requested_retreat_distance_m": float(
                self.get_parameter("retreat_distance_m").value
            ),
            "push_distance_m": self.push_distance_m,
            "push_contact_distance_m": self.push_contact_distance_m,
            "book_contact_gap_m": self.book_contact_gap_m,
            "book_push_distance_m": self.book_push_distance_m,
            "requested_book_push_distance_m": self.requested_book_push_distance_m,
            "push_target_trailing_depth_m": float(
                self.get_parameter("push_target_trailing_depth_m").value
            ),
            "task_complete": self.phase == "complete",
        }
        key = (self.phase, reason)
        self.status_publisher.publish(String(data=json.dumps(report, sort_keys=True)))
        if key == self.last_status_key:
            return
        self.last_status_key = key
        if self.phase == "failed":
            self.get_logger().error(reason)
        else:
            self.get_logger().info(reason)

    def destroy_node(self):
        self._publish_control(False)
        self._publish_zero_twist()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = FakeReleaseRetreatSequence()
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
