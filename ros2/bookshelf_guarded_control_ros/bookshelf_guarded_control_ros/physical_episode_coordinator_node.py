#!/usr/bin/env python3
"""Coordinate one live-marker xArm bookshelf episode."""

from __future__ import annotations

import json
import math

from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import TwistStamped
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

from .fake_release_retreat_sequence_node import (
    physical_release_guard_state,
    required_book_push_distance,
    retreat_progress,
)
from .physical_episode_coordinator_math import (
    validate_episode_operation,
    trailing_depth_target_reached,
)
from .policy_tool_control_math import make_transform


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


class PhysicalEpisodeCoordinator(Node):
    """Hand policy control through release, retreat, and marker-stopped push."""

    def __init__(self):
        super().__init__("physical_episode_coordinator")
        self._declare_parameters()
        self.operation = validate_episode_operation(
            self.get_parameter("operation").value,
            self.get_parameter("authorization_token").value,
        )
        self.control_mode = self.operation == "control"

        self.base_frame = str(self.get_parameter("base_frame").value)
        self.eef_frame = str(self.get_parameter("eef_frame").value)
        self.tcp_frame = str(self.get_parameter("tcp_frame").value)
        self.book_frame = str(self.get_parameter("book_frame").value)
        self.book_size_xyz = self._positive_vector_parameter(
            "book_size_xyz", length=3
        )
        self.transform_base_slot = make_transform(
            self.get_parameter("slot_translation_base_xyz").value,
            self.get_parameter("slot_quaternion_base_xyzw").value,
        )
        direction = self._finite_vector_parameter(
            "retreat_direction_base_xyz", length=3
        )
        magnitude = float(np.linalg.norm(direction))
        if magnitude <= 0.0:
            raise ValueError("retreat_direction_base_xyz must be nonzero")
        self.retreat_direction = direction / magnitude
        self._validate_motion_parameters()

        latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.status_publisher = self.create_publisher(
            String, str(self.get_parameter("status_topic").value), latched
        )
        self.complete_publisher = self.create_publisher(
            Bool, str(self.get_parameter("complete_topic").value), latched
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
            String,
            str(self.get_parameter("policy_debug_topic").value),
            self._policy_debug_callback,
            10,
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Calculate mode deliberately creates none of these command interfaces.
        self.control_enable_publisher = None
        self.mode_publisher = None
        self.twist_publisher = None
        self.gripper_client = None
        self.start_service = None
        if self.control_mode:
            self.control_enable_publisher = self.create_publisher(
                Bool,
                str(self.get_parameter("policy_control_enable_topic").value),
                latched,
            )
            self.mode_publisher = self.create_publisher(
                Int32, str(self.get_parameter("mode_topic").value), latched
            )
            self.twist_publisher = self.create_publisher(
                TwistStamped,
                str(self.get_parameter("twist_command_topic").value),
                10,
            )
            self.gripper_client = ActionClient(
                self,
                FollowJointTrajectory,
                str(self.get_parameter("gripper_action").value),
            )
            self.start_service = self.create_service(
                Trigger,
                str(self.get_parameter("start_episode_service").value),
                self._start_episode_callback,
            )

        self.phase = "calculate" if not self.control_mode else "waiting_for_start"
        self.start_requested = bool(
            self.control_mode and self.get_parameter("start_immediately").value
        )
        self.latest_observation_valid = False
        self.latest_observation_valid_ns = None
        self.latest_inference_valid = False
        self.latest_inference_valid_ns = None
        self.latest_policy_debug = None
        self.latest_policy_debug_ns = None
        self.policy_release_action = None
        self.release_pending = False
        self.release_trigger_source = None
        self.physical_release_state = None
        self.current_trailing_depth_m = None
        self.remaining_book_push_distance_m = None
        self.push_policy_ready = False
        self.phase_start_ns = self._now_ns()
        self.retreat_start_xyz = None
        self.retreat_distance_m = 0.0
        self.push_start_xyz = None
        self.push_tcp_travel_m = 0.0
        self.gripper_goal_pending = False
        self.gripper_goal_kind = None
        self.gripper_goal_handle = None
        self.hardware_commanded = False
        self.last_status_key = None
        self.last_status_publish_ns = None

        if self.control_mode:
            self._publish_control(False)
            self._publish_mode(0)
        self.complete_publisher.publish(Bool(data=False))
        self.timer = self.create_timer(0.02, self._timer_callback)
        if self.control_mode:
            self.get_logger().warning(
                "PHYSICAL XARM FULL EPISODE is hardware-capable. The episode "
                "starts only after explicit authorization and start request."
            )
        else:
            self.get_logger().info(
                "CALCULATE-ONLY episode monitor created no gripper, Servo command, "
                "control-enable, or mode publisher."
            )
        self._publish_status("waiting for live episode geometry")

    def _declare_parameters(self):
        self.declare_parameter("operation", "calculate")
        self.declare_parameter("authorization_token", "")
        self.declare_parameter("start_immediately", False)
        self.declare_parameter("base_frame", "link_base")
        self.declare_parameter("eef_frame", "link_eef")
        self.declare_parameter("tcp_frame", "link_tcp")
        self.declare_parameter("book_frame", "target_book_center")
        self.declare_parameter("book_size_xyz", [0.156, 0.034, 0.236])
        self.declare_parameter("slot_translation_base_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter(
            "slot_quaternion_base_xyzw", [0.0, 0.0, 0.0, 1.0]
        )
        self.declare_parameter("physical_release_tcp_x_limit_m", -0.006)
        self.declare_parameter("minimum_book_leading_penetration_m", 0.08)
        self.declare_parameter("push_target_trailing_depth_m", -0.012)
        self.declare_parameter("push_target_tolerance_m", 0.001)
        self.declare_parameter("release_threshold", 0.5)
        self.declare_parameter("retreat_direction_base_xyz", [-1.0, 0.0, 0.0])
        self.declare_parameter("retreat_distance_m", 0.09)
        self.declare_parameter("retreat_speed_m_s", 0.025)
        self.declare_parameter("retreat_timeout_s", 10.0)
        self.declare_parameter("insert_timeout_s", 120.0)
        self.declare_parameter("maximum_push_tcp_travel_m", 0.14)
        self.declare_parameter("push_timeout_s", 90.0)
        self.declare_parameter("status_publish_period_s", 0.2)
        self.declare_parameter("message_max_age_s", 0.5)
        self.declare_parameter("tf_max_age_s", 0.5)
        self.declare_parameter("tf_lookup_timeout_s", 0.02)
        self.declare_parameter(
            "policy_control_enable_topic", "/bookshelf_control/episode_enable"
        )
        self.declare_parameter("mode_topic", "/bookshelf_policy/mode")
        self.declare_parameter("policy_debug_topic", "/bookshelf_shadow/policy_debug")
        self.declare_parameter(
            "observation_valid_topic", "/bookshelf_policy/observation_valid"
        )
        self.declare_parameter(
            "inference_valid_topic", "/bookshelf_shadow/inference_valid"
        )
        self.declare_parameter("status_topic", "/bookshelf_control/task_status")
        self.declare_parameter("complete_topic", "/bookshelf_control/task_complete")
        self.declare_parameter(
            "start_episode_service", "/bookshelf_control/start_full_episode"
        )
        self.declare_parameter(
            "twist_command_topic", "/servo_server/delta_twist_cmds"
        )
        self.declare_parameter(
            "gripper_action",
            "/xarm_gripper_traj_controller/follow_joint_trajectory",
        )
        self.declare_parameter("gripper_joint_name", "drive_joint")
        self.declare_parameter("gripper_open_position", 0.0)
        self.declare_parameter("gripper_closed_position", 0.85)
        self.declare_parameter("gripper_move_duration_s", 1.5)
        self.declare_parameter("gripper_server_timeout_s", 10.0)

    def _finite_vector_parameter(self, name: str, *, length: int) -> np.ndarray:
        value = np.asarray(self.get_parameter(name).value, dtype=np.float64)
        if value.shape != (length,) or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must contain {length} finite values")
        return value

    def _positive_vector_parameter(self, name: str, *, length: int) -> np.ndarray:
        value = self._finite_vector_parameter(name, length=length)
        if np.any(value <= 0.0):
            raise ValueError(f"{name} must contain positive values")
        return value

    def _validate_motion_parameters(self):
        positive = (
            "retreat_distance_m",
            "retreat_speed_m_s",
            "retreat_timeout_s",
            "insert_timeout_s",
            "maximum_push_tcp_travel_m",
            "push_timeout_s",
            "gripper_move_duration_s",
            "gripper_server_timeout_s",
            "status_publish_period_s",
        )
        for name in positive:
            value = float(self.get_parameter(name).value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        nonnegative = (
            "minimum_book_leading_penetration_m",
            "push_target_tolerance_m",
        )
        for name in nonnegative:
            value = float(self.get_parameter(name).value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("physical_release_tcp_x_limit_m", "push_target_trailing_depth_m"):
            if not math.isfinite(float(self.get_parameter(name).value)):
                raise ValueError(f"{name} must be finite")

    def _now_ns(self) -> int:
        return int(self.get_clock().now().nanoseconds)

    def _fresh(self, timestamp_ns) -> bool:
        if timestamp_ns is None:
            return False
        maximum_age_s = float(self.get_parameter("message_max_age_s").value)
        return (self._now_ns() - timestamp_ns) * 1.0e-9 <= maximum_age_s

    def _observation_valid_callback(self, message: Bool):
        self.latest_observation_valid = bool(message.data)
        self.latest_observation_valid_ns = self._now_ns()

    def _inference_valid_callback(self, message: Bool):
        self.latest_inference_valid = bool(message.data)
        self.latest_inference_valid_ns = self._now_ns()

    def _policy_debug_callback(self, message: String):
        try:
            debug = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            return
        if not isinstance(debug, dict) or not bool(debug.get("valid", False)):
            return
        self.latest_policy_debug = debug
        self.latest_policy_debug_ns = self._now_ns()
        release_action = float(debug.get("release_action", float("nan")))
        if math.isfinite(release_action):
            self.policy_release_action = release_action
            if (
                self.phase == "insert"
                and release_action > float(self.get_parameter("release_threshold").value)
            ):
                self.release_pending = True
        if self.phase == "waiting_for_push_policy":
            observation = debug.get("observation_12d")
            if (
                isinstance(observation, list)
                and observation
                and math.isclose(float(observation[0]), 1.0, abs_tol=1.0e-4)
            ):
                self.push_policy_ready = True

    def _start_episode_callback(self, _request, response):
        if self.phase != "waiting_for_start":
            response.success = False
            response.message = f"episode cannot start from phase {self.phase}"
            return response
        self.start_requested = True
        self.phase_start_ns = self._now_ns()
        response.success = True
        response.message = "episode start requested; waiting for fresh inputs"
        self._publish_status(response.message)
        return response

    def _lookup_base_frame(self, frame: str):
        try:
            message = self.tf_buffer.lookup_transform(
                self.base_frame,
                frame,
                Time(),
                timeout=Duration(
                    seconds=float(self.get_parameter("tf_lookup_timeout_s").value)
                ),
            )
        except Exception as error:
            return None, f"TF {self.base_frame} <- {frame} unavailable: {error}"
        stamp_ns = int(message.header.stamp.sec) * 1_000_000_000 + int(
            message.header.stamp.nanosec
        )
        maximum_age_s = float(self.get_parameter("tf_max_age_s").value)
        if maximum_age_s > 0.0 and stamp_ns > 0:
            age_s = (self._now_ns() - stamp_ns) * 1.0e-9
            if age_s > maximum_age_s:
                return None, f"TF {self.base_frame} <- {frame} is stale"
        return _transform_message_to_matrix(message), None

    def _policy_inputs_error(self) -> str | None:
        if not self.latest_observation_valid:
            return "observation_valid is false"
        if not self._fresh(self.latest_observation_valid_ns):
            return "observation_valid is missing or stale"
        if not self.latest_inference_valid:
            return "inference_valid is false"
        if not self._fresh(self.latest_inference_valid_ns):
            return "inference_valid is missing or stale"
        if self.latest_policy_debug is None or not self._fresh(
            self.latest_policy_debug_ns
        ):
            return "policy debug is missing or stale"
        return None

    def _geometry(self):
        transform_base_eef, eef_error = self._lookup_base_frame(self.eef_frame)
        transform_base_tcp, tcp_error = self._lookup_base_frame(self.tcp_frame)
        transform_base_book, book_error = self._lookup_base_frame(self.book_frame)
        error = eef_error or tcp_error or book_error
        if error:
            return None, error
        state = physical_release_guard_state(
            self.transform_base_slot,
            transform_base_tcp,
            transform_base_book,
            self.book_size_xyz,
            float(self.get_parameter("physical_release_tcp_x_limit_m").value),
            float(
                self.get_parameter("minimum_book_leading_penetration_m").value
            ),
        )
        self.physical_release_state = state
        self.current_trailing_depth_m = float(state["book_trailing_depth_m"])
        self.remaining_book_push_distance_m = required_book_push_distance(
            self.current_trailing_depth_m,
            float(self.get_parameter("push_target_trailing_depth_m").value),
        )
        return (transform_base_eef, transform_base_tcp, transform_base_book), None

    def _timer_callback(self):
        geometry, geometry_error = self._geometry()
        if self.phase == "calculate":
            reason = geometry_error or "live geometry valid; no command interfaces exist"
            self._publish_status(reason)
            return

        if self.phase in ("complete", "failed"):
            return

        if self.phase == "waiting_for_start":
            if not self.start_requested:
                self._publish_status(
                    "authorized control is idle; call the start service or use start_immediately"
                )
                return
            input_error = self._policy_inputs_error()
            if geometry_error or input_error:
                self._publish_status(geometry_error or input_error)
                return
            if not self.gripper_client.server_is_ready():
                elapsed_s = (self._now_ns() - self.phase_start_ns) * 1.0e-9
                if elapsed_s > float(
                    self.get_parameter("gripper_server_timeout_s").value
                ):
                    self._fail(
                        "physical gripper trajectory action is unavailable; "
                        "insertion was not started"
                    )
                    return
                self._publish_status(
                    "waiting for physical gripper action before insertion"
                )
                return
            self.phase = "insert"
            self.phase_start_ns = self._now_ns()
            self._publish_mode(0)
            self._publish_control(True)
            self.hardware_commanded = True
            self._publish_status("live inputs valid; policy insertion running")
            return

        if geometry_error:
            self._fail(geometry_error)
            return

        transform_base_eef, transform_base_tcp, _ = geometry
        if self.phase in ("insert", "push"):
            input_error = self._policy_inputs_error()
            if input_error:
                self._fail(input_error)
                return

        if self.phase == "insert":
            self._run_insert_release_logic()
        elif self.phase == "opening":
            self._ensure_gripper_goal("open")
        elif self.phase == "retreat":
            self._run_retreat(transform_base_eef)
        elif self.phase == "closing":
            self._ensure_gripper_goal("close")
        elif self.phase == "waiting_for_push_policy":
            self._run_waiting_for_push_policy(transform_base_eef)
        elif self.phase == "push":
            self._run_push(transform_base_eef)

    def _run_insert_release_logic(self):
        elapsed_s = (self._now_ns() - self.phase_start_ns) * 1.0e-9
        if elapsed_s > float(self.get_parameter("insert_timeout_s").value):
            self._fail("policy insertion timed out before a supported release")
            return
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
                "physical gripper boundary reached before the live book had enough shelf support"
            )
            return
        self._publish_status("policy insertion running")

    def _begin_release(self, source: str):
        self.release_pending = False
        self.release_trigger_source = str(source)
        self.phase = "opening"
        self.phase_start_ns = self._now_ns()
        self.gripper_goal_pending = False
        self._publish_control(False)
        self._publish_mode(1)
        self._publish_zero_twist()
        if source == "policy":
            reason = "supported policy release accepted; opening physical gripper"
        else:
            reason = "physical gripper boundary reached; opening physical gripper"
        self._publish_status(reason)

    def _run_retreat(self, transform_base_eef):
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
            self._publish_status("physical retreat complete; closing gripper for push")
            return
        elapsed_s = (self._now_ns() - self.phase_start_ns) * 1.0e-9
        if elapsed_s > float(self.get_parameter("retreat_timeout_s").value):
            self._fail("physical retreat timed out")
            return
        speed = float(self.get_parameter("retreat_speed_m_s").value)
        remaining = max(requested_distance - self.retreat_distance_m, 0.0)
        speed = min(speed, remaining * 50.0)
        self._publish_direction_twist(self.retreat_direction, speed)
        self.hardware_commanded = True
        self._publish_status("physical retreat running")

    def _run_waiting_for_push_policy(self, transform_base_eef):
        elapsed_s = (self._now_ns() - self.phase_start_ns) * 1.0e-9
        input_error = self._policy_inputs_error()
        if input_error:
            if elapsed_s > float(self.get_parameter("push_timeout_s").value):
                self._fail(f"PUSH-mode policy inputs did not settle: {input_error}")
                return
            self._publish_status(f"waiting for PUSH-mode inputs: {input_error}")
            return
        if not self.push_policy_ready:
            if elapsed_s > float(self.get_parameter("push_timeout_s").value):
                self._fail("PUSH-mode policy did not become ready")
                return
            self._publish_status("waiting for PUSH-mode policy")
            return
        self.push_start_xyz = transform_base_eef[:3, 3].copy()
        self.phase = "push"
        self.phase_start_ns = self._now_ns()
        self._publish_control(True)
        self.hardware_commanded = True
        self._publish_status("live-marker policy push running")

    def _run_push(self, transform_base_eef):
        target_depth = float(self.get_parameter("push_target_trailing_depth_m").value)
        tolerance = float(self.get_parameter("push_target_tolerance_m").value)
        if trailing_depth_target_reached(
            self.current_trailing_depth_m,
            target_depth,
            tolerance,
        ):
            self.phase = "complete"
            self._publish_control(False)
            self._publish_zero_twist()
            self.complete_publisher.publish(Bool(data=True))
            self._publish_status(
                "live book reached target trailing depth; full episode complete"
            )
            return
        insertion_direction = -self.retreat_direction
        self.push_tcp_travel_m = max(
            0.0,
            retreat_progress(
                self.push_start_xyz,
                transform_base_eef[:3, 3],
                insertion_direction,
            ),
        )
        if self.push_tcp_travel_m >= float(
            self.get_parameter("maximum_push_tcp_travel_m").value
        ):
            self._fail("maximum physical push TCP travel reached before book target")
            return
        elapsed_s = (self._now_ns() - self.phase_start_ns) * 1.0e-9
        if elapsed_s > float(self.get_parameter("push_timeout_s").value):
            self._fail("live-marker policy push timed out")
            return
        self._publish_status("live-marker policy push running")

    def _ensure_gripper_goal(self, kind: str):
        if self.gripper_goal_pending:
            return
        elapsed_s = (self._now_ns() - self.phase_start_ns) * 1.0e-9
        if not self.gripper_client.server_is_ready():
            if elapsed_s > float(
                self.get_parameter("gripper_server_timeout_s").value
            ):
                self._fail("physical gripper trajectory action is unavailable")
                return
            self._publish_status(f"waiting for physical gripper before {kind}")
            return
        position_name = (
            "gripper_open_position" if kind == "open" else "gripper_closed_position"
        )
        position = float(self.get_parameter(position_name).value)
        duration_s = float(self.get_parameter("gripper_move_duration_s").value)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [
            str(self.get_parameter("gripper_joint_name").value)
        ]
        point = JointTrajectoryPoint()
        point.positions = [position]
        duration_ns = int(round(duration_s * 1.0e9))
        point.time_from_start.sec = duration_ns // 1_000_000_000
        point.time_from_start.nanosec = duration_ns % 1_000_000_000
        goal.trajectory.points = [point]
        self.gripper_goal_pending = True
        self.gripper_goal_kind = kind
        self.hardware_commanded = True
        future = self.gripper_client.send_goal_async(goal)
        future.add_done_callback(self._gripper_goal_response)
        self._publish_status(f"physical gripper {kind} goal sent")

    def _gripper_goal_response(self, future):
        try:
            goal_handle = future.result()
        except Exception as error:
            self._fail(f"physical gripper goal failed: {error}")
            return
        if goal_handle is None or not goal_handle.accepted:
            self._fail("physical gripper goal was rejected")
            return
        self.gripper_goal_handle = goal_handle
        if self.phase == "failed":
            goal_handle.cancel_goal_async()
            self.gripper_goal_pending = False
            self.gripper_goal_kind = None
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._gripper_goal_result)

    def _gripper_goal_result(self, future):
        if self.phase == "failed":
            self.gripper_goal_pending = False
            self.gripper_goal_kind = None
            return
        try:
            wrapped_result = future.result()
            error_code = int(wrapped_result.result.error_code)
        except Exception as error:
            self._fail(f"physical gripper result failed: {error}")
            return
        if error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            self._fail(f"physical gripper trajectory error {error_code}")
            return
        kind = self.gripper_goal_kind
        self.gripper_goal_pending = False
        self.gripper_goal_kind = None
        self.gripper_goal_handle = None
        if kind == "open":
            self.phase = "retreat"
            self.phase_start_ns = self._now_ns()
            self.retreat_start_xyz = None
            self._publish_status("physical gripper open; starting straight retreat")
        elif kind == "close":
            self.phase = "waiting_for_push_policy"
            self.phase_start_ns = self._now_ns()
            self.push_policy_ready = False
            self.push_start_xyz = None
            self._publish_mode(2)
            self._publish_control(False)
            self._publish_status("physical gripper closed; waiting for PUSH-mode policy")

    def _publish_control(self, enabled: bool):
        if self.control_enable_publisher is not None:
            self.control_enable_publisher.publish(Bool(data=bool(enabled)))

    def _publish_mode(self, mode: int):
        if self.mode_publisher is not None:
            self.mode_publisher.publish(Int32(data=int(mode)))

    def _publish_direction_twist(self, direction, speed: float):
        message = TwistStamped()
        message.header.frame_id = self.base_frame
        message.header.stamp = self.get_clock().now().to_msg()
        message.twist.linear.x = float(direction[0] * speed)
        message.twist.linear.y = float(direction[1] * speed)
        message.twist.linear.z = float(direction[2] * speed)
        self.twist_publisher.publish(message)

    def _publish_zero_twist(self):
        if self.twist_publisher is None:
            return
        message = TwistStamped()
        message.header.frame_id = self.base_frame
        message.header.stamp = self.get_clock().now().to_msg()
        self.twist_publisher.publish(message)

    def _fail(self, reason: str):
        if self.phase == "failed":
            return
        self.phase = "failed"
        self._cancel_gripper_goal()
        self.gripper_goal_pending = False
        self.gripper_goal_kind = None
        self._publish_control(False)
        self._publish_zero_twist()
        self.complete_publisher.publish(Bool(data=False))
        self._publish_status(reason)

    def _cancel_gripper_goal(self):
        if self.gripper_goal_handle is None:
            return
        try:
            self.gripper_goal_handle.cancel_goal_async()
        except Exception:
            pass
        self.gripper_goal_handle = None

    def _publish_status(self, reason: str):
        report = {
            "valid": self.phase != "failed",
            "phase": self.phase,
            "reason": str(reason),
            "operation": self.operation,
            "hardware_authorized": self.control_mode,
            "command_interfaces_created": self.control_mode,
            "hardware_commanded": self.hardware_commanded,
            "start_requested": self.start_requested,
            "policy_release_action": self.policy_release_action,
            "release_trigger_source": self.release_trigger_source,
            "physical_release_guard": self.physical_release_state,
            "current_book_trailing_depth_m": self.current_trailing_depth_m,
            "target_book_trailing_depth_m": float(
                self.get_parameter("push_target_trailing_depth_m").value
            ),
            "remaining_book_push_distance_m": self.remaining_book_push_distance_m,
            "retreat_distance_m": self.retreat_distance_m,
            "requested_retreat_distance_m": float(
                self.get_parameter("retreat_distance_m").value
            ),
            "push_tcp_travel_m": self.push_tcp_travel_m,
            "task_complete": self.phase == "complete",
            "book_pose_source": "live_marker_tf",
        }
        key = (self.phase, reason)
        now_ns = self._now_ns()
        period_ns = int(
            round(
                float(self.get_parameter("status_publish_period_s").value) * 1.0e9
            )
        )
        repeated = key == self.last_status_key
        if (
            repeated
            and self.last_status_publish_ns is not None
            and now_ns - self.last_status_publish_ns < period_ns
        ):
            return
        self.status_publisher.publish(String(data=json.dumps(report, sort_keys=True)))
        self.last_status_publish_ns = now_ns
        if repeated:
            return
        self.last_status_key = key
        if self.phase == "failed":
            self.get_logger().error(reason)
        else:
            self.get_logger().info(reason)

    def destroy_node(self):
        self._publish_control(False)
        self._publish_zero_twist()
        self._cancel_gripper_goal()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PhysicalEpisodeCoordinator()
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
