#!/usr/bin/env python3
"""Shared path-planning core with no trajectory execution interface."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.msg import MoveItErrorCodes, RobotState, RobotTrajectory
from moveit_msgs.srv import GetMotionPlan
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32MultiArray, String
import tf2_ros

from .policy_tool_control_math import (
    JointTrajectorySafetyLimits,
    PolicyToolTarget,
    TargetSafetyLimits,
    compute_policy_tool_target,
    joint_trajectory_sanity,
    make_transform,
    matrix_to_quaternion_xyzw,
    provenance_error,
    target_safety_error,
    transform_to_dict,
)
from .planning_scene_math import LOCAL_INSERTION, scene_status_error
from .pose_motion_plan import build_pose_motion_plan_request


@dataclass
class PlanSnapshot:
    target: PolicyToolTarget
    trajectory_start: RobotState
    trajectory: RobotTrajectory
    generated_ns: int
    planning_time_s: float
    report: dict


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


class PolicyToolPlannerBase(Node):
    """Convert a shadow delta into a collision-checked MoveIt path."""

    def __init__(self, node_name: str):
        super().__init__(node_name)
        self._declare_parameters()
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.tcp_frame = str(self.get_parameter("tcp_frame").value)
        self.group_name = str(self.get_parameter("group_name").value)
        self.planning_link = str(self.get_parameter("planning_link").value)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.plan_client = self.create_client(
            GetMotionPlan,
            str(self.get_parameter("planning_service").value),
        )

        self.latest_observation_valid = False
        self.latest_observation_valid_ns = None
        self.latest_inference_valid = False
        self.latest_inference_valid_ns = None
        self.latest_delta = None
        self.latest_delta_ns = None
        self.latest_slot_pose = None
        self.latest_slot_pose_ns = None
        self.latest_joint_state = None
        self.latest_joint_state_ns = None
        self.latest_adapter_debug = None
        self.latest_adapter_debug_ns = None
        self.latest_policy_debug = None
        self.latest_policy_debug_ns = None
        self.latest_scene_status = None
        self.latest_scene_status_ns = None

        self.plan_pending = False
        self.pending_target = None
        self.pending_report = None
        self.last_requested_target_id = None
        self.latest_plan = None
        self.last_status_key = None

        self.plan_valid_publisher = self.create_publisher(
            Bool,
            str(self.get_parameter("plan_valid_topic").value),
            10,
        )
        self.report_publisher = self.create_publisher(
            String,
            str(self.get_parameter("plan_report_topic").value),
            10,
        )
        self.target_tool_publisher = self.create_publisher(
            PoseStamped,
            str(self.get_parameter("target_policy_tool_topic").value),
            10,
        )
        self.target_tcp_publisher = self.create_publisher(
            PoseStamped,
            str(self.get_parameter("target_tcp_topic").value),
            10,
        )
        self.trajectory_publisher = self.create_publisher(
            RobotTrajectory,
            str(self.get_parameter("planned_trajectory_topic").value),
            1,
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
            JointState,
            str(self.get_parameter("joint_states_topic").value),
            self._joint_state_callback,
            20,
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
        self.create_subscription(
            String,
            str(self.get_parameter("scene_status_topic").value),
            self._scene_status_callback,
            10,
        )

        rate = max(float(self.get_parameter("planning_rate_hz").value), 0.1)
        self.timer = self.create_timer(1.0 / rate, self._timer_callback)

    def _declare_parameters(self):
        self.declare_parameter("base_frame", "link_base")
        self.declare_parameter("tcp_frame", "link_tcp")
        self.declare_parameter("planning_link", "link_tcp")
        self.declare_parameter("group_name", "xarm7")
        self.declare_parameter("planning_service", "/plan_kinematic_path")
        self.declare_parameter("planning_pipeline_id", "")
        self.declare_parameter("planner_id", "")
        self.declare_parameter("planning_attempts", 3)
        self.declare_parameter("allowed_planning_time_s", 2.0)
        self.declare_parameter("velocity_scaling", 0.05)
        self.declare_parameter("acceleration_scaling", 0.05)
        self.declare_parameter("position_tolerance_m", 0.0005)
        self.declare_parameter("orientation_tolerance_rad", math.radians(0.5))
        self.declare_parameter("planning_rate_hz", 1.0)
        self.declare_parameter("message_max_age_s", 0.50)
        self.declare_parameter("tf_max_age_s", 0.50)
        self.declare_parameter("tf_lookup_timeout_s", 0.10)
        self.declare_parameter("command_scale", 0.10)

        self.declare_parameter("require_trajectory_sanity", True)
        self.declare_parameter(
            "expected_arm_joint_names",
            [
                "joint1",
                "joint2",
                "joint3",
                "joint4",
                "joint5",
                "joint6",
                "joint7",
            ],
        )
        self.declare_parameter("minimum_trajectory_point_count", 2)
        self.declare_parameter("require_trajectory_velocities", True)
        self.declare_parameter("maximum_trajectory_start_error_rad", 0.02)
        self.declare_parameter(
            "maximum_trajectory_waypoint_joint_jump_rad", 0.05
        )
        self.declare_parameter(
            "maximum_trajectory_endpoint_joint_delta_rad", 0.10
        )
        self.declare_parameter("maximum_trajectory_joint_path_length_rad", 0.30)
        self.declare_parameter("minimum_trajectory_duration_s", 0.10)
        self.declare_parameter("maximum_trajectory_duration_s", 15.0)

        self.declare_parameter("tcp_policy_tool_translation_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter(
            "tcp_policy_tool_quaternion_xyzw", [0.0, 0.0, 0.0, 1.0]
        )
        self.declare_parameter("expected_policy_tool_status", "")
        self.declare_parameter("expected_slot_status", "")
        self.declare_parameter("expected_book_status", "")
        self.declare_parameter("expected_bundle_sha256", "")
        self.declare_parameter("allow_unverified_policy_tool", False)
        self.declare_parameter("planning_scene_complete", False)
        self.declare_parameter("require_scene_status", False)
        self.declare_parameter("required_scene_mode", LOCAL_INSERTION)
        self.declare_parameter("scene_status_max_age_s", 0.50)

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
                "policy_to_robot_node",
                "cartesian_action_executor_node",
                "action_executor_node",
            ],
        )

        self.declare_parameter("observation_valid_topic", "/bookshelf_policy/observation_valid")
        self.declare_parameter("inference_valid_topic", "/bookshelf_shadow/inference_valid")
        self.declare_parameter("final_delta_topic", "/bookshelf_shadow/final_delta")
        self.declare_parameter("slot_pose_base_topic", "/bookshelf_policy/slot_pose_base")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("adapter_debug_topic", "/bookshelf_policy/adapter_debug")
        self.declare_parameter("policy_debug_topic", "/bookshelf_shadow/policy_debug")
        self.declare_parameter("scene_status_topic", "/bookshelf_scene/status")
        self.declare_parameter("plan_valid_topic", "/bookshelf_guarded/plan_valid")
        self.declare_parameter("plan_report_topic", "/bookshelf_guarded/plan_report")
        self.declare_parameter("target_policy_tool_topic", "/bookshelf_guarded/target_policy_tool")
        self.declare_parameter("target_tcp_topic", "/bookshelf_guarded/target_tcp")
        self.declare_parameter("planned_trajectory_topic", "/bookshelf_guarded/planned_trajectory")

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

    def _slot_pose_callback(self, message: PoseStamped):
        self.latest_slot_pose = message
        self.latest_slot_pose_ns = self._now_ns()

    def _joint_state_callback(self, message: JointState):
        self.latest_joint_state = message
        self.latest_joint_state_ns = self._now_ns()

    def _adapter_debug_callback(self, message: String):
        self.latest_adapter_debug = self._parse_debug(message.data)
        self.latest_adapter_debug_ns = self._now_ns()

    def _policy_debug_callback(self, message: String):
        self.latest_policy_debug = self._parse_debug(message.data)
        self.latest_policy_debug_ns = self._now_ns()

    def _scene_status_callback(self, message: String):
        self.latest_scene_status = self._parse_debug(message.data)
        self.latest_scene_status_ns = self._now_ns()

    @staticmethod
    def _parse_debug(value: str):
        try:
            parsed = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return None
        return parsed if isinstance(parsed, dict) else None

    def _fresh(self, timestamp_ns, maximum_age_s=None) -> bool:
        if timestamp_ns is None:
            return False
        maximum_age_s = (
            float(self.get_parameter("message_max_age_s").value)
            if maximum_age_s is None
            else float(maximum_age_s)
        )
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
            (self.latest_joint_state, self.latest_joint_state_ns, "joint state"),
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
        if bool(self.get_parameter("require_scene_status").value):
            if not self._fresh(
                self.latest_scene_status_ns,
                float(self.get_parameter("scene_status_max_age_s").value),
            ):
                return "planning scene status is missing or stale"
            error = scene_status_error(
                self.latest_scene_status,
                required_mode=str(self.get_parameter("required_scene_mode").value),
            )
            if error:
                return error
        blocked = self._blocked_nodes_present()
        if blocked:
            return f"blocked legacy execution nodes are active: {blocked}"
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

    def _trajectory_safety_limits(self) -> JointTrajectorySafetyLimits:
        return JointTrajectorySafetyLimits(
            expected_joint_names=tuple(
                str(value)
                for value in self.get_parameter("expected_arm_joint_names").value
            ),
            minimum_point_count=int(
                self.get_parameter("minimum_trajectory_point_count").value
            ),
            require_velocities=bool(
                self.get_parameter("require_trajectory_velocities").value
            ),
            maximum_start_error_rad=float(
                self.get_parameter("maximum_trajectory_start_error_rad").value
            ),
            maximum_waypoint_joint_jump_rad=float(
                self.get_parameter(
                    "maximum_trajectory_waypoint_joint_jump_rad"
                ).value
            ),
            maximum_endpoint_joint_delta_rad=float(
                self.get_parameter(
                    "maximum_trajectory_endpoint_joint_delta_rad"
                ).value
            ),
            maximum_joint_path_length_rad=float(
                self.get_parameter(
                    "maximum_trajectory_joint_path_length_rad"
                ).value
            ),
            minimum_duration_s=float(
                self.get_parameter("minimum_trajectory_duration_s").value
            ),
            maximum_duration_s=float(
                self.get_parameter("maximum_trajectory_duration_s").value
            ),
        )

    def _trajectory_sanity(self, response) -> tuple[dict, str | None]:
        if not bool(self.get_parameter("require_trajectory_sanity").value):
            return {
                "passed": False,
                "skipped": True,
                "reasons": ["trajectory sanity validation is disabled"],
            }, "trajectory sanity validation is disabled"
        trajectory = response.trajectory.joint_trajectory
        start = response.trajectory_start.joint_state
        return joint_trajectory_sanity(
            trajectory.joint_names,
            [point.positions for point in trajectory.points],
            [point.velocities for point in trajectory.points],
            [
                float(point.time_from_start.sec)
                + float(point.time_from_start.nanosec) * 1.0e-9
                for point in trajectory.points
            ],
            start.name,
            start.position,
            limits=self._trajectory_safety_limits(),
        )

    def _timer_callback(self):
        if self.plan_pending:
            return
        error = self._input_error()
        if error:
            self._publish_invalid(error)
            return
        transform_base_tcp, error = self._lookup_base_tcp()
        if error:
            self._publish_invalid(error)
            return
        try:
            transform_base_slot = _pose_to_transform(self.latest_slot_pose.pose)
            target = compute_policy_tool_target(
                transform_base_slot,
                transform_base_tcp,
                self._tool_transform(),
                self.latest_delta,
                command_scale=float(self.get_parameter("command_scale").value),
            )
            report = self._base_report(target)
            error = target_safety_error(
                target,
                self.latest_delta,
                self._safety_limits(),
            )
        except ValueError as exception:
            self._publish_invalid(f"target geometry error: {exception}")
            return
        if error:
            self._publish_invalid(error, report=report)
            return

        self._publish_target_poses(target)
        if target.target_id == self.last_requested_target_id and self.latest_plan is not None:
            self._republish_latest_plan()
            return
        if not self.plan_client.wait_for_service(timeout_sec=0.05):
            self._publish_invalid("MoveIt planning service is unavailable")
            return

        request = self._motion_plan_request(target)
        self.pending_target = target
        self.pending_report = report
        self.plan_pending = True
        self.last_requested_target_id = target.target_id
        future = self.plan_client.call_async(request)
        future.add_done_callback(self._plan_response_callback)
        self._log_once(
            f"planning:{target.target_id}",
            f"Planning path for target {target.target_id[:12]} in PLAN-ONLY core.",
        )

    def _motion_plan_request(self, target: PolicyToolTarget):
        workspace_min = self.get_parameter("workspace_min_xyz").value
        workspace_max = self.get_parameter("workspace_max_xyz").value
        target_pose = _transform_to_pose(target.transform_base_tcp_target)
        return build_pose_motion_plan_request(
            target_pose=target_pose,
            start_joint_state=self.latest_joint_state,
            base_frame=self.base_frame,
            planning_link=self.planning_link,
            group_name=self.group_name,
            workspace_min_xyz=workspace_min,
            workspace_max_xyz=workspace_max,
            planning_pipeline_id=str(
                self.get_parameter("planning_pipeline_id").value
            ),
            planner_id=str(self.get_parameter("planner_id").value),
            planning_attempts=int(self.get_parameter("planning_attempts").value),
            allowed_planning_time_s=float(
                self.get_parameter("allowed_planning_time_s").value
            ),
            velocity_scaling=float(self.get_parameter("velocity_scaling").value),
            acceleration_scaling=float(
                self.get_parameter("acceleration_scaling").value
            ),
            position_tolerance_m=float(
                self.get_parameter("position_tolerance_m").value
            ),
            orientation_tolerance_rad=float(
                self.get_parameter("orientation_tolerance_rad").value
            ),
            constraint_name=f"policy_tool_{target.target_id[:12]}",
        )

    def _plan_response_callback(self, future):
        target = self.pending_target
        report = dict(self.pending_report or {})
        self.pending_target = None
        self.pending_report = None
        self.plan_pending = False
        try:
            response = future.result().motion_plan_response
        except Exception as error:
            self.latest_plan = None
            self._publish_invalid(f"MoveIt planning call failed: {error}", report=report)
            return

        success = int(response.error_code.val) == int(MoveItErrorCodes.SUCCESS)
        point_count = len(response.trajectory.joint_trajectory.points)
        report.update(
            {
                "moveit_error_code": int(response.error_code.val),
                "planning_time_s": float(response.planning_time),
                "trajectory_point_count": int(point_count),
                "path_planned": bool(success and point_count > 0),
                "planning_scene_complete": bool(
                    self.get_parameter("planning_scene_complete").value
                ),
            }
        )
        if not success or point_count == 0:
            self.latest_plan = None
            self._publish_invalid("MoveIt did not return a valid path", report=report)
            return

        trajectory_report, trajectory_error = self._trajectory_sanity(response)
        report["trajectory_sanity"] = trajectory_report
        if trajectory_error:
            self.latest_plan = None
            self._publish_invalid(trajectory_error, report=report)
            return

        report.update(
            {
                "valid": True,
                "execution_ready": bool(
                    self.get_parameter("planning_scene_complete").value
                ),
                "reason": (
                    "collision-checked path available"
                    if bool(self.get_parameter("planning_scene_complete").value)
                    else "path available, but planning scene completeness is unconfirmed"
                ),
            }
        )
        self.latest_plan = PlanSnapshot(
            target=target,
            trajectory_start=response.trajectory_start,
            trajectory=response.trajectory,
            generated_ns=self._now_ns(),
            planning_time_s=float(response.planning_time),
            report=report,
        )
        self.plan_valid_publisher.publish(Bool(data=True))
        self.trajectory_publisher.publish(response.trajectory)
        self.report_publisher.publish(String(data=json.dumps(report, sort_keys=True)))
        self._on_valid_plan(self.latest_plan)
        self._log_once(
            f"valid:{target.target_id}",
            f"Valid MoveIt path for target {target.target_id[:12]}; "
            f"points={point_count}, execution_ready={report['execution_ready']}.",
        )

    def _base_report(self, target: PolicyToolTarget) -> dict:
        return {
            "valid": False,
            "shadow_inputs_valid": True,
            "hardware_commanded": False,
            "gripper_command_interface": False,
            "target_id": target.target_id,
            "runtime_source_file": str(Path(__file__).resolve()),
            "base_frame": self.base_frame,
            "tcp_frame": self.tcp_frame,
            "command_scale": float(self.get_parameter("command_scale").value),
            "unscaled_delta": [float(value) for value in self.latest_delta],
            "scaled_delta": [float(value) for value in target.scaled_delta],
            "tcp_translation_step_m": target.tcp_translation_step_m,
            "tcp_rotation_step_deg": math.degrees(target.tcp_rotation_step_rad),
            "slot_pose_base": transform_to_dict(
                _pose_to_transform(self.latest_slot_pose.pose)
            ),
            "current_tcp_base": transform_to_dict(
                target.transform_base_policy_tool_current
                @ np.linalg.inv(self._tool_transform())
            ),
            "current_policy_tool_base": transform_to_dict(
                target.transform_base_policy_tool_current
            ),
            "current_policy_tool_slot": transform_to_dict(
                target.transform_slot_policy_tool_current
            ),
            "target_policy_tool_slot": transform_to_dict(
                target.transform_slot_policy_tool_target
            ),
            "tcp_policy_tool": transform_to_dict(self._tool_transform()),
            "target_policy_tool_base": transform_to_dict(
                target.transform_base_policy_tool_target
            ),
            "target_tcp_base": transform_to_dict(target.transform_base_tcp_target),
            "policy_tool_transform_status": self.latest_adapter_debug.get(
                "policy_tool_transform_status"
            ),
            "slot_transform_status": self.latest_adapter_debug.get(
                "static_slot_transform_status"
            ),
            "book_transform_status": self.latest_adapter_debug.get(
                "eef_book_transform_status"
            ),
            "policy_bundle_sha256": self.latest_policy_debug.get("bundle_sha256"),
            "blocked_nodes": self._blocked_nodes_present(),
            "scene_status_required": bool(
                self.get_parameter("require_scene_status").value
            ),
            "scene_status": self.latest_scene_status,
        }

    def _publish_target_poses(self, target: PolicyToolTarget):
        stamp = self.get_clock().now().to_msg()
        tool = PoseStamped()
        tool.header.frame_id = self.base_frame
        tool.header.stamp = stamp
        tool.pose = _transform_to_pose(target.transform_base_policy_tool_target)
        tcp = PoseStamped()
        tcp.header.frame_id = self.base_frame
        tcp.header.stamp = stamp
        tcp.pose = _transform_to_pose(target.transform_base_tcp_target)
        self.target_tool_publisher.publish(tool)
        self.target_tcp_publisher.publish(tcp)

    def _republish_latest_plan(self):
        if self.latest_plan is None:
            return
        self.plan_valid_publisher.publish(Bool(data=True))
        self.report_publisher.publish(
            String(data=json.dumps(self.latest_plan.report, sort_keys=True))
        )

    def _publish_invalid(self, reason: str, *, report=None):
        self.plan_valid_publisher.publish(Bool(data=False))
        value = dict(report or {})
        value.update(
            {
                "valid": False,
                "execution_ready": False,
                "hardware_commanded": False,
                "gripper_command_interface": False,
                "reason": str(reason),
            }
        )
        self.report_publisher.publish(String(data=json.dumps(value, sort_keys=True)))
        self._on_invalid_plan(str(reason))
        self._log_once(f"invalid:{reason}", f"Plan invalid: {reason}", warning=True)

    def _on_valid_plan(self, plan: PlanSnapshot):
        """Hook for the explicitly guarded executor."""

    def _on_invalid_plan(self, reason: str):
        """Hook for the explicitly guarded executor."""

    def _log_once(self, key: str, message: str, warning=False):
        if key == self.last_status_key:
            return
        self.last_status_key = key
        if warning:
            self.get_logger().warning(message)
        else:
            self.get_logger().info(message)
