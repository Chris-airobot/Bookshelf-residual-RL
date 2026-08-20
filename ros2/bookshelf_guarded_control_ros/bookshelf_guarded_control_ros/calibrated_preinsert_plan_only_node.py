#!/usr/bin/env python3
"""Plan to the calibrated global pre-insertion TCP pose without execution."""

from __future__ import annotations

import copy
from datetime import datetime
import json
import math
from pathlib import Path

from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.msg import DisplayTrajectory, MoveItErrorCodes, RobotTrajectory
from moveit_msgs.srv import GetMotionPlan, GetPositionIK
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
import tf2_ros

from .calibrated_preinsert_plan_math import (
    PreinsertTargetLimits,
    preinsert_target_error,
    preinsert_target_metrics,
    target_identifier,
)
from .planning_scene_math import global_scene_status_error
from .policy_tool_control_math import (
    JointTrajectorySafetyLimits,
    TRAJECTORY_FINGERPRINT_KIND,
    canonical_ros_message_sha256,
    joint_trajectory_sanity,
    make_transform,
    matrix_to_quaternion_xyzw,
    named_joint_target_branch_report,
    transform_to_dict,
)
from .pose_motion_plan import (
    build_joint_motion_plan_request,
    build_position_ik_request,
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
    pose = Pose()
    pose.position.x = float(transform[0, 3])
    pose.position.y = float(transform[1, 3])
    pose.position.z = float(transform[2, 3])
    quaternion = matrix_to_quaternion_xyzw(transform[:3, :3])
    pose.orientation.x = float(quaternion[0])
    pose.orientation.y = float(quaternion[1])
    pose.orientation.z = float(quaternion[2])
    pose.orientation.w = float(quaternion[3])
    return pose


class CalibratedPreinsertPlanOnlyNode(Node):
    """One-request-per-target MoveIt planner with no execution interface."""

    def __init__(self):
        super().__init__("calibrated_preinsert_plan_only")
        self._declare_parameters()
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.tcp_frame = str(self.get_parameter("tcp_frame").value)
        self.planning_link = str(self.get_parameter("planning_link").value)
        self.group_name = str(self.get_parameter("group_name").value)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.plan_client = self.create_client(
            GetMotionPlan,
            str(self.get_parameter("planning_service").value),
        )
        self.ik_client = self.create_client(
            GetPositionIK,
            str(self.get_parameter("ik_service").value),
        )

        self.latest_target_valid = False
        self.latest_target_valid_ns = None
        self.latest_target_pose = None
        self.latest_target_pose_ns = None
        self.latest_target_debug = None
        self.latest_target_debug_ns = None
        self.latest_joint_state = None
        self.latest_joint_state_ns = None
        self.latest_scene_status = None
        self.latest_scene_status_ns = None
        self.plan_pending = False
        self.pending = None
        self.completed_target_id = None
        self.latest_report = None
        self.latest_trajectory = None
        self.last_status_key = None

        latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.plan_valid_publisher = self.create_publisher(
            Bool, str(self.get_parameter("plan_valid_topic").value), latched
        )
        self.report_publisher = self.create_publisher(
            String, str(self.get_parameter("plan_report_topic").value), latched
        )
        self.target_publisher = self.create_publisher(
            PoseStamped, str(self.get_parameter("target_tcp_output_topic").value), latched
        )
        self.trajectory_publisher = self.create_publisher(
            RobotTrajectory,
            str(self.get_parameter("planned_trajectory_topic").value),
            latched,
        )
        self.display_publisher = self.create_publisher(
            DisplayTrajectory,
            str(self.get_parameter("display_trajectory_topic").value),
            latched,
        )

        self.create_subscription(
            Bool,
            str(self.get_parameter("target_valid_topic").value),
            self._target_valid_callback,
            10,
        )
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("target_tcp_topic").value),
            self._target_pose_callback,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("target_debug_topic").value),
            self._target_debug_callback,
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
            str(self.get_parameter("scene_status_topic").value),
            self._scene_status_callback,
            10,
        )
        rate = max(float(self.get_parameter("planning_rate_hz").value), 0.1)
        self.create_timer(1.0 / rate, self._timer_callback)

        self.output_path = (
            Path(str(self.get_parameter("output_dir").value)).expanduser()
            / "calibrated_preinsert_plan_report.json"
        )
        self.get_logger().info(
            "Automatic calibrated pre-insertion PLAN-ONLY bridge started."
        )
        self.get_logger().info(
            "It has no trajectory execution, controller, gripper, or robot-command client."
        )

    def _declare_parameters(self):
        self.declare_parameter("base_frame", "link_base")
        self.declare_parameter("tcp_frame", "link_tcp")
        self.declare_parameter("planning_link", "link_tcp")
        self.declare_parameter("group_name", "xarm7")
        self.declare_parameter("planning_service", "/plan_kinematic_path")
        self.declare_parameter("ik_service", "/compute_ik")
        self.declare_parameter("ik_timeout_s", 1.0)
        self.declare_parameter("ik_avoid_collisions", True)
        self.declare_parameter("planning_pipeline_id", "")
        self.declare_parameter("planner_id", "")
        self.declare_parameter("planning_attempts", 3)
        self.declare_parameter("allowed_planning_time_s", 5.0)
        self.declare_parameter("velocity_scaling", 0.05)
        self.declare_parameter("acceleration_scaling", 0.05)
        self.declare_parameter("position_tolerance_m", 0.001)
        self.declare_parameter("orientation_tolerance_rad", math.radians(1.0))
        self.declare_parameter("planning_rate_hz", 2.0)
        self.declare_parameter("message_max_age_s", 1.0)
        self.declare_parameter("scene_status_max_age_s", 1.0)
        self.declare_parameter("tf_max_age_s", 0.50)
        self.declare_parameter("tf_lookup_timeout_s", 0.10)
        self.declare_parameter("maximum_preserved_tcp_orientation_change_deg", 0.10)
        self.declare_parameter("maximum_target_translation_m", 0.75)
        self.declare_parameter("maximum_target_rotation_deg", 5.0)
        self.declare_parameter("workspace_min_xyz", [0.20, -0.60, 0.05])
        self.declare_parameter("workspace_max_xyz", [1.00, 0.60, 1.00])

        self.declare_parameter("require_trajectory_sanity", True)
        self.declare_parameter(
            "expected_arm_joint_names",
            ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"],
        )
        self.declare_parameter("minimum_trajectory_point_count", 2)
        self.declare_parameter("require_trajectory_velocities", True)
        self.declare_parameter("maximum_trajectory_start_error_rad", 0.02)
        self.declare_parameter("maximum_trajectory_waypoint_joint_jump_rad", 0.20)
        self.declare_parameter("maximum_trajectory_endpoint_joint_delta_rad", 3.0)
        self.declare_parameter("require_near_current_goal_joints", True)
        self.declare_parameter("maximum_goal_joint_delta_rad", 1.5)
        self.declare_parameter("joint_goal_tolerance_rad", 0.001)
        self.declare_parameter("maximum_trajectory_joint_path_length_rad", 10.0)
        self.declare_parameter("minimum_trajectory_duration_s", 0.10)
        self.declare_parameter("maximum_trajectory_duration_s", 90.0)
        self.declare_parameter(
            "blocked_node_names",
            [
                "guarded_policy_tool_executor",
                "policy_to_robot_node",
                "cartesian_action_executor_node",
                "action_executor_node",
            ],
        )

        self.declare_parameter("target_valid_topic", "/bookshelf_shadow/calibrated_target_valid")
        self.declare_parameter("target_tcp_topic", "/bookshelf_shadow/target_tcp_pose")
        self.declare_parameter("target_debug_topic", "/bookshelf_shadow/calibrated_target_debug")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("scene_status_topic", "/bookshelf_scene/status")
        self.declare_parameter("plan_valid_topic", "/bookshelf_preinsert/plan_valid")
        self.declare_parameter("plan_report_topic", "/bookshelf_preinsert/plan_report")
        self.declare_parameter("target_tcp_output_topic", "/bookshelf_preinsert/target_tcp")
        self.declare_parameter("planned_trajectory_topic", "/bookshelf_preinsert/planned_trajectory")
        self.declare_parameter("display_trajectory_topic", "/display_planned_path")
        self.declare_parameter("output_dir", "/tmp/bookshelf_preinsert_plan")

    def _now_ns(self) -> int:
        return int(self.get_clock().now().nanoseconds)

    def _fresh(self, timestamp_ns, maximum_age_s=None) -> bool:
        if timestamp_ns is None:
            return False
        maximum = (
            float(self.get_parameter("message_max_age_s").value)
            if maximum_age_s is None
            else float(maximum_age_s)
        )
        return maximum <= 0.0 or (self._now_ns() - timestamp_ns) * 1.0e-9 <= maximum

    @staticmethod
    def _json_object(value: str):
        try:
            parsed = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return None
        return parsed if isinstance(parsed, dict) else None

    def _target_valid_callback(self, message: Bool):
        self.latest_target_valid = bool(message.data)
        self.latest_target_valid_ns = self._now_ns()

    def _target_pose_callback(self, message: PoseStamped):
        self.latest_target_pose = message
        self.latest_target_pose_ns = self._now_ns()

    def _target_debug_callback(self, message: String):
        self.latest_target_debug = self._json_object(message.data)
        self.latest_target_debug_ns = self._now_ns()

    def _joint_state_callback(self, message: JointState):
        self.latest_joint_state = message
        self.latest_joint_state_ns = self._now_ns()

    def _scene_status_callback(self, message: String):
        self.latest_scene_status = self._json_object(message.data)
        self.latest_scene_status_ns = self._now_ns()

    def _blocked_nodes_present(self):
        blocked = {
            str(value).strip().lstrip("/")
            for value in self.get_parameter("blocked_node_names").value
        }
        active = {str(value).strip().lstrip("/") for value in self.get_node_names()}
        active.discard(self.get_name().lstrip("/"))
        return sorted(blocked.intersection(active))

    def _input_error(self):
        required = (
            (self.latest_target_pose, self.latest_target_pose_ns, "target TCP pose"),
            (self.latest_target_debug, self.latest_target_debug_ns, "target debug"),
            (self.latest_joint_state, self.latest_joint_state_ns, "joint state"),
        )
        for value, timestamp, label in required:
            if value is None or not self._fresh(timestamp):
                return f"{label} is missing or stale"
        if self.latest_target_pose.header.frame_id != self.base_frame:
            return (
                f"target TCP frame is {self.latest_target_pose.header.frame_id!r}, "
                f"expected {self.base_frame!r}"
            )
        debug = self.latest_target_debug
        if debug.get("hardware_commanded") is not False:
            return "target calculator does not prove hardware_commanded=false"
        if not bool(debug.get("geometric_target_valid")):
            return "calibrated geometric target is invalid"
        if debug.get("target_orientation_mode") != "preserve_current_tcp":
            return "target orientation mode is not preserve_current_tcp"
        if not bool(debug.get("orientation_latched")):
            return "preserved TCP orientation has not been latched"
        if debug.get("target_unexpected_clipped_labels"):
            return "calibrated target has unexpected clipped observation channels"
        if not bool(self.get_parameter("require_near_current_goal_joints").value):
            return "near-current IK branch validation is disabled"
        change = debug.get("preserved_tcp_orientation_change_deg")
        if change is None or not np.isfinite(float(change)):
            return "preserved TCP orientation diagnostic is unavailable"
        maximum = float(
            self.get_parameter("maximum_preserved_tcp_orientation_change_deg").value
        )
        if abs(float(change)) > maximum:
            return (
                "preserved TCP orientation changed unexpectedly: "
                f"{float(change):.6f} deg > {maximum:.6f} deg"
            )
        if not self._fresh(
            self.latest_scene_status_ns,
            float(self.get_parameter("scene_status_max_age_s").value),
        ):
            return "global planning scene status is missing or stale"
        error = global_scene_status_error(self.latest_scene_status)
        if error:
            return error
        blocked = self._blocked_nodes_present()
        if blocked:
            return f"execution nodes are active: {blocked}"
        return None

    def _lookup_current_tcp(self):
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
        stamp_ns = int(message.header.stamp.sec) * 1_000_000_000 + int(
            message.header.stamp.nanosec
        )
        maximum = float(self.get_parameter("tf_max_age_s").value)
        if maximum > 0.0 and stamp_ns > 0:
            age = (self._now_ns() - stamp_ns) * 1.0e-9
            if age > maximum:
                return None, f"TF {self.base_frame} <- {self.tcp_frame} is stale"
        return _transform_message_to_matrix(message), None

    def _target_limits(self):
        return PreinsertTargetLimits(
            maximum_translation_m=float(
                self.get_parameter("maximum_target_translation_m").value
            ),
            maximum_rotation_rad=math.radians(
                float(self.get_parameter("maximum_target_rotation_deg").value)
            ),
            workspace_min_xyz=tuple(
                float(value) for value in self.get_parameter("workspace_min_xyz").value
            ),
            workspace_max_xyz=tuple(
                float(value) for value in self.get_parameter("workspace_max_xyz").value
            ),
        )

    def _trajectory_limits(self):
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
                self.get_parameter("maximum_trajectory_waypoint_joint_jump_rad").value
            ),
            maximum_endpoint_joint_delta_rad=float(
                self.get_parameter("maximum_trajectory_endpoint_joint_delta_rad").value
            ),
            maximum_joint_path_length_rad=float(
                self.get_parameter("maximum_trajectory_joint_path_length_rad").value
            ),
            minimum_duration_s=float(
                self.get_parameter("minimum_trajectory_duration_s").value
            ),
            maximum_duration_s=float(
                self.get_parameter("maximum_trajectory_duration_s").value
            ),
        )

    def _trajectory_sanity(self, response):
        if not bool(self.get_parameter("require_trajectory_sanity").value):
            return {}, "trajectory sanity validation is disabled"
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
            limits=self._trajectory_limits(),
        )

    def _ik_request(self, transform_base_tcp_target, start_joint_state):
        return build_position_ik_request(
            target_pose=_transform_to_pose(transform_base_tcp_target),
            start_joint_state=start_joint_state,
            base_frame=self.base_frame,
            planning_link=self.planning_link,
            group_name=self.group_name,
            timeout_s=float(self.get_parameter("ik_timeout_s").value),
            avoid_collisions=bool(
                self.get_parameter("ik_avoid_collisions").value
            ),
        )

    def _motion_plan_request(self, target_joint_state, start_joint_state, target_id):
        expected_joint_names = tuple(
            str(value)
            for value in self.get_parameter("expected_arm_joint_names").value
        )
        target_positions = dict(
            zip(target_joint_state.name, target_joint_state.position)
        )
        return build_joint_motion_plan_request(
            target_joint_names=expected_joint_names,
            target_joint_positions=[target_positions[name] for name in expected_joint_names],
            start_joint_state=start_joint_state,
            group_name=self.group_name,
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
            joint_tolerance_rad=float(
                self.get_parameter("joint_goal_tolerance_rad").value
            ),
            constraint_name=f"calibrated_preinsert_ik_{target_id[:12]}",
        )

    def _timer_callback(self):
        if self.plan_pending:
            return
        error = self._input_error()
        if error:
            self._publish_invalid(error)
            return
        current, error = self._lookup_current_tcp()
        if error:
            self._publish_invalid(error)
            return
        try:
            target = _pose_to_transform(self.latest_target_pose.pose)
            target_id = target_identifier(target)
            metrics = preinsert_target_metrics(current, target)
            error = preinsert_target_error(current, target, limits=self._target_limits())
        except ValueError as exception:
            self._publish_invalid(f"pre-insertion target geometry error: {exception}")
            return
        report = self._base_report(current, target, target_id, metrics)
        if error:
            self._publish_invalid(error, report=report)
            return

        self._publish_target(target)
        if target_id == self.completed_target_id:
            self._republish_result()
            return
        if not self.ik_client.wait_for_service(timeout_sec=0.05):
            self._publish_invalid("MoveIt IK service is unavailable", report=report)
            return
        if not self.plan_client.wait_for_service(timeout_sec=0.05):
            self._publish_invalid("MoveIt planning service is unavailable", report=report)
            return

        start_joint_state = copy.deepcopy(self.latest_joint_state)
        self.plan_pending = True
        self.pending = (target, target_id, report, start_joint_state)
        future = self.ik_client.call_async(self._ik_request(target, start_joint_state))
        future.add_done_callback(self._ik_response_callback)
        self._log_once(
            f"ik:{target_id}",
            f"Requesting collision-aware seeded IK for target {target_id[:12]}.",
        )

    def _ik_response_callback(self, future):
        target, target_id, report, start_joint_state = self.pending
        try:
            response = future.result()
        except Exception as error:
            self._finish_pending_invalid(
                target_id, f"MoveIt IK call failed: {error}", report
            )
            return

        success = int(response.error_code.val) == int(MoveItErrorCodes.SUCCESS)
        report.update(
            {
                "ik_checked": True,
                "ik_collision_aware": bool(
                    self.get_parameter("ik_avoid_collisions").value
                ),
                "ik_error_code": int(response.error_code.val),
            }
        )
        if not success:
            self._finish_pending_invalid(
                target_id, "MoveIt did not return a valid IK solution", report
            )
            return

        expected = tuple(
            str(value)
            for value in self.get_parameter("expected_arm_joint_names").value
        )
        branch_report, error = named_joint_target_branch_report(
            start_joint_state.name,
            start_joint_state.position,
            response.solution.joint_state.name,
            response.solution.joint_state.position,
            expected,
            float(self.get_parameter("maximum_goal_joint_delta_rad").value),
        )
        report["ik_joint_branch"] = branch_report
        if error:
            self._finish_pending_invalid(target_id, error, report)
            return

        try:
            request = self._motion_plan_request(
                response.solution.joint_state, start_joint_state, target_id
            )
        except (KeyError, ValueError) as exception:
            self._finish_pending_invalid(
                target_id, f"joint motion plan request is invalid: {exception}", report
            )
            return
        future = self.plan_client.call_async(request)
        future.add_done_callback(self._plan_response_callback)
        self._log_once(
            f"planning:{target_id}",
            f"Planning to validated nearby IK branch for target {target_id[:12]}.",
        )

    def _finish_pending_invalid(self, target_id, reason, report):
        self.pending = None
        self.plan_pending = False
        self.completed_target_id = target_id
        self._publish_invalid(reason, report=report)

    def _plan_response_callback(self, future):
        target, target_id, report, start_joint_state = self.pending
        self.pending = None
        self.plan_pending = False
        self.completed_target_id = target_id
        try:
            response = future.result().motion_plan_response
        except Exception as error:
            self._publish_invalid(f"MoveIt planning call failed: {error}", report=report)
            return

        point_count = len(response.trajectory.joint_trajectory.points)
        success = int(response.error_code.val) == int(MoveItErrorCodes.SUCCESS)
        report.update(
            {
                "moveit_error_code": int(response.error_code.val),
                "planning_time_s": float(response.planning_time),
                "trajectory_point_count": int(point_count),
                "path_planned": bool(success and point_count > 0),
            }
        )
        if not success or point_count == 0:
            self._publish_invalid("MoveIt did not return a valid path", report=report)
            return
        trajectory_report, error = self._trajectory_sanity(response)
        report["trajectory_sanity"] = trajectory_report
        if error:
            self._publish_invalid(error, report=report)
            return

        report.update(
            {
                "valid": True,
                "collision_checked": True,
                "trajectory_fingerprint_kind": TRAJECTORY_FINGERPRINT_KIND,
                "trajectory_sha256": canonical_ros_message_sha256(
                    response.trajectory
                ),
                "execution_ready": False,
                "execution_authorized": False,
                "human_review_required": True,
                "reason": "collision-aware global path available for human review only",
            }
        )
        self.latest_report = report
        self.latest_trajectory = response.trajectory
        self.plan_valid_publisher.publish(Bool(data=True))
        self.trajectory_publisher.publish(response.trajectory)
        self.report_publisher.publish(String(data=json.dumps(report, sort_keys=True)))
        display = DisplayTrajectory()
        display.trajectory_start = response.trajectory_start
        display.trajectory = [response.trajectory]
        self.display_publisher.publish(display)
        self._write_report(report)
        self._log_once(
            f"valid:{target_id}",
            f"Valid PLAN-ONLY global path: points={point_count}; execution_ready=False.",
        )

    def _base_report(self, current, target, target_id, metrics):
        return {
            "schema_version": 1,
            "kind": "bookshelf_calibrated_preinsert_plan_only",
            "generated_at": datetime.now().astimezone().isoformat(),
            "valid": False,
            "hardware_commanded": False,
            "execution_ready": False,
            "execution_authorized": False,
            "gripper_command_interface": False,
            "target_id": target_id,
            "base_frame": self.base_frame,
            "tcp_frame": self.tcp_frame,
            "planning_link": self.planning_link,
            "current_tcp_base": transform_to_dict(current),
            "target_tcp_base": transform_to_dict(target),
            "target_translation_m": float(metrics["translation_m"]),
            "target_rotation_deg": math.degrees(float(metrics["rotation_rad"])),
            "target_calculator_debug": self.latest_target_debug,
            "target_policy_observation_valid": bool(self.latest_target_valid),
            "geometric_target_used_for_global_plan": True,
            "ik_checked": False,
            "scene_status": self.latest_scene_status,
            "blocked_nodes": self._blocked_nodes_present(),
            "goal_joint_branch_constraint": {
                "required": bool(
                    self.get_parameter("require_near_current_goal_joints").value
                ),
                "center": "current_joint_state",
                "maximum_delta_rad": float(
                    self.get_parameter("maximum_goal_joint_delta_rad").value
                ),
            },
            "planning_sequence": "seeded_collision_aware_ik_then_joint_goal_plan",
        }

    def _publish_target(self, transform):
        message = PoseStamped()
        message.header.frame_id = self.base_frame
        message.header.stamp = self.get_clock().now().to_msg()
        message.pose = _transform_to_pose(transform)
        self.target_publisher.publish(message)

    def _republish_result(self):
        if self.latest_report is None:
            return
        self.plan_valid_publisher.publish(Bool(data=bool(self.latest_report["valid"])))
        self.report_publisher.publish(
            String(data=json.dumps(self.latest_report, sort_keys=True))
        )
        if self.latest_trajectory is not None:
            self.trajectory_publisher.publish(self.latest_trajectory)

    def _publish_invalid(self, reason, *, report=None):
        value = dict(report or {})
        value.update(
            {
                "schema_version": 1,
                "kind": "bookshelf_calibrated_preinsert_plan_only",
                "generated_at": datetime.now().astimezone().isoformat(),
                "valid": False,
                "hardware_commanded": False,
                "execution_ready": False,
                "execution_authorized": False,
                "gripper_command_interface": False,
                "reason": str(reason),
            }
        )
        self.latest_report = value
        self.latest_trajectory = None
        self.plan_valid_publisher.publish(Bool(data=False))
        self.report_publisher.publish(String(data=json.dumps(value, sort_keys=True)))
        self._write_report(value)
        self._log_once(f"invalid:{reason}", f"Pre-insertion plan invalid: {reason}", True)

    def _write_report(self, report):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _log_once(self, key, message, warning=False):
        if key == self.last_status_key:
            return
        self.last_status_key = key
        if warning:
            self.get_logger().warning(message)
        else:
            self.get_logger().info(message)


def main(args=None):
    rclpy.init(args=args)
    node = CalibratedPreinsertPlanOnlyNode()
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
