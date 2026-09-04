#!/usr/bin/env python3
"""Detect, visualize, plan, and explicitly execute one pre-insertion move."""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Point, Pose, PoseStamped
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import (
    AttachedCollisionObject,
    CollisionObject,
    DisplayTrajectory,
    MoveItErrorCodes,
    PlanningScene,
    PlanningSceneComponents,
    RobotState,
    RobotTrajectory,
)
from moveit_msgs.srv import (
    ApplyPlanningScene,
    GetMotionPlan,
    GetPositionIK,
    GetPlanningScene,
    GetStateValidity,
)
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Float32, String
from std_srvs.srv import Trigger
import tf2_ros
from trajectory_msgs.msg import JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray
import yaml

from .geometry import (
    compute_preinsert_target,
    invert_transform,
    make_transform,
    matrix_to_quaternion_xyzw,
)
from .ik_branch_selection import (
    XArm7Kinematics,
    diverse_seeds,
    is_duplicate,
    select_candidate,
    trajectory_joint_path_length,
    wrapped_joint_delta,
)
from .moveit_requests import (
    build_joint_motion_plan_request,
    build_position_ik_request,
)
from .execution_gate import hardware_commands_allowed
from .joint_pose import JOINT_NAMES, load_joint_pose
from .per_grasp_calibration import FreshMarkerSampleGate, robust_average_transforms


def _pose_to_transform(pose) -> np.ndarray:
    return make_transform(
        [pose.position.x, pose.position.y, pose.position.z],
        [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
    )


def _transform_message_to_matrix(message) -> np.ndarray:
    value = message.transform
    return make_transform(
        [value.translation.x, value.translation.y, value.translation.z],
        [value.rotation.x, value.rotation.y, value.rotation.z, value.rotation.w],
    )


def _transform_to_pose(transform) -> Pose:
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = [
        float(value) for value in transform[:3, 3]
    ]
    quaternion = matrix_to_quaternion_xyzw(transform[:3, :3])
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = [
        float(value) for value in quaternion
    ]
    return pose


def _rotation_matrix_to_rpy_degrees(rotation) -> tuple[float, float, float]:
    """Return fixed-axis XYZ roll/pitch/yaw for diagnostics only."""
    rotation = np.asarray(rotation, dtype=np.float64)
    horizontal = math.hypot(float(rotation[0, 0]), float(rotation[1, 0]))
    pitch = math.atan2(-float(rotation[2, 0]), horizontal)
    if horizontal > 1.0e-9:
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        roll = math.atan2(-float(rotation[1, 2]), float(rotation[1, 1]))
        yaw = 0.0
    return tuple(math.degrees(value) for value in (roll, pitch, yaw))


def _compact_pose(transform) -> str:
    quaternion = matrix_to_quaternion_xyzw(transform[:3, :3])
    position = ",".join(f"{float(value):+.6f}" for value in transform[:3, 3])
    orientation = ",".join(f"{float(value):+.6f}" for value in quaternion)
    return f"p=[{position}] q=[{orientation}]"


def _frozen_slot_document(base_frame, transform_base_slot, width_m, confidence):
    quaternion = matrix_to_quaternion_xyzw(transform_base_slot[:3, :3])
    return {
        "static_slot_environment_check": {"ros__parameters": {
            "base_frame": str(base_frame),
            "static_slot_translation_xyz": [
                float(value) for value in transform_base_slot[:3, 3]
            ],
            "static_slot_quaternion_xyzw": [float(value) for value in quaternion],
            "static_slot_width_m": float(width_m),
        }},
        "calibrated_preinsert_target": {"ros__parameters": {
            "static_slot_confidence": float(confidence),
        }},
    }


def build_direct_joint_trajectory(joint_names, start, target, duration_s, sample_count):
    """Build the exact linearly interpolated trajectory used for preview and execution."""
    names = [str(name) for name in joint_names]
    start = np.asarray(start, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    count = int(sample_count)
    duration = float(duration_s)
    if len(names) == 0 or start.shape != target.shape or start.size != len(names):
        raise ValueError("direct trajectory joint names/start/target dimensions do not match")
    if count < 2:
        raise ValueError("direct trajectory requires at least two samples")
    if not np.all(np.isfinite(start)) or not np.all(np.isfinite(target)):
        raise ValueError("direct trajectory contains a non-finite joint value")
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("direct trajectory duration must be positive")

    trajectory = RobotTrajectory()
    trajectory.joint_trajectory.joint_names = names
    for index in range(count):
        alpha = float(index + 1) / float(count)
        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in start + alpha * (target - start)]
        point.time_from_start = Duration(seconds=duration * alpha).to_msg()
        trajectory.joint_trajectory.points.append(point)
    return trajectory


def collision_pairs(result):
    return {
        frozenset((str(contact.contact_body_1), str(contact.contact_body_2)))
        for contact in getattr(result, "contacts", [])
    }


class SimplePreinsertNode(Node):
    """A single explicit plan/execute boundary around proven target math."""

    def __init__(self):
        super().__init__("simple_preinsert")
        self._declare_parameters()
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.eef_frame = str(self.get_parameter("eef_frame").value)
        self.tcp_frame = str(self.get_parameter("tcp_frame").value)
        self.planning_link = str(self.get_parameter("planning_link").value)
        self.group_name = str(self.get_parameter("group_name").value)
        self.robot_model_id = str(self.get_parameter("robot_model_id").value)
        self.shadow_full_sequence = bool(
            self.get_parameter("shadow_full_sequence").value
        )
        self.requested_execution = bool(self.get_parameter("allow_execution").value)
        self.allow_execution = hardware_commands_allowed(
            self.requested_execution, self.shadow_full_sequence
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.ik_client = self.create_client(
            GetPositionIK, str(self.get_parameter("ik_service").value)
        )
        self.plan_client = self.create_client(
            GetMotionPlan, str(self.get_parameter("planning_service").value)
        )
        self.scene_client = self.create_client(
            ApplyPlanningScene,
            str(self.get_parameter("apply_planning_scene_service").value),
        )
        self.state_validity_client = self.create_client(
            GetStateValidity,
            str(self.get_parameter("state_validity_service").value),
        )
        self.planning_scene_client = self.create_client(
            GetPlanningScene,
            str(self.get_parameter("get_planning_scene_service").value),
        )
        self.execution_client = (
            ActionClient(
                self,
                ExecuteTrajectory,
                str(self.get_parameter("execution_action").value),
            )
            if self.allow_execution
            else None
        )
        self.direct_execution_client = (
            ActionClient(
                self,
                FollowJointTrajectory,
                str(self.get_parameter("direct_trajectory_action").value),
            )
            if self.allow_execution
            else None
        )

        self.latest_slot_pose = None
        self.latest_slot_receive_ns = None
        self.latest_slot_width = None
        self.latest_confidence = None
        self.latest_joint_state = None
        self.latest_joint_state_ns = None
        self.latest_target = None
        self.latest_target_ns = None
        self.latest_slot_candidate = None
        self.frozen_slot = None
        self.planned_trajectory = None
        self.planned_kind = None
        self.planned_type = None
        self.executing_kind = None
        self.executing_type = None
        self.diagnostics_printed = False
        self.phase = "waiting_for_slot"
        self.pending = None
        self.branch_search = None
        self.branch_kinematics = None
        self.direct_check = None
        self.book_detach_pending = False
        self.book_scene_transition_state = "idle"
        self.per_grasp_samples = []
        self.per_grasp_capture_active = False
        self.per_grasp_capture_requested = False
        self.frozen_eef_book = None
        self.per_grasp_diagnostics = None
        self.per_grasp_sample_gate = FreshMarkerSampleGate(
            self.get_parameter("per_grasp_marker_max_age_s").value
        )
        self.scan_positions = load_joint_pose(
            self.get_parameter("scan_joint_state_path").value
        )
        self.loading_positions = load_joint_pose(
            self.get_parameter("loading_joint_state_path").value
        )

        latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.slot_base_publisher = self.create_publisher(
            PoseStamped, "/bookshelf_simple/slot_pose_base", latched
        )
        self.book_target_publisher = self.create_publisher(
            PoseStamped, "/bookshelf_simple/target_book_pose", latched
        )
        self.eef_target_publisher = self.create_publisher(
            PoseStamped, "/bookshelf_simple/target_eef_pose", latched
        )
        self.tcp_target_publisher = self.create_publisher(
            PoseStamped, "/bookshelf_simple/target_tcp_pose", latched
        )
        self.marker_publisher = self.create_publisher(
            MarkerArray, "/bookshelf_simple/markers", latched
        )
        self.display_publisher = self.create_publisher(
            DisplayTrajectory, "/display_planned_path", latched
        )
        self.status_publisher = self.create_publisher(
            String, "/bookshelf_simple/status", latched
        )
        self.per_grasp_publisher = self.create_publisher(
            PoseStamped, "/bookshelf_simple/per_grasp_eef_book", latched
        )
        self.per_grasp_status_publisher = self.create_publisher(
            String, "/bookshelf_simple/per_grasp_status", latched
        )

        self.create_subscription(
            PoseStamped, "/slot_detector/slot_pose", self._slot_pose_callback, 10
        )
        self.create_subscription(
            Float32, "/slot_detector/slot_width", self._slot_width_callback, 10
        )
        self.create_subscription(
            Float32, "/slot_detector/confidence", self._confidence_callback, 10
        )
        self.create_subscription(
            JointState,
            str(self.get_parameter("joint_states_topic").value),
            self._joint_state_callback,
            20,
        )
        self.create_subscription(
            String,
            "/bookshelf_simple/policy/status",
            self._policy_status_callback,
            10,
        )
        self.create_subscription(
            String,
            "/bookshelf_simple/operator_action_status",
            self._operator_action_status_callback,
            10,
        )
        self.create_service(
            Trigger,
            "/bookshelf_simple/plan_and_execute_preinsert",
            self._plan_trigger_callback,
        )
        self.create_service(
            Trigger,
            "/bookshelf_simple/accept_slot",
            self._accept_slot_callback,
        )
        self.create_service(
            Trigger,
            "/bookshelf_simple/plan_preinsert",
            self._plan_trigger_callback,
        )
        self.create_service(
            Trigger, "/bookshelf_simple/plan_scan", self._plan_scan_callback
        )
        self.create_service(
            Trigger, "/bookshelf_simple/plan_loading", self._plan_loading_callback
        )
        self.create_service(
            Trigger,
            "/bookshelf_simple/plan_return_loading",
            self._plan_return_loading_callback,
        )
        self.create_service(
            Trigger,
            "/bookshelf_simple/execute_preinsert",
            self._execute_trigger_callback,
        )
        self.create_timer(0.1, self._update_target)
        mode = "PLAN+EXECUTE" if self.allow_execution else "PLAN-ONLY"
        self.get_logger().warning(
            f"Simple pre-insertion node ready in {mode} mode. Motion requires the Trigger service."
        )
        self._publish_status("waiting_for_slot")

    def _declare_parameters(self):
        self.declare_parameter("base_frame", "link_base")
        self.declare_parameter("eef_frame", "link_eef")
        self.declare_parameter("tcp_frame", "link_tcp")
        self.declare_parameter("planning_link", "link_tcp")
        self.declare_parameter("group_name", "xarm7")
        self.declare_parameter("robot_model_id", "UF_ROBOT")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter(
            "scan_joint_state_path",
            "~/BookshelfFiles/experiment_configs/operator_joint_poses/scan_joint_state.yaml",
        )
        self.declare_parameter(
            "loading_joint_state_path",
            "~/BookshelfFiles/experiment_configs/operator_joint_poses/loading_joint_state.yaml",
        )
        self.declare_parameter("ik_service", "/compute_ik")
        self.declare_parameter("planning_service", "/plan_kinematic_path")
        self.declare_parameter("apply_planning_scene_service", "/apply_planning_scene")
        self.declare_parameter("execution_action", "/execute_trajectory")
        self.declare_parameter("state_validity_service", "/check_state_validity")
        self.declare_parameter("get_planning_scene_service", "/get_planning_scene")
        self.declare_parameter(
            "direct_trajectory_action",
            "/xarm7_traj_controller/follow_joint_trajectory",
        )
        self.declare_parameter("direct_joint_move_duration_s", 5.0)
        self.declare_parameter("direct_trajectory_sample_count", 21)
        self.declare_parameter("held_book_collision_id", "bookshelf_simple_held_book")
        self.declare_parameter("allow_execution", False)
        self.declare_parameter("shadow_full_sequence", False)
        self.declare_parameter("require_slot_acceptance", False)
        self.declare_parameter("separate_execution_confirmation", False)
        self.declare_parameter(
            "frozen_slot_output_path", "/tmp/bookshelf_simple_frozen_slot.yaml"
        )
        self.declare_parameter("print_target_diagnostics", False)
        self.declare_parameter("attach_book_collision", True)
        self.declare_parameter("eef_book_translation_xyz", [
            0.006189808263520789, 0.004397635899244547, 0.18076520526773382,
        ])
        self.declare_parameter("eef_book_quaternion_xyzw", [
            0.7170947434170492, 0.01281329455160485,
            0.6961397093730864, 0.03162994594249451,
        ])
        self.declare_parameter("book_marker_frame", "target_book_center")
        self.declare_parameter("per_grasp_min_samples", 20)
        self.declare_parameter("per_grasp_target_samples", 30)
        self.declare_parameter("per_grasp_marker_max_age_s", 0.25)
        self.declare_parameter("per_grasp_translation_outlier_m", 0.005)
        self.declare_parameter(
            "per_grasp_orientation_outlier_rad", math.radians(5.0)
        )
        self.declare_parameter("book_size_xyz", [0.156, 0.034, 0.236])
        self.declare_parameter("preinsert_standoff_m", 0.030)
        self.declare_parameter("preinsert_vertical_offset_m", 0.006)
        self.declare_parameter("minimum_slot_width_m", 0.020)
        self.declare_parameter("maximum_slot_width_m", 0.090)
        self.declare_parameter("minimum_confidence", 0.60)
        self.declare_parameter("message_max_age_s", 0.50)
        self.declare_parameter("tf_lookup_timeout_s", 0.10)
        self.declare_parameter("maximum_target_translation_m", 0.75)
        self.declare_parameter("workspace_min_xyz", [0.20, -0.60, 0.05])
        self.declare_parameter("workspace_max_xyz", [1.00, 0.60, 1.00])
        self.declare_parameter("ik_timeout_s", 1.0)
        self.declare_parameter("planning_pipeline_id", "")
        self.declare_parameter("planner_id", "")
        self.declare_parameter("planning_attempts", 3)
        self.declare_parameter("allowed_planning_time_s", 5.0)
        self.declare_parameter("velocity_scaling", 0.05)
        self.declare_parameter("acceleration_scaling", 0.05)
        self.declare_parameter("joint_goal_tolerance_rad", 0.001)
        self.declare_parameter("maximum_goal_joint_delta_rad", 1.5)
        self.declare_parameter("ik_branch_seed_count", 24)
        self.declare_parameter("ik_branch_random_seed", 7)
        self.declare_parameter("ik_branch_deduplication_rad", 0.01)
        self.declare_parameter("ik_branch_path_samples", 11)
        self.declare_parameter("predicted_insertion_distance_m", 0.10)
        self.declare_parameter("minimum_predicted_joint_margin_rad", 0.05)
        self.declare_parameter("maximum_predicted_condition", 27.0)
        self.declare_parameter("similar_condition_band", 1.0)
        self.declare_parameter("expected_arm_joint_names", [
            "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7",
        ])
        self.declare_parameter("book_touch_links", [
            "link_eef",
            "link_tcp",
            "xarm_gripper_base_link",
            "left_finger",
            "right_finger",
            "left_inner_knuckle",
            "right_inner_knuckle",
            "left_outer_knuckle",
            "right_outer_knuckle",
        ])

    def _now_ns(self):
        return int(self.get_clock().now().nanoseconds)

    def _fresh(self, timestamp_ns):
        if timestamp_ns is None:
            return False
        return (self._now_ns() - timestamp_ns) * 1.0e-9 <= float(
            self.get_parameter("message_max_age_s").value
        )

    def _slot_pose_callback(self, message):
        self.latest_slot_pose = message
        self.latest_slot_receive_ns = self._now_ns()

    def _slot_width_callback(self, message):
        self.latest_slot_width = float(message.data)

    def _confidence_callback(self, message):
        self.latest_confidence = float(message.data)

    def _joint_state_callback(self, message):
        self.latest_joint_state = message
        self.latest_joint_state_ns = self._now_ns()

    def _policy_status_callback(self, message):
        try:
            phase = str(json.loads(message.data).get("phase", ""))
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        if phase != "release_complete":
            return
        if not bool(self.get_parameter("attach_book_collision").value):
            self.book_scene_transition_state = "verified"
            self._publish_status("book_scene_transition_verified")
            return
        self._begin_release_detach()

    def _operator_action_status_callback(self, message):
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        if payload.get("action") != "close" or not bool(payload.get("success")):
            return
        self.per_grasp_samples = []
        self.per_grasp_sample_gate.reset()
        self.per_grasp_capture_requested = True
        self.frozen_eef_book = None
        self.per_grasp_diagnostics = None
        self.per_grasp_capture_active = True
        self.per_grasp_status_publisher.publish(String(data=json.dumps({
            "source": "collecting", "sample_count": 0,
            **self.per_grasp_sample_gate.diagnostics(),
        }, sort_keys=True)))
        self.get_logger().info(
            "Book grasp complete; collecting stable EEF-to-book marker samples"
        )

    def _fixed_eef_book(self):
        return make_transform(
            self.get_parameter("eef_book_translation_xyz").value,
            self.get_parameter("eef_book_quaternion_xyzw").value,
        )

    def _active_eef_book(self):
        return (
            self.frozen_eef_book
            if self.frozen_eef_book is not None
            else self._fixed_eef_book()
        )

    def _collect_per_grasp_sample(self):
        if not self.per_grasp_capture_active:
            return
        try:
            base_eef = self._lookup(self.base_frame, self.eef_frame)
            book_message = self._lookup_message(
                self.base_frame, str(self.get_parameter("book_marker_frame").value)
            )
        except (tf2_ros.TransformException, ValueError):
            self.per_grasp_sample_gate.reject_lookup()
            self._finish_per_grasp_capture_window_if_needed()
            return
        marker_stamp_ns = Time.from_msg(book_message.header.stamp).nanoseconds
        if not self.per_grasp_sample_gate.accept(marker_stamp_ns, self._now_ns()):
            self._finish_per_grasp_capture_window_if_needed()
            return
        base_book = _transform_message_to_matrix(book_message)
        self.per_grasp_samples.append(invert_transform(base_eef) @ base_book)
        self._finish_per_grasp_capture_window_if_needed()

    def _finish_per_grasp_capture_window_if_needed(self):
        target = int(self.get_parameter("per_grasp_target_samples").value)
        if (self.per_grasp_sample_gate.accepted_count >= target
                or self.per_grasp_sample_gate.total_reads_attempted >= target):
            self._freeze_per_grasp_transform()

    def _fail_per_grasp_transform(self, reason):
        self.frozen_eef_book = None
        self.per_grasp_capture_active = False
        diagnostics = {
            "source": "calibration_failed",
            "result": "FAIL",
            "reason": str(reason),
            "required_count": int(self.get_parameter("per_grasp_min_samples").value),
            **self.per_grasp_sample_gate.diagnostics(self._now_ns()),
        }
        self.per_grasp_diagnostics = diagnostics
        self.per_grasp_status_publisher.publish(
            String(data=json.dumps(diagnostics, sort_keys=True))
        )
        self.get_logger().error(
            "PER-GRASP EEF->BOOK CALIBRATION FAIL: "
            f"{reason}; reads={diagnostics['total_reads_attempted']} "
            f"fresh_unique={diagnostics['unique_fresh_samples']} "
            f"duplicate={diagnostics['duplicate_samples_rejected']} "
            f"stale={diagnostics['stale_samples_rejected']} "
            f"accepted_age_s=[{diagnostics['newest_accepted_sample_age_s']},"
            f"{diagnostics['oldest_accepted_sample_age_s']}]"
        )
        return False

    def _freeze_per_grasp_transform(self):
        if self.frozen_eef_book is not None:
            return True
        minimum = int(self.get_parameter("per_grasp_min_samples").value)
        try:
            self.per_grasp_sample_gate.require_minimum(minimum)
        except ValueError as error:
            return self._fail_per_grasp_transform(error)
        try:
            transform, diagnostics = robust_average_transforms(
                self.per_grasp_samples,
                self.get_parameter("per_grasp_translation_outlier_m").value,
                self.get_parameter("per_grasp_orientation_outlier_rad").value,
            )
        except ValueError as error:
            return self._fail_per_grasp_transform(f"robust averaging invalid: {error}")
        if int(diagnostics["accepted_count"]) < minimum:
            return self._fail_per_grasp_transform(
                "robust averaging rejected too many samples: "
                f"{diagnostics['accepted_count']}/{minimum} required"
            )
        self.frozen_eef_book = transform
        diagnostics = {
            **diagnostics,
            **self.per_grasp_sample_gate.diagnostics(self._now_ns()),
            "result": "PASS",
        }
        self.per_grasp_diagnostics = diagnostics
        self.per_grasp_capture_active = False
        message = PoseStamped()
        message.header.frame_id = self.eef_frame
        message.header.stamp = self.get_clock().now().to_msg()
        message.pose = _transform_to_pose(transform)
        self.per_grasp_publisher.publish(message)
        report = {
            "source": "per_grasp",
            "transform_eef_book": transform.tolist(),
            **diagnostics,
        }
        self.per_grasp_status_publisher.publish(
            String(data=json.dumps(report, sort_keys=True))
        )
        self.get_logger().info(
            "PER-GRASP EEF->BOOK FROZEN: "
            f"reads={diagnostics['total_reads_attempted']} "
            f"fresh_unique={diagnostics['unique_fresh_samples']} "
            f"duplicate={diagnostics['duplicate_samples_rejected']} "
            f"stale={diagnostics['stale_samples_rejected']} "
            f"accepted_age_s=[{diagnostics['newest_accepted_sample_age_s']:.3f},"
            f"{diagnostics['oldest_accepted_sample_age_s']:.3f}] "
            f"accepted={diagnostics['accepted_count']} "
            f"rejected={diagnostics['rejected_count']} "
            f"translation_rms_mm={diagnostics['translation_rms_m'] * 1000.0:.3f} "
            f"orientation_rms_deg={math.degrees(diagnostics['orientation_rms_rad']):.3f} "
            f"{_compact_pose(transform)}"
        )
        return True

    def _planning_scene_request(self):
        request = GetPlanningScene.Request()
        request.components.components = int(
            PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS
            | PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
        )
        return request

    def _begin_release_detach(self):
        if self.book_detach_pending or self.book_scene_transition_state in (
            "transitioning",
            "verified",
            "warning",
        ):
            return
        if not self.planning_scene_client.wait_for_service(timeout_sec=0.25):
            self._book_scene_transition_warning(
                "MoveIt planning-scene query service is unavailable"
            )
            return
        self.book_scene_transition_state = "transitioning"
        self.book_detach_pending = True
        self.get_logger().warning("BOOK SCENE TRANSITION START")
        self._publish_status("book_scene_transition_started")
        future = self.planning_scene_client.call_async(self._planning_scene_request())
        future.add_done_callback(self._release_scene_response_callback)

    def _book_scene_transition_warning(self, reason):
        self.book_detach_pending = False
        self.book_scene_transition_state = "warning"
        self.get_logger().warning(f"BOOK SCENE UPDATE WARNING: {reason}")
        self._publish_status("book_scene_transition_warning", reason=reason)

    def _release_scene_response_callback(self, future):
        book_id = str(self.get_parameter("held_book_collision_id").value)
        try:
            scene = future.result().scene
        except Exception as error:
            self._book_scene_transition_warning(f"scene query failed: {error}")
            return
        attached = next(
            (
                item
                for item in scene.robot_state.attached_collision_objects
                if item.object.id == book_id
            ),
            None,
        )
        if attached is None:
            world = any(item.id == book_id for item in scene.world.collision_objects)
            if world:
                self.get_logger().warning("BOOK DETACHED")
                self.get_logger().warning("BOOK WORLD ADDED")
                self.book_detach_pending = False
                self.book_scene_transition_state = "verified"
                self.get_logger().warning("BOOK SCENE TRANSITION VERIFIED")
                self._publish_status("book_scene_transition_verified")
            else:
                self._book_scene_transition_warning(
                    "book is neither attached nor present in the world planning scene"
                )
            return
        try:
            request = self._detach_book_to_world_request(attached)
        except Exception as error:
            self._book_scene_transition_warning(f"detach preparation failed: {error}")
            return
        future = self.scene_client.call_async(request)
        future.add_done_callback(self._release_detach_response_callback)

    def _detach_book_to_world_request(self, attached):
        remove = AttachedCollisionObject()
        remove.link_name = attached.link_name
        remove.object.id = attached.object.id
        remove.object.operation = CollisionObject.REMOVE
        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects = [remove]
        request = ApplyPlanningScene.Request()
        request.scene = scene
        return request

    def _release_detach_response_callback(self, future):
        try:
            success = bool(future.result().success)
        except Exception as error:
            self._book_scene_transition_warning(f"detach failed: {error}")
            return
        if not success:
            self._book_scene_transition_warning("MoveIt rejected released-book detach")
            return
        self.get_logger().warning("BOOK DETACHED")
        future = self.planning_scene_client.call_async(self._planning_scene_request())
        future.add_done_callback(self._verify_release_detach_callback)

    def _verify_release_detach_callback(self, future):
        book_id = str(self.get_parameter("held_book_collision_id").value)
        try:
            scene = future.result().scene
        except Exception as error:
            self._book_scene_transition_warning(f"detach verification failed: {error}")
            return
        attached = any(
            item.object.id == book_id
            for item in scene.robot_state.attached_collision_objects
        )
        world = any(item.id == book_id for item in scene.world.collision_objects)
        if attached or not world:
            self._book_scene_transition_warning(
                f"post-detach state attached={attached} world={world}"
            )
            return
        self.book_detach_pending = False
        self.book_scene_transition_state = "verified"
        self.get_logger().warning("BOOK WORLD ADDED")
        self.get_logger().warning("BOOK SCENE TRANSITION VERIFIED")
        self._publish_status("book_scene_transition_verified")

    def _lookup(self, parent, child):
        return _transform_message_to_matrix(self._lookup_message(parent, child))

    def _lookup_message(self, parent, child):
        timeout = Duration(seconds=float(self.get_parameter("tf_lookup_timeout_s").value))
        return self.tf_buffer.lookup_transform(parent, child, Time(), timeout=timeout)

    def _live_slot_candidate(self):
        if not self._fresh(self.latest_slot_receive_ns):
            return None
        if self.latest_confidence is None or self.latest_slot_width is None:
            return None
        if self.latest_confidence < float(self.get_parameter("minimum_confidence").value):
            return None
        if not (float(self.get_parameter("minimum_slot_width_m").value)
                <= self.latest_slot_width
                <= float(self.get_parameter("maximum_slot_width_m").value)):
            return None
        try:
            source_frame = self.latest_slot_pose.header.frame_id
            base_source = (
                np.eye(4, dtype=np.float64)
                if source_frame == self.base_frame
                else self._lookup(self.base_frame, source_frame)
            )
            base_slot = base_source @ _pose_to_transform(self.latest_slot_pose.pose)
        except (tf2_ros.TransformException, ValueError) as error:
            self._publish_status("waiting_for_tf", reason=str(error))
            return None
        return (
            base_slot,
            float(self.latest_slot_width),
            float(self.latest_confidence),
            self._now_ns(),
        )

    def _update_target(self):
        self._collect_per_grasp_sample()
        if self.frozen_slot is None:
            candidate = self._live_slot_candidate()
            if candidate is None:
                return
            self.latest_slot_candidate = candidate
            base_slot, slot_width, _, _ = candidate
            self._publish_slot_only(base_slot, slot_width)
            if bool(self.get_parameter("require_slot_acceptance").value):
                if self.phase in ("waiting_for_slot", "waiting_for_tf"):
                    self._publish_status("slot_candidate_ready")
                return
        else:
            base_slot, slot_width, _ = self.frozen_slot

        try:
            base_eef = self._lookup(self.base_frame, self.eef_frame)
            base_tcp = self._lookup(self.base_frame, self.tcp_frame)
            eef_tcp = invert_transform(base_eef) @ base_tcp
            eef_book = self._active_eef_book()
            book_size = self.get_parameter("book_size_xyz").value
            target = compute_preinsert_target(
                base_slot,
                eef_book,
                eef_tcp,
                book_depth_m=float(book_size[0]),
                standoff_m=float(self.get_parameter("preinsert_standoff_m").value),
                vertical_offset_m=float(self.get_parameter("preinsert_vertical_offset_m").value),
            )
        except (tf2_ros.TransformException, ValueError) as error:
            self._publish_status("waiting_for_tf", reason=str(error))
            return
        self.latest_target = (base_slot, target, base_tcp)
        self.latest_target_ns = self._now_ns()
        self._print_target_diagnostics_once(base_slot, target)
        self._publish_poses_and_markers(base_slot, target, slot_width)
        if self.phase in ("waiting_for_slot", "waiting_for_tf", "slot_frozen"):
            self._publish_status("awaiting_plan_confirmation")

    def _print_target_diagnostics_once(self, base_slot, target):
        if self.diagnostics_printed or not bool(
            self.get_parameter("print_target_diagnostics").value
        ):
            return
        relative_rpy = _rotation_matrix_to_rpy_degrees(
            target.transform_slot_book[:3, :3]
        )
        rpy = ",".join(f"{value:+.3f}" for value in relative_rpy)
        self.get_logger().info(
            "PREINSERT "
            f"slot_base({_compact_pose(base_slot)}) | "
            f"book_base({_compact_pose(target.transform_base_book)}) | "
            f"tcp_base({_compact_pose(target.transform_base_tcp)}) | "
            f"book_slot({_compact_pose(target.transform_slot_book)}) | "
            f"book_vs_slot_rpy_deg=[{rpy}]"
        )
        self.diagnostics_printed = True

    def _stamped_pose(self, transform):
        message = PoseStamped()
        message.header.frame_id = self.base_frame
        message.header.stamp = self.get_clock().now().to_msg()
        message.pose = _transform_to_pose(transform)
        return message

    def _publish_slot_only(self, base_slot, slot_width):
        slot = self._stamped_pose(base_slot)
        self.slot_base_publisher.publish(slot)
        self.marker_publisher.publish(MarkerArray(markers=[self._box_marker(
            0,
            "detected_slot",
            slot.pose,
            [0.008, slot_width, 0.25],
            [0.1, 0.9, 0.2, 0.30],
        )]))

    def _publish_poses_and_markers(self, base_slot, target, slot_width):
        slot = self._stamped_pose(base_slot)
        book = self._stamped_pose(target.transform_base_book)
        eef = self._stamped_pose(target.transform_base_eef)
        tcp = self._stamped_pose(target.transform_base_tcp)
        self.slot_base_publisher.publish(slot)
        self.book_target_publisher.publish(book)
        self.eef_target_publisher.publish(eef)
        self.tcp_target_publisher.publish(tcp)
        markers = MarkerArray()
        markers.markers.append(self._box_marker(
            0, "detected_slot", slot.pose,
            [0.008, slot_width, 0.25], [0.1, 0.9, 0.2, 0.30],
        ))
        markers.markers.append(self._box_marker(
            1, "target_book", book.pose,
            self.get_parameter("book_size_xyz").value, [0.1, 0.4, 1.0, 0.45],
        ))
        markers.markers.extend(self._axis_markers(tcp.pose, 10, "target_tcp"))
        self.marker_publisher.publish(markers)

    def _box_marker(self, marker_id, namespace, pose, scale, rgba):
        marker = Marker()
        marker.header.frame_id = self.base_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose = pose
        marker.scale.x, marker.scale.y, marker.scale.z = [float(v) for v in scale]
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = [float(v) for v in rgba]
        return marker

    def _axis_markers(self, pose, first_id, namespace):
        base = _pose_to_transform(pose)
        colors = ([1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.3, 1.0, 1.0])
        markers = []
        for axis in range(3):
            marker = Marker()
            marker.header.frame_id = self.base_frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = namespace
            marker.id = first_id + axis
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose = pose
            direction = base[:3, axis]
            start = Point(x=float(base[0,3]), y=float(base[1,3]), z=float(base[2,3]))
            end = Point(x=float(base[0,3] + 0.08*direction[0]),
                        y=float(base[1,3] + 0.08*direction[1]),
                        z=float(base[2,3] + 0.08*direction[2]))
            marker.pose = Pose()
            marker.pose.orientation.w = 1.0
            marker.points = [start, end]
            marker.scale.x, marker.scale.y, marker.scale.z = 0.006, 0.012, 0.018
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = colors[axis]
            markers.append(marker)
        return markers

    def _clear_trial_state_for_scan(self):
        """Release only per-trial state while preserving configuration and ROS I/O."""

        self.frozen_slot = None
        self.latest_target = None
        self.latest_target_ns = None
        self.planned_trajectory = None
        self.planned_kind = None
        self.planned_type = None
        self.executing_kind = None
        self.executing_type = None
        self.pending = None
        self.branch_search = None
        self.direct_check = None
        self.book_detach_pending = False
        self.book_scene_transition_state = "idle"
        self.diagnostics_printed = False

    def _plan_scan_callback(self, _request, response):
        self._clear_trial_state_for_scan()
        return self._plan_joint_pose("scan", self.scan_positions, response)

    def _plan_loading_callback(self, _request, response):
        return self._plan_joint_pose("loading", self.loading_positions, response)

    def _plan_return_loading_callback(self, _request, response):
        return self._plan_joint_pose("return_loading", self.loading_positions, response)

    def _plan_joint_pose(self, kind, positions, response):
        if self.phase in (
            "clearing_book",
            "attaching_book",
            "requesting_ik",
            "requesting_ik_branches",
            "planning",
            "planning_ik_branches",
            "executing",
        ):
            response.success = False
            response.message = f"workflow is busy ({self.phase})"
            return response
        if self.latest_joint_state is None or not self._fresh(self.latest_joint_state_ns):
            response.success = False
            response.message = "no fresh joint state is available"
            return response
        if not self.state_validity_client.wait_for_service(timeout_sec=0.25):
            response.success = False
            response.message = "MoveIt state-validity service is unavailable"
            return response
        self.planned_trajectory = None
        self.planned_kind = None
        self.planned_type = None
        self.executing_kind = None
        self.executing_type = None
        self.pending = {
            "kind": str(kind),
            "joint_state": copy.deepcopy(self.latest_joint_state),
            "joint_positions": [float(value) for value in positions],
        }
        if kind == "return_loading":
            if self.book_detach_pending or self.book_scene_transition_state != "verified":
                self.get_logger().warning(
                    "RETURN: released-book planning-scene state is stale or incomplete; "
                    "ignoring that object while preserving robot/environment checks"
                )
            self._prepare_direct_joint_trajectory()
            response.success = True
            response.message = (
                f"{kind} trajectory verification accepted; execution requires E"
            )
            return response
        if not self.planning_scene_client.wait_for_service(timeout_sec=0.25):
            self.pending = None
            response.success = False
            response.message = "MoveIt planning-scene query service is unavailable"
            return response
        self._publish_status("checking_book_lifecycle")
        future = self.planning_scene_client.call_async(self._planning_scene_request())
        future.add_done_callback(self._direct_scene_state_response_callback)
        response.success = True
        response.message = f"{kind} trajectory verification accepted; execution requires E"
        return response

    def _book_scene_cleanup_request(self, scene_state):
        book_id = str(self.get_parameter("held_book_collision_id").value)
        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        for item in scene_state.robot_state.attached_collision_objects:
            if item.object.id != book_id:
                continue
            remove = AttachedCollisionObject()
            remove.link_name = item.link_name
            remove.object.id = book_id
            remove.object.operation = CollisionObject.REMOVE
            scene.robot_state.attached_collision_objects.append(remove)
        if any(item.id == book_id for item in scene_state.world.collision_objects):
            remove = CollisionObject()
            remove.header.frame_id = self.base_frame
            remove.id = book_id
            remove.operation = CollisionObject.REMOVE
            scene.world.collision_objects.append(remove)
        request = ApplyPlanningScene.Request()
        request.scene = scene
        return request

    def _direct_scene_state_response_callback(self, future):
        book_id = str(self.get_parameter("held_book_collision_id").value)
        try:
            scene = future.result().scene
        except Exception as error:
            self._fail(f"planning-scene query failed: {error}")
            return
        attached = [
            item
            for item in scene.robot_state.attached_collision_objects
            if item.object.id == book_id
        ]
        world = [item for item in scene.world.collision_objects if item.id == book_id]
        if not attached and not world:
            self._prepare_direct_joint_trajectory()
            return
        if not self.scene_client.wait_for_service(timeout_sec=0.25):
            self._fail("MoveIt planning-scene cleanup service is unavailable")
            return
        self._publish_status("clearing_book")
        future = self.scene_client.call_async(self._book_scene_cleanup_request(scene))
        future.add_done_callback(self._joint_scene_response_callback)

    def _joint_scene_response_callback(self, future):
        try:
            success = bool(future.result().success)
        except Exception as error:
            self._fail(f"planning-scene cleanup failed: {error}")
            return
        if not success:
            self._fail("MoveIt rejected held-book cleanup")
            return
        self._prepare_direct_joint_trajectory()

    def _prepare_direct_joint_trajectory(self):
        current = dict(zip(self.pending["joint_state"].name, self.pending["joint_state"].position))
        missing = [name for name in JOINT_NAMES if name not in current]
        if missing:
            self._fail(f"current joint state is missing {missing}")
            return
        start = np.asarray([current[name] for name in JOINT_NAMES], dtype=np.float64)
        target = np.asarray(self.pending["joint_positions"], dtype=np.float64)
        try:
            if self.branch_kinematics is None:
                self.branch_kinematics = XArm7Kinematics(JOINT_NAMES, self.planning_link)
            if np.any(target < self.branch_kinematics.lower) or np.any(
                target > self.branch_kinematics.upper
            ):
                offenders = [
                    name
                    for name, value, lower, upper in zip(
                        JOINT_NAMES,
                        target,
                        self.branch_kinematics.lower,
                        self.branch_kinematics.upper,
                    )
                    if value < lower or value > upper
                ]
                raise ValueError(f"target joints outside model limits: {offenders}")
            trajectory = build_direct_joint_trajectory(
                JOINT_NAMES,
                start,
                target,
                float(self.get_parameter("direct_joint_move_duration_s").value),
                int(self.get_parameter("direct_trajectory_sample_count").value),
            )
        except Exception as error:
            self._fail(f"direct trajectory generation failed: {error}")
            return

        states = [self._direct_robot_state(self.pending["joint_state"], start)]
        states.extend(
            self._direct_robot_state(self.pending["joint_state"], point.positions)
            for point in trajectory.joint_trajectory.points
        )
        self.direct_check = {
            "kind": self.pending["kind"],
            "trajectory": trajectory,
            "trajectory_start": RobotState(
                joint_state=copy.deepcopy(self.pending["joint_state"]), is_diff=False
            ),
            "states": states,
            "index": 0,
        }
        self._publish_status("verifying_direct_trajectory")
        self._request_next_direct_state_check()

    @staticmethod
    def _direct_robot_state(template, positions):
        state = copy.deepcopy(template)
        values = dict(zip(state.name, state.position))
        values.update(dict(zip(JOINT_NAMES, positions)))
        state.position = [float(values[name]) for name in state.name]
        state.velocity = []
        state.effort = []
        return RobotState(joint_state=state, is_diff=False)

    def _request_next_direct_state_check(self):
        check = self.direct_check
        if check is None:
            return
        if check["index"] >= len(check["states"]):
            self._accept_direct_trajectory()
            return
        request = GetStateValidity.Request()
        request.robot_state = check["states"][check["index"]]
        request.group_name = self.group_name
        future = self.state_validity_client.call_async(request)
        future.add_done_callback(self._direct_state_check_callback)

    def _direct_state_check_callback(self, future):
        check = self.direct_check
        if check is None:
            return
        sample = int(check["index"])
        try:
            result = future.result()
        except Exception as error:
            self._fail(f"direct trajectory collision check failed at sample {sample}: {error}")
            return
        pairs = collision_pairs(result)
        if check["kind"] == "return_loading":
            error = self._return_contact_error(result, pairs)
            if error:
                self._fail(error)
                return
        elif not bool(result.valid):
            self._fail(
                self._collision_failure_message(
                    "direct trajectory is invalid/colliding", sample, pairs
                )
            )
            return
        check["index"] += 1
        self._request_next_direct_state_check()

    @staticmethod
    def _pair_labels(pairs):
        return sorted("<->".join(sorted(pair)) for pair in pairs)

    def _collision_failure_message(self, prefix, sample, pairs):
        detail = f"; contacts={self._pair_labels(pairs)[:4]}" if pairs else ""
        return (
            f"{prefix} at sample {sample}/{len(self.direct_check['states']) - 1}"
            f"{detail}"
        )

    def _return_contact_error(self, result, pairs):
        check = self.direct_check
        sample = int(check["index"])
        book_id = str(self.get_parameter("held_book_collision_id").value)
        book_pairs = {pair for pair in pairs if book_id in pair}
        blocking_pairs = pairs - book_pairs
        if blocking_pairs:
            return self._collision_failure_message(
                "return trajectory is invalid/colliding", sample, blocking_pairs
            )
        if not bool(result.valid) and not book_pairs:
            return self._collision_failure_message(
                "return trajectory is invalid without reported held-book contact",
                sample,
                pairs,
            )
        if book_pairs:
            self.get_logger().warning(
                "RETURN: ignoring stale held-book planning-scene contacts at sample "
                f"{sample}: {self._pair_labels(book_pairs)[:4]}"
            )
        return None

    def _accept_direct_trajectory(self):
        check = self.direct_check
        if check is None:
            return
        display = DisplayTrajectory()
        display.model_id = self.robot_model_id
        display.trajectory_start = check["trajectory_start"]
        display.trajectory = [check["trajectory"]]
        self.display_publisher.publish(display)
        self.planned_trajectory = check["trajectory"]
        self.planned_kind = str(check["kind"])
        self.planned_type = "direct_joint"
        sample_count = len(check["trajectory"].joint_trajectory.points)
        self.direct_check = None
        self.pending = None
        self.get_logger().warning(
            f"{self.planned_kind.upper()} DIRECT TRAJECTORY READY - "
            f"{sample_count} collision-checked samples; waiting for E"
        )
        self._publish_status("awaiting_execute_confirmation")

    def _accept_slot_callback(self, _request, response):
        if self.phase in (
            "attaching_book",
            "requesting_ik",
            "requesting_ik_branches",
            "planning",
            "planning_ik_branches",
            "executing",
        ):
            response.success = False
            response.message = f"workflow is busy ({self.phase})"
            return response
        candidate = self.latest_slot_candidate
        if candidate is None or not self._fresh(candidate[3]):
            response.success = False
            response.message = "no fresh valid slot candidate is available"
            return response
        base_slot, width, confidence, _ = candidate
        document = _frozen_slot_document(
            self.base_frame, base_slot, width, confidence
        )
        output_path = Path(os.path.expandvars(os.path.expanduser(str(
            self.get_parameter("frozen_slot_output_path").value
        ))))
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as stream:
                yaml.safe_dump(document, stream, sort_keys=False)
        except OSError as error:
            response.success = False
            response.message = f"could not save frozen slot: {error}"
            return response
        self.frozen_slot = (base_slot.copy(), float(width), float(confidence))
        self.latest_target = None
        self.planned_trajectory = None
        self.planned_kind = None
        self.planned_type = None
        self.diagnostics_printed = False
        self._publish_status("slot_frozen", reason=str(output_path))
        response.success = True
        response.message = f"slot frozen and saved to {output_path}"
        return response

    def _target_error(self):
        if self.phase in (
            "attaching_book",
            "requesting_ik",
            "requesting_ik_branches",
            "planning",
            "planning_ik_branches",
            "executing",
        ):
            return f"workflow is busy ({self.phase})"
        if (bool(self.get_parameter("require_slot_acceptance").value)
                and self.frozen_slot is None):
            return "slot has not been accepted and frozen"
        if self.latest_target is None or not self._fresh(self.latest_target_ns):
            return "no fresh valid detected-slot target is available"
        if self.latest_joint_state is None or not self._fresh(self.latest_joint_state_ns):
            return "no fresh joint state is available"
        _, target, current_tcp = self.latest_target
        lower = np.asarray(self.get_parameter("workspace_min_xyz").value, dtype=float)
        upper = np.asarray(self.get_parameter("workspace_max_xyz").value, dtype=float)
        xyz = target.transform_base_tcp[:3, 3]
        if np.any(xyz < lower) or np.any(xyz > upper):
            return "target TCP lies outside the configured workspace"
        distance = float(np.linalg.norm(xyz - current_tcp[:3, 3]))
        if distance > float(self.get_parameter("maximum_target_translation_m").value):
            return f"target TCP translation {distance:.3f} m exceeds the limit"
        return None

    def _plan_trigger_callback(self, _request, response):
        newly_frozen = (
            self.per_grasp_capture_active
            and self.frozen_eef_book is None
            and self._freeze_per_grasp_transform()
        )
        if newly_frozen:
            # Refresh the Cartesian target with the just-frozen grasp transform.
            self._update_target()
        if self.frozen_eef_book is None:
            if self.per_grasp_capture_requested:
                reason = (
                    self.per_grasp_diagnostics or {}
                ).get("reason", "per-grasp calibration did not produce a transform")
                response.success = False
                response.message = f"per-grasp calibration failed: {reason}; press C to retry"
                self._publish_status("rejected", reason=response.message)
                return response
            self.per_grasp_capture_active = False
            report = {
                "source": "fixed_fallback",
                "sample_count": len(self.per_grasp_samples),
                "required_count": int(
                    self.get_parameter("per_grasp_min_samples").value
                ),
                "transform_eef_book": self._fixed_eef_book().tolist(),
            }
            self.per_grasp_status_publisher.publish(
                String(data=json.dumps(report, sort_keys=True))
            )
            self.get_logger().warning(
                "PER-GRASP EEF->BOOK unavailable at plan request; explicitly "
                f"using fixed fallback ({len(self.per_grasp_samples)} valid samples)"
            )
        error = self._target_error()
        if error:
            response.success = False
            response.message = error
            self._publish_status("rejected", reason=error)
            return response
        required = ((self.ik_client, "IK"), (self.plan_client, "planning"))
        for client, label in required:
            if not client.wait_for_service(timeout_sec=0.25):
                response.success = False
                response.message = f"MoveIt {label} service is unavailable"
                return response
        self.pending = {
            "kind": "preinsert",
            "target": copy.deepcopy(self.latest_target[1]),
            "joint_state": copy.deepcopy(self.latest_joint_state),
            "insertion_direction": np.asarray(
                self.latest_target[0][:3, 0], dtype=np.float64
            ).copy(),
        }
        self.planned_trajectory = None
        self.planned_kind = None
        self.planned_type = None
        self.executing_kind = None
        self.executing_type = None
        if bool(self.get_parameter("attach_book_collision").value):
            if not self.scene_client.wait_for_service(timeout_sec=0.25):
                response.success = False
                response.message = "MoveIt planning-scene service is unavailable"
                self.pending = None
                return response
            self._publish_status("attaching_book")
            future = self.scene_client.call_async(self._book_scene_request())
            future.add_done_callback(self._scene_response_callback)
        else:
            self._request_ik()
        response.success = True
        response.message = (
            "plan request accepted; execution requires separate confirmation"
            if bool(self.get_parameter("separate_execution_confirmation").value)
            else (
                "plan and execution request accepted" if self.allow_execution
                else "plan request accepted; execution is disabled"
            )
        )
        return response

    def _execute_trigger_callback(self, _request, response):
        if not bool(self.get_parameter("separate_execution_confirmation").value):
            response.success = False
            response.message = "separate execution confirmation is not enabled"
            return response
        if (
            getattr(self, "shadow_full_sequence", False)
            and self.phase == "awaiting_execute_confirmation"
            and self.planned_trajectory is not None
        ):
            kind = self.planned_kind
            trajectory_type = self.planned_type
            self.planned_trajectory = None
            self.planned_kind = None
            self.planned_type = None
            self.executing_kind = kind
            self.executing_type = trajectory_type
            self.get_logger().warning(
                f"SHADOW: would execute {kind} {trajectory_type} trajectory"
            )
            self._publish_status(
                "executing",
                reason=f"SHADOW: would execute {kind} {trajectory_type} trajectory",
            )
            self._publish_status(
                "done", reason=f"SHADOW: {kind} logical completion"
            )
            self.executing_kind = None
            self.executing_type = None
            response.success = True
            response.message = f"SHADOW: {kind} execution suppressed"
            return response
        if not self.allow_execution:
            response.success = False
            response.message = "trajectory execution is disabled"
            return response
        if self.phase != "awaiting_execute_confirmation" or self.planned_trajectory is None:
            response.success = False
            response.message = "no confirmed trajectory is awaiting execution"
            return response
        trajectory = self.planned_trajectory
        self.executing_kind = self.planned_kind
        self.executing_type = self.planned_type
        self.planned_trajectory = None
        self.planned_kind = None
        self.planned_type = None
        if self.executing_type == "direct_joint":
            if (
                self.direct_execution_client is None
                or not self.direct_execution_client.wait_for_server(timeout_sec=0.5)
            ):
                self._fail("direct joint trajectory action is unavailable")
                response.success = False
                response.message = "direct joint trajectory action is unavailable"
                return response
            self._send_direct_execution(trajectory)
            response.message = "reviewed direct joint trajectory submitted"
        else:
            if (
                self.execution_client is None
                or not self.execution_client.wait_for_server(timeout_sec=0.5)
            ):
                self._fail("MoveIt execution action is unavailable")
                response.success = False
                response.message = "MoveIt execution action is unavailable"
                return response
            self._send_execution(trajectory)
            response.message = "reviewed MoveIt trajectory submitted"
        response.success = True
        return response

    def _book_scene_request(self):
        book_size = self.get_parameter("book_size_xyz").value
        primitive = SolidPrimitive(type=SolidPrimitive.BOX)
        primitive.dimensions = [float(value) for value in book_size]
        collision = CollisionObject()
        collision.header.frame_id = self.eef_frame
        collision.id = "bookshelf_simple_held_book"
        collision.primitives = [primitive]
        collision.primitive_poses = [_transform_to_pose(self._active_eef_book())]
        collision.operation = CollisionObject.ADD
        attached = AttachedCollisionObject()
        attached.link_name = self.eef_frame
        attached.object = collision
        attached.touch_links = [str(value) for value in self.get_parameter("book_touch_links").value]
        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects = [attached]
        request = ApplyPlanningScene.Request()
        request.scene = scene
        return request

    def _scene_response_callback(self, future):
        try:
            success = bool(future.result().success)
        except Exception as error:
            self._fail(f"planning-scene call failed: {error}")
            return
        if not success:
            self._fail("MoveIt rejected the held-book collision object")
            return
        self.book_detach_pending = False
        self.book_scene_transition_state = "attached"
        self._request_ik()

    def _request_ik(self):
        expected = [str(v) for v in self.get_parameter("expected_arm_joint_names").value]
        current = dict(zip(self.pending["joint_state"].name, self.pending["joint_state"].position))
        missing = [name for name in expected if name not in current]
        if missing:
            self._fail(f"current joint state is missing {missing}")
            return
        try:
            if self.branch_kinematics is None:
                self.branch_kinematics = XArm7Kinematics(expected, self.planning_link)
        except Exception as error:
            self._fail(f"could not load xArm7 branch-scoring model: {error}")
            return
        current_arm = np.asarray([current[name] for name in expected], dtype=np.float64)
        seeds = diverse_seeds(
            current_arm,
            self.branch_kinematics.lower,
            self.branch_kinematics.upper,
            int(self.get_parameter("ik_branch_seed_count").value),
            int(self.get_parameter("ik_branch_random_seed").value),
        )
        direction = np.asarray(self.pending["insertion_direction"], dtype=np.float64)
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 0.0:
            self._fail("slot +X insertion direction is invalid")
            return
        self.branch_search = {
            "expected": expected,
            "current_arm": current_arm,
            "seeds": seeds,
            "seed_index": 0,
            "unique": [],
            "candidates": [],
            "generated": 0,
            "surviving_checks": 0,
            "plan_index": 0,
            "direction": direction / direction_norm,
        }
        self._publish_status("requesting_ik_branches")
        self._request_next_seed_ik()

    def _joint_state_with_arm_positions(self, positions):
        state = copy.deepcopy(self.pending["joint_state"])
        values = dict(zip(state.name, state.position))
        values.update(dict(zip(self.branch_search["expected"], positions)))
        state.position = [float(values[name]) for name in state.name]
        state.velocity = []
        state.effort = []
        return state

    def _arm_solution(self, result):
        if int(result.error_code.val) != int(MoveItErrorCodes.SUCCESS):
            return None
        values = dict(zip(result.solution.joint_state.name, result.solution.joint_state.position))
        expected = self.branch_search["expected"]
        if any(name not in values for name in expected):
            return None
        return np.asarray([values[name] for name in expected], dtype=np.float64)

    def _ik_request_for(self, transform_base_tcp, seed):
        request = build_position_ik_request(
            target_pose=_transform_to_pose(transform_base_tcp),
            start_joint_state=self._joint_state_with_arm_positions(seed),
            base_frame=self.base_frame,
            planning_link=self.planning_link,
            group_name=self.group_name,
            timeout_s=float(self.get_parameter("ik_timeout_s").value),
        )
        # Preserve the held-book attachment already applied to the planning scene.
        request.ik_request.robot_state.is_diff = True
        return request

    def _request_next_seed_ik(self):
        search = self.branch_search
        if search is None:
            return
        if search["seed_index"] >= len(search["seeds"]):
            self._begin_candidate_plans()
            return
        seed = search["seeds"][search["seed_index"]]
        search["seed_index"] += 1
        request = self._ik_request_for(
            self.pending["target"].transform_base_tcp, seed
        )
        future = self.ik_client.call_async(request)
        future.add_done_callback(self._seed_ik_response_callback)

    def _seed_ik_response_callback(self, future):
        try:
            joints = self._arm_solution(future.result())
        except Exception as error:
            self._fail(f"MoveIt IK branch call failed: {error}")
            return
        search = self.branch_search
        if joints is None or is_duplicate(
            joints,
            search["unique"],
            float(self.get_parameter("ik_branch_deduplication_rad").value),
        ):
            self._request_next_seed_ik()
            return
        search["unique"].append(joints.copy())
        search["generated"] += 1
        margin = self.branch_kinematics.joint_limit_margin(joints)
        maximum_delta = float(np.max(np.abs(wrapped_joint_delta(
            joints, search["current_arm"]
        ))))
        if (
            margin < float(self.get_parameter("minimum_predicted_joint_margin_rad").value)
            or maximum_delta > float(self.get_parameter("maximum_goal_joint_delta_rad").value)
        ):
            self._request_next_seed_ik()
            return
        candidate = {
            "candidate_id": search["generated"],
            "joints": joints.copy(),
            "last_joints": joints.copy(),
            "conditions": [self.branch_kinematics.condition_number(joints)],
            "minimum_margin": margin,
            "plan": None,
            "transition_cost": math.inf,
        }
        self._request_candidate_path_ik(candidate, 1)

    def _request_candidate_path_ik(self, candidate, sample_index):
        sample_count = max(int(self.get_parameter("ik_branch_path_samples").value), 2)
        if sample_index >= sample_count:
            candidate["max_condition"] = max(candidate["conditions"])
            candidate["final_condition"] = candidate["conditions"][-1]
            self.branch_search["candidates"].append(candidate)
            self.branch_search["surviving_checks"] += 1
            self._request_next_seed_ik()
            return
        distance = (
            float(self.get_parameter("predicted_insertion_distance_m").value)
            * sample_index
            / (sample_count - 1)
        )
        target = self.pending["target"].transform_base_tcp.copy()
        target[:3, 3] += self.branch_search["direction"] * distance
        request = self._ik_request_for(target, candidate["last_joints"])
        future = self.ik_client.call_async(request)
        future.add_done_callback(
            lambda completed, candidate=candidate, sample_index=sample_index: (
                self._candidate_path_ik_response(candidate, sample_index, completed)
            )
        )

    def _candidate_path_ik_response(self, candidate, sample_index, future):
        try:
            joints = self._arm_solution(future.result())
        except Exception as error:
            self._fail(f"MoveIt insertion-path IK call failed: {error}")
            return
        if joints is None:
            self._request_next_seed_ik()
            return
        margin = self.branch_kinematics.joint_limit_margin(joints)
        if margin < float(self.get_parameter("minimum_predicted_joint_margin_rad").value):
            self._request_next_seed_ik()
            return
        candidate["last_joints"] = joints
        candidate["minimum_margin"] = min(candidate["minimum_margin"], margin)
        candidate["conditions"].append(
            self.branch_kinematics.condition_number(joints)
        )
        self._request_candidate_path_ik(candidate, sample_index + 1)

    def _begin_candidate_plans(self):
        search = self.branch_search
        maximum_condition = float(
            self.get_parameter("maximum_predicted_condition").value
        )
        search["candidates"] = [
            candidate
            for candidate in search["candidates"]
            if math.isfinite(candidate["max_condition"])
            and candidate["max_condition"] < maximum_condition
        ]
        self.get_logger().info(
            "PREINSERT IK SEARCH "
            f"generated={search['generated']} "
            f"surviving_collision_joint_checks={search['surviving_checks']} "
            f"below_condition_limit={len(search['candidates'])}"
        )
        if not search["candidates"]:
            self._fail(
                "no acceptable preinsert IK branch survived collision, joint-limit, "
                "insertion-path, and singularity checks"
            )
            return
        search["candidates"].sort(key=lambda candidate: candidate["max_condition"])
        search["plan_index"] = 0
        self._publish_status("planning_ik_branches")
        self._request_next_candidate_plan()

    def _request_next_candidate_plan(self):
        search = self.branch_search
        if search["plan_index"] >= len(search["candidates"]):
            self._finish_branch_selection()
            return
        candidate = search["candidates"][search["plan_index"]]
        search["plan_index"] += 1
        request = build_joint_motion_plan_request(
            target_joint_names=search["expected"],
            target_joint_positions=candidate["joints"],
            start_joint_state=self.pending["joint_state"],
            group_name=self.group_name,
            planning_pipeline_id=str(self.get_parameter("planning_pipeline_id").value),
            planner_id=str(self.get_parameter("planner_id").value),
            planning_attempts=int(self.get_parameter("planning_attempts").value),
            allowed_planning_time_s=float(self.get_parameter("allowed_planning_time_s").value),
            velocity_scaling=float(self.get_parameter("velocity_scaling").value),
            acceleration_scaling=float(self.get_parameter("acceleration_scaling").value),
            joint_tolerance_rad=float(self.get_parameter("joint_goal_tolerance_rad").value),
        )
        # Preserve the held-book attachment while planning from the supplied joints.
        request.motion_plan_request.start_state.is_diff = True
        future = self.plan_client.call_async(request)
        future.add_done_callback(
            lambda completed, candidate=candidate: self._candidate_plan_callback(
                candidate, completed
            )
        )

    def _candidate_plan_callback(self, candidate, future):
        try:
            result = future.result().motion_plan_response
        except Exception as error:
            self._fail(f"MoveIt candidate planning call failed: {error}")
            return
        if (
            int(result.error_code.val) == int(MoveItErrorCodes.SUCCESS)
            and len(result.trajectory.joint_trajectory.points) >= 2
        ):
            candidate["plan"] = result
            candidate["transition_cost"] = trajectory_joint_path_length(
                result.trajectory, self.branch_search["expected"]
            )
        self._request_next_candidate_plan()

    def _finish_branch_selection(self):
        selected = select_candidate(
            self.branch_search["candidates"],
            float(self.get_parameter("similar_condition_band").value),
        )
        if selected is None:
            self._fail("no singularity-safe preinsert IK branch could be planned")
            return
        self.get_logger().info(
            "PREINSERT IK SELECTED "
            f"candidate={selected['candidate_id']} "
            f"max_predicted_condition={selected['max_condition']:.3f} "
            f"transition_cost={selected['transition_cost']:.3f} "
            f"joints={np.array2string(selected['joints'], precision=6, separator=',')}"
        )
        result = selected["plan"]
        self.branch_search = None
        SimplePreinsertNode._accept_plan_result(self, result)

    def _plan_response_callback(self, future):
        try:
            result = future.result().motion_plan_response
        except Exception as error:
            self._fail(f"MoveIt planning call failed: {error}")
            return
        SimplePreinsertNode._accept_plan_result(self, result)

    def _accept_plan_result(self, result):
        points = result.trajectory.joint_trajectory.points
        if int(result.error_code.val) != int(MoveItErrorCodes.SUCCESS) or len(points) < 2:
            self._fail(f"MoveIt planning failed with code {int(result.error_code.val)}")
            return
        display = DisplayTrajectory()
        display.model_id = self.robot_model_id
        display.trajectory_start = result.trajectory_start
        display.trajectory = [result.trajectory]
        self.display_publisher.publish(display)
        if bool(self.get_parameter("separate_execution_confirmation").value):
            kind = str(self.pending.get("kind", "preinsert"))
            self.planned_trajectory = copy.deepcopy(result.trajectory)
            self.planned_kind = kind
            self.planned_type = "moveit"
            self.pending = None
            self.get_logger().warning(
                f"{kind.upper()} PLAN READY - waiting for execution confirmation"
            )
            if getattr(self, "shadow_full_sequence", False):
                self.get_logger().warning(f"SHADOW: {kind} plan ready")
            self._publish_status("awaiting_execute_confirmation")
            return
        if not self.allow_execution:
            self.pending = None
            self._publish_status("planned", reason="execution disabled")
            return
        if self.execution_client is None or not self.execution_client.wait_for_server(timeout_sec=0.5):
            self._fail("MoveIt execution action is unavailable")
            return
        self._send_execution(result.trajectory)

    def _send_execution(self, trajectory):
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory
        self._publish_status("executing")
        future = self.execution_client.send_goal_async(goal)
        future.add_done_callback(self._execution_goal_callback)

    def _send_direct_execution(self, trajectory):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory.joint_trajectory
        self._publish_status("executing")
        future = self.direct_execution_client.send_goal_async(goal)
        future.add_done_callback(self._direct_execution_goal_callback)

    def _direct_execution_goal_callback(self, future):
        try:
            goal_handle = future.result()
        except Exception as error:
            self._fail(f"direct trajectory submission failed: {error}")
            return
        if not goal_handle.accepted:
            self._fail("trajectory controller rejected the direct trajectory")
            return
        goal_handle.get_result_async().add_done_callback(
            self._direct_execution_result_callback
        )

    def _direct_execution_result_callback(self, future):
        try:
            error_code = int(future.result().result.error_code)
        except Exception as error:
            self._fail(f"direct trajectory result failed: {error}")
            return
        self.pending = None
        if error_code == int(FollowJointTrajectory.Result.SUCCESSFUL):
            self._publish_status("done")
            self.executing_kind = None
            self.executing_type = None
        else:
            self._fail(f"direct trajectory execution code {error_code}")

    def _execution_goal_callback(self, future):
        try:
            goal_handle = future.result()
        except Exception as error:
            self._fail(f"trajectory submission failed: {error}")
            return
        if not goal_handle.accepted:
            self._fail("MoveIt rejected the trajectory")
            return
        goal_handle.get_result_async().add_done_callback(self._execution_result_callback)

    def _execution_result_callback(self, future):
        try:
            error_code = int(future.result().result.error_code.val)
        except Exception as error:
            self._fail(f"trajectory result failed: {error}")
            return
        self.pending = None
        if error_code == int(MoveItErrorCodes.SUCCESS):
            self._publish_status("done")
            self.executing_kind = None
            self.executing_type = None
        else:
            self._fail(f"MoveIt execution code {error_code}")

    def _fail(self, reason):
        kind = self.planned_kind or self.executing_kind or (
            self.pending.get("kind") if isinstance(self.pending, dict) else None
        )
        trajectory_type = self.planned_type or self.executing_type or (
            "direct_joint" if self.direct_check is not None else None
        )
        self.pending = None
        self.branch_search = None
        self.planned_trajectory = None
        self.planned_kind = None
        self.planned_type = None
        self.direct_check = None
        self.executing_kind = kind
        self.executing_type = trajectory_type
        self._publish_status("failed", reason=reason)
        self.executing_kind = None
        self.executing_type = None

    def _publish_status(self, phase, reason=None):
        self.phase = phase
        payload = {
            "phase": phase,
            "allow_execution": self.allow_execution,
            "requested_execution": self.requested_execution,
            "shadow_full_sequence": self.shadow_full_sequence,
            "motion_requires_trigger": True,
            "slot_frozen": self.frozen_slot is not None,
            "plan_kind": self.planned_kind or self.executing_kind or (
                self.pending.get("kind") if isinstance(self.pending, dict) else None
            ),
            "trajectory_type": self.planned_type or self.executing_type,
        }
        if reason:
            payload["reason"] = str(reason)
        message = String(data=json.dumps(payload, sort_keys=True))
        self.status_publisher.publish(message)
        if phase in ("failed", "rejected"):
            self.get_logger().warning(str(reason))


def main(args=None):
    rclpy.init(args=args)
    node = SimplePreinsertNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
