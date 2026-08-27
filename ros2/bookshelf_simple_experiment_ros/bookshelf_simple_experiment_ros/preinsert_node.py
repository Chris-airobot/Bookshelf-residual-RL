#!/usr/bin/env python3
"""Detect, visualize, plan, and explicitly execute one pre-insertion move."""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from geometry_msgs.msg import Point, Pose, PoseStamped
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import (
    AttachedCollisionObject,
    CollisionObject,
    DisplayTrajectory,
    MoveItErrorCodes,
    PlanningScene,
)
from moveit_msgs.srv import ApplyPlanningScene, GetMotionPlan, GetPositionIK
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
from visualization_msgs.msg import Marker, MarkerArray
import yaml

from .geometry import (
    compute_preinsert_target,
    invert_transform,
    make_transform,
    matrix_to_quaternion_xyzw,
)
from .moveit_requests import (
    build_joint_motion_plan_request,
    build_position_ik_request,
)


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
        self.allow_execution = bool(self.get_parameter("allow_execution").value)

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
        self.execution_client = (
            ActionClient(
                self,
                ExecuteTrajectory,
                str(self.get_parameter("execution_action").value),
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
        self.diagnostics_printed = False
        self.phase = "waiting_for_slot"
        self.pending = None

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
        self.declare_parameter("ik_service", "/compute_ik")
        self.declare_parameter("planning_service", "/plan_kinematic_path")
        self.declare_parameter("apply_planning_scene_service", "/apply_planning_scene")
        self.declare_parameter("execution_action", "/execute_trajectory")
        self.declare_parameter("allow_execution", False)
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

    def _lookup(self, parent, child):
        timeout = Duration(seconds=float(self.get_parameter("tf_lookup_timeout_s").value))
        message = self.tf_buffer.lookup_transform(parent, child, Time(), timeout=timeout)
        return _transform_message_to_matrix(message)

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
            eef_book = make_transform(
                self.get_parameter("eef_book_translation_xyz").value,
                self.get_parameter("eef_book_quaternion_xyzw").value,
            )
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

    def _accept_slot_callback(self, _request, response):
        if self.phase in ("attaching_book", "requesting_ik", "planning", "executing"):
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
        self.diagnostics_printed = False
        self._publish_status("slot_frozen", reason=str(output_path))
        response.success = True
        response.message = f"slot frozen and saved to {output_path}"
        return response

    def _target_error(self):
        if self.phase in ("attaching_book", "requesting_ik", "planning", "executing"):
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
            "target": copy.deepcopy(self.latest_target[1]),
            "joint_state": copy.deepcopy(self.latest_joint_state),
        }
        self.planned_trajectory = None
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
        if not self.allow_execution or self.execution_client is None:
            response.success = False
            response.message = "trajectory execution is disabled"
            return response
        if self.phase != "awaiting_execute_confirmation" or self.planned_trajectory is None:
            response.success = False
            response.message = "no confirmed MoveIt plan is awaiting execution"
            return response
        if not self.execution_client.wait_for_server(timeout_sec=0.5):
            response.success = False
            response.message = "MoveIt execution action is unavailable"
            return response
        trajectory = self.planned_trajectory
        self.planned_trajectory = None
        self._send_execution(trajectory)
        response.success = True
        response.message = "trajectory submitted to MoveIt"
        return response

    def _book_scene_request(self):
        book_size = self.get_parameter("book_size_xyz").value
        primitive = SolidPrimitive(type=SolidPrimitive.BOX)
        primitive.dimensions = [float(value) for value in book_size]
        collision = CollisionObject()
        collision.header.frame_id = self.eef_frame
        collision.id = "bookshelf_simple_held_book"
        collision.primitives = [primitive]
        collision.primitive_poses = [_transform_to_pose(make_transform(
            self.get_parameter("eef_book_translation_xyz").value,
            self.get_parameter("eef_book_quaternion_xyzw").value,
        ))]
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
        self._request_ik()

    def _request_ik(self):
        self._publish_status("requesting_ik")
        request = build_position_ik_request(
            target_pose=_transform_to_pose(self.pending["target"].transform_base_tcp),
            start_joint_state=self.pending["joint_state"],
            base_frame=self.base_frame,
            planning_link=self.planning_link,
            group_name=self.group_name,
            timeout_s=float(self.get_parameter("ik_timeout_s").value),
        )
        future = self.ik_client.call_async(request)
        future.add_done_callback(self._ik_response_callback)

    def _ik_response_callback(self, future):
        try:
            result = future.result()
        except Exception as error:
            self._fail(f"MoveIt IK call failed: {error}")
            return
        if int(result.error_code.val) != int(MoveItErrorCodes.SUCCESS):
            self._fail(f"MoveIt IK failed with code {int(result.error_code.val)}")
            return
        expected = [str(v) for v in self.get_parameter("expected_arm_joint_names").value]
        current = dict(zip(self.pending["joint_state"].name, self.pending["joint_state"].position))
        target = dict(zip(result.solution.joint_state.name, result.solution.joint_state.position))
        missing = [name for name in expected if name not in current or name not in target]
        if missing:
            self._fail(f"IK joint state is missing {missing}")
            return
        maximum_delta = max(abs(float(target[name]) - float(current[name])) for name in expected)
        if maximum_delta > float(self.get_parameter("maximum_goal_joint_delta_rad").value):
            self._fail(f"IK branch jump {maximum_delta:.3f} rad exceeds the limit")
            return
        request = build_joint_motion_plan_request(
            target_joint_names=expected,
            target_joint_positions=[target[name] for name in expected],
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
        self._publish_status("planning")
        future = self.plan_client.call_async(request)
        future.add_done_callback(self._plan_response_callback)

    def _plan_response_callback(self, future):
        try:
            result = future.result().motion_plan_response
        except Exception as error:
            self._fail(f"MoveIt planning call failed: {error}")
            return
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
            self.planned_trajectory = copy.deepcopy(result.trajectory)
            self.pending = None
            self.get_logger().warning(
                "PREINSERT PLAN READY - waiting for execution confirmation"
            )
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
        else:
            self._publish_status("failed", reason=f"MoveIt execution code {error_code}")

    def _fail(self, reason):
        self.pending = None
        self.planned_trajectory = None
        self._publish_status("failed", reason=reason)

    def _publish_status(self, phase, reason=None):
        self.phase = phase
        payload = {
            "phase": phase,
            "allow_execution": self.allow_execution,
            "motion_requires_trigger": True,
            "slot_frozen": self.frozen_slot is not None,
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
