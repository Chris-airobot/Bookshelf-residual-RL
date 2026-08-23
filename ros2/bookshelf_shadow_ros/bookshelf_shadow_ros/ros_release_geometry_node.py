#!/usr/bin/env python3
"""Capture one read-only xArm geometry snapshot at a policy release request."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import time

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
import tf2_ros
import yaml

from .policy_observation_math import make_transform
from .policy_shadow_math import POLICY_ACTION_LABELS
from .ros_release_geometry import (
    XARM_GRIPPER_COLLISION_MESHES,
    book_snapshot,
    gripper_collision_snapshot,
    pose_dict,
    relative_pose,
    stl_local_bounds,
)


def _pose_message_to_transform(message: PoseStamped) -> np.ndarray:
    pose = message.pose
    return make_transform(
        [pose.position.x, pose.position.y, pose.position.z],
        [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
    )


def _tf_message_to_transform(message) -> np.ndarray:
    value = message.transform
    return make_transform(
        [value.translation.x, value.translation.y, value.translation.z],
        [value.rotation.x, value.rotation.y, value.rotation.z, value.rotation.w],
    )


def _load_parameters(path: Path, node_name: str) -> dict:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"approved configuration is not a mapping: {path}")
    node = document.get(node_name)
    if not isinstance(node, dict) or not isinstance(node.get("ros__parameters"), dict):
        raise ValueError(f"approved configuration has no {node_name}.ros__parameters")
    return dict(node["ros__parameters"])


class RosReleaseGeometryNode(Node):
    """Subscriber-only capture; it has no controller or command interfaces."""

    def __init__(self):
        super().__init__("ros_release_geometry")
        self._declare_parameters()
        self.output_path = Path(str(self.get_parameter("output_path").value)).expanduser()
        self.approved_config_path = Path(
            str(self.get_parameter("approved_config_path").value)
        ).expanduser()
        if not self.approved_config_path.is_file():
            raise FileNotFoundError(
                f"approved configuration does not exist: {self.approved_config_path}"
            )
        adapter = _load_parameters(self.approved_config_path, "policy_observation_adapter")
        environment = _load_parameters(
            self.approved_config_path, "static_slot_environment_check"
        )
        self.book_size_xyz = np.asarray(adapter["book_size_xyz"], dtype=np.float64)
        self.slot_depth_m = float(adapter["slot_depth_m"])
        self.slot_height_m = float(environment["visual_slot_height_m"])
        self.transform_eef_policy_tool = make_transform(
            adapter["tool_offset_xyz"], adapter["tool_offset_quaternion_xyzw"]
        )
        self.policy_tool_transform_status = str(
            adapter.get("policy_tool_transform_status", "unknown")
        )

        self.base_frame = str(self.get_parameter("base_frame").value)
        self.eef_frame = str(self.get_parameter("eef_frame").value)
        self.tcp_frame = str(self.get_parameter("tcp_frame").value)
        self.capture_condition = str(
            self.get_parameter("capture_condition").value
        ).strip()
        if self.capture_condition not in (
            "release_requested",
            "release_above_threshold",
            "first_valid",
            "task_release",
        ):
            raise ValueError(
                "capture_condition must be release_requested, "
                "release_above_threshold, first_valid, or task_release"
            )

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.latest = {}
        self.pending_policy_debug = None
        self.pending_task_status = None
        self.pending_since = None
        self.completed = False
        self.report_written = False
        self.mesh_bounds = self._load_mesh_bounds()

        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("book_pose_topic").value),
            lambda message: self._remember("book_pose", message),
            10,
        )
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("slot_pose_topic").value),
            lambda message: self._remember("slot_pose", message),
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("adapter_debug_topic").value),
            lambda message: self._remember_json("adapter_debug", message),
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
            str(self.get_parameter("task_status_topic").value),
            self._task_status_callback,
            10,
        )
        self.create_timer(0.05, self._try_capture)
        self.get_logger().info(
            f"READ-ONLY release geometry capture waiting for {self.capture_condition}."
        )
        self.get_logger().info(f"Output: {self.output_path}")
        self.get_logger().info(
            "No publisher, service, action, IK, trajectory, controller, gripper, "
            "or robot-command interface is created."
        )

    def _declare_parameters(self):
        self.declare_parameter("approved_config_path", "")
        self.declare_parameter(
            "output_path", "/tmp/" + "xarm_ros_release_geometry.json"
        )
        self.declare_parameter("capture_condition", "release_requested")
        self.declare_parameter("base_frame", "link_base")
        self.declare_parameter("eef_frame", "link_eef")
        self.declare_parameter("tcp_frame", "link_tcp")
        self.declare_parameter("book_pose_topic", "/bookshelf_policy/book_pose_base")
        self.declare_parameter("slot_pose_topic", "/bookshelf_policy/slot_pose_base")
        self.declare_parameter("adapter_debug_topic", "/bookshelf_policy/adapter_debug")
        self.declare_parameter("policy_debug_topic", "/bookshelf_shadow/policy_debug")
        self.declare_parameter("task_status_topic", "/bookshelf_sim/task_status")
        self.declare_parameter("pair_max_age_s", 0.25)
        self.declare_parameter("pending_timeout_s", 2.0)
        self.declare_parameter("tf_lookup_timeout_s", 0.05)
        self.declare_parameter(
            "gripper_frames", list(XARM_GRIPPER_COLLISION_MESHES)
        )

    def _load_mesh_bounds(self):
        share = Path(get_package_share_directory("xarm_description"))
        result = {}
        for frame, relative_path in XARM_GRIPPER_COLLISION_MESHES.items():
            mesh_path = share / "meshes" / relative_path
            result[frame] = stl_local_bounds(mesh_path)
        return result

    def _remember(self, key, value):
        self.latest[key] = (value, time.monotonic())

    def _remember_json(self, key, message):
        try:
            value = json.loads(message.data)
        except json.JSONDecodeError:
            return
        self._remember(key, value)

    def _policy_debug_callback(self, message):
        if self.completed or self.pending_policy_debug is not None:
            return
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            return
        if not bool(payload.get("valid", False)):
            return
        self._remember("policy_debug", payload)
        if self.capture_condition == "task_release":
            if self.pending_task_status is not None:
                self.pending_policy_debug = payload
                self.pending_since = time.monotonic()
                self.get_logger().info(
                    "Actual task release and policy state synchronized."
                )
            return
        triggered = {
            "release_requested": bool(payload.get("release_requested_diagnostic", False)),
            "release_above_threshold": bool(
                payload.get("release_action_above_threshold", False)
            ),
            "first_valid": True,
        }[self.capture_condition]
        if triggered:
            self.pending_policy_debug = payload
            self.pending_since = time.monotonic()
            self.get_logger().info(
                f"Capture condition met: {self.capture_condition}; synchronizing TF."
            )

    def _task_status_callback(self, message):
        if (
            self.completed
            or self.pending_policy_debug is not None
            or self.capture_condition != "task_release"
        ):
            return
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            return
        if not bool(payload.get("valid", False)) or payload.get("phase") != "opening":
            return
        self.pending_task_status = payload
        try:
            policy_debug = self._recent("policy_debug")
        except ValueError:
            return
        self.pending_policy_debug = policy_debug
        self.pending_since = time.monotonic()
        self.get_logger().info(
            "Actual task release observed; synchronizing release geometry TF."
        )

    def _recent(self, key):
        if key not in self.latest:
            raise ValueError(f"waiting for {key}")
        value, recorded = self.latest[key]
        maximum_age = float(self.get_parameter("pair_max_age_s").value)
        if maximum_age > 0.0 and time.monotonic() - recorded > maximum_age:
            raise ValueError(f"{key} is stale")
        return value

    def _lookup(self, child_frame):
        message = self.tf_buffer.lookup_transform(
            self.base_frame,
            child_frame,
            Time(),
            timeout=Duration(
                seconds=float(self.get_parameter("tf_lookup_timeout_s").value)
            ),
        )
        return _tf_message_to_transform(message)

    def _try_capture(self):
        if self.completed or self.pending_policy_debug is None:
            return
        try:
            report = self._snapshot()
        except Exception as error:
            timeout = float(self.get_parameter("pending_timeout_s").value)
            if time.monotonic() - self.pending_since < timeout:
                return
            self._write_failure(f"capture condition met but geometry was unavailable: {error}")
            self.completed = True
            return
        self._write(report)
        self.completed = True
        self.get_logger().info(f"Release geometry written to {self.output_path}")
        rclpy.shutdown()

    def _snapshot(self):
        book_message = self._recent("book_pose")
        slot_message = self._recent("slot_pose")
        adapter_debug = self._recent("adapter_debug")
        if book_message.header.frame_id != self.base_frame:
            raise ValueError(f"book pose is in {book_message.header.frame_id!r}")
        if slot_message.header.frame_id != self.base_frame:
            raise ValueError(f"slot pose is in {slot_message.header.frame_id!r}")
        if not bool(adapter_debug.get("valid", False)):
            raise ValueError("adapter_debug is invalid")

        transform_base_book = _pose_message_to_transform(book_message)
        transform_base_slot = _pose_message_to_transform(slot_message)
        transform_base_eef = self._lookup(self.eef_frame)
        transform_base_tcp = self._lookup(self.tcp_frame)
        link_transforms = {}
        unavailable_frames = {}
        for frame in self.get_parameter("gripper_frames").value:
            name = str(frame)
            try:
                link_transforms[name] = self._lookup(name)
            except Exception as error:
                unavailable_frames[name] = str(error)
        for required in ("xarm_gripper_base_link", "left_finger", "right_finger"):
            if required not in link_transforms:
                raise ValueError(f"required gripper TF is unavailable: {required}")

        transform_base_policy_tool = transform_base_eef @ self.transform_eef_policy_tool
        policy_debug = self.pending_policy_debug
        policy_mapping = policy_debug.get("policy_action", {})
        policy_action = [
            float(policy_mapping[label]) for label in POLICY_ACTION_LABELS
        ]
        book = book_snapshot(
            transform_base_book,
            transform_base_slot,
            self.book_size_xyz,
            self.slot_depth_m,
        )
        book["task_metrics"] = {
            str(name): float(value)
            for name, value in adapter_debug["raw_metrics"].items()
        }
        slot_width_m = float(adapter_debug["slot_width_m"])
        collision = gripper_collision_snapshot(
            transform_base_slot=transform_base_slot,
            link_transforms_base=link_transforms,
            mesh_bounds=self.mesh_bounds,
            slot_depth_m=self.slot_depth_m,
            slot_width_m=slot_width_m,
            slot_height_m=self.slot_height_m,
        )
        physical_frames = {
            "coordinate_frame": self.base_frame,
            "palm_body_name": "xarm_gripper_base_link",
            "palm": pose_dict(link_transforms["xarm_gripper_base_link"]),
            "left_finger_body_name": "left_finger",
            "left_finger": pose_dict(link_transforms["left_finger"]),
            "right_finger_body_name": "right_finger",
            "right_finger": pose_dict(link_transforms["right_finger"]),
            "eef": pose_dict(transform_base_eef),
            "tcp": pose_dict(transform_base_tcp),
            "all_available_gripper_links": {
                name: pose_dict(value) for name, value in sorted(link_transforms.items())
            },
            "unavailable_gripper_frames": unavailable_frames,
        }
        return {
            "schema_version": 1,
            "kind": "bookshelf_release_geometry_diagnostic",
            "source": "ros_xarm_read_only",
            "generated_at": datetime.now().astimezone().isoformat(),
            "approved_config": str(self.approved_config_path.resolve()),
            "release": {
                "accepted": True,
                "capture_condition": self.capture_condition,
                "mode_observation": float(policy_debug["mode_observation"]),
                "release_action": float(policy_debug["release_action"]),
                "release_requested_diagnostic": bool(
                    policy_debug["release_requested_diagnostic"]
                ),
                "normalized_observation_before_action": [
                    float(value) for value in policy_debug["normalized_observation"]
                ],
                "policy_action": policy_action,
                "policy_debug": policy_debug,
                "adapter_debug": adapter_debug,
                "task_status": self.pending_task_status,
            },
            "book": book,
            "physical_frames": physical_frames,
            "virtual_policy_tool": {
                "coordinate_frame": self.base_frame,
                "pose": pose_dict(transform_base_policy_tool),
                "policy_tool_transform_status": self.policy_tool_transform_status,
                "book_to_policy_tool": relative_pose(
                    transform_base_book, transform_base_policy_tool
                ),
                "book_to_tcp": relative_pose(transform_base_book, transform_base_tcp),
                "tcp_to_policy_tool": relative_pose(
                    transform_base_tcp, transform_base_policy_tool
                ),
            },
            "slot_opening": {
                "coordinate_frame": "slot",
                "pose_base": pose_dict(transform_base_slot),
                "mouth_x_m": 0.0,
                "back_x_m": self.slot_depth_m,
                "center_y_m": 0.0,
                "total_extra_lateral_clearance_m": slot_width_m,
                "minimum_y_m": -0.5 * slot_width_m,
                "maximum_y_m": 0.5 * slot_width_m,
                "minimum_z_m": -0.5 * self.slot_height_m,
                "maximum_z_m": 0.5 * self.slot_height_m,
                "height_source": "approved visual_slot_height_m proxy",
            },
            "static_shelf_obstacle_envelopes": collision,
            "physical_gripper_to_shelf": collision,
            "read_only": True,
            "hardware_commanded_by_capture": False,
        }

    def _write(self, report):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self.report_written = True

    def _write_failure(self, reason):
        self._write(
            {
                "schema_version": 1,
                "kind": "bookshelf_release_geometry_diagnostic",
                "source": "ros_xarm_read_only",
                "generated_at": datetime.now().astimezone().isoformat(),
                "approved_config": str(self.approved_config_path.resolve()),
                "release": {"accepted": False, "reason": reason},
                "read_only": True,
                "hardware_commanded_by_capture": False,
            }
        )
        self.get_logger().error(reason)

    def write_incomplete_report(self):
        if not self.report_written:
            self._write_failure(
                f"capture stopped before condition {self.capture_condition!r} was observed"
            )


def main(args=None):
    rclpy.init(args=args)
    node = RosReleaseGeometryNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.write_incomplete_report()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
