#!/usr/bin/env python3
"""Build the trained 12D bookshelf observation in read-only shadow mode."""

import json
import math

from geometry_msgs.msg import PoseStamped, TransformStamped
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32, Float32MultiArray, Int32, String
import tf2_ros

from .policy_observation_math import (
    OBSERVATION_LABELS,
    ObservationScales,
    compute_policy_observation,
    invert_transform,
    make_transform,
    matrix_to_quaternion_xyzw,
)


_MODE_OBSERVATIONS = {0: 0.0, 1: 0.5, 2: 1.0}
_MODE_NAMES = {0: "insert", 1: "scripted", 2: "push"}


def _stamp_nanoseconds(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _pose_to_transform(pose) -> np.ndarray:
    return make_transform(
        [pose.position.x, pose.position.y, pose.position.z],
        [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
    )


def _transform_message_to_matrix(transform: TransformStamped) -> np.ndarray:
    value = transform.transform
    return make_transform(
        [value.translation.x, value.translation.y, value.translation.z],
        [value.rotation.x, value.rotation.y, value.rotation.z, value.rotation.w],
    )


def _matrix_to_pose_stamped(transform: np.ndarray, frame_id: str, stamp) -> PoseStamped:
    message = PoseStamped()
    message.header.frame_id = frame_id
    message.header.stamp = stamp
    message.pose.position.x = float(transform[0, 3])
    message.pose.position.y = float(transform[1, 3])
    message.pose.position.z = float(transform[2, 3])
    quaternion = matrix_to_quaternion_xyzw(transform[:3, :3])
    message.pose.orientation.x = float(quaternion[0])
    message.pose.orientation.y = float(quaternion[1])
    message.pose.orientation.z = float(quaternion[2])
    message.pose.orientation.w = float(quaternion[3])
    return message


class PolicyObservationAdapterNode(Node):
    """Read-only adapter from real ROS perception/state to the policy observation."""

    def __init__(self):
        super().__init__("policy_observation_adapter")

        self._declare_parameters()
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.ee_frame = str(self.get_parameter("ee_frame").value)
        self.target_book_frame = str(self.get_parameter("target_book_frame").value)
        self.book_pose_source = str(self.get_parameter("book_pose_source").value).strip().lower()
        self.eef_book_transform_status = str(
            self.get_parameter("eef_book_transform_status").value
        ).strip()
        if self.book_pose_source not in ("auto", "marker", "eef_fixed"):
            raise ValueError("book_pose_source must be auto, marker, or eef_fixed.")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.latest_slot_pose = None
        self.latest_slot_arrival_ns = None
        self.latest_slot_width = None
        self.latest_slot_width_arrival_ns = None
        self.latest_confidence = None
        self.latest_confidence_arrival_ns = None
        self.latest_joint_state = None
        self.latest_joint_arrival_ns = None
        self.mode = int(self.get_parameter("default_mode").value)
        self.latched_transform_eef_book = None
        self.last_status_key = None

        self.raw_metrics_publisher = self.create_publisher(
            Float32MultiArray,
            str(self.get_parameter("raw_metrics_topic").value),
            10,
        )
        self.observation_publisher = self.create_publisher(
            Float32MultiArray,
            str(self.get_parameter("observation_topic").value),
            10,
        )
        self.valid_publisher = self.create_publisher(
            Bool,
            str(self.get_parameter("valid_topic").value),
            10,
        )
        self.slot_pose_base_publisher = self.create_publisher(
            PoseStamped,
            str(self.get_parameter("slot_pose_base_topic").value),
            10,
        )
        self.book_pose_base_publisher = self.create_publisher(
            PoseStamped,
            str(self.get_parameter("book_pose_base_topic").value),
            10,
        )
        self.debug_publisher = self.create_publisher(
            String,
            str(self.get_parameter("debug_topic").value),
            10,
        )

        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("slot_pose_topic").value),
            self._slot_pose_callback,
            10,
        )
        self.create_subscription(
            Float32,
            str(self.get_parameter("slot_width_topic").value),
            self._slot_width_callback,
            10,
        )
        self.create_subscription(
            Float32,
            str(self.get_parameter("confidence_topic").value),
            self._confidence_callback,
            10,
        )
        self.create_subscription(
            JointState,
            str(self.get_parameter("joint_states_topic").value),
            self._joint_state_callback,
            20,
        )
        self.create_subscription(
            Int32,
            str(self.get_parameter("mode_topic").value),
            self._mode_callback,
            10,
        )

        frequency = max(float(self.get_parameter("publish_rate_hz").value), 1.0)
        self.timer = self.create_timer(1.0 / frequency, self._timer_callback)

        self.get_logger().info("Policy observation adapter started in SHADOW-ONLY mode.")
        self.get_logger().info("This node has no action, IK, trajectory, or gripper command interfaces.")
        self.get_logger().info(
            f"Book source={self.book_pose_source}, base={self.base_frame}, "
            f"tool={self.ee_frame}, target_book={self.target_book_frame}"
        )

    def _declare_parameters(self):
        self.declare_parameter("base_frame", "link_base")
        self.declare_parameter("ee_frame", "link_eef")
        self.declare_parameter("target_book_frame", "target_book_center")
        self.declare_parameter("book_pose_source", "auto")
        self.declare_parameter("default_mode", 0)

        self.declare_parameter("slot_pose_topic", "/slot_detector/slot_pose")
        self.declare_parameter("slot_width_topic", "/slot_detector/slot_width")
        self.declare_parameter("confidence_topic", "/slot_detector/confidence")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("mode_topic", "/bookshelf_policy/mode")
        self.declare_parameter("raw_metrics_topic", "/bookshelf_policy/raw_metrics")
        self.declare_parameter("observation_topic", "/bookshelf_policy/observation_12d")
        self.declare_parameter("valid_topic", "/bookshelf_policy/observation_valid")
        self.declare_parameter("slot_pose_base_topic", "/bookshelf_policy/slot_pose_base")
        self.declare_parameter("book_pose_base_topic", "/bookshelf_policy/book_pose_base")
        self.declare_parameter("debug_topic", "/bookshelf_policy/adapter_debug")

        self.declare_parameter("book_size_xyz", [0.156, 0.034, 0.236])
        self.declare_parameter("slot_depth_m", 0.20)
        self.declare_parameter("slot_target_offset_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter("tool_offset_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter("minimum_slot_width_m", 0.020)
        self.declare_parameter("maximum_slot_width_m", 0.090)
        self.declare_parameter("minimum_confidence", 0.60)

        self.declare_parameter("gripper_joint_name", "drive_joint")
        self.declare_parameter("gripper_open_joint_position", 0.0)
        self.declare_parameter("gripper_closed_joint_position", 0.85)

        self.declare_parameter("latch_eef_book_from_marker", True)
        self.declare_parameter("use_configured_eef_book_transform", False)
        self.declare_parameter("eef_book_translation_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter("eef_book_quaternion_xyzw", [0.0, 0.0, 0.0, 1.0])
        self.declare_parameter("eef_book_transform_status", "unconfigured")

        self.declare_parameter("rear_to_mouth_obs_scale", 0.08)
        self.declare_parameter("front_to_back_obs_scale", 0.08)
        self.declare_parameter("lat_err_obs_scale", 0.05)
        self.declare_parameter("z_err_obs_scale", 0.05)
        self.declare_parameter("yaw_err_obs_scale_deg", 30.0)
        self.declare_parameter("tool_to_book_obs_scale", 0.25)

        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("message_max_age_s", 0.50)
        self.declare_parameter("tf_max_age_s", 0.50)
        self.declare_parameter("check_source_header_age", True)
        self.declare_parameter("tf_lookup_timeout_s", 0.05)

    def _now_nanoseconds(self) -> int:
        return int(self.get_clock().now().nanoseconds)

    def _slot_pose_callback(self, message: PoseStamped):
        self.latest_slot_pose = message
        self.latest_slot_arrival_ns = self._now_nanoseconds()

    def _slot_width_callback(self, message: Float32):
        self.latest_slot_width = float(message.data)
        self.latest_slot_width_arrival_ns = self._now_nanoseconds()

    def _confidence_callback(self, message: Float32):
        self.latest_confidence = float(message.data)
        self.latest_confidence_arrival_ns = self._now_nanoseconds()

    def _joint_state_callback(self, message: JointState):
        self.latest_joint_state = message
        self.latest_joint_arrival_ns = self._now_nanoseconds()

    def _mode_callback(self, message: Int32):
        mode = int(message.data)
        if mode not in _MODE_OBSERVATIONS:
            self.get_logger().warning(f"Ignoring invalid policy mode {mode}; expected 0, 1, or 2.")
            return
        self.mode = mode

    def _arrival_is_fresh(self, arrival_ns) -> bool:
        if arrival_ns is None:
            return False
        maximum_age = float(self.get_parameter("message_max_age_s").value)
        if maximum_age <= 0.0:
            return True
        return (self._now_nanoseconds() - arrival_ns) * 1.0e-9 <= maximum_age

    def _header_is_fresh(self, stamp) -> bool:
        if not bool(self.get_parameter("check_source_header_age").value):
            return True
        maximum_age = float(self.get_parameter("message_max_age_s").value)
        stamp_ns = _stamp_nanoseconds(stamp)
        if maximum_age <= 0.0 or stamp_ns == 0:
            return True
        age = abs(self._now_nanoseconds() - stamp_ns) * 1.0e-9
        return age <= maximum_age

    def _transform_is_fresh(self, transform: TransformStamped) -> bool:
        maximum_age = float(self.get_parameter("tf_max_age_s").value)
        stamp_ns = _stamp_nanoseconds(transform.header.stamp)
        if maximum_age <= 0.0 or stamp_ns == 0:
            return True
        age = abs(self._now_nanoseconds() - stamp_ns) * 1.0e-9
        return age <= maximum_age

    def _lookup_transform(self, target_frame: str, source_frame: str):
        if target_frame == source_frame:
            return np.eye(4, dtype=np.float64), None
        timeout = max(float(self.get_parameter("tf_lookup_timeout_s").value), 0.0)
        try:
            message = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=timeout),
            )
        except Exception as error:
            return None, f"TF {target_frame} <- {source_frame} unavailable: {error}"
        if not self._transform_is_fresh(message):
            return None, f"TF {target_frame} <- {source_frame} is stale"
        return _transform_message_to_matrix(message), None

    def _slot_transform_in_base(self):
        if self.latest_slot_pose is None:
            return None, "waiting for slot pose"
        if not self._arrival_is_fresh(self.latest_slot_arrival_ns):
            return None, "slot pose callback is stale"
        if not self._header_is_fresh(self.latest_slot_pose.header.stamp):
            return None, "slot pose source timestamp is stale"
        source_frame = self.latest_slot_pose.header.frame_id
        if not source_frame:
            return None, "slot pose has an empty frame_id"

        transform_base_source, error = self._lookup_transform(self.base_frame, source_frame)
        if error:
            return None, error
        transform_source_slot = _pose_to_transform(self.latest_slot_pose.pose)
        offset = self._three_vector_parameter("slot_target_offset_xyz")
        return transform_base_source @ transform_source_slot @ make_transform(offset), None

    def _configured_eef_book_transform(self):
        if not bool(self.get_parameter("use_configured_eef_book_transform").value):
            return None
        translation = self._three_vector_parameter("eef_book_translation_xyz")
        quaternion = self._four_vector_parameter("eef_book_quaternion_xyzw")
        return make_transform(translation, quaternion)

    def _marker_book_transform(self):
        return self._lookup_transform(self.base_frame, self.target_book_frame)

    def _book_transform_in_base(self, transform_base_eef: np.ndarray):
        marker_transform = None
        marker_error = "marker lookup skipped for eef_fixed source"
        if self.book_pose_source != "eef_fixed":
            marker_transform, marker_error = self._marker_book_transform()
            if marker_transform is not None:
                if (
                    self.mode == 0
                    and bool(self.get_parameter("latch_eef_book_from_marker").value)
                ):
                    self.latched_transform_eef_book = (
                        invert_transform(transform_base_eef) @ marker_transform
                    )
                return marker_transform, "marker", None

        if self.book_pose_source == "marker":
            return None, None, marker_error

        transform_eef_book = self.latched_transform_eef_book
        source_name = "latched_eef_book"
        if transform_eef_book is None:
            transform_eef_book = self._configured_eef_book_transform()
            source_name = "configured_eef_book"

        if transform_eef_book is None:
            return None, None, f"{marker_error}; no EEF-to-book transform is available"
        if self.book_pose_source == "auto" and self.mode != 0:
            return (
                None,
                None,
                f"{marker_error}; EEF fallback is disabled after release because mode={_MODE_NAMES[self.mode]}",
            )
        return transform_base_eef @ transform_eef_book, source_name, None

    def _gripper_open_value(self):
        if self.latest_joint_state is None:
            return None, "waiting for joint states"
        if not self._arrival_is_fresh(self.latest_joint_arrival_ns):
            return None, "joint state callback is stale"

        joint_name = str(self.get_parameter("gripper_joint_name").value)
        try:
            index = self.latest_joint_state.name.index(joint_name)
            position = float(self.latest_joint_state.position[index])
        except (ValueError, IndexError):
            return None, f"joint state does not contain {joint_name}"

        opened = float(self.get_parameter("gripper_open_joint_position").value)
        closed = float(self.get_parameter("gripper_closed_joint_position").value)
        denominator = closed - opened
        if abs(denominator) < 1.0e-9:
            return None, "gripper open and closed joint positions are identical"
        gripper_open = (closed - position) / denominator
        return float(np.clip(gripper_open, 0.0, 1.0)), None

    def _three_vector_parameter(self, name: str) -> np.ndarray:
        value = np.asarray(self.get_parameter(name).value, dtype=np.float64)
        if value.shape != (3,):
            raise ValueError(f"Parameter {name} must contain exactly 3 values.")
        return value

    def _four_vector_parameter(self, name: str) -> np.ndarray:
        value = np.asarray(self.get_parameter(name).value, dtype=np.float64)
        if value.shape != (4,):
            raise ValueError(f"Parameter {name} must contain exactly 4 values.")
        return value

    def _observation_scales(self) -> ObservationScales:
        return ObservationScales(
            rear_to_mouth=float(self.get_parameter("rear_to_mouth_obs_scale").value),
            front_to_back=float(self.get_parameter("front_to_back_obs_scale").value),
            lateral=float(self.get_parameter("lat_err_obs_scale").value),
            vertical=float(self.get_parameter("z_err_obs_scale").value),
            yaw=math.radians(float(self.get_parameter("yaw_err_obs_scale_deg").value)),
            tool_to_book=float(self.get_parameter("tool_to_book_obs_scale").value),
        )

    def _validate_detector_values(self):
        if self.latest_slot_width is None or self.latest_confidence is None:
            return "waiting for slot width/confidence"
        if not self._arrival_is_fresh(self.latest_slot_width_arrival_ns):
            return "slot width callback is stale"
        if not self._arrival_is_fresh(self.latest_confidence_arrival_ns):
            return "slot confidence callback is stale"
        minimum_width = float(self.get_parameter("minimum_slot_width_m").value)
        maximum_width = float(self.get_parameter("maximum_slot_width_m").value)
        if not minimum_width <= self.latest_slot_width <= maximum_width:
            return (
                f"slot width {self.latest_slot_width:.4f} m is outside "
                f"[{minimum_width:.4f}, {maximum_width:.4f}] m"
            )
        minimum_confidence = float(self.get_parameter("minimum_confidence").value)
        if self.latest_confidence < minimum_confidence:
            return (
                f"slot confidence {self.latest_confidence:.3f} is below "
                f"{minimum_confidence:.3f}"
            )
        return None

    def _timer_callback(self):
        if self.mode not in _MODE_OBSERVATIONS:
            self._publish_invalid(f"invalid mode {self.mode}")
            return

        detector_error = self._validate_detector_values()
        if detector_error:
            self._publish_invalid(detector_error)
            return

        transform_base_slot, error = self._slot_transform_in_base()
        if error:
            self._publish_invalid(error)
            return

        transform_base_eef, error = self._lookup_transform(self.base_frame, self.ee_frame)
        if error:
            self._publish_invalid(error)
            return

        transform_base_book, book_source, error = self._book_transform_in_base(transform_base_eef)
        if error:
            self._publish_invalid(error)
            return

        gripper_open, error = self._gripper_open_value()
        if error:
            self._publish_invalid(error)
            return

        tool_offset = self._three_vector_parameter("tool_offset_xyz")
        transform_base_tool = transform_base_eef @ make_transform(tool_offset)
        transform_slot_base = invert_transform(transform_base_slot)
        transform_slot_book = transform_slot_base @ transform_base_book
        transform_slot_tool = transform_slot_base @ transform_base_tool

        try:
            raw, observation = compute_policy_observation(
                transform_slot_book,
                transform_slot_tool,
                book_size=self._three_vector_parameter("book_size_xyz"),
                slot_depth=float(self.get_parameter("slot_depth_m").value),
                mode_observation=_MODE_OBSERVATIONS[self.mode],
                gripper_open=gripper_open,
                scales=self._observation_scales(),
            )
        except ValueError as error:
            self._publish_invalid(f"observation geometry error: {error}")
            return

        stamp = self.get_clock().now().to_msg()
        self.raw_metrics_publisher.publish(Float32MultiArray(data=raw.tolist()))
        self.observation_publisher.publish(Float32MultiArray(data=observation.tolist()))
        self.slot_pose_base_publisher.publish(
            _matrix_to_pose_stamped(transform_base_slot, self.base_frame, stamp)
        )
        self.book_pose_base_publisher.publish(
            _matrix_to_pose_stamped(transform_base_book, self.base_frame, stamp)
        )
        self.valid_publisher.publish(Bool(data=True))

        debug = {
            "valid": True,
            "shadow_only": True,
            "mode": _MODE_NAMES[self.mode],
            "book_pose_source": book_source,
            "eef_book_transform_status": self.eef_book_transform_status,
            "slot_width_m": self.latest_slot_width,
            "slot_confidence": self.latest_confidence,
            "raw_metrics": {
                label: round(float(value), 7)
                for label, value in zip(OBSERVATION_LABELS, raw)
            },
            "observation_12d": [round(float(value), 7) for value in observation],
            "vecnormalize_applied": False,
        }
        if self.latched_transform_eef_book is not None:
            quaternion = matrix_to_quaternion_xyzw(
                self.latched_transform_eef_book[:3, :3]
            )
            debug["latched_eef_book"] = {
                "translation_xyz_m": [
                    round(float(value), 7)
                    for value in self.latched_transform_eef_book[:3, 3]
                ],
                "quaternion_xyzw": [
                    round(float(value), 7) for value in quaternion
                ],
            }
        self.debug_publisher.publish(String(data=json.dumps(debug, sort_keys=True)))
        self._log_status_once(
            f"valid:{book_source}",
            f"Valid 12D observation using {book_source}; "
            f"slot={self.latest_slot_width * 1000.0:.1f} mm, "
            f"confidence={self.latest_confidence:.2f}",
        )

    def _publish_invalid(self, reason: str):
        self.valid_publisher.publish(Bool(data=False))
        debug = {
            "valid": False,
            "shadow_only": True,
            "mode": _MODE_NAMES.get(self.mode, "invalid"),
            "reason": reason,
            "vecnormalize_applied": False,
        }
        self.debug_publisher.publish(String(data=json.dumps(debug, sort_keys=True)))
        self._log_status_once(f"invalid:{reason}", f"Observation invalid: {reason}", warning=True)

    def _log_status_once(self, key: str, message: str, warning: bool = False):
        if key == self.last_status_key:
            return
        self.last_status_key = key
        if warning:
            self.get_logger().warning(message)
        else:
            self.get_logger().info(message)


def main(args=None):
    rclpy.init(args=args)
    node = PolicyObservationAdapterNode()
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
