#!/usr/bin/env python3
"""Visualize and sanity-check an immutable static slot without robot commands."""

from datetime import datetime
import json
from pathlib import Path
import time

from geometry_msgs.msg import Point, Pose, PoseStamped, TransformStamped
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import Bool, Float32, String
import tf2_ros
from visualization_msgs.msg import Marker, MarkerArray

from .policy_observation_math import make_transform, matrix_to_quaternion_xyzw
from .static_slot_environment_check import (
    ConsecutiveMatchGate,
    SlotCheckTolerances,
    compare_slot_measurement,
)


def _pose_to_transform(pose: Pose) -> np.ndarray:
    return make_transform(
        [pose.position.x, pose.position.y, pose.position.z],
        [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
    )


def _transform_message_to_matrix(message: TransformStamped) -> np.ndarray:
    value = message.transform
    return make_transform(
        [value.translation.x, value.translation.y, value.translation.z],
        [value.rotation.x, value.rotation.y, value.rotation.z, value.rotation.w],
    )


def _transform_to_pose(transform: np.ndarray) -> Pose:
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = (
        float(value) for value in transform[:3, 3]
    )
    quaternion = matrix_to_quaternion_xyzw(transform[:3, :3])
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = (
        float(value) for value in quaternion
    )
    return pose


class StaticSlotEnvironmentCheckNode(Node):
    def __init__(self):
        super().__init__("static_slot_environment_check")
        self._declare_parameters()

        self.base_frame = str(self.get_parameter("base_frame").value).strip()
        self.reference_status = str(
            self.get_parameter("static_slot_transform_status").value
        ).strip()
        if not self.base_frame:
            raise ValueError("base_frame must not be empty")
        if self.reference_status.lower() in ("", "unknown", "unconfigured"):
            raise ValueError("The configured static slot requires provenance status")

        self.reference_transform = make_transform(
            self._vector_parameter("static_slot_translation_xyz", 3),
            self._vector_parameter("static_slot_quaternion_xyzw", 4),
        )
        self.reference_width_m = float(
            self.get_parameter("static_slot_width_m").value
        )
        self.slot_depth_m = float(self.get_parameter("slot_depth_m").value)
        self.visual_slot_height_m = float(
            self.get_parameter("visual_slot_height_m").value
        )
        if min(
            self.reference_width_m, self.slot_depth_m, self.visual_slot_height_m
        ) <= 0.0:
            raise ValueError("Static slot visualization dimensions must be positive")

        self.tolerances = SlotCheckTolerances(
            maximum_translation_error_m=float(
                self.get_parameter("maximum_translation_error_m").value
            ),
            maximum_rotation_error_deg=float(
                self.get_parameter("maximum_rotation_error_deg").value
            ),
            maximum_width_error_m=float(
                self.get_parameter("maximum_width_error_m").value
            ),
            minimum_confidence=float(
                self.get_parameter("minimum_confidence").value
            ),
        )
        self.gate = ConsecutiveMatchGate(
            int(self.get_parameter("required_matching_samples").value)
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.marker_publisher = self.create_publisher(
            MarkerArray,
            str(self.get_parameter("marker_topic").value),
            latched_qos,
        )
        self.static_pose_publisher = self.create_publisher(
            PoseStamped,
            str(self.get_parameter("static_pose_topic").value),
            latched_qos,
        )
        self.live_pose_publisher = self.create_publisher(
            PoseStamped,
            str(self.get_parameter("live_pose_base_topic").value),
            10,
        )
        self.passed_publisher = self.create_publisher(
            Bool,
            str(self.get_parameter("passed_topic").value),
            latched_qos,
        )
        self.status_publisher = self.create_publisher(
            String,
            str(self.get_parameter("status_topic").value),
            latched_qos,
        )

        self.latest_pose = None
        self.latest_pose_arrival = None
        self.latest_width = None
        self.latest_width_arrival = None
        self.latest_confidence = None
        self.latest_confidence_arrival = None
        self.latest_live_transform = None
        self.latest_comparison = None
        self.latest_reason = None
        self.last_report_state = None

        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("live_pose_topic").value),
            self._pose_callback,
            10,
        )
        self.create_subscription(
            Float32,
            str(self.get_parameter("live_width_topic").value),
            self._width_callback,
            10,
        )
        self.create_subscription(
            Float32,
            str(self.get_parameter("live_confidence_topic").value),
            self._confidence_callback,
            10,
        )
        self.create_timer(0.5, self._timer_callback)
        self._publish("waiting for a synchronized live slot estimate")
        self.get_logger().info(
            "READ-ONLY static-slot check started. It cannot plan or command robot motion."
        )
        self.get_logger().info(
            f"RViz MarkerArray: {self.get_parameter('marker_topic').value}"
        )

    def _declare_parameters(self):
        self.declare_parameter("base_frame", "link_base")
        self.declare_parameter("static_slot_translation_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter(
            "static_slot_quaternion_xyzw", [0.0, 0.0, 0.0, 1.0]
        )
        self.declare_parameter("static_slot_width_m", 0.0)
        self.declare_parameter("static_slot_transform_status", "unconfigured")
        self.declare_parameter("slot_depth_m", 0.20)
        self.declare_parameter("visual_slot_height_m", 0.25)
        self.declare_parameter("minimum_confidence", 0.60)
        self.declare_parameter("maximum_translation_error_m", 0.010)
        self.declare_parameter("maximum_rotation_error_deg", 5.0)
        self.declare_parameter("maximum_width_error_m", 0.005)
        self.declare_parameter("required_matching_samples", 30)
        self.declare_parameter("pair_max_age_s", 0.10)
        self.declare_parameter("stream_max_age_s", 1.0)
        self.declare_parameter("message_max_age_s", 0.50)
        self.declare_parameter("tf_lookup_timeout_s", 0.05)
        self.declare_parameter("output_dir", "/tmp/bookshelf_static_slot_check")
        self.declare_parameter("live_pose_topic", "/slot_detector/slot_pose")
        self.declare_parameter("live_width_topic", "/slot_detector/slot_width")
        self.declare_parameter("live_confidence_topic", "/slot_detector/confidence")
        self.declare_parameter(
            "marker_topic", "/bookshelf_environment/slot_markers"
        )
        self.declare_parameter(
            "static_pose_topic", "/bookshelf_environment/static_slot_pose"
        )
        self.declare_parameter(
            "live_pose_base_topic", "/bookshelf_environment/live_slot_pose_base"
        )
        self.declare_parameter(
            "passed_topic", "/bookshelf_environment/static_slot_check_passed"
        )
        self.declare_parameter(
            "status_topic", "/bookshelf_environment/static_slot_check_status"
        )

    def _vector_parameter(self, name: str, length: int) -> np.ndarray:
        values = np.asarray(self.get_parameter(name).value, dtype=np.float64)
        if values.shape != (length,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain {length} finite values")
        return values

    def _pose_callback(self, message: PoseStamped):
        self.latest_pose = message
        self.latest_pose_arrival = time.monotonic()

    def _width_callback(self, message: Float32):
        self.latest_width = float(message.data)
        self.latest_width_arrival = time.monotonic()

    def _confidence_callback(self, message: Float32):
        self.latest_confidence = float(message.data)
        self.latest_confidence_arrival = time.monotonic()
        self._evaluate_latest()

    def _timer_callback(self):
        now = time.monotonic()
        maximum_age = float(self.get_parameter("stream_max_age_s").value)
        arrivals = (
            self.latest_pose_arrival,
            self.latest_width_arrival,
            self.latest_confidence_arrival,
        )
        if any(value is None for value in arrivals) or any(
            now - value > maximum_age for value in arrivals
        ):
            self.gate.reset()
            self.latest_comparison = None
            self.latest_live_transform = None
            self._publish("live slot estimate is missing or stale")
        else:
            self._publish(self.latest_reason)

    def _evaluate_latest(self):
        now = time.monotonic()
        maximum_pair_age = float(self.get_parameter("pair_max_age_s").value)
        arrivals = (self.latest_pose_arrival, self.latest_width_arrival)
        if self.latest_pose is None or self.latest_width is None or any(
            value is None or now - value > maximum_pair_age for value in arrivals
        ):
            self.gate.reset()
            self.latest_comparison = None
            self.latest_live_transform = None
            self._publish("waiting for synchronized pose, width, and confidence")
            return
        if not self.latest_pose.header.frame_id:
            self._fail("live slot pose has an empty frame_id")
            return

        stamp_ns = (
            int(self.latest_pose.header.stamp.sec) * 1_000_000_000
            + int(self.latest_pose.header.stamp.nanosec)
        )
        message_max_age = float(self.get_parameter("message_max_age_s").value)
        if stamp_ns > 0 and message_max_age > 0.0:
            age_s = (self.get_clock().now().nanoseconds - stamp_ns) * 1.0e-9
            if age_s < -0.05 or age_s > message_max_age:
                self._fail(f"live slot source timestamp age is {age_s:.3f} s")
                return

        try:
            if self.latest_pose.header.frame_id == self.base_frame:
                transform_base_source = np.eye(4, dtype=np.float64)
            else:
                timeout = float(self.get_parameter("tf_lookup_timeout_s").value)
                tf_message = self.tf_buffer.lookup_transform(
                    self.base_frame,
                    self.latest_pose.header.frame_id,
                    Time(),
                    timeout=Duration(seconds=max(timeout, 0.0)),
                )
                transform_base_source = _transform_message_to_matrix(tf_message)
            transform_base_live = transform_base_source @ _pose_to_transform(
                self.latest_pose.pose
            )
            comparison = compare_slot_measurement(
                self.reference_transform,
                transform_base_live,
                reference_width_m=self.reference_width_m,
                measured_width_m=self.latest_width,
                confidence=self.latest_confidence,
                tolerances=self.tolerances,
            )
        except Exception as error:
            self._fail(f"live comparison unavailable: {error}")
            return

        self.latest_live_transform = transform_base_live
        self.latest_comparison = comparison
        self.gate.update(comparison["matches"])
        live_pose = PoseStamped()
        live_pose.header.frame_id = self.base_frame
        live_pose.header.stamp = self.get_clock().now().to_msg()
        live_pose.pose = _transform_to_pose(transform_base_live)
        self.live_pose_publisher.publish(live_pose)
        reason = None
        if not comparison["matches"]:
            reason = "live slot disagrees: " + ", ".join(
                comparison["failed_checks"]
            )
        self._publish(reason)

    def _fail(self, reason: str):
        self.gate.reset()
        self.latest_comparison = None
        self.latest_live_transform = None
        self._publish(reason)

    def _status(self, reason=None) -> dict:
        status = {
            "schema_version": 1,
            "generated_at": datetime.now().astimezone().isoformat(),
            "hardware_commanded": False,
            "saved_static_slot_modified": False,
            "human_approval_required": True,
            "check_passed": self.gate.passed,
            "matching_samples": self.gate.matching_samples,
            "required_matching_samples": self.gate.required_matches,
            "reason": reason,
            "base_frame": self.base_frame,
            "static_slot": {
                "translation_xyz": self.reference_transform[:3, 3].tolist(),
                "quaternion_xyzw": matrix_to_quaternion_xyzw(
                    self.reference_transform[:3, :3]
                ).tolist(),
                "width_m": self.reference_width_m,
                "transform_status": self.reference_status,
            },
            "tolerances": {
                "minimum_confidence": self.tolerances.minimum_confidence,
                "maximum_translation_error_m": (
                    self.tolerances.maximum_translation_error_m
                ),
                "maximum_rotation_error_deg": (
                    self.tolerances.maximum_rotation_error_deg
                ),
                "maximum_width_error_m": self.tolerances.maximum_width_error_m,
            },
            "latest_comparison": self.latest_comparison,
            "limitations": [
                "Agreement is a repeatability check, not absolute pose ground truth.",
                "A human must visually compare the RViz slot with the physical shelf.",
                "This process never updates the configured static slot pose.",
            ],
        }
        if self.latest_live_transform is not None:
            status["live_slot_base"] = {
                "translation_xyz": self.latest_live_transform[:3, 3].tolist(),
                "quaternion_xyzw": matrix_to_quaternion_xyzw(
                    self.latest_live_transform[:3, :3]
                ).tolist(),
            }
        return status

    def _publish(self, reason=None):
        self.latest_reason = reason
        now = self.get_clock().now().to_msg()
        static_pose = PoseStamped()
        static_pose.header.frame_id = self.base_frame
        static_pose.header.stamp = now
        static_pose.pose = _transform_to_pose(self.reference_transform)
        self.static_pose_publisher.publish(static_pose)
        self.passed_publisher.publish(Bool(data=self.gate.passed))
        status = self._status(reason)
        self.status_publisher.publish(String(data=json.dumps(status, sort_keys=True)))
        self.marker_publisher.publish(self._markers(now, reason))

        report_state = (
            self.gate.passed,
            reason,
            self.gate.matching_samples,
        )
        if (
            self.last_report_state is None
            or report_state[:2] != self.last_report_state[:2]
        ):
            self._write_report(status)
        self.last_report_state = report_state

    def _markers(self, stamp, reason=None) -> MarkerArray:
        markers = []
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        markers.append(delete_all)

        volume_transform = self.reference_transform @ make_transform(
            [0.5 * self.slot_depth_m, 0.0, 0.0]
        )
        volume = self._marker("configured_static_slot", 0, Marker.CUBE, stamp)
        volume.pose = _transform_to_pose(volume_transform)
        volume.scale.x = self.slot_depth_m
        volume.scale.y = self.reference_width_m
        volume.scale.z = self.visual_slot_height_m
        volume.color.r, volume.color.g, volume.color.b, volume.color.a = (
            0.05,
            0.75,
            1.0,
            0.16,
        )
        markers.append(volume)

        outline = self._slot_outline(
            namespace="configured_static_slot",
            marker_id=1,
            stamp=stamp,
            transform=self.reference_transform,
            width=self.reference_width_m,
            color=(0.05, 0.75, 1.0, 1.0),
        )
        markers.append(outline)

        axis = self._marker("configured_static_slot", 2, Marker.ARROW, stamp)
        axis.pose = _transform_to_pose(self.reference_transform)
        axis.points = [
            Point(x=0.0, y=0.0, z=0.0),
            Point(x=min(self.slot_depth_m, 0.10), y=0.0, z=0.0),
        ]
        axis.scale.x, axis.scale.y, axis.scale.z = 0.006, 0.012, 0.018
        axis.color.r, axis.color.g, axis.color.b, axis.color.a = 0.0, 0.35, 1.0, 1.0
        markers.append(axis)

        if self.latest_live_transform is not None and self.latest_comparison is not None:
            matching = bool(self.latest_comparison["matches"])
            color = (0.1, 1.0, 0.2, 1.0) if matching else (1.0, 0.1, 0.05, 1.0)
            markers.append(
                self._slot_outline(
                    namespace="live_slot_check",
                    marker_id=0,
                    stamp=stamp,
                    transform=self.latest_live_transform,
                    width=float(self.latest_comparison["measured_width_m"]),
                    color=color,
                )
            )
            deviation = self._marker(
                "live_slot_check", 1, Marker.LINE_LIST, stamp
            )
            deviation.points = [
                Point(
                    x=float(self.reference_transform[0, 3]),
                    y=float(self.reference_transform[1, 3]),
                    z=float(self.reference_transform[2, 3]),
                ),
                Point(
                    x=float(self.latest_live_transform[0, 3]),
                    y=float(self.latest_live_transform[1, 3]),
                    z=float(self.latest_live_transform[2, 3]),
                ),
            ]
            deviation.scale.x = 0.006
            (
                deviation.color.r,
                deviation.color.g,
                deviation.color.b,
                deviation.color.a,
            ) = color
            markers.append(deviation)

        label_transform = self.reference_transform @ make_transform(
            [0.0, 0.0, 0.5 * self.visual_slot_height_m + 0.04]
        )
        label = self._marker("configured_static_slot", 3, Marker.TEXT_VIEW_FACING, stamp)
        label.pose.position.x, label.pose.position.y, label.pose.position.z = (
            float(value) for value in label_transform[:3, 3]
        )
        label.pose.orientation.w = 1.0
        label.scale.z = 0.025
        label.color.r, label.color.g, label.color.b, label.color.a = 1.0, 1.0, 1.0, 1.0
        state = "PASS" if self.gate.passed else "CHECKING"
        if reason and not self.gate.passed:
            state = "NO MATCH"
        label.text = (
            f"STATIC SLOT (cyan) | LIVE (green/red) | {state} "
            f"{self.gate.matching_samples}/{self.gate.required_matches}"
        )
        if self.latest_comparison is not None:
            comparison = self.latest_comparison
            label.text += (
                f" | dt={comparison['translation_error_m'] * 1000.0:.1f} mm"
                f" dr={comparison['rotation_error_deg']:.1f} deg"
                f" dw={comparison['width_error_m'] * 1000.0:.1f} mm"
                f" conf={comparison['confidence']:.2f}"
            )
        markers.append(label)
        return MarkerArray(markers=markers)

    def _marker(self, namespace: str, marker_id: int, marker_type: int, stamp) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.base_frame
        marker.header.stamp = stamp
        marker.ns = namespace
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        return marker

    def _slot_outline(self, *, namespace, marker_id, stamp, transform, width, color):
        marker = self._marker(namespace, marker_id, Marker.LINE_LIST, stamp)
        marker.pose = _transform_to_pose(transform)
        marker.scale.x = 0.004
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        half_width = 0.5 * width
        half_height = 0.5 * self.visual_slot_height_m
        corners = [
            (x, y, z)
            for x in (0.0, self.slot_depth_m)
            for y in (-half_width, half_width)
            for z in (-half_height, half_height)
        ]
        edges = (
            (0, 1), (0, 2), (1, 3), (2, 3),
            (4, 5), (4, 6), (5, 7), (6, 7),
            (0, 4), (1, 5), (2, 6), (3, 7),
        )
        for start, end in edges:
            marker.points.append(Point(x=corners[start][0], y=corners[start][1], z=corners[start][2]))
            marker.points.append(Point(x=corners[end][0], y=corners[end][1], z=corners[end][2]))
        return marker

    def _write_report(self, status):
        output_dir = Path(str(self.get_parameter("output_dir").value)).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "static_slot_environment_check.json"
        report_path.write_text(
            json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def main(args=None):
    rclpy.init(args=args)
    node = StaticSlotEnvironmentCheckNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
