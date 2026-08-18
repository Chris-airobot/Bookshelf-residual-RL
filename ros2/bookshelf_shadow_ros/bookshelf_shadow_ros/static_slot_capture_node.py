#!/usr/bin/env python3
"""Capture a robust static slot candidate without changing robot state."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import subprocess
import time

from geometry_msgs.msg import Point, Pose, PoseStamped, TransformStamped
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Bool, Float32, String
import tf2_ros
from visualization_msgs.msg import Marker, MarkerArray

from .policy_observation_math import make_transform, matrix_to_quaternion_xyzw
from .static_slot_capture import (
    StaticSlotCaptureAccumulator,
    StaticSlotSample,
    serializable_capture_result,
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


def _git_revision(repository_path: str) -> dict | None:
    """Read branch and commit once without invoking the LFS-sensitive status filter."""

    repository_path = str(repository_path or "").strip()
    if not repository_path:
        return None
    repository = Path(repository_path).expanduser().resolve()
    result = {"repository": str(repository)}
    commands = {
        "commit": ["git", "-C", str(repository), "rev-parse", "HEAD"],
        "branch": ["git", "-C", str(repository), "branch", "--show-current"],
    }
    for key, command in commands.items():
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            result[key] = completed.stdout.strip() if completed.returncode == 0 else None
            if completed.returncode != 0:
                result[f"{key}_error"] = completed.stderr.strip()
        except (OSError, subprocess.TimeoutExpired) as error:
            result[key] = None
            result[f"{key}_error"] = str(error)
    return result


class StaticSlotCaptureNode(Node):
    """Subscriber-only capture node for the unobstructed shelf scan stage."""

    def __init__(self):
        super().__init__("static_slot_capture")
        self._declare_parameters()
        self.base_frame = str(self.get_parameter("base_frame").value).strip()
        if not self.base_frame:
            raise ValueError("base_frame must not be empty")
        self.target_samples = int(self.get_parameter("target_samples").value)
        self.finalization_retry_interval_samples = int(
            self.get_parameter("finalization_retry_interval_samples").value
        )
        self.use_latest_tf = bool(self.get_parameter("use_latest_tf").value)
        self.minimum_confidence = float(
            self.get_parameter("minimum_confidence").value
        )
        self.minimum_slot_width_m = float(
            self.get_parameter("minimum_slot_width_m").value
        )
        self.maximum_slot_width_m = float(
            self.get_parameter("maximum_slot_width_m").value
        )
        if self.target_samples < 1:
            raise ValueError("target_samples must be at least one")
        if self.finalization_retry_interval_samples < 1:
            raise ValueError(
                "finalization_retry_interval_samples must be at least one"
            )
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be in [0, 1]")
        if not 0.0 < self.minimum_slot_width_m < self.maximum_slot_width_m:
            raise ValueError("slot width limits are invalid")

        self.accumulator = StaticSlotCaptureAccumulator(
            minimum_samples=int(self.get_parameter("minimum_samples").value),
            minimum_inlier_fraction=float(
                self.get_parameter("minimum_inlier_fraction").value
            ),
            maximum_translation_deviation_m=float(
                self.get_parameter("maximum_translation_deviation_m").value
            ),
            maximum_rotation_deviation_deg=float(
                self.get_parameter("maximum_rotation_deviation_deg").value
            ),
            maximum_width_deviation_m=float(
                self.get_parameter("maximum_width_deviation_m").value
            ),
        )
        if self.target_samples < self.accumulator.minimum_samples:
            raise ValueError("target_samples must be at least minimum_samples")

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=60.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.ready_publisher = self.create_publisher(
            Bool, str(self.get_parameter("ready_topic").value), latched_qos
        )
        self.status_publisher = self.create_publisher(
            String, str(self.get_parameter("status_topic").value), latched_qos
        )
        self.pose_publisher = self.create_publisher(
            PoseStamped,
            str(self.get_parameter("candidate_pose_topic").value),
            latched_qos,
        )
        self.marker_publisher = self.create_publisher(
            MarkerArray,
            str(self.get_parameter("marker_topic").value),
            latched_qos,
        )

        self.latest_pose = None
        self.latest_pose_arrival = None
        self.latest_width = None
        self.latest_width_arrival = None
        self.latest_confidence = None
        self.latest_confidence_arrival = None
        self.camera_info = None
        self.last_stamp_ns = None
        self.last_transform_base_source = None
        self.last_source_frame = None
        self.completed = False
        self.last_finalize_attempt_sample_count = 0
        self.last_filter_error = None
        self.report = None
        self.git_provenance = _git_revision(
            str(self.get_parameter("repository_path").value)
        )
        self.counters = {
            "pose_messages": 0,
            "accepted_samples": 0,
            "duplicate_stamp": 0,
            "unsynchronized": 0,
            "low_confidence": 0,
            "invalid_width": 0,
            "invalid_pose": 0,
            "stale_message": 0,
            "tf_unavailable": 0,
            "finalization_attempts": 0,
        }

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
        self.create_subscription(
            CameraInfo,
            str(self.get_parameter("camera_info_topic").value),
            self._camera_info_callback,
            10,
        )
        self.create_timer(0.5, self._timer_callback)
        self._publish_status("collecting static slot samples")
        self.get_logger().info(
            "READ-ONLY static-slot capture started. It cannot plan or command motion."
        )
        self.get_logger().info(
            f"Collecting {self.target_samples} accepted samples in {self.base_frame}."
        )
        if self.use_latest_tf:
            self.get_logger().warning(
                "Using latest-available TF for an explicitly stationary replay. "
                "Do not enable this mode for a moving robot."
            )

    def _declare_parameters(self):
        self.declare_parameter("base_frame", "link_base")
        self.declare_parameter("target_samples", 120)
        self.declare_parameter("finalization_retry_interval_samples", 30)
        self.declare_parameter("minimum_samples", 60)
        self.declare_parameter("minimum_inlier_fraction", 0.80)
        self.declare_parameter("minimum_confidence", 0.60)
        self.declare_parameter("minimum_slot_width_m", 0.020)
        self.declare_parameter("maximum_slot_width_m", 0.090)
        self.declare_parameter("maximum_translation_deviation_m", 0.010)
        self.declare_parameter("maximum_rotation_deviation_deg", 5.0)
        self.declare_parameter("maximum_width_deviation_m", 0.005)
        self.declare_parameter("pair_max_age_s", 0.10)
        self.declare_parameter("message_max_age_s", 0.50)
        self.declare_parameter("tf_lookup_timeout_s", 0.05)
        self.declare_parameter("use_latest_tf", False)
        self.declare_parameter("slot_depth_m", 0.20)
        self.declare_parameter("visual_slot_height_m", 0.25)
        self.declare_parameter("repository_path", "")
        self.declare_parameter("output_dir", "/tmp/bookshelf_static_slot_capture")
        self.declare_parameter("live_pose_topic", "/slot_detector/slot_pose")
        self.declare_parameter("live_width_topic", "/slot_detector/slot_width")
        self.declare_parameter("live_confidence_topic", "/slot_detector/confidence")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter(
            "ready_topic", "/bookshelf_environment/static_slot_capture_ready"
        )
        self.declare_parameter(
            "status_topic", "/bookshelf_environment/static_slot_capture_status"
        )
        self.declare_parameter(
            "candidate_pose_topic",
            "/bookshelf_environment/static_slot_candidate_pose",
        )
        self.declare_parameter(
            "marker_topic", "/bookshelf_environment/static_slot_candidate_markers"
        )

    def _pose_callback(self, message: PoseStamped):
        self.latest_pose = message
        self.latest_pose_arrival = time.monotonic()
        self.counters["pose_messages"] += 1

    def _width_callback(self, message: Float32):
        self.latest_width = float(message.data)
        self.latest_width_arrival = time.monotonic()

    def _confidence_callback(self, message: Float32):
        self.latest_confidence = float(message.data)
        self.latest_confidence_arrival = time.monotonic()
        self._capture_latest()

    def _camera_info_callback(self, message: CameraInfo):
        if self.camera_info is None:
            self.camera_info = message

    def _timer_callback(self):
        if not self.completed:
            self._publish_status("collecting static slot samples")

    def _capture_latest(self):
        if self.completed:
            return
        now = time.monotonic()
        arrivals = (self.latest_pose_arrival, self.latest_width_arrival)
        pair_max_age = float(self.get_parameter("pair_max_age_s").value)
        if self.latest_pose is None or self.latest_width is None or any(
            value is None or now - value > pair_max_age for value in arrivals
        ):
            self.counters["unsynchronized"] += 1
            return
        confidence = float(self.latest_confidence)
        if not np.isfinite(confidence) or confidence < self.minimum_confidence:
            self.counters["low_confidence"] += 1
            return
        width = float(self.latest_width)
        if (
            not np.isfinite(width)
            or not self.minimum_slot_width_m <= width <= self.maximum_slot_width_m
        ):
            self.counters["invalid_width"] += 1
            return
        if not self.latest_pose.header.frame_id:
            self.counters["invalid_pose"] += 1
            return

        stamp_ns = Time.from_msg(self.latest_pose.header.stamp).nanoseconds
        if stamp_ns == self.last_stamp_ns:
            self.counters["duplicate_stamp"] += 1
            return
        message_max_age = float(self.get_parameter("message_max_age_s").value)
        if stamp_ns > 0 and message_max_age > 0.0:
            age_s = (self.get_clock().now().nanoseconds - stamp_ns) * 1.0e-9
            if age_s < -0.05 or age_s > message_max_age:
                self.counters["stale_message"] += 1
                return

        source_frame = self.latest_pose.header.frame_id
        try:
            if source_frame == self.base_frame:
                transform_base_source = np.eye(4, dtype=np.float64)
            else:
                timeout = float(self.get_parameter("tf_lookup_timeout_s").value)
                lookup_time = (
                    Time()
                    if self.use_latest_tf or stamp_ns <= 0
                    else Time.from_msg(self.latest_pose.header.stamp)
                )
                tf_message = self.tf_buffer.lookup_transform(
                    self.base_frame,
                    source_frame,
                    lookup_time,
                    timeout=Duration(seconds=max(timeout, 0.0)),
                )
                transform_base_source = _transform_message_to_matrix(tf_message)
            transform_base_slot = transform_base_source @ _pose_to_transform(
                self.latest_pose.pose
            )
            self.accumulator.add(
                StaticSlotSample(
                    stamp_ns=int(stamp_ns),
                    transform_base_slot=transform_base_slot,
                    width_m=width,
                    confidence=confidence,
                )
            )
        except Exception as error:
            self.counters["tf_unavailable"] += 1
            self._publish_status(f"sample rejected: {error}")
            return

        self.last_stamp_ns = int(stamp_ns)
        self.last_transform_base_source = transform_base_source
        self.last_source_frame = source_frame
        self.counters["accepted_samples"] += 1
        sample_count = len(self.accumulator.samples)
        if (
            sample_count >= self.target_samples
            and (
                self.last_finalize_attempt_sample_count == 0
                or sample_count - self.last_finalize_attempt_sample_count
                >= self.finalization_retry_interval_samples
            )
        ):
            self._finalize(allow_retry=True)

    def _finalize(self, *, allow_retry: bool = False):
        self.last_finalize_attempt_sample_count = len(self.accumulator.samples)
        self.counters["finalization_attempts"] += 1
        try:
            result = self.accumulator.result()
            candidate = serializable_capture_result(result)
            report = self._base_report(valid=True, reason=None)
            report["candidate"] = candidate
            report["statistics"] = {
                key: value
                for key, value in result.items()
                if key
                not in {
                    "transform_base_slot",
                    "translation_xyz",
                    "quaternion_xyzw",
                    "width_m",
                    "confidence",
                }
            }
        except ValueError as error:
            self.last_filter_error = str(error)
            if allow_retry:
                self.get_logger().warning(
                    "Robust filtering is not yet consistent; continuing the "
                    f"stationary capture: {error}"
                )
                self._publish_status(
                    "collecting additional samples after robust filtering: "
                    f"{error}"
                )
                return False
            report = self._base_report(valid=False, reason=str(error))
        except Exception as error:
            report = self._base_report(valid=False, reason=str(error))
        self.report = report
        self.completed = True
        self._write_report(report)
        self._publish_status(report.get("reason"))
        if report["valid"]:
            self.get_logger().info(
                "Static slot candidate is ready for RViz review and explicit promotion."
            )
        else:
            self.get_logger().error(f"Static slot capture failed: {report['reason']}")
        return bool(report["valid"])

    def _base_report(self, *, valid: bool, reason: str | None) -> dict:
        report = {
            "schema_version": 1,
            "kind": "bookshelf_static_slot_capture_candidate",
            "generated_at": datetime.now().astimezone().isoformat(),
            "hardware_commanded": False,
            "active_configuration_modified": False,
            "human_approval_required": True,
            "valid": bool(valid),
            "reason": reason,
            "base_frame": self.base_frame,
            "source_frame": self.last_source_frame,
            "counters": dict(self.counters),
            "target_samples": self.target_samples,
            "finalization_retry_interval_samples": (
                self.finalization_retry_interval_samples
            ),
            "tf_lookup_mode": (
                "latest_available_stationary"
                if self.use_latest_tf
                else "message_timestamp"
            ),
            "source_topics": {
                "pose": str(self.get_parameter("live_pose_topic").value),
                "width": str(self.get_parameter("live_width_topic").value),
                "confidence": str(
                    self.get_parameter("live_confidence_topic").value
                ),
                "camera_info": str(self.get_parameter("camera_info_topic").value),
            },
            "limitations": [
                "This is a repeatability estimate, not independent absolute ground truth.",
                "A human must compare the candidate marker with the physical slot in RViz.",
                "This node never changes active ROS or policy configuration.",
            ],
        }
        if self.last_transform_base_source is not None:
            report["last_transform_base_source"] = {
                "translation_xyz": self.last_transform_base_source[:3, 3].tolist(),
                "quaternion_xyzw": matrix_to_quaternion_xyzw(
                    self.last_transform_base_source[:3, :3]
                ).tolist(),
            }
        if self.camera_info is not None:
            report["camera_info"] = {
                "frame_id": self.camera_info.header.frame_id,
                "width": int(self.camera_info.width),
                "height": int(self.camera_info.height),
                "distortion_model": self.camera_info.distortion_model,
                "d": list(self.camera_info.d),
                "k": list(self.camera_info.k),
                "p": list(self.camera_info.p),
            }
        if self.git_provenance is not None:
            report["git"] = self.git_provenance
        if self.last_filter_error is not None:
            report["last_filter_error"] = self.last_filter_error
        return report

    def _publish_status(self, reason: str | None):
        ready = bool(self.report and self.report.get("valid"))
        self.ready_publisher.publish(Bool(data=ready))
        status = self.report or self._base_report(valid=False, reason=reason)
        status["accepted_samples"] = len(self.accumulator.samples)
        self.status_publisher.publish(String(data=json.dumps(status, sort_keys=True)))
        if ready:
            transform = make_transform(
                self.report["candidate"]["translation_xyz"],
                self.report["candidate"]["quaternion_xyzw"],
            )
            pose = PoseStamped()
            pose.header.frame_id = self.base_frame
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose = _transform_to_pose(transform)
            self.pose_publisher.publish(pose)
            self.marker_publisher.publish(self._candidate_markers(transform, pose.header.stamp))

    def _candidate_markers(self, transform: np.ndarray, stamp) -> MarkerArray:
        width = float(self.report["candidate"]["width_m"])
        depth = float(self.get_parameter("slot_depth_m").value)
        height = float(self.get_parameter("visual_slot_height_m").value)
        markers = []
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        markers.append(delete_all)

        arrow = Marker()
        arrow.header.frame_id = self.base_frame
        arrow.header.stamp = stamp
        arrow.ns = "static_slot_candidate"
        arrow.id = 0
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        arrow.pose = _transform_to_pose(transform)
        arrow.points = [Point(x=0.0, y=0.0, z=0.0), Point(x=min(depth, 0.10), y=0.0, z=0.0)]
        arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.006, 0.012, 0.018
        arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = 0.1, 1.0, 0.2, 1.0
        markers.append(arrow)

        outline = Marker()
        outline.header.frame_id = self.base_frame
        outline.header.stamp = stamp
        outline.ns = "static_slot_candidate"
        outline.id = 1
        outline.type = Marker.LINE_LIST
        outline.action = Marker.ADD
        outline.pose = _transform_to_pose(transform)
        outline.scale.x = 0.004
        outline.color.r, outline.color.g, outline.color.b, outline.color.a = 0.1, 1.0, 0.2, 1.0
        half_width = 0.5 * width
        half_height = 0.5 * height
        corners = [
            (x, y, z)
            for x in (0.0, depth)
            for y in (-half_width, half_width)
            for z in (-half_height, half_height)
        ]
        edges = (
            (0, 1), (0, 2), (1, 3), (2, 3),
            (4, 5), (4, 6), (5, 7), (6, 7),
            (0, 4), (1, 5), (2, 6), (3, 7),
        )
        for start, end in edges:
            outline.points.append(
                Point(
                    x=corners[start][0],
                    y=corners[start][1],
                    z=corners[start][2],
                )
            )
            outline.points.append(
                Point(
                    x=corners[end][0],
                    y=corners[end][1],
                    z=corners[end][2],
                )
            )
        markers.append(outline)
        return MarkerArray(markers=markers)

    def _write_report(self, report: dict):
        output_dir = Path(str(self.get_parameter("output_dir").value)).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "static_slot_capture_candidate.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def write_incomplete_report(self):
        if self.completed:
            return
        if len(self.accumulator.samples) >= self.target_samples:
            self._finalize(allow_retry=False)
            return
        report = self._base_report(
            valid=False,
            reason=(
                "capture stopped before target sample count: "
                f"{len(self.accumulator.samples)}/{self.target_samples}"
            ),
        )
        self._write_report(report)


def main(args=None):
    rclpy.init(args=args)
    node = StaticSlotCaptureNode()
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
