#!/usr/bin/env python3
"""Fail-closed comparison of live and configured held-book geometry."""

from __future__ import annotations

from collections import deque
from datetime import datetime
import hashlib
import json
from pathlib import Path

from geometry_msgs.msg import Pose, PoseStamped
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Bool, Header, String
import tf2_ros

from .held_book_pose_check import (
    compare_transforms,
    load_configured_transform,
    mean_transform,
    transform_spread,
)
from .policy_tool_control_math import make_transform, matrix_to_quaternion_xyzw


def _transform_message_to_matrix(message) -> np.ndarray:
    translation = message.transform.translation
    rotation = message.transform.rotation
    return make_transform(
        [translation.x, translation.y, translation.z],
        [rotation.x, rotation.y, rotation.z, rotation.w],
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


def _transform_payload(transform: np.ndarray) -> dict:
    quaternion = matrix_to_quaternion_xyzw(transform[:3, :3])
    return {
        "translation_xyz_m": [float(value) for value in transform[:3, 3]],
        "quaternion_xyzw": [float(value) for value in quaternion],
    }


class HeldBookPoseCheckNode(Node):
    """Compare a live marker-derived book pose with the frozen MoveIt box pose."""

    def __init__(self):
        super().__init__("held_book_pose_check")
        self._declare_parameters()
        self.tcp_frame = str(self.get_parameter("tcp_frame").value)
        self.detected_book_frame = str(
            self.get_parameter("detected_book_frame").value
        )
        self.required_samples = max(
            int(self.get_parameter("required_stable_samples").value), 1
        )
        self.samples = deque(maxlen=self.required_samples)
        self.last_stamp_ns = None
        self.passed_latched = False
        self.last_verified_at = None
        self.last_reason = "waiting for live marker-derived book pose"
        self.latest_status = None

        configured_path = Path(
            str(self.get_parameter("scene_config_path").value)
        ).expanduser()
        self.scene_config_path = configured_path.resolve()
        self.scene_config_sha256 = None
        self.configured_transform = None
        try:
            if not self.scene_config_path.is_file():
                raise ValueError(f"scene config does not exist: {self.scene_config_path}")
            self.configured_transform = load_configured_transform(
                self.scene_config_path
            )
            self.scene_config_sha256 = hashlib.sha256(
                self.scene_config_path.read_bytes()
            ).hexdigest()
        except (OSError, ValueError) as error:
            self.last_reason = f"invalid scene configuration: {error}"

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.passed_publisher = self.create_publisher(
            Bool, str(self.get_parameter("passed_topic").value), 10
        )
        self.status_publisher = self.create_publisher(
            String, str(self.get_parameter("status_topic").value), 10
        )
        self.live_pose_publisher = self.create_publisher(
            PoseStamped, str(self.get_parameter("live_pose_topic").value), 10
        )
        self.configured_pose_publisher = self.create_publisher(
            PoseStamped,
            str(self.get_parameter("configured_pose_topic").value),
            10,
        )

        output_dir = Path(str(self.get_parameter("output_dir").value)).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        self.report_path = output_dir / "held_book_pose_check.json"
        rate = max(float(self.get_parameter("publish_rate_hz").value), 0.2)
        self.create_timer(1.0 / rate, self._timer_callback)
        self.get_logger().warning(
            "Held-book pose check is READ-ONLY. It cannot update MoveIt geometry "
            "or command the robot."
        )

    def _declare_parameters(self):
        self.declare_parameter("scene_config_path", "")
        self.declare_parameter("tcp_frame", "link_tcp")
        self.declare_parameter("detected_book_frame", "calibration_detected_book")
        self.declare_parameter("required_stable_samples", 30)
        self.declare_parameter("latch_success_through_marker_occlusion", True)
        self.declare_parameter("maximum_sample_translation_spread_m", 0.003)
        self.declare_parameter("maximum_sample_rotation_spread_deg", 2.0)
        self.declare_parameter("maximum_config_translation_error_m", 0.010)
        self.declare_parameter("maximum_config_rotation_error_deg", 5.0)
        self.declare_parameter("tf_lookup_timeout_s", 0.05)
        self.declare_parameter("tf_max_age_s", 0.50)
        self.declare_parameter("publish_rate_hz", 5.0)
        self.declare_parameter("output_dir", "/tmp/bookshelf_held_book_pose_check")
        self.declare_parameter(
            "passed_topic", "/bookshelf_scene/held_book_pose_check_passed"
        )
        self.declare_parameter(
            "status_topic", "/bookshelf_scene/held_book_pose_check_status"
        )
        self.declare_parameter(
            "live_pose_topic", "/bookshelf_scene/live_held_book_pose_tcp"
        )
        self.declare_parameter(
            "configured_pose_topic", "/bookshelf_scene/configured_held_book_pose_tcp"
        )

    def _lookup_live_transform(self):
        try:
            message = self.tf_buffer.lookup_transform(
                self.tcp_frame,
                self.detected_book_frame,
                Time(),
                timeout=Duration(
                    seconds=float(self.get_parameter("tf_lookup_timeout_s").value)
                ),
            )
        except Exception as error:
            return None, None, f"live TF unavailable: {error}"
        stamp_ns = Time.from_msg(message.header.stamp).nanoseconds
        maximum_age_s = float(self.get_parameter("tf_max_age_s").value)
        if maximum_age_s > 0.0:
            age_s = (self.get_clock().now().nanoseconds - stamp_ns) * 1.0e-9
            if age_s > maximum_age_s:
                return None, stamp_ns, f"live TF is stale: age={age_s:.3f} s"
        return _transform_message_to_matrix(message), stamp_ns, None

    def _timer_callback(self):
        passed = False
        status = self._base_status()
        if self.configured_transform is None:
            status["reason"] = self.last_reason
            self._publish(passed, status)
            return

        live_transform, stamp_ns, error = self._lookup_live_transform()
        if error:
            if self.passed_latched and bool(
                self.get_parameter("latch_success_through_marker_occlusion").value
            ):
                passed = True
                self.last_reason = (
                    "using latched stable agreement while live marker is unavailable"
                )
                status.update(
                    {
                        "reason": self.last_reason,
                        "live_detection_available": False,
                        "live_detection_error": error,
                    }
                )
            else:
                self.samples.clear()
                self.last_reason = error
                status["reason"] = error
                status["live_detection_available"] = False
            self._publish(passed, status)
            return

        stamp = self.get_clock().now().to_msg()
        self.live_pose_publisher.publish(
            PoseStamped(
                header=self._header(stamp), pose=_transform_to_pose(live_transform)
            )
        )
        if stamp_ns != self.last_stamp_ns:
            self.samples.append(live_transform)
            self.last_stamp_ns = stamp_ns

        status["accepted_unique_samples"] = len(self.samples)
        status["live_detection_available"] = True
        status["latest_live_transform_tcp_book"] = _transform_payload(live_transform)
        if len(self.samples) < self.required_samples:
            self.last_reason = (
                f"collecting stable live book poses: {len(self.samples)}/"
                f"{self.required_samples}"
            )
            status["reason"] = self.last_reason
            self._publish(self.passed_latched, status)
            return

        candidate = mean_transform(self.samples)
        spread = transform_spread(self.samples, candidate)
        comparison = compare_transforms(self.configured_transform, candidate)
        stable = (
            spread.translation_error_m
            <= float(
                self.get_parameter("maximum_sample_translation_spread_m").value
            )
            and spread.rotation_error_deg
            <= float(
                self.get_parameter("maximum_sample_rotation_spread_deg").value
            )
        )
        matches = (
            stable
            and comparison.translation_error_m
            <= float(self.get_parameter("maximum_config_translation_error_m").value)
            and comparison.rotation_error_deg
            <= float(self.get_parameter("maximum_config_rotation_error_deg").value)
        )
        if stable and matches:
            self.passed_latched = True
            self.last_verified_at = datetime.now().astimezone().isoformat()
        elif stable and not matches:
            self.passed_latched = False
        passed = bool(self.passed_latched)
        if not stable:
            self.last_reason = (
                "live marker-derived book pose is not stable"
                if not self.passed_latched
                else "using latched agreement while current live samples are unstable"
            )
        elif not matches:
            self.last_reason = "live book pose disagrees with configured MoveIt box"
        else:
            self.last_reason = "live book pose agrees with configured MoveIt box"
        status.update(
            {
                "reason": self.last_reason,
                "live_candidate_stable": stable,
                "live_candidate_transform_tcp_book": _transform_payload(candidate),
                "sample_spread": {
                    "translation_m": spread.translation_error_m,
                    "rotation_deg": spread.rotation_error_deg,
                },
                "configured_difference": {
                    "translation_m": comparison.translation_error_m,
                    "rotation_deg": comparison.rotation_error_deg,
                },
            }
        )
        self._publish(passed, status)

    def _base_status(self):
        return {
            "schema_version": 1,
            "generated_at": datetime.now().astimezone().isoformat(),
            "check_passed": False,
            "tcp_frame": self.tcp_frame,
            "detected_book_frame": self.detected_book_frame,
            "required_stable_samples": self.required_samples,
            "accepted_unique_samples": len(self.samples),
            "check_latched": self.passed_latched,
            "last_verified_at": self.last_verified_at,
            "scene_config": {
                "path": str(self.scene_config_path),
                "sha256": self.scene_config_sha256,
            },
            "configured_transform_tcp_book": (
                _transform_payload(self.configured_transform)
                if self.configured_transform is not None
                else None
            ),
            "tolerances": {
                "maximum_sample_translation_spread_m": float(
                    self.get_parameter("maximum_sample_translation_spread_m").value
                ),
                "maximum_sample_rotation_spread_deg": float(
                    self.get_parameter("maximum_sample_rotation_spread_deg").value
                ),
                "maximum_config_translation_error_m": float(
                    self.get_parameter("maximum_config_translation_error_m").value
                ),
                "maximum_config_rotation_error_deg": float(
                    self.get_parameter("maximum_config_rotation_error_deg").value
                ),
            },
            "human_approval_required": True,
            "active_configuration_modified": False,
            "execution_authorized": False,
            "hardware_commanded": False,
        }

    def _header(self, stamp):
        return Header(stamp=stamp, frame_id=self.tcp_frame)

    def _publish(self, passed: bool, status: dict):
        status["check_passed"] = bool(passed)
        status["check_latched"] = self.passed_latched
        status["last_verified_at"] = self.last_verified_at
        self.latest_status = status
        self.passed_publisher.publish(Bool(data=bool(passed)))
        self.status_publisher.publish(String(data=json.dumps(status, sort_keys=True)))
        if self.configured_transform is not None:
            stamp = self.get_clock().now().to_msg()
            self.configured_pose_publisher.publish(
                PoseStamped(
                    header=self._header(stamp),
                    pose=_transform_to_pose(self.configured_transform),
                )
            )
        temporary = self.report_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
        temporary.replace(self.report_path)


def main(args=None):
    rclpy.init(args=args)
    node = HeldBookPoseCheckNode()
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
