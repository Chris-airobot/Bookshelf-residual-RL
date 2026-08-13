#!/usr/bin/env python3
"""Publish and report a calibrated pre-insertion target without commanding motion."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from geometry_msgs.msg import PoseStamped, TransformStamped
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Bool, Float32MultiArray, String
import tf2_ros

from .calibrated_preinsert_target_math import (
    PreinsertTargetSpec,
    calibration_sensitivity,
    compare_current_eef_to_target,
    compute_calibrated_preinsert_target,
    compute_preserved_tcp_orientation_preinsert_target,
    labelled_values,
    transform_to_dict,
)
from .policy_observation_math import (
    ObservationScales,
    invert_transform,
    make_transform,
    matrix_to_quaternion_xyzw,
)


def _stamp_nanoseconds(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _transform_message_to_matrix(message: TransformStamped) -> np.ndarray:
    value = message.transform
    return make_transform(
        [value.translation.x, value.translation.y, value.translation.z],
        [value.rotation.x, value.rotation.y, value.rotation.z, value.rotation.w],
    )


def _matrix_to_pose_stamped(transform, frame_id: str, stamp) -> PoseStamped:
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


class CalibratedPreinsertTargetNode(Node):
    """Read-only geometric target calculator and TF comparison reporter."""

    def __init__(self):
        super().__init__("calibrated_preinsert_target")
        self._declare_parameters()

        self.base_frame = str(self.get_parameter("base_frame").value)
        self.ee_frame = str(self.get_parameter("ee_frame").value)
        self.tcp_frame = str(self.get_parameter("tcp_frame").value)
        self.target_orientation_mode = str(
            self.get_parameter("target_orientation_mode").value
        ).strip()
        if self.target_orientation_mode not in (
            "preserve_current_tcp",
            "book_aligned",
        ):
            raise ValueError(
                "target_orientation_mode must be preserve_current_tcp or "
                "book_aligned."
            )
        maximum_orientation_error = float(
            self.get_parameter("maximum_preserved_book_orientation_error_deg").value
        )
        if not np.isfinite(maximum_orientation_error) or maximum_orientation_error <= 0.0:
            raise ValueError(
                "maximum_preserved_book_orientation_error_deg must be positive."
            )
        self.slot_status = str(
            self.get_parameter("static_slot_transform_status").value
        ).strip()
        self.eef_book_status = str(
            self.get_parameter("eef_book_transform_status").value
        ).strip()
        self.policy_tool_status = str(
            self.get_parameter("policy_tool_transform_status").value
        ).strip()
        for name, value in (
            ("static_slot_transform_status", self.slot_status),
            ("eef_book_transform_status", self.eef_book_status),
            ("policy_tool_transform_status", self.policy_tool_status),
        ):
            if value.lower() in ("", "unknown", "unconfigured"):
                raise ValueError(f"{name} must explicitly describe its provenance.")

        self.transform_base_slot = make_transform(
            self._vector_parameter("static_slot_translation_xyz", 3),
            self._vector_parameter("static_slot_quaternion_xyzw", 4),
        )
        self.transform_eef_book = make_transform(
            self._vector_parameter("eef_book_translation_xyz", 3),
            self._vector_parameter("eef_book_quaternion_xyzw", 4),
        )
        self.transform_eef_policy_tool = make_transform(
            self._vector_parameter("eef_policy_tool_translation_xyz", 3),
            self._vector_parameter("eef_policy_tool_quaternion_xyzw", 4),
        )
        self.policy_tool_verified = self.policy_tool_status.lower().startswith(
            ("verified_", "validated_")
        )
        self.spec = self._target_spec()
        self.book_aligned_target = compute_calibrated_preinsert_target(
            self.transform_base_slot,
            self.transform_eef_book,
            transform_eef_policy_tool=self.transform_eef_policy_tool,
            spec=self.spec,
        )
        self.target = (
            self.book_aligned_target
            if self.target_orientation_mode == "book_aligned"
            else None
        )
        self.preserved_orientation_diagnostics = None
        self.geometric_target_valid = False
        self.target_valid = False
        self._refresh_target_validity()
        self.sensitivity = calibration_sensitivity(
            self.transform_base_slot,
            self.transform_eef_book,
            transform_eef_policy_tool=self.transform_eef_policy_tool,
            spec=self.spec,
            samples=int(self.get_parameter("sensitivity_samples").value),
            translation_uncertainty_m=float(
                self.get_parameter("translation_uncertainty_m").value
            ),
            rotation_uncertainty_deg=float(
                self.get_parameter("rotation_uncertainty_deg").value
            ),
            seed=int(self.get_parameter("sensitivity_seed").value),
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.current_comparison = None
        self.current_tcp_transform = None
        self.current_lookup_error = "waiting for current base-to-EEF/TCP transforms"
        self.last_status_key = None
        self.last_report_write_ns = None

        self.valid_publisher = self.create_publisher(
            Bool, str(self.get_parameter("valid_topic").value), 10
        )
        self.target_book_publisher = self.create_publisher(
            PoseStamped, str(self.get_parameter("target_book_pose_topic").value), 10
        )
        self.target_eef_publisher = self.create_publisher(
            PoseStamped, str(self.get_parameter("target_eef_pose_topic").value), 10
        )
        self.target_tcp_publisher = self.create_publisher(
            PoseStamped, str(self.get_parameter("target_tcp_pose_topic").value), 10
        )
        self.target_policy_tool_publisher = self.create_publisher(
            PoseStamped,
            str(self.get_parameter("target_policy_tool_pose_topic").value),
            10,
        )
        self.current_book_publisher = self.create_publisher(
            PoseStamped, str(self.get_parameter("current_book_pose_topic").value), 10
        )
        self.current_eef_publisher = self.create_publisher(
            PoseStamped, str(self.get_parameter("current_eef_pose_topic").value), 10
        )
        self.current_tcp_publisher = self.create_publisher(
            PoseStamped, str(self.get_parameter("current_tcp_pose_topic").value), 10
        )
        self.target_raw_publisher = self.create_publisher(
            Float32MultiArray,
            str(self.get_parameter("target_raw_metrics_topic").value),
            10,
        )
        self.target_observation_publisher = self.create_publisher(
            Float32MultiArray,
            str(self.get_parameter("target_observation_topic").value),
            10,
        )
        self.debug_publisher = self.create_publisher(
            String, str(self.get_parameter("debug_topic").value), 10
        )

        output_dir = Path(str(self.get_parameter("output_dir").value)).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        self.report_path = output_dir / "calibrated_preinsert_target_report.json"
        self._write_report()

        frequency = max(float(self.get_parameter("publish_rate_hz").value), 1.0)
        self.timer = self.create_timer(1.0 / frequency, self._timer_callback)

        self.get_logger().info(
            "Calibrated pre-insertion target calculator started in READ-ONLY mode."
        )
        self.get_logger().info(
            "No IK, planning, trajectory, controller, gripper, or robot-command "
            "interface is created."
        )
        self.get_logger().info(f"Static target report: {self.report_path}")
        self.get_logger().info(
            f"Target orientation mode={self.target_orientation_mode}; "
            "preserve_current_tcp latches the first fresh live TCP orientation."
        )
        if not self.policy_tool_verified:
            self.get_logger().warning(
                "PPO observation is fail-closed because the policy-tool "
                f"transform is not verified: {self.policy_tool_status}"
            )
        if self.target is None:
            self.get_logger().warning(
                "Target remains invalid until fresh link_eef and link_tcp TF are available."
            )
        elif self.target.unexpected_clipped_labels:
            self.get_logger().warning(
                "Target has unexpected clipped observations: "
                + ", ".join(self.target.unexpected_clipped_labels)
            )
        else:
            self.get_logger().info(
                "Only expected pre-insertion depth channels are clipped: "
                + ", ".join(self.target.expected_clipped_labels)
            )

    def _declare_parameters(self):
        self.declare_parameter("base_frame", "link_base")
        self.declare_parameter("ee_frame", "link_eef")
        self.declare_parameter("tcp_frame", "link_tcp")
        self.declare_parameter("target_orientation_mode", "preserve_current_tcp")
        self.declare_parameter("maximum_preserved_book_orientation_error_deg", 15.0)
        self.declare_parameter("static_slot_translation_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter(
            "static_slot_quaternion_xyzw", [0.0, 0.0, 0.0, 1.0]
        )
        self.declare_parameter("static_slot_width_m", 0.0)
        self.declare_parameter("static_slot_confidence", 0.0)
        self.declare_parameter("static_slot_transform_status", "unconfigured")
        self.declare_parameter("eef_book_translation_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter("eef_book_quaternion_xyzw", [0.0, 0.0, 0.0, 1.0])
        self.declare_parameter("eef_book_transform_status", "unconfigured")
        self.declare_parameter("eef_policy_tool_translation_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter(
            "eef_policy_tool_quaternion_xyzw", [0.0, 0.0, 0.0, 1.0]
        )
        self.declare_parameter("policy_tool_transform_status", "unconfigured")

        self.declare_parameter("book_size_xyz", [0.156, 0.034, 0.236])
        self.declare_parameter("slot_depth_m", 0.20)
        self.declare_parameter("preinsert_standoff_m", 0.030)
        self.declare_parameter("preinsert_vertical_offset_m", 0.006)
        self.declare_parameter("target_gripper_open", 0.0)
        self.declare_parameter("rear_to_mouth_obs_scale", 0.08)
        self.declare_parameter("front_to_back_obs_scale", 0.08)
        self.declare_parameter("lat_err_obs_scale", 0.05)
        self.declare_parameter("z_err_obs_scale", 0.05)
        self.declare_parameter("yaw_err_obs_scale_deg", 30.0)
        self.declare_parameter("tool_to_book_obs_scale", 0.25)

        self.declare_parameter("sensitivity_samples", 2000)
        self.declare_parameter("translation_uncertainty_m", 0.002)
        self.declare_parameter("rotation_uncertainty_deg", 2.0)
        self.declare_parameter("sensitivity_seed", 42)
        self.declare_parameter("publish_rate_hz", 5.0)
        self.declare_parameter("tf_lookup_timeout_s", 0.05)
        self.declare_parameter("tf_max_age_s", 0.50)
        self.declare_parameter("report_write_period_s", 1.0)
        self.declare_parameter("output_dir", "/tmp/bookshelf_calibrated_target")

        self.declare_parameter(
            "valid_topic", "/bookshelf_shadow/calibrated_target_valid"
        )
        self.declare_parameter(
            "target_book_pose_topic", "/bookshelf_shadow/target_book_pose"
        )
        self.declare_parameter(
            "target_eef_pose_topic", "/bookshelf_shadow/target_eef_pose"
        )
        self.declare_parameter(
            "target_tcp_pose_topic", "/bookshelf_shadow/target_tcp_pose"
        )
        self.declare_parameter(
            "target_policy_tool_pose_topic",
            "/bookshelf_shadow/target_policy_tool_pose",
        )
        self.declare_parameter(
            "current_book_pose_topic", "/bookshelf_shadow/current_book_pose"
        )
        self.declare_parameter(
            "current_eef_pose_topic", "/bookshelf_shadow/current_eef_pose"
        )
        self.declare_parameter(
            "current_tcp_pose_topic", "/bookshelf_shadow/current_tcp_pose"
        )
        self.declare_parameter(
            "target_raw_metrics_topic", "/bookshelf_shadow/target_raw_metrics"
        )
        self.declare_parameter(
            "target_observation_topic",
            "/bookshelf_shadow/target_observation_12d",
        )
        self.declare_parameter(
            "debug_topic", "/bookshelf_shadow/calibrated_target_debug"
        )

    def _target_spec(self) -> PreinsertTargetSpec:
        import math

        return PreinsertTargetSpec(
            book_size=tuple(self._vector_parameter("book_size_xyz", 3)),
            slot_depth=float(self.get_parameter("slot_depth_m").value),
            standoff=float(self.get_parameter("preinsert_standoff_m").value),
            vertical_offset=float(
                self.get_parameter("preinsert_vertical_offset_m").value
            ),
            gripper_open=float(self.get_parameter("target_gripper_open").value),
            observation_scales=ObservationScales(
                rear_to_mouth=float(
                    self.get_parameter("rear_to_mouth_obs_scale").value
                ),
                front_to_back=float(
                    self.get_parameter("front_to_back_obs_scale").value
                ),
                lateral=float(self.get_parameter("lat_err_obs_scale").value),
                vertical=float(self.get_parameter("z_err_obs_scale").value),
                yaw=math.radians(
                    float(self.get_parameter("yaw_err_obs_scale_deg").value)
                ),
                tool_to_book=float(
                    self.get_parameter("tool_to_book_obs_scale").value
                ),
            ),
        )

    def _vector_parameter(self, name: str, size: int) -> np.ndarray:
        value = np.asarray(self.get_parameter(name).value, dtype=np.float64)
        if value.shape != (size,) or not np.all(np.isfinite(value)):
            raise ValueError(f"Parameter {name} must contain {size} finite values.")
        return value

    def _refresh_target_validity(self):
        self.geometric_target_valid = bool(
            self.target is not None and not self.target.unexpected_clipped_labels
        )
        if self.preserved_orientation_diagnostics is not None:
            maximum = float(
                self.get_parameter(
                    "maximum_preserved_book_orientation_error_deg"
                ).value
            )
            self.geometric_target_valid = bool(
                self.geometric_target_valid
                and self.preserved_orientation_diagnostics[
                    "book_orientation_error_deg"
                ]
                <= maximum
            )
        self.target_valid = bool(
            self.geometric_target_valid and self.policy_tool_verified
        )

    def _lookup_current_frame(self, frame: str):
        timeout = max(float(self.get_parameter("tf_lookup_timeout_s").value), 0.0)
        try:
            message = self.tf_buffer.lookup_transform(
                self.base_frame,
                frame,
                Time(),
                timeout=Duration(seconds=timeout),
            )
        except Exception as error:
            return None, f"TF {self.base_frame} <- {frame} unavailable: {error}"

        maximum_age = float(self.get_parameter("tf_max_age_s").value)
        stamp_ns = _stamp_nanoseconds(message.header.stamp)
        if maximum_age > 0.0 and stamp_ns != 0:
            age = abs(int(self.get_clock().now().nanoseconds) - stamp_ns) * 1.0e-9
            if age > maximum_age:
                return None, f"TF {self.base_frame} <- {frame} is stale"
        return _transform_message_to_matrix(message), None

    def _latch_preserved_orientation_target(
        self, transform_base_eef, transform_base_tcp
    ):
        if self.target is not None:
            return
        self.target, self.preserved_orientation_diagnostics = (
            compute_preserved_tcp_orientation_preinsert_target(
                self.transform_base_slot,
                self.transform_eef_book,
                transform_base_eef,
                transform_base_tcp,
                transform_eef_policy_tool=self.transform_eef_policy_tool,
                spec=self.spec,
            )
        )
        self._refresh_target_validity()
        self.get_logger().warning(
            "Latched current link_tcp orientation for the read-only "
            "pre-insertion target. Restart this node to capture a new orientation."
        )
        error = self.preserved_orientation_diagnostics["book_orientation_error_deg"]
        maximum = float(
            self.get_parameter("maximum_preserved_book_orientation_error_deg").value
        )
        if error > maximum:
            self.get_logger().warning(
                f"Preserved TCP orientation makes the book differ from the slot "
                f"by {error:.3f} deg; limit is {maximum:.3f} deg. Target is invalid."
            )

    def _timer_callback(self):
        stamp = self.get_clock().now().to_msg()
        transform_base_eef, eef_error = self._lookup_current_frame(self.ee_frame)
        transform_base_tcp, tcp_error = self._lookup_current_frame(self.tcp_frame)
        errors = [value for value in (eef_error, tcp_error) if value]
        if errors:
            self.current_lookup_error = "; ".join(errors)
            self.valid_publisher.publish(Bool(data=False))
            self._publish_debug()
            self._log_status_once(
                f"invalid:{self.current_lookup_error}",
                self.current_lookup_error,
                warning=True,
            )
            return

        self.current_lookup_error = None
        self.current_tcp_transform = transform_base_tcp
        if self.target_orientation_mode == "preserve_current_tcp":
            self._latch_preserved_orientation_target(
                transform_base_eef, transform_base_tcp
            )

        self.valid_publisher.publish(Bool(data=self.target_valid))
        self.current_eef_publisher.publish(
            _matrix_to_pose_stamped(transform_base_eef, self.base_frame, stamp)
        )
        self.current_tcp_publisher.publish(
            _matrix_to_pose_stamped(transform_base_tcp, self.base_frame, stamp)
        )
        if self.target is None:
            self._publish_debug()
            return

        transform_eef_tcp = invert_transform(transform_base_eef) @ transform_base_tcp
        transform_base_tcp_target = (
            self.target.transform_base_eef_target @ transform_eef_tcp
        )
        self.target_book_publisher.publish(
            _matrix_to_pose_stamped(
                self.target.transform_base_book_target, self.base_frame, stamp
            )
        )
        self.target_eef_publisher.publish(
            _matrix_to_pose_stamped(
                self.target.transform_base_eef_target, self.base_frame, stamp
            )
        )
        self.target_tcp_publisher.publish(
            _matrix_to_pose_stamped(
                transform_base_tcp_target, self.base_frame, stamp
            )
        )
        self.target_policy_tool_publisher.publish(
            _matrix_to_pose_stamped(
                self.target.transform_base_policy_tool_target,
                self.base_frame,
                stamp,
            )
        )
        self.target_raw_publisher.publish(
            Float32MultiArray(data=self.target.raw_metrics.tolist())
        )
        self.target_observation_publisher.publish(
            Float32MultiArray(data=self.target.observation_12d.tolist())
        )

        self.current_comparison = compare_current_eef_to_target(
            transform_base_eef,
            self.transform_base_slot,
            self.transform_eef_book,
            self.target,
            transform_eef_policy_tool=self.transform_eef_policy_tool,
            spec=self.spec,
        )
        self.current_book_publisher.publish(
            _matrix_to_pose_stamped(
                self.current_comparison["transform_base_book_current"],
                self.base_frame,
                stamp,
            )
        )
        self._publish_debug()
        self._log_status_once(
            "comparison_available",
            "Current EEF comparison is available; results remain diagnostic only.",
        )
        self._maybe_write_report()

    def _publish_debug(self):
        payload = {
            "valid": self.target_valid,
            "geometric_target_valid": self.geometric_target_valid,
            "policy_observation_valid": self.target_valid,
            "target_orientation_mode": self.target_orientation_mode,
            "orientation_latched": self.target is not None,
            "policy_tool_transform_status": self.policy_tool_status,
            "shadow_only": True,
            "hardware_commanded": False,
            "ik_checked": False,
            "reachability_checked": False,
            "report_path": str(self.report_path),
            "target_unexpected_clipped_labels": (
                []
                if self.target is None
                else list(self.target.unexpected_clipped_labels)
            ),
            "current_comparison_available": self.current_comparison is not None,
            "current_lookup_error": self.current_lookup_error,
        }
        if self.preserved_orientation_diagnostics is not None:
            payload.update(
                {
                    "preserved_tcp_orientation_change_deg": round(
                        self.preserved_orientation_diagnostics[
                            "tcp_orientation_change_deg"
                        ],
                        7,
                    ),
                    "preserved_book_orientation_error_deg": round(
                        self.preserved_orientation_diagnostics[
                            "book_orientation_error_deg"
                        ],
                        5,
                    ),
                    "maximum_preserved_book_orientation_error_deg": float(
                        self.get_parameter(
                            "maximum_preserved_book_orientation_error_deg"
                        ).value
                    ),
                }
            )
        if self.current_comparison is not None:
            payload["target_minus_current_translation_norm_m"] = round(
                self.current_comparison[
                    "target_minus_current_translation_norm_m"
                ],
                7,
            )
            payload["target_minus_current_rotation_deg"] = round(
                self.current_comparison["target_minus_current_rotation_deg"], 5
            )
            payload["current_clipped_labels"] = list(
                self.current_comparison["clipped_labels"]
            )
        self.debug_publisher.publish(String(data=json.dumps(payload, sort_keys=True)))

    def _maybe_write_report(self):
        now_ns = int(self.get_clock().now().nanoseconds)
        period = max(float(self.get_parameter("report_write_period_s").value), 0.0)
        if (
            self.last_report_write_ns is not None
            and period > 0.0
            and (now_ns - self.last_report_write_ns) * 1.0e-9 < period
        ):
            return
        self._write_report()
        self.last_report_write_ns = now_ns

    def _write_report(self):
        target_payload = {
            "available": False,
            "reason": self.current_lookup_error,
        }
        if self.target is not None:
            target_payload = {
                "available": True,
                "orientation_mode": self.target_orientation_mode,
                "transform_slot_book": transform_to_dict(
                    self.target.transform_slot_book_target
                ),
                "transform_base_book": transform_to_dict(
                    self.target.transform_base_book_target
                ),
                "transform_base_eef": transform_to_dict(
                    self.target.transform_base_eef_target
                ),
                "transform_base_policy_tool": transform_to_dict(
                    self.target.transform_base_policy_tool_target
                ),
                "transform_slot_eef": transform_to_dict(
                    self.target.transform_slot_eef_target
                ),
                "transform_slot_policy_tool": transform_to_dict(
                    self.target.transform_slot_policy_tool_target
                ),
                "geometric_target_valid": self.geometric_target_valid,
                "policy_observation_valid": self.target_valid,
                "raw_metrics": labelled_values(self.target.raw_metrics),
                "observation_12d": labelled_values(
                    self.target.observation_12d
                ),
                "clipped_labels": list(self.target.clipped_labels),
                "expected_clipped_labels": list(
                    self.target.expected_clipped_labels
                ),
                "unexpected_clipped_labels": list(
                    self.target.unexpected_clipped_labels
                ),
                "expected_clip_explanation": (
                    f"At a {self.spec.standoff * 1000.0:.1f} mm pre-insertion "
                    "standoff, rear_to_mouth and front_to_back exceed the "
                    f"policy's {self.spec.observation_scales.rear_to_mouth * 1000.0:.1f} "
                    "mm depth observation scales and are expected to clip. "
                    "Other clipped channels indicate a configuration or geometry problem."
                ),
            }
        if self.preserved_orientation_diagnostics is not None:
            diagnostics = self.preserved_orientation_diagnostics
            target_payload["preserved_tcp_orientation"] = {
                "transform_base_tcp_current": transform_to_dict(
                    diagnostics["transform_base_tcp_current"]
                ),
                "transform_base_tcp_target": transform_to_dict(
                    diagnostics["transform_base_tcp_target"]
                ),
                "transform_eef_tcp": transform_to_dict(
                    diagnostics["transform_eef_tcp"]
                ),
                "transform_tcp_book": transform_to_dict(
                    diagnostics["transform_tcp_book"]
                ),
                "tcp_orientation_change_deg": diagnostics[
                    "tcp_orientation_change_deg"
                ],
                "book_orientation_error_deg": diagnostics[
                    "book_orientation_error_deg"
                ],
                "book_center_error_m": diagnostics["book_center_error_m"],
                "maximum_book_orientation_error_deg": float(
                    self.get_parameter(
                        "maximum_preserved_book_orientation_error_deg"
                    ).value
                ),
            }
        payload = {
            "schema_version": 2,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "safety": {
                "shadow_only": True,
                "hardware_commanded": False,
                "ik_checked": False,
                "reachability_checked": False,
                "collision_checked": False,
                "execution_authorized": False,
            },
            "frames": {
                "base": self.base_frame,
                "eef": self.ee_frame,
                "tcp": self.tcp_frame,
                "slot": "configured static slot frame",
                "book": "policy book frame: +X depth, +Y thickness, +Z up",
            },
            "provenance": {
                "static_slot_transform_status": self.slot_status,
                "eef_book_transform_status": self.eef_book_status,
                "policy_tool_transform_status": self.policy_tool_status,
            },
            "configuration": {
                "target_orientation_mode": self.target_orientation_mode,
                "maximum_preserved_book_orientation_error_deg": float(
                    self.get_parameter(
                        "maximum_preserved_book_orientation_error_deg"
                    ).value
                ),
                "book_size_xyz_m": [float(value) for value in self.spec.book_size],
                "slot_depth_m": self.spec.slot_depth,
                "static_slot_width_m": float(
                    self.get_parameter("static_slot_width_m").value
                ),
                "static_slot_confidence": float(
                    self.get_parameter("static_slot_confidence").value
                ),
                "preinsert_standoff_m": self.spec.standoff,
                "preinsert_vertical_offset_m": self.spec.vertical_offset,
                "target_gripper_open": self.spec.gripper_open,
            },
            "calibration": {
                "transform_base_slot": transform_to_dict(
                    self.transform_base_slot
                ),
                "transform_eef_book": transform_to_dict(self.transform_eef_book),
                "transform_eef_policy_tool": transform_to_dict(
                    self.transform_eef_policy_tool
                ),
            },
            "target": target_payload,
            "book_aligned_reference_calibration_sensitivity": self.sensitivity,
            "current_comparison": self._serializable_current_comparison(),
            "limitations": [
                "The static slot pose has no independent absolute ground truth.",
                "The EEF-to-book transform is valid only for the measured rigid grasp.",
                "Preserving TCP orientation does not independently prove book-slot alignment.",
                "This report does not prove IK reachability, collision freedom, or safe execution.",
                "No result in this report authorizes robot motion.",
            ],
        }
        temporary = self.report_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(self.report_path)

    def _serializable_current_comparison(self):
        if self.current_comparison is None:
            return {
                "available": False,
                "reason": self.current_lookup_error,
            }
        value = self.current_comparison
        return {
            "available": True,
            "transform_base_eef_current": transform_to_dict(
                value["transform_base_eef_current"]
            ),
            "transform_base_book_current": transform_to_dict(
                value["transform_base_book_current"]
            ),
            "transform_base_policy_tool_current": transform_to_dict(
                value["transform_base_policy_tool_current"]
            ),
            "transform_slot_book_current": transform_to_dict(
                value["transform_slot_book_current"]
            ),
            "transform_current_eef_to_target_eef": transform_to_dict(
                value["transform_current_eef_to_target_eef"]
            ),
            "target_minus_current_translation_base_m": [
                float(component)
                for component in value["target_minus_current_translation_base_m"]
            ],
            "target_minus_current_translation_norm_m": value[
                "target_minus_current_translation_norm_m"
            ],
            "target_minus_current_rotation_deg": value[
                "target_minus_current_rotation_deg"
            ],
            "raw_metrics": labelled_values(value["raw_metrics"]),
            "observation_12d": labelled_values(value["observation_12d"]),
            "clipped_labels": list(value["clipped_labels"]),
        }

    def _log_status_once(self, key: str, message: str, warning=False):
        if key == self.last_status_key:
            return
        self.last_status_key = key
        if warning:
            self.get_logger().warning(message)
        else:
            self.get_logger().info(message)


def main(args=None):
    rclpy.init(args=args)
    node = CalibratedPreinsertTargetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
