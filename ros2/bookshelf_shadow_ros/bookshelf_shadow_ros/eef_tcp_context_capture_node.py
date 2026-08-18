#!/usr/bin/env python3
"""Capture the fixed link_eef to link_tcp transform during bag replay."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
import tf2_ros

from .calibrated_preinsert_target_math import transform_to_dict
from .policy_observation_math import make_transform
from .stationary_capture_bundle import summarize_fixed_transform


def _transform_message_to_matrix(message) -> np.ndarray:
    value = message.transform
    return make_transform(
        [value.translation.x, value.translation.y, value.translation.z],
        [value.rotation.x, value.rotation.y, value.rotation.z, value.rotation.w],
    )


class EefTcpContextCaptureNode(Node):
    """Read TF only and write an unapproved fixed-transform context report."""

    def __init__(self):
        super().__init__("eef_tcp_context_capture")
        self.declare_parameter("parent_frame", "link_eef")
        self.declare_parameter("child_frame", "link_tcp")
        self.declare_parameter("output_path", "/tmp/eef_tcp_context.json")
        self.declare_parameter("required_samples", 10)
        self.declare_parameter("maximum_translation_spread_m", 0.0005)
        self.declare_parameter("maximum_rotation_spread_deg", 0.25)
        self.declare_parameter("tf_lookup_timeout_s", 0.10)

        self.parent_frame = str(self.get_parameter("parent_frame").value).strip()
        self.child_frame = str(self.get_parameter("child_frame").value).strip()
        if not self.parent_frame or not self.child_frame:
            raise ValueError("parent_frame and child_frame must not be empty")
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=60.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.transforms = []
        self.completed = False
        self.report_written = False
        self.create_timer(0.20, self._timer_callback)
        self.get_logger().info(
            "READ-ONLY EEF/TCP context capture started; no motion interface exists."
        )

    def _timer_callback(self):
        if self.completed:
            return
        try:
            message = self.tf_buffer.lookup_transform(
                self.parent_frame,
                self.child_frame,
                Time(),
                timeout=Duration(
                    seconds=float(
                        self.get_parameter("tf_lookup_timeout_s").value
                    )
                ),
            )
        except Exception:
            return
        self.transforms.append(_transform_message_to_matrix(message))
        required = max(int(self.get_parameter("required_samples").value), 1)
        if len(self.transforms) >= required:
            self._write_report()
            self.completed = True

    def _write_report(self, reason: str | None = None):
        if self.report_written:
            return
        self.report_written = True
        required = max(int(self.get_parameter("required_samples").value), 1)
        valid = False
        summary = None
        if len(self.transforms) >= required:
            try:
                summary = summarize_fixed_transform(
                    self.transforms,
                    minimum_samples=required,
                    maximum_translation_spread_m=float(
                        self.get_parameter(
                            "maximum_translation_spread_m"
                        ).value
                    ),
                    maximum_rotation_spread_deg=float(
                        self.get_parameter(
                            "maximum_rotation_spread_deg"
                        ).value
                    ),
                )
                valid = True
            except ValueError as error:
                reason = str(error)
        elif reason is None:
            reason = f"capture stopped at {len(self.transforms)}/{required} samples"

        report = {
            "schema_version": 1,
            "kind": "bookshelf_eef_tcp_context_capture",
            "generated_at": datetime.now().astimezone().isoformat(),
            "valid": valid,
            "reason": None if valid else reason,
            "parent_frame": self.parent_frame,
            "child_frame": self.child_frame,
            "sample_count": len(self.transforms),
            "required_samples": required,
            "read_only": True,
            "active_configuration_modified": False,
            "execution_authorized": False,
            "hardware_commanded": False,
        }
        if summary is not None:
            report["transform_eef_tcp"] = transform_to_dict(summary["transform"])
            report["sample_spread"] = {
                "translation_m": summary["translation_spread_m"],
                "rotation_deg": summary["rotation_spread_deg"],
            }
            report["tolerances"] = {
                "maximum_translation_spread_m": summary[
                    "maximum_translation_spread_m"
                ],
                "maximum_rotation_spread_deg": summary[
                    "maximum_rotation_spread_deg"
                ],
            }

        output_path = Path(
            str(self.get_parameter("output_path").value)
        ).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if valid:
            self.get_logger().info(f"EEF/TCP context written to {output_path}")
        else:
            self.get_logger().error(f"EEF/TCP context is invalid: {reason}")

    def write_incomplete_report(self):
        if not self.report_written:
            self._write_report("capture interrupted before a valid context was ready")


def main(args=None):
    rclpy.init(args=args)
    node = EefTcpContextCaptureNode()
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
