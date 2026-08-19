#!/usr/bin/env python3
"""Audit frozen-slot plus live-marker observations during offline Bag C replay."""

from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
import time

from geometry_msgs.msg import Point, PoseStamped
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from .stationary_shadow_replay import StationaryShadowReplayAccumulator


def _pose_values(message: PoseStamped):
    pose = message.pose
    return (
        [pose.position.x, pose.position.y, pose.position.z],
        [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ],
    )


class StationaryShadowReplayAuditNode(Node):
    """Subscriber-only audit that never exposes a robot command interface."""

    def __init__(self):
        super().__init__("stationary_shadow_replay_audit")
        self._declare_parameters()
        self.accumulator = StationaryShadowReplayAccumulator(
            minimum_valid_samples=int(
                self.get_parameter("minimum_valid_samples").value
            ),
            maximum_book_translation_jump_m=float(
                self.get_parameter("maximum_book_translation_jump_m").value
            ),
            maximum_book_rotation_jump_deg=float(
                self.get_parameter("maximum_book_rotation_jump_deg").value
            ),
        )
        self.latest_book_pose = None
        self.latest_slot_pose = None
        self.report_written = False
        self.marker_publisher = self.create_publisher(
            MarkerArray,
            str(self.get_parameter("visualization_topic").value),
            10,
        )
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("book_pose_topic").value),
            self._book_pose_callback,
            10,
        )
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("slot_pose_topic").value),
            self._slot_pose_callback,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("adapter_debug_topic").value),
            self._adapter_debug_callback,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("policy_debug_topic").value),
            self._policy_debug_callback,
            10,
        )
        self.get_logger().info(
            "Stationary shadow replay audit started: frozen slot plus live marker book."
        )
        self.get_logger().info(
            "This node only subscribes, publishes diagnostics, and writes files; "
            "it has no action, IK, trajectory, controller, gripper, or command client."
        )

    def _declare_parameters(self):
        self.declare_parameter(
            "adapter_debug_topic", "/bookshelf_policy/adapter_debug"
        )
        self.declare_parameter(
            "policy_debug_topic", "/bookshelf_shadow/policy_debug"
        )
        self.declare_parameter(
            "book_pose_topic", "/bookshelf_policy/book_pose_base"
        )
        self.declare_parameter(
            "slot_pose_topic", "/bookshelf_policy/slot_pose_base"
        )
        self.declare_parameter(
            "visualization_topic",
            "/bookshelf_shadow/stationary_replay_markers",
        )
        self.declare_parameter("output_dir", "/tmp/bookshelf_stationary_shadow")
        self.declare_parameter("candidate_id", "unknown")
        self.declare_parameter("minimum_valid_samples", 30)
        self.declare_parameter("pair_max_age_s", 0.25)
        self.declare_parameter("maximum_book_translation_jump_m", 0.010)
        self.declare_parameter("maximum_book_rotation_jump_deg", 5.0)
        self.declare_parameter("slot_visualization_depth_m", 0.20)
        self.declare_parameter("book_height_m", 0.236)

    def _book_pose_callback(self, message):
        self.latest_book_pose = (message, time.monotonic())

    def _slot_pose_callback(self, message):
        self.latest_slot_pose = (message, time.monotonic())

    def _recent_pose(self, value, label):
        if value is None:
            raise ValueError(f"waiting for {label} pose")
        message, arrival = value
        maximum_age = float(self.get_parameter("pair_max_age_s").value)
        if maximum_age > 0.0 and time.monotonic() - arrival > maximum_age:
            raise ValueError(f"{label} pose is stale")
        expected_frame = "link_base"
        if message.header.frame_id != expected_frame:
            raise ValueError(
                f"{label} pose frame is {message.header.frame_id!r}, "
                f"expected {expected_frame!r}"
            )
        return message

    def _adapter_debug_callback(self, message):
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError as error:
            self.accumulator.add_invalid(f"adapter debug JSON failed: {error}")
            return
        if not bool(payload.get("valid", False)):
            self.accumulator.add_invalid(payload.get("reason", "unspecified"))
            return
        try:
            book_pose = self._recent_pose(self.latest_book_pose, "book")
            slot_pose = self._recent_pose(self.latest_slot_pose, "slot")
        except ValueError as error:
            self.accumulator.add_invalid(str(error))
            return
        book_position, book_quaternion = _pose_values(book_pose)
        slot_position, slot_quaternion = _pose_values(slot_pose)
        accepted = self.accumulator.add_adapter_sample(
            payload,
            book_position=book_position,
            book_quaternion=book_quaternion,
            slot_position=slot_position,
            slot_quaternion=slot_quaternion,
        )
        if accepted:
            self._publish_markers(slot_pose, float(payload["slot_width_m"]))

    def _policy_debug_callback(self, message):
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            return
        self.accumulator.add_policy_debug(payload)

    def _publish_markers(self, slot_pose, slot_width):
        line = Marker()
        line.header = slot_pose.header
        line.ns = "stationary_shadow_frozen_slot"
        line.id = 0
        line.type = Marker.LINE_LIST
        line.action = Marker.ADD
        line.pose = slot_pose.pose
        line.scale.x = 0.004
        line.color.r = 0.1
        line.color.g = 1.0
        line.color.b = 0.2
        line.color.a = 1.0
        half_width = 0.5 * float(slot_width)
        half_height = 0.5 * float(self.get_parameter("book_height_m").value)
        corners = [
            Point(x=0.0, y=-half_width, z=-half_height),
            Point(x=0.0, y=half_width, z=-half_height),
            Point(x=0.0, y=half_width, z=half_height),
            Point(x=0.0, y=-half_width, z=half_height),
        ]
        for first, second in ((0, 1), (1, 2), (2, 3), (3, 0)):
            line.points.extend([corners[first], corners[second]])

        arrow = Marker()
        arrow.header = slot_pose.header
        arrow.ns = "stationary_shadow_insertion_axis"
        arrow.id = 1
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        arrow.pose = slot_pose.pose
        depth = float(self.get_parameter("slot_visualization_depth_m").value)
        arrow.points = [
            Point(x=-0.5 * depth, y=0.0, z=0.0),
            Point(x=0.5 * depth, y=0.0, z=0.0),
        ]
        arrow.scale.x = 0.008
        arrow.scale.y = 0.016
        arrow.scale.z = 0.025
        arrow.color.r = 0.1
        arrow.color.g = 1.0
        arrow.color.b = 0.2
        arrow.color.a = 0.9
        self.marker_publisher.publish(MarkerArray(markers=[line, arrow]))

    def write_report(self):
        if self.report_written:
            return
        self.report_written = True
        output_dir = Path(str(self.get_parameter("output_dir").value)).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = self.accumulator.summary()
        report = {
            "schema_version": 1,
            "kind": "bookshelf_stationary_shadow_replay_audit",
            "generated_at": datetime.now().astimezone().isoformat(),
            "candidate_id": str(self.get_parameter("candidate_id").value),
            "passed": bool(summary["passed"]),
            "reason": (
                None
                if summary["passed"]
                else "; ".join(summary["failure_reasons"])
            ),
            "observation_pipeline": summary,
            "interpretation": {
                "slot": "frozen View A RGB-D estimate",
                "book": "continuous marker-derived semantic book frame",
                "policy_activation": (
                    "diagnostic only; Bag C may be outside the local insertion envelope"
                ),
            },
            "safety": {
                "shadow_only": True,
                "plan_requested": False,
                "execution_authorized": False,
                "hardware_commanded": False,
                "active_configuration_modified": False,
                "candidate_selected": False,
            },
        }
        report_path = output_dir / "stationary_shadow_replay_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if self.accumulator.rows:
            csv_path = output_dir / "stationary_shadow_observations.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=list(self.accumulator.rows[0])
                )
                writer.writeheader()
                writer.writerows(self.accumulator.rows)
        self.get_logger().info(f"Stationary shadow report written to {report_path}")
        self.get_logger().info(
            f"OBSERVATION REPLAY: passed={summary['passed']}, "
            f"valid={summary['valid_samples']}, "
            f"marker_sources={summary['book_pose_sources']}, "
            f"activation_ready="
            f"{summary['policy_diagnostics']['activation_ready_messages']}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = StationaryShadowReplayAuditNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.write_report()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
