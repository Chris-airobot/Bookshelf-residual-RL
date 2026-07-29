#!/usr/bin/env python3
"""Record read-only slot-detector stability metrics during rosbag replay."""

import csv
from datetime import datetime
import json
from pathlib import Path
import time

from geometry_msgs.msg import PoseStamped
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

from .offline_validation import SlotAuditAccumulator


class SlotDetectionAuditNode(Node):
    def __init__(self):
        super().__init__("slot_detection_audit")
        self.declare_parameter("confidence_topic", "/slot_detector/confidence")
        self.declare_parameter("width_topic", "/slot_detector/slot_width")
        self.declare_parameter("pose_topic", "/slot_detector/slot_pose")
        self.declare_parameter("minimum_confidence", 0.60)
        self.declare_parameter("target_samples", 1200)
        self.declare_parameter("pair_max_age_s", 0.10)
        self.declare_parameter("output_dir", "/tmp/bookshelf_slot_audit")

        self.accumulator = SlotAuditAccumulator(
            minimum_confidence=float(self.get_parameter("minimum_confidence").value)
        )
        self.latest_width = None
        self.latest_width_time = None
        self.latest_pose = None
        self.latest_pose_time = None
        self.completed = False

        self.create_subscription(
            Float32,
            str(self.get_parameter("width_topic").value),
            self._width_callback,
            10,
        )
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("pose_topic").value),
            self._pose_callback,
            10,
        )
        self.create_subscription(
            Float32,
            str(self.get_parameter("confidence_topic").value),
            self._confidence_callback,
            10,
        )
        self.get_logger().info(
            "Slot detector audit started. It only subscribes and writes CSV/JSON diagnostics."
        )

    def _width_callback(self, message):
        self.latest_width = float(message.data)
        self.latest_width_time = time.monotonic()

    def _pose_callback(self, message):
        self.latest_pose = message
        self.latest_pose_time = time.monotonic()

    def _confidence_callback(self, message):
        now = time.monotonic()
        maximum_age = float(self.get_parameter("pair_max_age_s").value)
        paired = (
            self.latest_width is not None
            and self.latest_pose is not None
            and self.latest_width_time is not None
            and self.latest_pose_time is not None
            and now - self.latest_width_time <= maximum_age
            and now - self.latest_pose_time <= maximum_age
        )
        if paired:
            pose = self.latest_pose.pose
            self.accumulator.add(
                message.data,
                width=self.latest_width,
                position=[pose.position.x, pose.position.y, pose.position.z],
                quaternion_xyzw=[
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ],
            )
        else:
            self.accumulator.add(message.data)

        target = max(int(self.get_parameter("target_samples").value), 1)
        count = len(self.accumulator.rows)
        if count % 100 == 0:
            summary = self.accumulator.summary()
            self.get_logger().info(
                f"samples={count}, valid={summary['valid_samples']}, "
                f"valid_fraction={summary['valid_fraction']:.3f}"
            )
        if count >= target and not self.completed:
            self.completed = True
            self._write_report()

    def _write_report(self):
        output_dir = Path(str(self.get_parameter("output_dir").value)).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        rows_path = output_dir / "slot_detection_samples.csv"
        with rows_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.accumulator.rows[0]))
            writer.writeheader()
            writer.writerows(self.accumulator.rows)

        summary = {
            "schema_version": 1,
            "generated_at": datetime.now().astimezone().isoformat(),
            "hardware_commanded": False,
            "ground_truth_available": False,
            "detector_stability": self.accumulator.summary(),
            "limitations": [
                "This report measures repeatability, not absolute slot-pose accuracy.",
                "No physical slot-width or pose ground truth is available in the rosbag.",
            ],
        }
        summary_path = output_dir / "slot_detection_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self.get_logger().info(f"Slot audit report written to {summary_path}")
        self.get_logger().info("Target reached; press Ctrl-C after rosbag playback is stopped.")


def main(args=None):
    rclpy.init(args=args)
    node = SlotDetectionAuditNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node.accumulator.rows and not node.completed:
            node._write_report()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
