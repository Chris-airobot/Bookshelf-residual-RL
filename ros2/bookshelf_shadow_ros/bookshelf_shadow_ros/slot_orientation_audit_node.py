#!/usr/bin/env python3
"""Write a read-only live-versus-reference slot orientation audit."""

from datetime import datetime
import json
from pathlib import Path
import time

from geometry_msgs.msg import PoseStamped
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

from .slot_orientation_audit import SlotOrientationAuditAccumulator


def _quaternion(message):
    orientation = message.pose.orientation
    return [orientation.x, orientation.y, orientation.z, orientation.w]


class SlotOrientationAuditNode(Node):
    def __init__(self):
        super().__init__("slot_orientation_audit")
        self.declare_parameter(
            "live_pose_topic", "/bookshelf_environment/live_slot_pose_base"
        )
        self.declare_parameter(
            "reference_pose_topic", "/bookshelf_environment/static_slot_pose"
        )
        self.declare_parameter("confidence_topic", "/slot_detector/confidence")
        self.declare_parameter("minimum_confidence", 0.60)
        self.declare_parameter("stable_spread_p95_deg", 1.0)
        self.declare_parameter("meaningful_disagreement_deg", 2.0)
        self.declare_parameter("pair_max_age_s", 0.20)
        self.declare_parameter("target_samples", 700)
        self.declare_parameter("output_dir", "/tmp/bookshelf_slot_orientation_audit")

        self.accumulator = SlotOrientationAuditAccumulator(
            minimum_confidence=float(
                self.get_parameter("minimum_confidence").value
            ),
            stable_spread_p95_deg=float(
                self.get_parameter("stable_spread_p95_deg").value
            ),
            meaningful_disagreement_deg=float(
                self.get_parameter("meaningful_disagreement_deg").value
            ),
        )
        self.latest_reference = None
        self.latest_reference_time = None
        self.latest_confidence = None
        self.latest_confidence_time = None
        self.missing_pair_samples = 0
        self.completed = False

        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("reference_pose_topic").value),
            self._reference_callback,
            10,
        )
        self.create_subscription(
            Float32,
            str(self.get_parameter("confidence_topic").value),
            self._confidence_callback,
            10,
        )
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("live_pose_topic").value),
            self._live_callback,
            10,
        )
        self.get_logger().warning(
            "READ-ONLY slot orientation audit started. It has no planning, "
            "trajectory, controller, gripper, or robot-command interface."
        )

    def _reference_callback(self, message):
        self.latest_reference = message
        self.latest_reference_time = time.monotonic()

    def _confidence_callback(self, message):
        self.latest_confidence = float(message.data)
        self.latest_confidence_time = time.monotonic()

    def _live_callback(self, message):
        now = time.monotonic()
        maximum_age = float(self.get_parameter("pair_max_age_s").value)
        paired = (
            self.latest_reference is not None
            and self.latest_confidence is not None
            and self.latest_reference_time is not None
            and self.latest_confidence_time is not None
            and now - self.latest_reference_time <= maximum_age
            and now - self.latest_confidence_time <= maximum_age
        )
        if not paired:
            self.missing_pair_samples += 1
            return

        self.accumulator.add(
            _quaternion(message),
            _quaternion(self.latest_reference),
            self.latest_confidence,
        )
        target = max(int(self.get_parameter("target_samples").value), 1)
        if len(self.accumulator.rows) >= target and not self.completed:
            self.completed = True
            self._write_report()

    def _write_report(self):
        output_dir = Path(str(self.get_parameter("output_dir").value)).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "schema_version": 1,
            "kind": "bookshelf_slot_orientation_audit",
            "generated_at": datetime.now().astimezone().isoformat(),
            "orientation_audit": self.accumulator.summary(),
            "missing_pair_samples": self.missing_pair_samples,
            "safety": {
                "shadow_only": True,
                "plan_requested": False,
                "execution_authorized": False,
                "hardware_commanded": False,
            },
            "limitations": [
                "This audit separates temporal variation from a stable reference disagreement.",
                "It does not establish absolute physical slot orientation ground truth.",
                "Lighting and distance are not independently varied in this recording.",
            ],
        }
        path = output_dir / "slot_orientation_audit.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        summary = report["orientation_audit"]
        self.get_logger().info(
            f"Slot orientation audit written to {path}; "
            f"classification={summary['classification']}"
        )

    def write_partial_report(self):
        if self.accumulator.rows and not self.completed:
            self._write_report()


def main(args=None):
    rclpy.init(args=args)
    node = SlotOrientationAuditNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.write_partial_report()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
