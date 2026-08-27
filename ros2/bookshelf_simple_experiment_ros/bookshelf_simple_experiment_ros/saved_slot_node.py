#!/usr/bin/env python3
"""Publish one reviewed slot pose from a saved experiment YAML."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path

from geometry_msgs.msg import PoseStamped
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32
import yaml


DEFAULT_SAVED_SLOT_CONFIG = os.path.join(
    "~",
    "BookshelfFiles",
    "experiment_configs",
    "stationary_approved_53e7fe80d56d_20260819_142355",
    "trial_static_slot.yaml",
)


@dataclass(frozen=True)
class SavedSlot:
    base_frame: str
    translation_xyz: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    width_m: float
    confidence: float


def _finite_tuple(value, length, label):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{label} must contain {length} values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} contains a non-finite value")
    return result


def load_saved_slot(path) -> SavedSlot:
    """Load the static slot and its reviewed confidence from the approved YAML."""
    resolved = Path(os.path.expandvars(os.path.expanduser(str(path))))
    with resolved.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    try:
        static = document["static_slot_environment_check"]["ros__parameters"]
        target = document["calibrated_preinsert_target"]["ros__parameters"]
        translation = _finite_tuple(
            static["static_slot_translation_xyz"], 3, "slot translation"
        )
        quaternion = _finite_tuple(
            static["static_slot_quaternion_xyzw"], 4, "slot quaternion"
        )
        width = float(static["static_slot_width_m"])
        confidence = float(target["static_slot_confidence"])
        base_frame = str(static["base_frame"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"saved slot YAML has an invalid schema: {error}") from error
    if not base_frame or not math.isfinite(width) or width <= 0.0:
        raise ValueError("saved slot frame or width is invalid")
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("saved slot confidence is invalid")
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm < 1.0e-12:
        raise ValueError("saved slot quaternion is zero")
    quaternion = tuple(value / norm for value in quaternion)
    return SavedSlot(base_frame, translation, quaternion, width, confidence)


class SavedSlotNode(Node):
    def __init__(self):
        super().__init__("saved_slot_publisher")
        self.declare_parameter("slot_config", DEFAULT_SAVED_SLOT_CONFIG)
        self.declare_parameter("publish_rate_hz", 5.0)
        config_path = str(self.get_parameter("slot_config").value)
        self.slot = load_saved_slot(config_path)
        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.pose_publisher = self.create_publisher(
            PoseStamped, "/slot_detector/slot_pose", qos
        )
        self.width_publisher = self.create_publisher(
            Float32, "/slot_detector/slot_width", qos
        )
        self.confidence_publisher = self.create_publisher(
            Float32, "/slot_detector/confidence", qos
        )
        rate = max(float(self.get_parameter("publish_rate_hz").value), 0.1)
        self.create_timer(1.0 / rate, self._publish)
        self.get_logger().info(f"Loaded reviewed saved slot from {config_path}")

    def _publish(self):
        message = PoseStamped()
        message.header.frame_id = self.slot.base_frame
        message.header.stamp = self.get_clock().now().to_msg()
        message.pose.position.x, message.pose.position.y, message.pose.position.z = (
            self.slot.translation_xyz
        )
        (
            message.pose.orientation.x,
            message.pose.orientation.y,
            message.pose.orientation.z,
            message.pose.orientation.w,
        ) = self.slot.quaternion_xyzw
        self.pose_publisher.publish(message)
        self.width_publisher.publish(Float32(data=self.slot.width_m))
        self.confidence_publisher.publish(Float32(data=self.slot.confidence))


def main(args=None):
    rclpy.init(args=args)
    node = SavedSlotNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
