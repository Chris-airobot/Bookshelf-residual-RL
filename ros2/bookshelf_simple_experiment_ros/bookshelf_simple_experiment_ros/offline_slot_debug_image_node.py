#!/usr/bin/env python3
"""Publish a static synthetic slot-detector-style image for offline RViz preview."""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


def _make_preview_image() -> np.ndarray:
    height, width = 540, 960
    image = np.full((height, width, 3), (188, 183, 176), dtype=np.uint8)

    shelf_color = (154, 181, 207)
    cv2.rectangle(image, (45, 45), (915, 90), shelf_color, -1)
    cv2.rectangle(image, (45, 445), (915, 505), shelf_color, -1)
    cv2.rectangle(image, (45, 45), (78, 505), shelf_color, -1)
    cv2.rectangle(image, (882, 45), (915, 505), shelf_color, -1)

    books = [
        (88, 106, (167, 195, 217)),
        (159, 118, (184, 206, 219)),
        (230, 100, (152, 189, 211)),
        (301, 112, (192, 211, 219)),
        (372, 104, (161, 196, 216)),
        (570, 110, (187, 208, 219)),
        (641, 101, (157, 191, 213)),
        (712, 116, (194, 212, 220)),
        (783, 105, (170, 198, 216)),
    ]
    for x, top, color in books:
        cv2.rectangle(image, (x, top), (x + 62, 444), color, -1)
        cv2.rectangle(image, (x, top), (x + 62, 444), (115, 120, 125), 2)

    left_x, center_x, right_x = 454, 501, 548
    green = (40, 220, 55)
    cv2.line(image, (left_x, 95), (left_x, 450), green, 4)
    cv2.line(image, (right_x, 95), (right_x, 450), green, 4)
    cv2.line(image, (center_x, 95), (center_x, 450), green, 2)
    cv2.circle(image, (center_x, 272), 7, green, -1)
    cv2.putText(
        image,
        "slot width: 37.2 mm",
        (378, 480),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        green,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "confidence: 0.96  |  OFFLINE PREVIEW",
        (300, 525),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (35, 110, 35),
        1,
        cv2.LINE_AA,
    )
    return image


class OfflineSlotDebugImageNode(Node):
    def __init__(self):
        super().__init__("offline_slot_debug_image")
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self.publisher = self.create_publisher(
            Image, "/slot_detector/debug_image", qos
        )
        self.image = _make_preview_image()
        self.create_timer(1.0, self._publish)
        self._publish()
        self.get_logger().info(
            "publishing synthetic offline preview on /slot_detector/debug_image"
        )

    def _publish(self):
        message = Image()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "offline_preview"
        message.height, message.width = self.image.shape[:2]
        message.encoding = "bgr8"
        message.is_bigendian = False
        message.step = message.width * 3
        message.data = self.image.tobytes()
        self.publisher.publish(message)


def main(args=None):
    rclpy.init(args=args)
    node = OfflineSlotDebugImageNode()
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
