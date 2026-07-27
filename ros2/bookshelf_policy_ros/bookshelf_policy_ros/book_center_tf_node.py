#!/usr/bin/env python3

import argparse
import sys

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.utilities import remove_ros_args
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros


def parse_bool(value):
    return str(value).strip().lower() in ("1", "true", "yes", "on")


class BookCenterTFNode(Node):
    def __init__(self, args):
        super().__init__("book_center_tf_node")
        self.args = args
        self.static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        marker_qos = QoSProfile(depth=10)
        marker_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.marker_pub = self.create_publisher(
            MarkerArray,
            args.marker_topic,
            marker_qos,
        )
        self.debug_pub = self.create_publisher(String, args.debug_topic, marker_qos)

        self.book_specs = [
            (
                args.target_marker_frame,
                args.target_center_frame,
                [args.target_offset_x, args.target_offset_y, args.target_offset_z],
                [args.target_size_x, args.target_size_y, args.target_size_z],
                [0.0, 1.0, 0.15, 0.75],
                [args.book_qx, args.book_qy, args.book_qz, args.book_qw],
            ),
            (
                args.left_marker_frame,
                args.left_center_frame,
                [args.left_offset_x, args.left_offset_y, args.left_offset_z],
                [args.left_size_x, args.left_size_y, args.left_size_z],
                [0.1, 0.45, 1.0, 0.65],
                [args.book_qx, args.book_qy, args.book_qz, args.book_qw],
            ),
            (
                args.right_marker_frame,
                args.right_center_frame,
                [args.right_offset_x, args.right_offset_y, args.right_offset_z],
                [args.right_size_x, args.right_size_y, args.right_size_z],
                [1.0, 0.35, 0.05, 0.65],
                [args.book_qx, args.book_qy, args.book_qz, args.book_qw],
            ),
        ]

        if args.publish_static_tfs:
            self.publish_static_center_tfs()
        self.timer = self.create_timer(args.period, self.timer_callback)

        self.get_logger().info("Book center TF node started")
        tf_mode = "static TF" if args.publish_static_tfs else "book offset"
        for marker_frame, center_frame, offset, size, _color, quat in self.book_specs:
            self.get_logger().info(
                f"{tf_mode} {marker_frame} -> {center_frame}: "
                f"translation [{offset[0]:.4f}, {offset[1]:.4f}, {offset[2]:.4f}], "
                f"rotation xyzw [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}], "
                f"box size [{size[0]:.4f}, {size[1]:.4f}, {size[2]:.4f}]"
            )
        self.get_logger().info(f"Publishing RViz book boxes on {args.marker_topic}")

    def publish_static_center_tfs(self):
        transforms = []
        stamp = self.get_clock().now().to_msg()
        for marker_frame, center_frame, offset, _size, _color, quat in self.book_specs:
            msg = TransformStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = marker_frame
            msg.child_frame_id = center_frame
            msg.transform.translation.x = float(offset[0])
            msg.transform.translation.y = float(offset[1])
            msg.transform.translation.z = float(offset[2])
            msg.transform.rotation.x = float(quat[0])
            msg.transform.rotation.y = float(quat[1])
            msg.transform.rotation.z = float(quat[2])
            msg.transform.rotation.w = float(quat[3])
            transforms.append(msg)
        self.static_tf_broadcaster.sendTransform(transforms)

    def timer_callback(self):
        marker_array = MarkerArray()
        for marker_id, (marker_frame, _center_frame, offset, size, color, quat) in enumerate(self.book_specs):
            marker_array.markers.append(
                self.make_book_marker(marker_id, marker_frame, offset, size, color, quat)
            )

        self.marker_pub.publish(marker_array)
        msg = String()
        msg.data = f"published {len(marker_array.markers)} book box markers"
        self.debug_pub.publish(msg)

    def make_book_marker(self, marker_id, marker_frame, offset, size, color, quat):
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = marker_frame
        marker.ns = "bookshelf_books"
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = float(offset[0])
        marker.pose.position.y = float(offset[1])
        marker.pose.position.z = float(offset[2])
        marker.pose.orientation.x = float(quat[0])
        marker.pose.orientation.y = float(quat[1])
        marker.pose.orientation.z = float(quat[2])
        marker.pose.orientation.w = float(quat[3])
        marker.scale.x = float(size[0])
        marker.scale.y = float(size[1])
        marker.scale.z = float(size[2])
        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = float(color[3])
        marker.frame_locked = True
        return marker


def parse_args(argv=None):
    argv = remove_ros_args(args=sys.argv if argv is None else argv)[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", type=float, default=0.1)
    parser.add_argument("--marker_topic", default="/bookshelf_policy/book_boxes")
    parser.add_argument("--debug_topic", default="/bookshelf_policy/book_boxes_debug")
    parser.add_argument("--publish_static_tfs", type=parse_bool, default=True)

    parser.add_argument("--target_marker_frame", default="target_book_marker")
    parser.add_argument("--left_marker_frame", default="left_side_book_marker")
    parser.add_argument("--right_marker_frame", default="right_side_book_marker")

    parser.add_argument("--target_center_frame", default="target_book_center")
    parser.add_argument("--left_center_frame", default="left_side_book_center")
    parser.add_argument("--right_center_frame", default="right_side_book_center")

    parser.add_argument("--target_offset_x", type=float, default=0.0)
    parser.add_argument("--target_offset_y", type=float, default=-0.097)
    parser.add_argument("--target_offset_z", type=float, default=-0.156 / 2.0)

    parser.add_argument("--left_offset_x", type=float, default=0.0)
    parser.add_argument("--left_offset_y", type=float, default=0.0)
    parser.add_argument("--left_offset_z", type=float, default=-0.179 / 2.0)

    parser.add_argument("--right_offset_x", type=float, default=0.0)
    parser.add_argument("--right_offset_y", type=float, default=0.0)
    parser.add_argument("--right_offset_z", type=float, default=-0.179 / 2.0)

    parser.add_argument("--book_qx", type=float, default=0.5)
    parser.add_argument("--book_qy", type=float, default=0.5)
    parser.add_argument("--book_qz", type=float, default=-0.5)
    parser.add_argument("--book_qw", type=float, default=0.5)

    parser.add_argument("--target_size_x", type=float, default=0.156)
    parser.add_argument("--target_size_y", type=float, default=0.034)
    parser.add_argument("--target_size_z", type=float, default=0.236)

    parser.add_argument("--left_size_x", type=float, default=0.179)
    parser.add_argument("--left_size_y", type=float, default=0.050)
    parser.add_argument("--left_size_z", type=float, default=0.230)

    parser.add_argument("--right_size_x", type=float, default=0.179)
    parser.add_argument("--right_size_y", type=float, default=0.065)
    parser.add_argument("--right_size_z", type=float, default=0.230)
    return parser.parse_args(argv)


def main(args=None):
    rclpy.init(args=args)
    cli_args = parse_args()
    node = BookCenterTFNode(cli_args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
