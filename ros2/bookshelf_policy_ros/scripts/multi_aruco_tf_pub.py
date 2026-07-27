#!/usr/bin/env python3

import argparse
import math

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import TransformBroadcaster


def rotation_matrix_to_quaternion(R):
    trace = np.trace(R)

    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    return qx, qy, qz, qw


class MultiArucoTFPublisher(Node):
    def __init__(self, args):
        super().__init__("multi_aruco_tf_publisher")

        self.bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)

        self.K = None
        self.D = None

        self.image_topic = args.image_topic
        self.camera_info_topic = args.camera_info_topic
        self.camera_frame = args.camera_frame
        self.debug_image_topic = args.debug_image_topic

        self.marker_frames = {
            int(args.left_id): args.left_frame,
            int(args.right_id): args.right_frame,
            int(args.target_id): args.target_frame,
        }
        self.marker_lengths = {
            int(args.left_id): float(args.left_marker_length),
            int(args.right_id): float(args.right_marker_length),
            int(args.target_id): float(args.target_marker_length),
        }

        self.dictionary = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_ARUCO_ORIGINAL
        )
        self.detector_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(
            self.dictionary,
            self.detector_params,
        )

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            qos,
        )
        self.create_subscription(Image, self.image_topic, self.image_callback, qos)
        self.debug_image_pub = self.create_publisher(
            Image,
            self.debug_image_topic,
            10,
        )

        self.get_logger().info("Multi-ArUco TF publisher started")
        self.get_logger().info(f"Image topic: {self.image_topic}")
        self.get_logger().info(f"Camera info topic: {self.camera_info_topic}")
        self.get_logger().info(f"Debug image topic: {self.debug_image_topic}")
        self.get_logger().info(f"Camera frame: {self.camera_frame}")
        for marker_id, frame in sorted(self.marker_frames.items()):
            self.get_logger().info(
                f"Publishing marker id {marker_id} as {frame}, "
                f"size={self.marker_lengths[marker_id]:.3f} m"
            )

    def camera_info_callback(self, msg):
        self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.D = np.array(msg.d, dtype=np.float64)

    def image_callback(self, msg):
        if self.K is None:
            return

        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        marker_corners, marker_ids, _ = self.aruco_detector.detectMarkers(gray)
        if marker_ids is None:
            self.publish_debug_image(msg, image)
            return

        marker_ids = marker_ids.flatten()
        rvecs_by_index = {}
        tvecs_by_index = {}

        for i, marker_id in enumerate(marker_ids):
            marker_id = int(marker_id)
            if marker_id not in self.marker_frames:
                continue

            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [marker_corners[i]],
                self.marker_lengths[marker_id],
                self.K,
                self.D,
            )
            rvecs_by_index[i] = rvecs[0]
            tvecs_by_index[i] = tvecs[0]

            R, _ = cv2.Rodrigues(rvecs_by_index[i].reshape(3, 1))
            qx, qy, qz, qw = rotation_matrix_to_quaternion(R)
            tvec = tvecs_by_index[i].reshape(3)

            tf_msg = TransformStamped()
            tf_msg.header.stamp = msg.header.stamp
            tf_msg.header.frame_id = self.camera_frame
            tf_msg.child_frame_id = self.marker_frames[marker_id]

            tf_msg.transform.translation.x = float(tvec[0])
            tf_msg.transform.translation.y = float(tvec[1])
            tf_msg.transform.translation.z = float(tvec[2])

            tf_msg.transform.rotation.x = float(qx)
            tf_msg.transform.rotation.y = float(qy)
            tf_msg.transform.rotation.z = float(qz)
            tf_msg.transform.rotation.w = float(qw)

            self.tf_broadcaster.sendTransform(tf_msg)

        self.publish_debug_image(
            msg,
            image,
            marker_corners,
            marker_ids,
            rvecs_by_index,
            tvecs_by_index,
        )

    def publish_debug_image(
        self,
        msg,
        image,
        marker_corners=None,
        marker_ids=None,
        rvecs=None,
        tvecs=None,
    ):
        debug_image = image.copy()

        if marker_ids is not None:
            selected_corners = []
            selected_ids = []
            selected_poses = []

            for i, marker_id in enumerate(marker_ids):
                marker_id = int(marker_id)
                if marker_id not in self.marker_frames:
                    continue

                selected_corners.append(marker_corners[i])
                selected_ids.append([marker_id])
                selected_poses.append(i)

            if selected_corners:
                cv2.aruco.drawDetectedMarkers(
                    debug_image,
                    selected_corners,
                    np.array(selected_ids, dtype=np.int32),
                )

                for i in selected_poses:
                    marker_id = int(marker_ids[i])
                    marker_length = self.marker_lengths[marker_id]
                    cv2.drawFrameAxes(
                        debug_image,
                        self.K,
                        self.D,
                        rvecs[i],
                        tvecs[i],
                        0.5 * marker_length,
                    )

                    corner = marker_corners[i][0][0]
                    label = self.marker_frames[marker_id]
                    cv2.putText(
                        debug_image,
                        label,
                        (int(corner[0]), int(corner[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

        debug_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding="bgr8")
        debug_msg.header = msg.header
        self.debug_image_pub.publish(debug_msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_topic", default="/camera/color/image_raw")
    parser.add_argument("--camera_info_topic", default="/camera/color/camera_info")
    parser.add_argument(
        "--debug_image_topic",
        default="/bookshelf_policy/aruco_debug_image",
    )
    parser.add_argument("--camera_frame", default="camera_color_optical_frame")

    parser.add_argument("--left_id", type=int, default=0)
    parser.add_argument("--right_id", type=int, default=1)
    parser.add_argument("--target_id", type=int, default=2)

    parser.add_argument("--left_marker_length", type=float, default=0.040)
    parser.add_argument("--right_marker_length", type=float, default=0.040)
    parser.add_argument("--target_marker_length", type=float, default=0.030)

    parser.add_argument("--left_frame", default="left_side_book_marker")
    parser.add_argument("--right_frame", default="right_side_book_marker")
    parser.add_argument("--target_frame", default="target_book_marker")

    args = parser.parse_args()

    rclpy.init()
    node = MultiArucoTFPublisher(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
