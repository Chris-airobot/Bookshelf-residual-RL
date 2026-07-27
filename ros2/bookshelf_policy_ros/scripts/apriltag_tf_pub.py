#!/usr/bin/env python3

import argparse
import math

import cv2
import numpy as np

import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import TransformBroadcaster


DICT_NAMES = {}
if hasattr(cv2.aruco, "DICT_APRILTAG_36h11"):
    DICT_NAMES["tag36h11"] = cv2.aruco.DICT_APRILTAG_36h11
if hasattr(cv2.aruco, "DICT_APRILTAG_25h9"):
    DICT_NAMES["tag25h9"] = cv2.aruco.DICT_APRILTAG_25h9


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


class AprilTagTFPublisher(Node):
    def __init__(self, args):
        super().__init__("apriltag_tf_publisher")
        if args.dictionary not in DICT_NAMES:
            raise RuntimeError(
                f"OpenCV does not expose AprilTag dictionary {args.dictionary}. "
                "Install/use an opencv-contrib build."
            )

        self.bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)
        self.K = None
        self.D = None
        self.args = args

        dictionary = cv2.aruco.getPredefinedDictionary(DICT_NAMES[args.dictionary])
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(dictionary, self.detector_params)

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.create_subscription(CameraInfo, args.camera_info_topic, self.camera_info_callback, qos)
        self.create_subscription(Image, args.image_topic, self.image_callback, qos)

        self.get_logger().info(
            f"Publishing AprilTag {args.dictionary} id={args.tag_id}: "
            f"{args.camera_frame} -> {args.tag_frame}"
        )

    def camera_info_callback(self, msg):
        self.K = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        self.D = np.asarray(msg.d, dtype=np.float64)

    def image_callback(self, msg):
        if self.K is None:
            return

        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return

        flat_ids = ids.reshape(-1)
        matches = np.where(flat_ids == self.args.tag_id)[0]
        if len(matches) == 0:
            return

        idx = int(matches[0])
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            [corners[idx]],
            self.args.tag_length,
            self.K,
            self.D,
        )
        rvec = rvecs[0].reshape(3, 1)
        tvec = tvecs[0].reshape(3)
        R, _ = cv2.Rodrigues(rvec)
        qx, qy, qz, qw = rotation_matrix_to_quaternion(R)

        tf_msg = TransformStamped()
        tf_msg.header.stamp = msg.header.stamp
        tf_msg.header.frame_id = self.args.camera_frame
        tf_msg.child_frame_id = self.args.tag_frame
        tf_msg.transform.translation.x = float(tvec[0])
        tf_msg.transform.translation.y = float(tvec[1])
        tf_msg.transform.translation.z = float(tvec[2])
        tf_msg.transform.rotation.x = float(qx)
        tf_msg.transform.rotation.y = float(qy)
        tf_msg.transform.rotation.z = float(qz)
        tf_msg.transform.rotation.w = float(qw)
        self.tf_broadcaster.sendTransform(tf_msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_topic", default="/camera/color/image_raw")
    parser.add_argument("--camera_info_topic", default="/camera/color/camera_info")
    parser.add_argument("--camera_frame", default="camera_color_optical_frame")
    parser.add_argument("--dictionary", choices=sorted(DICT_NAMES), default="tag36h11")
    parser.add_argument("--tag_id", type=int, default=0)
    parser.add_argument("--tag_length", type=float, default=0.019)
    parser.add_argument("--tag_frame", default="apriltag_36h11_0")
    args = parser.parse_args()

    rclpy.init()
    node = AprilTagTFPublisher(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
