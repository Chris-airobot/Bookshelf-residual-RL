import argparse
import math
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
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


class CharucoTFPublisher(Node):
    def __init__(self, image_topic, camera_info_topic, camera_frame, board_frame):
        super().__init__("charuco_tf_publisher")

        self.bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)

        self.K = None
        self.D = None

        self.camera_frame = camera_frame
        self.board_frame = board_frame

        # Your real printed ChArUco board parameters
        self.squares_x = 5
        self.squares_y = 7
        self.square_length = 0.040
        self.marker_length = 0.030

        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

        self.board = cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            self.square_length,
            self.marker_length,
            self.dictionary
        )

        self.detector_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(
            self.dictionary,
            self.detector_params
        )

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        self.create_subscription(CameraInfo, camera_info_topic, self.camera_info_callback, qos)
        self.create_subscription(Image, image_topic, self.image_callback, qos)

        self.get_logger().info("ChArUco TF publisher started")
        self.get_logger().info(f"Image topic: {image_topic}")
        self.get_logger().info(f"Camera info topic: {camera_info_topic}")
        self.get_logger().info(f"Publishing TF: {camera_frame} -> {board_frame}")

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
            return

        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners,
            marker_ids,
            gray,
            self.board
        )

        if retval < 4:
            return

        ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            charuco_corners,
            charuco_ids,
            self.board,
            self.K,
            self.D,
            None,
            None
        )

        if not ok:
            return

        R, _ = cv2.Rodrigues(rvec)
        qx, qy, qz, qw = rotation_matrix_to_quaternion(R)

        tf_msg = TransformStamped()
        tf_msg.header.stamp = msg.header.stamp
        tf_msg.header.frame_id = self.camera_frame
        tf_msg.child_frame_id = self.board_frame

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
    parser.add_argument("--board_frame", default="charuco_board")
    args = parser.parse_args()

    rclpy.init()
    node = CharucoTFPublisher(
        args.image_topic,
        args.camera_info_topic,
        args.camera_frame,
        args.board_frame
    )
    rclpy.spin(node)


if __name__ == "__main__":
    main()
