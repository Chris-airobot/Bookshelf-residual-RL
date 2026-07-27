#!/usr/bin/env python3

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image
import tf2_ros


def quat_xyzw_to_matrix(q):
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat_xyzw(R):
    trace = float(np.trace(R))
    if trace > 0.0:
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

    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    return q / np.linalg.norm(q)


def transform_to_matrix(tf_msg):
    t = tf_msg.transform.translation
    q = tf_msg.transform.rotation

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_xyzw_to_matrix(np.array([q.x, q.y, q.z, q.w]))
    T[:3, 3] = np.array([t.x, t.y, t.z], dtype=np.float64)
    return T


def rvec_tvec_to_matrix(rvec, tvec):
    T = np.eye(4, dtype=np.float64)
    R, _ = cv2.Rodrigues(rvec)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T


def invert_transform(T):
    T_inv = np.eye(4, dtype=np.float64)
    R = T[:3, :3]
    p = T[:3, 3]
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -(R.T @ p)
    return T_inv


def rotation_angle_between(q_ref, q):
    dot = abs(float(np.dot(q_ref, q)))
    dot = max(-1.0, min(1.0, dot))
    return 2.0 * math.acos(dot)


def average_quaternions(quats):
    if not quats:
        raise ValueError("Cannot average zero quaternions")

    q_ref = quats[0]
    aligned = []
    for q in quats:
        if float(np.dot(q_ref, q)) < 0.0:
            aligned.append(-q)
        else:
            aligned.append(q)

    q_mean = np.mean(np.stack(aligned, axis=0), axis=0)
    return q_mean / np.linalg.norm(q_mean)


class ExternalCameraCalibrator(Node):
    def __init__(self, args):
        super().__init__("external_camera_calibrator")
        self.args = args
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.samples = []

        self.wrist_K = None
        self.wrist_D = None
        self.external_K = None
        self.external_D = None
        self.latest_T_wrist_board = None
        self.latest_T_external_board = None

        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        self.ids = np.arange(
            args.start_id,
            args.start_id + (args.squares_x * args.squares_y) // 2,
            dtype=np.int32,
        )
        self.board = cv2.aruco.CharucoBoard(
            (args.squares_x, args.squares_y),
            args.square_length,
            args.marker_length,
            self.dictionary,
            self.ids,
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
            args.wrist_camera_info_topic,
            self.wrist_camera_info_callback,
            qos,
        )
        self.create_subscription(
            Image,
            args.wrist_image_topic,
            self.wrist_image_callback,
            qos,
        )
        self.create_subscription(
            CameraInfo,
            args.external_camera_info_topic,
            self.external_camera_info_callback,
            qos,
        )
        self.create_subscription(
            Image,
            args.external_image_topic,
            self.external_image_callback,
            qos,
        )

        self.get_logger().info("External camera calibrator started")
        self.get_logger().info(f"Wrist image topic: {args.wrist_image_topic}")
        self.get_logger().info(f"External image topic: {args.external_image_topic}")
        self.get_logger().info(
            "Detecting ChArUco board: "
            f"{args.squares_x}x{args.squares_y}, "
            f"square={args.square_length:.3f} m, "
            f"marker={args.marker_length:.3f} m"
        )

    def wrist_camera_info_callback(self, msg):
        self.wrist_K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.wrist_D = np.array(msg.d, dtype=np.float64)

    def external_camera_info_callback(self, msg):
        self.external_K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.external_D = np.array(msg.d, dtype=np.float64)

    def wrist_image_callback(self, msg):
        if self.wrist_K is None:
            return
        T = self.detect_board(msg, self.wrist_K, self.wrist_D)
        if T is not None:
            self.latest_T_wrist_board = T

    def external_image_callback(self, msg):
        if self.external_K is None:
            return
        T = self.detect_board(msg, self.external_K, self.external_D)
        if T is not None:
            self.latest_T_external_board = T

    def detect_board(self, msg, K, D):
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        marker_corners, marker_ids, _ = self.aruco_detector.detectMarkers(gray)
        if marker_ids is None:
            return None

        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners,
            marker_ids,
            gray,
            self.board,
        )
        if retval < self.args.min_charuco_corners:
            return None

        try:
            ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners,
                charuco_ids,
                self.board,
                K,
                D,
                None,
                None,
            )
        except cv2.error:
            return None
        if not ok:
            return None

        return rvec_tvec_to_matrix(rvec, tvec)

    def lookup_matrix(self, target_frame, source_frame):
        tf_msg = self.tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            rclpy.time.Time(),
            timeout=Duration(seconds=self.args.tf_timeout),
        )
        return transform_to_matrix(tf_msg)

    def take_sample(self):
        if self.latest_T_wrist_board is None:
            raise RuntimeError("No wrist camera ChArUco pose yet")
        if self.latest_T_external_board is None:
            raise RuntimeError("No external camera ChArUco pose yet")

        T_base_wrist_camera = self.lookup_matrix(
            self.args.base_frame,
            self.args.wrist_camera_frame,
        )
        T_wrist_camera_board = self.latest_T_wrist_board
        T_external_board = self.latest_T_external_board
        T_base_board = T_base_wrist_camera @ T_wrist_camera_board
        T_base_external = T_base_board @ invert_transform(T_external_board)
        self.samples.append(T_base_external)
        self.print_sample(len(self.samples), T_base_external)

    def print_sample(self, index, T):
        p = T[:3, 3]
        q = matrix_to_quat_xyzw(T[:3, :3])
        self.get_logger().info(
            f"sample {index:03d}: "
            f"xyz=[{p[0]: .6f}, {p[1]: .6f}, {p[2]: .6f}] "
            f"xyzw=[{q[0]: .6f}, {q[1]: .6f}, {q[2]: .6f}, {q[3]: .6f}]"
        )

    def summarize(self):
        translations = np.stack([T[:3, 3] for T in self.samples], axis=0)
        quats = [matrix_to_quat_xyzw(T[:3, :3]) for T in self.samples]

        p_mean = np.mean(translations, axis=0)
        q_mean = average_quaternions(quats)

        trans_err = np.linalg.norm(translations - p_mean.reshape(1, 3), axis=1)
        rot_err = np.array(
            [rotation_angle_between(q_mean, q) for q in quats],
            dtype=np.float64,
        )

        self.get_logger().info("External camera calibration summary")
        self.get_logger().info(f"samples: {len(self.samples)}")
        self.get_logger().info(
            "translation mean xyz="
            f"[{p_mean[0]:.9f}, {p_mean[1]:.9f}, {p_mean[2]:.9f}] m"
        )
        self.get_logger().info(
            "rotation mean xyzw="
            f"[{q_mean[0]:.9f}, {q_mean[1]:.9f}, {q_mean[2]:.9f}, {q_mean[3]:.9f}]"
        )
        self.get_logger().info(
            "translation error: "
            f"mean={np.mean(trans_err):.6f} m, max={np.max(trans_err):.6f} m"
        )
        self.get_logger().info(
            "rotation error: "
            f"mean={math.degrees(np.mean(rot_err)):.4f} deg, "
            f"max={math.degrees(np.max(rot_err)):.4f} deg"
        )

        static_cmd = (
            "ros2 run tf2_ros static_transform_publisher "
            f"{p_mean[0]:.9f} {p_mean[1]:.9f} {p_mean[2]:.9f} "
            f"{q_mean[0]:.9f} {q_mean[1]:.9f} {q_mean[2]:.9f} {q_mean[3]:.9f} "
            f"{self.args.base_frame} {self.args.external_camera_frame}"
        )
        self.get_logger().info("Static TF command:")
        self.get_logger().info(static_cmd)

        if self.args.save_file:
            self.save_result(Path(self.args.save_file), p_mean, q_mean, trans_err, rot_err)

    def save_result(self, path, p, q, trans_err, rot_err):
        path.parent.mkdir(parents=True, exist_ok=True)
        text = (
            "external_camera_static_tf:\n"
            f"  parent_frame: {self.args.base_frame}\n"
            f"  child_frame: {self.args.external_camera_frame}\n"
            "  translation:\n"
            f"    x: {p[0]:.9f}\n"
            f"    y: {p[1]:.9f}\n"
            f"    z: {p[2]:.9f}\n"
            "  rotation_xyzw:\n"
            f"    x: {q[0]:.9f}\n"
            f"    y: {q[1]:.9f}\n"
            f"    z: {q[2]:.9f}\n"
            f"    w: {q[3]:.9f}\n"
            "  diagnostics:\n"
            f"    samples: {len(self.samples)}\n"
            f"    translation_error_mean_m: {np.mean(trans_err):.9f}\n"
            f"    translation_error_max_m: {np.max(trans_err):.9f}\n"
            f"    rotation_error_mean_deg: {math.degrees(np.mean(rot_err)):.9f}\n"
            f"    rotation_error_max_deg: {math.degrees(np.max(rot_err)):.9f}\n"
        )
        path.write_text(text)
        self.get_logger().info(f"Saved calibration result to {path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_frame", default="link_base")
    parser.add_argument(
        "--wrist_camera_frame",
        default="wrist_camera_color_optical_frame",
    )
    parser.add_argument(
        "--external_camera_frame",
        default="external_camera_color_optical_frame",
    )
    parser.add_argument("--wrist_image_topic", default="/wrist_camera/color/image_raw")
    parser.add_argument(
        "--wrist_camera_info_topic",
        default="/wrist_camera/color/camera_info",
    )
    parser.add_argument(
        "--external_image_topic",
        default="/external_camera/color/image_raw",
    )
    parser.add_argument(
        "--external_camera_info_topic",
        default="/external_camera/color/camera_info",
    )
    parser.add_argument("--squares_x", type=int, default=14)
    parser.add_argument("--squares_y", type=int, default=4)
    parser.add_argument("--square_length", type=float, default=0.015)
    parser.add_argument("--marker_length", type=float, default=0.011)
    parser.add_argument("--start_id", type=int, default=40)
    parser.add_argument("--min_charuco_corners", type=int, default=6)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--sample_period", type=float, default=0.25)
    parser.add_argument("--tf_timeout", type=float, default=1.0)
    parser.add_argument("--save_file", default="")
    return parser.parse_args()


def main(args=None):
    rclpy.init(args=args)
    cli_args = parse_args()
    node = ExternalCameraCalibrator(cli_args)

    node.get_logger().info(
        "Computing "
        f"{cli_args.base_frame} -> {cli_args.external_camera_frame} from "
        f"{cli_args.base_frame} -> {cli_args.wrist_camera_frame}, "
        "live wrist ChArUco detection, and live external ChArUco detection"
    )

    try:
        while len(node.samples) < cli_args.samples and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=cli_args.sample_period)
            try:
                node.take_sample()
            except Exception as exc:
                node.get_logger().warn(f"Skipping sample: {exc}")

        if node.samples:
            node.summarize()
        else:
            node.get_logger().error("No valid samples collected.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
