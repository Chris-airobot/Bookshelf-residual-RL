#!/usr/bin/env python3
"""Detect the ChArUco board attached to the box/jig bottom plane.

This is the first calibration step: verify that the phone camera can estimate
the jig frame defined by the ChArUco board.
"""

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np


DICT_NAMES = {
    "4x4_50": cv2.aruco.DICT_4X4_50,
    "4x4_100": cv2.aruco.DICT_4X4_100,
    "4x4_250": cv2.aruco.DICT_4X4_250,
    "4x4_1000": cv2.aruco.DICT_4X4_1000,
    "5x5_50": cv2.aruco.DICT_5X5_50,
    "5x5_100": cv2.aruco.DICT_5X5_100,
    "6x6_50": cv2.aruco.DICT_6X6_50,
    "6x6_100": cv2.aruco.DICT_6X6_100,
}

APRILTAG_DICT_NAMES = {}
if hasattr(cv2.aruco, "DICT_APRILTAG_36h11"):
    APRILTAG_DICT_NAMES["tag36h11"] = cv2.aruco.DICT_APRILTAG_36h11
if hasattr(cv2.aruco, "DICT_APRILTAG_25h9"):
    APRILTAG_DICT_NAMES["tag25h9"] = cv2.aruco.DICT_APRILTAG_25h9

BOOK_MARKER_DICT_NAMES = dict(DICT_NAMES)
if hasattr(cv2.aruco, "DICT_ARUCO_ORIGINAL"):
    BOOK_MARKER_DICT_NAMES["aruco_original"] = cv2.aruco.DICT_ARUCO_ORIGINAL
BOOK_MARKER_DICT_NAMES.update(APRILTAG_DICT_NAMES)


# Printed board used by the RealSense/hand-eye calibration setup.
# Target Type: ChArUco
# Columns: 14, Rows: 4
# Checker width: 15 mm
# Marker width: 11 mm
# Dictionary: DICT_4X4_250, Start Id: 40
DEFAULT_DICTIONARY = "4x4_250"
DEFAULT_SQUARES_X = 14
DEFAULT_SQUARES_Y = 4
DEFAULT_SQUARE_LENGTH = 0.015
DEFAULT_MARKER_LENGTH = 0.011
DEFAULT_START_ID = 40
DEFAULT_BOOK_MARKER_DICTIONARY = (
    "tag36h11"
    if "tag36h11" in BOOK_MARKER_DICT_NAMES
    else "aruco_original"
    if "aruco_original" in BOOK_MARKER_DICT_NAMES
    else None
)
DEFAULT_BOOK_MARKER_IDS = "0"
DEFAULT_BOOK_MARKER_LENGTH = 0.019
# New calibration placement:
#   book dimensions are 179 x 65 x 230 mm in the Isaac convention below.
#   The visible 230 x 65 mm face sits just outside the ChArUco board and is
#   aligned to a board corner/edge.
DEFAULT_BOOK_DEPTH = 0.179
DEFAULT_BOOK_HEIGHT = 0.230
DEFAULT_BOOK_THICKNESS = 0.065
DEFAULT_BOOK_CORNER_BOARD = "0,0,-0.179"
# Isaac book convention:
#   +X_book = depth / insertion length
#   +Y_book = height
#   +Z_book = thickness
# Current physical calibration placement:
#   the book is outside the ChArUco board, with its aligned corner on the board
#   corner. The book height follows board +Y/green, thickness follows board
#   -X/red, and the corner is offset below the board plane so the book center
#   remains at negative board Z while keeping a right-handed Isaac-style book
#   frame.
DEFAULT_BOOK_X_AXIS_BOARD = "0,0,1"
DEFAULT_BOOK_Y_AXIS_BOARD = "0,1,0"
DEFAULT_BOOK_Z_AXIS_BOARD = "-1,0,0"


def open_capture(source):
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera source: {source}")
    return cap


class RosImageCapture:
    def __init__(self, image_topic, camera_info_topic, require_camera_info=True):
        try:
            import rclpy
            from cv_bridge import CvBridge
            from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
            from sensor_msgs.msg import CameraInfo, Image
        except ImportError as exc:
            raise RuntimeError(
                "ROS input requires rclpy, cv_bridge, and sensor_msgs in the active environment."
            ) from exc

        self.rclpy = rclpy
        if not rclpy.ok():
            rclpy.init()
            self._owns_rclpy = True
        else:
            self._owns_rclpy = False

        self.node = rclpy.create_node("phone_charuco_board_detect")
        self.bridge = CvBridge()
        self.frame = None
        self.K = None
        self.D = None
        self.image_error = None
        self.require_camera_info = require_camera_info

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.node.create_subscription(Image, image_topic, self._image_callback, qos)
        self.node.create_subscription(CameraInfo, camera_info_topic, self._camera_info_callback, qos)

    def _image_callback(self, msg):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.image_error = exc

    def _camera_info_callback(self, msg):
        self.K = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        self.D = np.asarray(msg.d, dtype=np.float64).reshape(-1, 1)

    def read(self, timeout_sec=None):
        deadline = None if timeout_sec is None else time.time() + timeout_sec
        while True:
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
            if self.image_error is not None:
                raise RuntimeError(f"Could not convert ROS image: {self.image_error}") from self.image_error
            if self.frame is not None and (not self.require_camera_info or self.K is not None):
                return True, self.frame.copy()
            if deadline is not None and time.time() >= deadline:
                return False, None

    def release(self):
        self.node.destroy_node()
        if self._owns_rclpy:
            self.rclpy.shutdown()


def load_camera_yaml(path):
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Could not open camera yaml: {path}")

    K = fs.getNode("camera_matrix").mat()
    D = fs.getNode("distortion_coefficients").mat()
    if K is None:
        K = fs.getNode("K").mat()
    if D is None:
        D = fs.getNode("D").mat()
    fs.release()

    if K is None:
        raise RuntimeError("Camera yaml must contain camera_matrix or K.")
    if D is None:
        D = np.zeros((5, 1), dtype=np.float64)
    return np.asarray(K, dtype=np.float64), np.asarray(D, dtype=np.float64).reshape(-1, 1)


def fallback_camera_matrix(width, height, focal_scale):
    focal = focal_scale * max(width, height)
    return np.array(
        [[focal, 0.0, width * 0.5], [0.0, focal, height * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def rvec_to_rpy_deg(rvec):
    R, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    if sy > 1e-6:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def transform_from_rvec_tvec(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T


def transform_to_xyz_rpy(T):
    x, y, z = T[:3, 3]
    roll, pitch, yaw = rvec_to_rpy_deg(cv2.Rodrigues(T[:3, :3])[0])
    return x, y, z, roll, pitch, yaw


def draw_frame_label(frame, K, D, rvec, tvec, label, color):
    points, _ = cv2.projectPoints(
        np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        rvec,
        tvec,
        K,
        D,
    )
    x, y = points.reshape(-1, 2)[0]
    cv2.putText(
        frame,
        label,
        (int(x) + 6, int(y) - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_frame_axes_if_visible(frame, K, D, rvec, tvec, length):
    axis_points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [length, 0.0, 0.0],
            [0.0, length, 0.0],
            [0.0, 0.0, length],
        ],
        dtype=np.float32,
    )
    projected, _ = cv2.projectPoints(axis_points, rvec, tvec, K, D)
    points = projected.reshape(-1, 2)
    height, width = frame.shape[:2]
    in_frame = (
        (points[:, 0] >= 0.0)
        & (points[:, 0] < float(width))
        & (points[:, 1] >= 0.0)
        & (points[:, 1] < float(height))
    )
    if bool(np.all(in_frame)):
        cv2.drawFrameAxes(frame, K, D, rvec, tvec, length)


def get_charuco_board_corners(board):
    if hasattr(board, "getChessboardCorners"):
        return np.asarray(board.getChessboardCorners(), dtype=np.float32)
    if hasattr(board, "chessboardCorners"):
        return np.asarray(board.chessboardCorners, dtype=np.float32)
    raise RuntimeError("This OpenCV ChArUco board object does not expose chessboard corners.")


def charuco_reprojection_error(board, charuco_corners, charuco_ids, rvec, tvec, K, D):
    if charuco_ids is None or len(charuco_ids) == 0:
        return None

    board_corners = get_charuco_board_corners(board)
    ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
    object_points = board_corners[ids].reshape(-1, 1, 3)
    image_points = np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 2)
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, D)
    projected = projected.reshape(-1, 2)

    errors = np.linalg.norm(projected - image_points, axis=1)
    return {
        "mean_px": float(errors.mean()),
        "max_px": float(errors.max()),
        "errors_px": errors,
        "projected_points": projected,
    }


def rotation_matrix_to_quat_wxyz(R):
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
    q = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    return q / np.linalg.norm(q)


def quat_wxyz_to_rotation_matrix(q):
    qw, qx, qy, qz = q / np.linalg.norm(q)
    return np.asarray(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def average_quat_wxyz(quats):
    if not quats:
        raise ValueError("Cannot average an empty quaternion list.")
    ref = quats[0]
    aligned = []
    for q in quats:
        qn = q / np.linalg.norm(q)
        if float(np.dot(ref, qn)) < 0.0:
            qn = -qn
        aligned.append(qn)
    q_mean = np.mean(np.asarray(aligned), axis=0)
    return q_mean / np.linalg.norm(q_mean)


def average_transforms(transforms):
    translations = np.asarray([T[:3, 3] for T in transforms], dtype=np.float64)
    quats = [rotation_matrix_to_quat_wxyz(T[:3, :3]) for T in transforms]
    T_mean = np.eye(4, dtype=np.float64)
    T_mean[:3, 3] = translations.mean(axis=0)
    T_mean[:3, :3] = quat_wxyz_to_rotation_matrix(average_quat_wxyz(quats))
    return T_mean


def rotation_error_deg(R_ref, R):
    R_err = R_ref.T @ R
    cos_angle = (float(np.trace(R_err)) - 1.0) * 0.5
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def summarize_transform_samples(transforms):
    T_mean = average_transforms(transforms)
    trans = np.asarray([T[:3, 3] for T in transforms], dtype=np.float64)
    trans_err = np.linalg.norm(trans - T_mean[:3, 3], axis=1)
    rot_err = np.asarray([rotation_error_deg(T_mean[:3, :3], T[:3, :3]) for T in transforms])
    return {
        "mean": T_mean,
        "translation_std_m": trans.std(axis=0),
        "translation_error_mean_m": float(trans_err.mean()),
        "translation_error_max_m": float(trans_err.max()),
        "rotation_error_mean_deg": float(rot_err.mean()),
        "rotation_error_max_deg": float(rot_err.max()),
    }


def parse_int_list(text):
    if text.strip() == "":
        return []
    return [int(item.strip()) for item in text.split(",")]


def parse_vec3(text):
    values = [float(item.strip()) for item in text.split(",")]
    if len(values) != 3:
        raise ValueError(f"Expected three comma-separated values, got: {text}")
    return np.asarray(values, dtype=np.float64)


def normalized(vec, name):
    norm = float(np.linalg.norm(vec))
    if norm < 1.0e-9:
        raise ValueError(f"{name} must be nonzero.")
    return vec / norm


def make_book_transform_in_board(
    corner_board,
    x_axis_board,
    y_axis_board,
    z_axis_board,
    depth,
    height,
    thickness,
):
    x_axis = normalized(x_axis_board, "book_x_axis_board")
    y_axis = normalized(y_axis_board, "book_y_axis_board")
    z_axis = normalized(z_axis_board, "book_z_axis_board")
    R = np.column_stack((x_axis, y_axis, z_axis))
    if np.linalg.det(R) < 0.0:
        raise ValueError("Book axes are left-handed. Flip one axis so X cross Y = Z.")

    center_board = (
        corner_board
        + 0.5 * depth * x_axis
        + 0.5 * height * y_axis
        + 0.5 * thickness * z_axis
    )
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = center_board
    return T


def charuco_marker_ids(squares_x, squares_y, start_id):
    marker_count = (squares_x * squares_y) // 2
    return np.arange(start_id, start_id + marker_count, dtype=np.int32)


def create_charuco_board(squares_x, squares_y, square_length, marker_length, dictionary, start_id):
    ids = charuco_marker_ids(squares_x, squares_y, start_id)
    if hasattr(cv2.aruco, "CharucoBoard"):
        try:
            return cv2.aruco.CharucoBoard(
                (squares_x, squares_y),
                square_length,
                marker_length,
                dictionary,
                ids,
            )
        except cv2.error:
            if start_id != 0:
                raise RuntimeError(
                    "This OpenCV CharucoBoard constructor does not accept custom marker IDs. "
                    "Use an OpenCV build with ChArUco custom ids support."
                )
            return cv2.aruco.CharucoBoard(
                (squares_x, squares_y),
                square_length,
                marker_length,
                dictionary,
            )
    if hasattr(cv2.aruco, "CharucoBoard_create"):
        try:
            return cv2.aruco.CharucoBoard_create(
                squares_x,
                squares_y,
                square_length,
                marker_length,
                dictionary,
                start_id,
            )
        except cv2.error:
            if start_id != 0:
                raise RuntimeError(
                    "This OpenCV CharucoBoard_create does not accept a nonzero first marker id. "
                    "Use an OpenCV build with ChArUco custom ids support."
                )
            return cv2.aruco.CharucoBoard_create(
                squares_x,
                squares_y,
                square_length,
                marker_length,
                dictionary,
            )
    raise RuntimeError(
        "This OpenCV install does not provide ChArUco support. "
        "Install opencv-contrib-python, not plain opencv-python."
    )


def create_detector_params():
    if hasattr(cv2.aruco, "DetectorParameters"):
        return cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        return cv2.aruco.DetectorParameters_create()
    raise RuntimeError("This OpenCV install does not provide ArUco detector parameters.")


def detect_markers(gray, dictionary, detector_params):
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
        return detector.detectMarkers(gray)
    return cv2.aruco.detectMarkers(gray, dictionary, parameters=detector_params)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="OpenCV camera index, MJPEG URL, or RTSP URL.")
    parser.add_argument(
        "--use_ros",
        action="store_true",
        help="Read images and intrinsics from ROS topics instead of OpenCV VideoCapture.",
    )
    parser.add_argument("--ros_image_topic", default="/camera/color/image_raw")
    parser.add_argument("--ros_camera_info_topic", default="/camera/color/camera_info")
    parser.add_argument(
        "--image_topic",
        dest="ros_image_topic",
        help="Alias for --ros_image_topic, matching the RealSense launch scripts.",
    )
    parser.add_argument(
        "--camera_info_topic",
        dest="ros_camera_info_topic",
        help="Alias for --ros_camera_info_topic, matching the RealSense launch scripts.",
    )
    parser.add_argument("--ros_start_timeout", type=float, default=10.0)
    parser.add_argument("--dictionary", choices=sorted(DICT_NAMES), default=DEFAULT_DICTIONARY)
    parser.add_argument(
        "--squares_x",
        type=int,
        default=DEFAULT_SQUARES_X,
        help="Number of chessboard squares along board X.",
    )
    parser.add_argument(
        "--squares_y",
        type=int,
        default=DEFAULT_SQUARES_Y,
        help="Number of chessboard squares along board Y.",
    )
    parser.add_argument(
        "--square_length",
        type=float,
        default=DEFAULT_SQUARE_LENGTH,
        help="Chessboard square side length in meters.",
    )
    parser.add_argument(
        "--marker_length",
        type=float,
        default=DEFAULT_MARKER_LENGTH,
        help="Inner ArUco marker side length in meters.",
    )
    parser.add_argument(
        "--start_id",
        type=int,
        default=DEFAULT_START_ID,
        help="First ArUco marker ID on the ChArUco board.",
    )
    parser.add_argument("--camera_yaml", default=None, help="Optional OpenCV/ROS camera calibration yaml.")
    parser.add_argument("--focal_scale", type=float, default=1.2, help="Fallback focal ~= focal_scale * max(width,height).")
    parser.add_argument("--min_corners", type=int, default=6, help="Minimum ChArUco corners needed for pose.")
    parser.add_argument("--axis_length", type=float, default=0.06, help="Displayed board axis length in meters.")
    parser.add_argument(
        "--book_marker_dictionary",
        choices=sorted(BOOK_MARKER_DICT_NAMES),
        default=DEFAULT_BOOK_MARKER_DICTIONARY,
        help="Dictionary for object/book markers; RealSense marker launch uses aruco_original.",
    )
    parser.add_argument(
        "--book_marker_ids",
        default=DEFAULT_BOOK_MARKER_IDS,
        help="Comma-separated object/book marker IDs to detect.",
    )
    parser.add_argument(
        "--book_marker_length",
        type=float,
        default=DEFAULT_BOOK_MARKER_LENGTH,
        help="Object/book marker black/coded square side length in meters.",
    )
    parser.add_argument("--apriltag_dictionary", choices=sorted(APRILTAG_DICT_NAMES), default=None, help=argparse.SUPPRESS)
    parser.add_argument("--apriltag_ids", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--apriltag_length", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--book_depth",
        type=float,
        default=DEFAULT_BOOK_DEPTH,
        help="Book depth/insertion length in meters. Isaac book local +X.",
    )
    parser.add_argument(
        "--book_height",
        type=float,
        default=DEFAULT_BOOK_HEIGHT,
        help="Book height in meters. Isaac book local +Y.",
    )
    parser.add_argument(
        "--book_thickness",
        type=float,
        default=DEFAULT_BOOK_THICKNESS,
        help="Book thickness in meters. Isaac book local +Z.",
    )
    parser.add_argument(
        "--book_corner_board",
        default=DEFAULT_BOOK_CORNER_BOARD,
        help="Aligned physical book corner in board frame, meters, as x,y,z.",
    )
    parser.add_argument(
        "--book_x_axis_board",
        default=DEFAULT_BOOK_X_AXIS_BOARD,
        help="Book +X/depth direction expressed in board frame, as x,y,z.",
    )
    parser.add_argument(
        "--book_y_axis_board",
        default=DEFAULT_BOOK_Y_AXIS_BOARD,
        help="Book +Y/height direction expressed in board frame, as x,y,z.",
    )
    parser.add_argument(
        "--book_z_axis_board",
        default=DEFAULT_BOOK_Z_AXIS_BOARD,
        help="Book +Z/thickness direction expressed in board frame, as x,y,z.",
    )
    parser.add_argument("--print_interval", type=float, default=0.5)
    parser.add_argument(
        "--collect_samples",
        type=int,
        default=0,
        help="If >0, collect this many T_tag_book samples per tag, average, save, and exit.",
    )
    parser.add_argument(
        "--output_npz",
        default="data/marker_to_book_calibration.npz",
        help="Output NPZ for calibrated T_tag_book transforms and raw samples.",
    )
    parser.add_argument(
        "--output_json",
        default="data/marker_to_book_calibration.json",
        help="Human-readable JSON summary for calibrated T_tag_book transforms.",
    )
    parser.add_argument("--no_preview", action="store_true")
    args = parser.parse_args()

    dictionary = cv2.aruco.getPredefinedDictionary(DICT_NAMES[args.dictionary])
    board = create_charuco_board(
        args.squares_x,
        args.squares_y,
        args.square_length,
        args.marker_length,
        dictionary,
        args.start_id,
    )
    detector_params = create_detector_params()
    marker_dictionary_name = args.apriltag_dictionary or args.book_marker_dictionary
    marker_ids_text = args.apriltag_ids if args.apriltag_ids is not None else args.book_marker_ids
    marker_length = args.apriltag_length if args.apriltag_length is not None else args.book_marker_length
    book_marker_ids = parse_int_list(marker_ids_text)
    book_marker_dictionary = None
    if book_marker_ids:
        if marker_dictionary_name is None:
            raise RuntimeError(
                "No object/book marker dictionary is available in this OpenCV install. "
                "Install opencv-contrib-python, not plain opencv-python."
            )
        book_marker_dictionary = cv2.aruco.getPredefinedDictionary(
            BOOK_MARKER_DICT_NAMES[marker_dictionary_name]
        )
    print(
        "[book marker] "
        f"dictionary={marker_dictionary_name} ids={book_marker_ids} length={marker_length:.4f} m"
    )

    T_board_book = make_book_transform_in_board(
        parse_vec3(args.book_corner_board),
        parse_vec3(args.book_x_axis_board),
        parse_vec3(args.book_y_axis_board),
        parse_vec3(args.book_z_axis_board),
        args.book_depth,
        args.book_height,
        args.book_thickness,
    )
    T_book_board = np.linalg.inv(T_board_book)

    if args.use_ros:
        cap = RosImageCapture(
            args.ros_image_topic,
            args.ros_camera_info_topic,
            require_camera_info=args.camera_yaml is None,
        )
        ok, frame = cap.read(timeout_sec=args.ros_start_timeout)
    else:
        cap = open_capture(args.source)
        ok, frame = cap.read()
    if not ok:
        if args.use_ros:
            raise RuntimeError(
                "ROS topics did not return an image"
                + (" and camera info" if args.camera_yaml is None else "")
                + f" within {args.ros_start_timeout:.1f} seconds."
            )
        raise RuntimeError("Camera opened but did not return a frame.")

    height, width = frame.shape[:2]
    if args.camera_yaml:
        K, D = load_camera_yaml(Path(args.camera_yaml))
        print(f"[charuco board] loaded camera intrinsics from {args.camera_yaml}")
    elif args.use_ros and cap.K is not None:
        K, D = cap.K, cap.D
        print(f"[charuco board] loaded camera intrinsics from {args.ros_camera_info_topic}")
    else:
        K = fallback_camera_matrix(width, height, args.focal_scale)
        D = np.zeros((5, 1), dtype=np.float64)
        print("[charuco board] WARNING: using approximate camera intrinsics.")
        print("[charuco board] This is fine for axis/debug, but not final calibration accuracy.")

    last_print = 0.0
    pose_count = 0
    tag_samples = {tag_id: [] for tag_id in book_marker_ids}

    while True:
        ok, frame = cap.read(timeout_sec=1.0) if args.use_ros else cap.read()
        if not ok:
            print("[charuco board] frame read failed")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        marker_corners, marker_ids, _ = detect_markers(gray, dictionary, detector_params)
        book_marker_corners = None
        book_marker_detected_ids = None
        if book_marker_dictionary is not None:
            book_marker_corners, book_marker_detected_ids, _ = detect_markers(
                gray, book_marker_dictionary, detector_params
            )
        board_found = False
        num_charuco = 0
        reproj_mean_px = None
        reproj_max_px = None
        tag_summaries = []

        if marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)
            retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners,
                marker_ids,
                gray,
                board,
                K,
                D,
            )
            num_charuco = int(retval)

            if retval >= args.min_corners:
                cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)
                try:
                    ok_pose, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                        charuco_corners,
                        charuco_ids,
                        board,
                        K,
                        D,
                        None,
                        None,
                    )
                except cv2.error:
                    ok_pose = False
                if ok_pose:
                    board_found = True
                    pose_count += 1
                    reproj = charuco_reprojection_error(
                        board,
                        charuco_corners,
                        charuco_ids,
                        rvec,
                        tvec,
                        K,
                        D,
                    )
                    if reproj is not None:
                        reproj_mean_px = reproj["mean_px"]
                        reproj_max_px = reproj["max_px"]
                    draw_frame_axes_if_visible(frame, K, D, rvec, tvec, args.axis_length)
                    draw_frame_label(frame, K, D, rvec, tvec, "board", (0, 255, 0))

                    T_camera_board = transform_from_rvec_tvec(rvec, tvec)
                    T_camera_book = T_camera_board @ T_board_book
                    book_rvec, book_tvec = cv2.Rodrigues(T_camera_book[:3, :3])[0], T_camera_book[
                        :3, 3
                    ].reshape(3, 1)
                    draw_frame_axes_if_visible(frame, K, D, book_rvec, book_tvec, args.axis_length * 0.8)
                    draw_frame_label(frame, K, D, book_rvec, book_tvec, "book", (255, 255, 0))

                    if book_marker_detected_ids is not None:
                        cv2.aruco.drawDetectedMarkers(frame, book_marker_corners, book_marker_detected_ids)
                        flat_tag_ids = book_marker_detected_ids.reshape(-1)
                        for tag_id in book_marker_ids:
                            matches = np.where(flat_tag_ids == tag_id)[0]
                            if len(matches) == 0:
                                continue
                            idx = int(matches[0])
                            tag_rvecs, tag_tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                                [book_marker_corners[idx]],
                                marker_length,
                                K,
                                D,
                            )
                            tag_rvec = tag_rvecs[0].reshape(3, 1)
                            tag_tvec = tag_tvecs[0].reshape(3, 1)
                            T_camera_tag = transform_from_rvec_tvec(tag_rvec, tag_tvec)
                            T_board_tag = np.linalg.inv(T_camera_board) @ T_camera_tag
                            T_book_tag = T_book_board @ T_board_tag
                            T_tag_book = np.linalg.inv(T_book_tag)
                            if args.collect_samples > 0 and len(tag_samples[tag_id]) < args.collect_samples:
                                tag_samples[tag_id].append(T_tag_book)
                            tag_draw_len = min(marker_length, args.axis_length * 0.45)
                            draw_frame_axes_if_visible(frame, K, D, tag_rvec, tag_tvec, tag_draw_len)
                            draw_frame_label(frame, K, D, tag_rvec, tag_tvec, f"tag {tag_id}", (0, 255, 255))
                            bx, by, bz, broll, bpitch, byaw = transform_to_xyz_rpy(T_board_tag)
                            x, y, z, roll_t, pitch_t, yaw_t = transform_to_xyz_rpy(T_book_tag)
                            tag_summaries.append(
                                (tag_id, bx, by, bz, x, y, z, roll_t, pitch_t, yaw_t)
                            )

                    now = time.time()
                    if now - last_print >= args.print_interval:
                        roll, pitch, yaw = rvec_to_rpy_deg(rvec)
                        t = np.asarray(tvec).reshape(3)
                        reproj_text = ""
                        if reproj_mean_px is not None:
                            reproj_text = (
                                f" reproj_mean={reproj_mean_px:.2f}px"
                                f" reproj_max={reproj_max_px:.2f}px"
                            )
                        print(
                            "[charuco board] "
                            f"T_camera_board t=({t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f}) m "
                            f"rpy=({roll:+.2f}, {pitch:+.2f}, {yaw:+.2f}) deg "
                            f"corners={num_charuco}"
                            f"{reproj_text}"
                        )
                        for tag_id, bx, by, bz, x, y, z, roll_t, pitch_t, yaw_t in tag_summaries:
                            print(
                                "[book marker] "
                                f"id={tag_id} "
                                f"T_board_tag t=({bx:+.4f}, {by:+.4f}, {bz:+.4f}) m "
                                f"T_book_tag t=({x:+.4f}, {y:+.4f}, {z:+.4f}) m "
                                f"rpy=({roll_t:+.2f}, {pitch_t:+.2f}, {yaw_t:+.2f}) deg"
                            )
                        last_print = now

        if args.collect_samples > 0:
            counts_text = ",".join(f"{tag_id}:{len(tag_samples[tag_id])}" for tag_id in book_marker_ids)
            if book_marker_ids and all(len(tag_samples[tag_id]) >= args.collect_samples for tag_id in book_marker_ids):
                break
        else:
            counts_text = ""

        status = (
            f"charuco board found={int(board_found)} corners={num_charuco} "
            f"markers={len(tag_summaries)}/{len(book_marker_ids)} poses={pose_count}"
        )
        if reproj_mean_px is not None:
            status += f" reproj={reproj_mean_px:.2f}/{reproj_max_px:.2f}px"
        if counts_text:
            status += f" samples={counts_text}"
        cv2.putText(
            frame,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if board_found else (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        if not args.no_preview:
            cv2.imshow("charuco board detect", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    cap.release()
    if not args.no_preview:
        cv2.destroyAllWindows()

    if args.collect_samples > 0:
        summaries = {}
        save_arrays = {
            "tag_ids": np.asarray(book_marker_ids, dtype=np.int64),
            "T_board_book": T_board_book,
            "T_book_board": T_book_board,
            "book_size_depth_height_thickness": np.asarray(
                [args.book_depth, args.book_height, args.book_thickness],
                dtype=np.float64,
            ),
        }
        for tag_id in book_marker_ids:
            samples = tag_samples[tag_id]
            if not samples:
                print(f"[calibration] tag {tag_id}: no samples, skipping")
                continue
            stats = summarize_transform_samples(samples)
            T_mean = stats["mean"]
            x, y, z, roll, pitch, yaw = transform_to_xyz_rpy(T_mean)
            save_arrays[f"T_tag{tag_id}_book"] = T_mean
            save_arrays[f"T_tag{tag_id}_book_samples"] = np.asarray(samples, dtype=np.float64)
            summaries[str(tag_id)] = {
                "count": len(samples),
                "T_tag_book": T_mean.tolist(),
                "translation_xyz_m": [float(x), float(y), float(z)],
                "rotation_rpy_deg": [float(roll), float(pitch), float(yaw)],
                "translation_std_m": stats["translation_std_m"].tolist(),
                "translation_error_mean_m": stats["translation_error_mean_m"],
                "translation_error_max_m": stats["translation_error_max_m"],
                "rotation_error_mean_deg": stats["rotation_error_mean_deg"],
                "rotation_error_max_deg": stats["rotation_error_max_deg"],
            }
            print(
                "[calibration] "
                f"tag={tag_id} n={len(samples)} "
                f"T_tag_book t=({x:+.4f}, {y:+.4f}, {z:+.4f}) m "
                f"rpy=({roll:+.2f}, {pitch:+.2f}, {yaw:+.2f}) deg "
                f"std=({stats['translation_std_m'][0]:.4f}, "
                f"{stats['translation_std_m'][1]:.4f}, "
                f"{stats['translation_std_m'][2]:.4f}) m "
                f"rot_err_mean={stats['rotation_error_mean_deg']:.2f} deg"
            )

        output_npz = Path(args.output_npz)
        output_json = Path(args.output_json)
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_npz, **save_arrays)
        with output_json.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "book_frame": {
                        "convention": "Isaac: +X depth/insertion, +Y height, +Z thickness",
                        "corner_board": parse_vec3(args.book_corner_board).tolist(),
                        "x_axis_board": parse_vec3(args.book_x_axis_board).tolist(),
                        "y_axis_board": parse_vec3(args.book_y_axis_board).tolist(),
                        "z_axis_board": parse_vec3(args.book_z_axis_board).tolist(),
                        "depth_height_thickness_m": [
                            args.book_depth,
                            args.book_height,
                            args.book_thickness,
                        ],
                    },
                    "tags": summaries,
                },
                f,
                indent=2,
            )
        print(f"[calibration] wrote {output_npz}")
        print(f"[calibration] wrote {output_json}")


if __name__ == "__main__":
    main()
