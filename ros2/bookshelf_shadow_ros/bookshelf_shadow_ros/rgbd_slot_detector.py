#!/usr/bin/env python3
"""Detect a bookshelf opening from aligned RGB and depth images in shadow mode."""

from collections import deque
from dataclasses import dataclass
import math
import time

import cv2
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped, PoseStamped
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32


@dataclass
class SlotDetection:
    left_pixel: int
    right_pixel: int
    center_pixel: int
    left_line: np.ndarray
    right_line: np.ndarray
    left_point: np.ndarray
    right_point: np.ndarray
    center_point: np.ndarray
    plane_normal: np.ndarray
    vertical_axis: np.ndarray
    slot_width: float
    confidence: float
    profile: np.ndarray
    opening_mask: np.ndarray
    plane_inlier_ratio: float


def _stamp_nanoseconds(message) -> int:
    return int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)


def _normalise(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length < 1.0e-9:
        raise ValueError("Cannot normalise a zero-length vector.")
    return vector / length


def _rotation_matrix_to_quaternion(matrix: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            qw = (matrix[2, 1] - matrix[1, 2]) / scale
            qx = 0.25 * scale
            qy = (matrix[0, 1] + matrix[1, 0]) / scale
            qz = (matrix[0, 2] + matrix[2, 0]) / scale
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            qw = (matrix[0, 2] - matrix[2, 0]) / scale
            qx = (matrix[0, 1] + matrix[1, 0]) / scale
            qy = 0.25 * scale
            qz = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            qw = (matrix[1, 0] - matrix[0, 1]) / scale
            qx = (matrix[0, 2] + matrix[2, 0]) / scale
            qy = (matrix[1, 2] + matrix[2, 1]) / scale
            qz = 0.25 * scale
    quaternion = np.array([qx, qy, qz, qw], dtype=np.float64)
    quaternion /= max(float(np.linalg.norm(quaternion)), 1.0e-12)
    return tuple(float(value) for value in quaternion)


class RgbdSlotDetector(Node):
    def __init__(self):
        super().__init__("rgbd_slot_detector")

        self.declare_parameter("image_topic", "/sim_camera/color/image_raw")
        self.declare_parameter("depth_topic", "/sim_camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/sim_camera/color/camera_info")
        self.declare_parameter("debug_image_topic", "/slot_detector/debug_image")
        self.declare_parameter("roi_x_min", 0.12)
        self.declare_parameter("roi_x_max", 0.88)
        self.declare_parameter("roi_y_min", 0.18)
        self.declare_parameter("roi_y_max", 0.84)
        self.declare_parameter("minimum_depth_m", 0.15)
        self.declare_parameter("maximum_depth_m", 1.50)
        self.declare_parameter("front_plane_inlier_m", 0.012)
        self.declare_parameter("opening_depth_m", 0.025)
        self.declare_parameter("opening_ratio", 0.48)
        self.declare_parameter("use_missing_depth_as_opening", True)
        self.declare_parameter("boundary_front_tolerance_m", 0.018)
        self.declare_parameter("boundary_front_ratio", 0.50)
        self.declare_parameter("boundary_search_px", 32)
        self.declare_parameter("rgb_edge_refine_px", 14)
        self.declare_parameter("minimum_rgb_edge_strength", 12.0)
        self.declare_parameter("boundary_line_search_px", 24)
        self.declare_parameter("rgb_line_refine_px", 6)
        self.declare_parameter("maximum_boundary_slope", 0.25)
        self.declare_parameter("minimum_slot_width_m", 0.020)
        self.declare_parameter("maximum_slot_width_m", 0.090)
        self.declare_parameter("minimum_slot_width_px", 12)
        self.declare_parameter("sync_tolerance_s", 0.025)
        self.declare_parameter("history_length", 5)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self.debug_image_topic = str(self.get_parameter("debug_image_topic").value)

        self.bridge = CvBridge()
        self.camera_info = None
        self.latest_rgb = None
        self.latest_rgb_message = None
        self.latest_depth = None
        self.latest_depth_message = None
        self.last_processed_pair = None
        self.last_log_time = 0.0
        self.consecutive_failures = 0

        history_length = max(int(self.get_parameter("history_length").value), 1)
        self.detection_history = deque(maxlen=history_length)

        self.pose_publisher = self.create_publisher(PoseStamped, "/slot_detector/slot_pose", 10)
        self.left_publisher = self.create_publisher(PointStamped, "/slot_detector/left_boundary", 10)
        self.right_publisher = self.create_publisher(PointStamped, "/slot_detector/right_boundary", 10)
        self.width_publisher = self.create_publisher(Float32, "/slot_detector/slot_width", 10)
        self.confidence_publisher = self.create_publisher(Float32, "/slot_detector/confidence", 10)
        self.debug_publisher = self.create_publisher(Image, self.debug_image_topic, qos_profile_sensor_data)

        self.create_subscription(Image, self.image_topic, self._rgb_callback, qos_profile_sensor_data)
        self.create_subscription(Image, self.depth_topic, self._depth_callback, qos_profile_sensor_data)
        self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self._camera_info_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info(f"RGB: {self.image_topic}")
        self.get_logger().info(f"Aligned depth: {self.depth_topic}")
        self.get_logger().info(f"CameraInfo: {self.camera_info_topic}")
        self.get_logger().info(
            "Slot frame convention: +X enters the shelf, +Z points up, and +Y completes a right-handed frame."
        )

    def _camera_info_callback(self, message: CameraInfo):
        self.camera_info = message
        self._try_process()

    def _rgb_callback(self, message: Image):
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        except CvBridgeError as error:
            self.get_logger().error(f"Could not convert RGB image: {error}")
            return
        self.latest_rgb_message = message
        self._try_process()

    def _depth_callback(self, message: Image):
        try:
            depth = self.bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")
        except CvBridgeError as error:
            self.get_logger().error(f"Could not convert depth image: {error}")
            return

        depth = np.asarray(depth)
        if message.encoding.upper() in ("16UC1", "MONO16"):
            depth = depth.astype(np.float32) * 0.001
        else:
            depth = depth.astype(np.float32)
        self.latest_depth = depth
        self.latest_depth_message = message
        self._try_process()

    def _try_process(self):
        if (
            self.camera_info is None
            or self.latest_rgb is None
            or self.latest_depth is None
            or self.latest_rgb_message is None
            or self.latest_depth_message is None
        ):
            return

        rgb_stamp = _stamp_nanoseconds(self.latest_rgb_message)
        depth_stamp = _stamp_nanoseconds(self.latest_depth_message)
        tolerance_ns = int(float(self.get_parameter("sync_tolerance_s").value) * 1.0e9)
        if abs(rgb_stamp - depth_stamp) > tolerance_ns:
            return

        pair = (rgb_stamp, depth_stamp)
        if pair == self.last_processed_pair:
            return
        self.last_processed_pair = pair

        if self.latest_rgb.shape[:2] != self.latest_depth.shape[:2]:
            self.get_logger().warning(
                f"RGB shape {self.latest_rgb.shape[:2]} does not match depth shape {self.latest_depth.shape[:2]}"
            )
            return

        detection = self._detect_slot(self.latest_depth, self.latest_rgb)
        if detection is None:
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.detection_history.maxlen:
                self.detection_history.clear()
            self.confidence_publisher.publish(Float32(data=0.0))
            debug = self._make_debug_image(self.latest_rgb, None)
            self._publish_debug(debug, self.latest_rgb_message)
            self._log_status("no valid slot candidate")
            return

        self.consecutive_failures = 0
        self.detection_history.append(detection)
        filtered = self._filtered_detection(detection)
        self._publish_detection(filtered, self.latest_rgb_message)
        debug = self._make_debug_image(self.latest_rgb, filtered)
        self._publish_debug(debug, self.latest_rgb_message)
        self._log_status(
            f"width={filtered.slot_width * 1000.0:.1f} mm, "
            f"confidence={filtered.confidence:.2f}, "
            f"pixels=({filtered.left_pixel}, {filtered.right_pixel})"
        )

    def _fit_front_plane(
        self,
        depth: np.ndarray,
        valid: np.ndarray,
        x0: int,
        y0: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ):
        rows, columns = np.nonzero(valid)
        values = depth[rows, columns]
        if values.size < 500:
            return None

        near_limit = float(np.percentile(values, 65.0)) + 0.008
        front_seed = values <= near_limit
        rows = rows[front_seed]
        columns = columns[front_seed]
        values = values[front_seed]
        if values.size < 300:
            return None

        if values.size > 25000:
            indices = np.linspace(0, values.size - 1, 25000, dtype=np.int64)
            rows = rows[indices]
            columns = columns[indices]
            values = values[indices]

        u = columns.astype(np.float64) + float(x0)
        v = rows.astype(np.float64) + float(y0)
        points = np.column_stack(
            (
                (u - cx) * values / fx,
                (v - cy) * values / fy,
                values,
            )
        )

        inlier_threshold = float(self.get_parameter("front_plane_inlier_m").value)
        inliers = np.ones(points.shape[0], dtype=bool)
        normal = None
        offset = None
        for _ in range(4):
            selected = points[inliers]
            if selected.shape[0] < 200:
                return None
            centroid = selected.mean(axis=0)
            _, _, vh = np.linalg.svd(selected - centroid, full_matrices=False)
            normal = _normalise(vh[-1])
            if normal[2] < 0.0:
                normal = -normal
            offset = -float(np.dot(normal, centroid))
            distances = np.abs(points @ normal + offset)
            inliers = distances < inlier_threshold

        return normal, float(offset), float(np.mean(inliers))

    @staticmethod
    def _plane_depth(
        normal: np.ndarray,
        offset: float,
        u: np.ndarray,
        v: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ) -> np.ndarray:
        ray_x = (u - cx) / fx
        ray_y = (v - cy) / fy
        denominator = normal[0] * ray_x + normal[1] * ray_y + normal[2]
        depth = np.full(np.broadcast(ray_x, ray_y).shape, np.nan, dtype=np.float64)
        usable = np.abs(denominator) > 1.0e-8
        depth[usable] = -offset / denominator[usable]
        return depth

    @staticmethod
    def _robust_line_fit(rows: np.ndarray, columns: np.ndarray) -> np.ndarray | None:
        if rows.size < 30:
            return None
        coefficients = np.polyfit(rows, columns, 1)
        for _ in range(5):
            residual = columns - np.polyval(coefficients, rows)
            median = float(np.median(residual))
            mad = float(np.median(np.abs(residual - median)))
            tolerance = max(2.0, 2.5 * 1.4826 * mad)
            inliers = np.abs(residual - median) <= tolerance
            if int(np.count_nonzero(inliers)) < 30:
                break
            rows = rows[inliers]
            columns = columns[inliers]
            coefficients = np.polyfit(rows, columns, 1)
        return np.asarray(coefficients, dtype=np.float64)

    def _fit_boundary_lines(
        self,
        opening_mask: np.ndarray,
        rgb: np.ndarray | None,
        x0: int,
        y0: int,
        left_guess: int,
        right_guess: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        center_local = int(round(0.5 * (left_guess + right_guess))) - x0
        line_search = max(int(self.get_parameter("boundary_line_search_px").value), 1)
        minimum_width_px = int(self.get_parameter("minimum_slot_width_px").value)
        rows = []
        left_columns = []
        right_columns = []

        for row_index, row in enumerate(opening_mask):
            if center_local < 0 or center_local >= row.size or not row[center_local]:
                continue
            left_local = center_local
            right_local = center_local
            while left_local > 0 and row[left_local - 1]:
                left_local -= 1
            while right_local + 1 < row.size and row[right_local + 1]:
                right_local += 1
            left_column = x0 + left_local
            right_column = x0 + right_local
            if right_column - left_column + 1 < minimum_width_px:
                continue
            if abs(left_column - left_guess) > line_search:
                continue
            if abs(right_column - right_guess) > line_search:
                continue
            rows.append(y0 + row_index)
            left_columns.append(left_column)
            right_columns.append(right_column)

        row_values = np.asarray(rows, dtype=np.float64)
        left_line = self._robust_line_fit(row_values, np.asarray(left_columns, dtype=np.float64))
        right_line = self._robust_line_fit(row_values, np.asarray(right_columns, dtype=np.float64))
        if left_line is None:
            left_line = np.array([0.0, float(left_guess)], dtype=np.float64)
        if right_line is None:
            right_line = np.array([0.0, float(right_guess)], dtype=np.float64)

        maximum_slope = float(self.get_parameter("maximum_boundary_slope").value)
        if abs(float(left_line[0])) > maximum_slope:
            left_line = np.array([0.0, float(left_guess)], dtype=np.float64)
        if abs(float(right_line[0])) > maximum_slope:
            right_line = np.array([0.0, float(right_guess)], dtype=np.float64)

        if rgb is None:
            return left_line, right_line

        gray = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY), (3, 3), 0.0)
        horizontal_gradient = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
        rgb_radius = max(int(self.get_parameter("rgb_line_refine_px").value), 0)
        minimum_strength = float(self.get_parameter("minimum_rgb_edge_strength").value)
        image_height, image_width = gray.shape

        def refine_with_rgb(line: np.ndarray) -> np.ndarray:
            if rgb_radius == 0:
                return line
            edge_rows = []
            edge_columns = []
            for row in range(max(y0, 0), min(y0 + opening_mask.shape[0], image_height)):
                predicted = int(round(float(np.polyval(line, row))))
                start = max(0, predicted - rgb_radius)
                end = min(image_width - 1, predicted + rgb_radius)
                scores = horizontal_gradient[row, start : end + 1]
                if scores.size == 0:
                    continue
                column = start + int(np.argmax(scores))
                if float(horizontal_gradient[row, column]) < minimum_strength:
                    continue
                edge_rows.append(row)
                edge_columns.append(column)
            refined = self._robust_line_fit(
                np.asarray(edge_rows, dtype=np.float64),
                np.asarray(edge_columns, dtype=np.float64),
            )
            if refined is None or abs(float(refined[0])) > maximum_slope:
                return line
            return refined

        left_line = refine_with_rgb(left_line)
        right_line = refine_with_rgb(right_line)

        # The column-level RGB/depth fusion provides the most stable boundary
        # positions at the reference height. Preserve those anchors while
        # using the row-wise fits only to recover each edge's perspective
        # slope; otherwise depth erosion can pull a fitted edge into the gap.
        center_row = y0 + 0.5 * (opening_mask.shape[0] - 1)
        left_line[1] += float(left_guess) - float(np.polyval(left_line, center_row))
        right_line[1] += float(right_guess) - float(np.polyval(right_line, center_row))
        return left_line, right_line

    def _detect_slot(self, depth: np.ndarray, rgb: np.ndarray | None = None):
        height, width = depth.shape
        fx = float(self.camera_info.k[0])
        fy = float(self.camera_info.k[4])
        cx = float(self.camera_info.k[2])
        cy = float(self.camera_info.k[5])
        if fx <= 0.0 or fy <= 0.0:
            return None

        x0 = int(round(float(self.get_parameter("roi_x_min").value) * width))
        x1 = int(round(float(self.get_parameter("roi_x_max").value) * width))
        y0 = int(round(float(self.get_parameter("roi_y_min").value) * height))
        y1 = int(round(float(self.get_parameter("roi_y_max").value) * height))
        x0, x1 = np.clip((x0, x1), 0, width)
        y0, y1 = np.clip((y0, y1), 0, height)
        if x1 - x0 < 40 or y1 - y0 < 40:
            return None

        roi_depth = depth[y0:y1, x0:x1]
        minimum_depth = float(self.get_parameter("minimum_depth_m").value)
        maximum_depth = float(self.get_parameter("maximum_depth_m").value)
        valid = np.isfinite(roi_depth) & (roi_depth > minimum_depth) & (roi_depth < maximum_depth)
        valid_coverage = float(np.mean(valid))
        if valid_coverage < 0.35:
            return None

        plane = self._fit_front_plane(roi_depth, valid, x0, y0, fx, fy, cx, cy)
        if plane is None:
            return None
        normal, offset, plane_inlier_ratio = plane

        roi_u, roi_v = np.meshgrid(np.arange(x0, x1, dtype=np.float64), np.arange(y0, y1, dtype=np.float64))
        expected_depth = self._plane_depth(normal, offset, roi_u, roi_v, fx, fy, cx, cy)
        depth_residual = roi_depth - expected_depth
        opening_depth = float(self.get_parameter("opening_depth_m").value)
        recessed = valid & (depth_residual > opening_depth)
        if bool(self.get_parameter("use_missing_depth_as_opening").value):
            # RealSense stereo often returns zero depth inside dark, textureless
            # openings. Within the fitted shelf plane, those holes are useful
            # opening evidence rather than automatic rejection.
            missing_depth = ~np.isfinite(roi_depth) | (roi_depth <= 0.0)
            recessed |= missing_depth
        opening_mask = np.isfinite(expected_depth) & recessed

        profile = opening_mask.mean(axis=0).astype(np.float64)
        kernel_size = 9
        kernel = np.ones(kernel_size, dtype=np.float64) / float(kernel_size)
        smooth_profile = np.convolve(profile, kernel, mode="same")
        front_tolerance = float(self.get_parameter("boundary_front_tolerance_m").value)
        front_profile = (valid & (np.abs(depth_residual) <= front_tolerance)).mean(axis=0).astype(np.float64)
        front_profile = np.convolve(front_profile, np.ones(5, dtype=np.float64) / 5.0, mode="same")

        rgb_edge_profile = None
        if rgb is not None and rgb.shape[:2] == depth.shape:
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            horizontal_gradient = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
            rgb_edge_profile = np.median(horizontal_gradient[y0:y1], axis=0)
            rgb_edge_profile = np.convolve(
                rgb_edge_profile,
                np.ones(5, dtype=np.float64) / 5.0,
                mode="same",
            )

        threshold = float(self.get_parameter("opening_ratio").value)
        active = smooth_profile >= threshold

        runs = []
        start = None
        for index, enabled in enumerate(active):
            if enabled and start is None:
                start = index
            if start is not None and (not enabled or index == active.size - 1):
                end = index if enabled and index == active.size - 1 else index - 1
                runs.append((start, end))
                start = None

        minimum_width_px = int(self.get_parameter("minimum_slot_width_px").value)
        image_center = 0.5 * (width - 1)
        candidates = []
        for start, end in runs:
            run_width = end - start + 1
            if run_width < minimum_width_px:
                continue
            global_center = x0 + 0.5 * (start + end)
            centre_score = math.exp(-abs(global_center - image_center) / max(0.25 * width, 1.0))
            opening_score = float(np.mean(smooth_profile[start : end + 1]))
            candidates.append((opening_score * centre_score, start, end))
        if not candidates:
            return None

        center_v = 0.5 * (y0 + y1 - 1)
        minimum_width = float(self.get_parameter("minimum_slot_width_m").value)
        maximum_width = float(self.get_parameter("maximum_slot_width_m").value)
        refine_radius = kernel_size
        raw_threshold = 0.85 * threshold
        boundary_search = max(int(self.get_parameter("boundary_search_px").value), 0)
        boundary_front_ratio = float(self.get_parameter("boundary_front_ratio").value)
        rgb_edge_refine = max(int(self.get_parameter("rgb_edge_refine_px").value), 0)
        minimum_rgb_edge = float(self.get_parameter("minimum_rgb_edge_strength").value)
        selected = None

        # A central visual gap can contain only a narrow strip of valid depth.
        # Check every ranked candidate geometrically instead of returning as
        # soon as the highest-scoring candidate fails the metric-width gate.
        for _, candidate_left, candidate_right in sorted(candidates, key=lambda item: item[0], reverse=True):
            left_local = candidate_left
            right_local = candidate_right
            refine_start = max(0, left_local - refine_radius)
            refine_end = min(profile.size - 1, right_local + refine_radius)
            supported = np.flatnonzero(profile[refine_start : refine_end + 1] >= raw_threshold)
            if supported.size:
                left_local = refine_start + int(supported[0])
                right_local = refine_start + int(supported[-1])

            # Expand the strong recessed core to the transitions where the
            # neighboring book faces stop supporting the fitted front plane.
            left_search_start = max(0, left_local - boundary_search)
            left_front = np.flatnonzero(
                front_profile[left_search_start:left_local] >= boundary_front_ratio
            )
            if left_front.size:
                left_local = left_search_start + int(left_front[-1]) + 1

            right_search_end = min(profile.size, right_local + 1 + boundary_search)
            right_front = np.flatnonzero(
                front_profile[right_local + 1 : right_search_end] >= boundary_front_ratio
            )
            if right_front.size:
                right_local = right_local + int(right_front[0])

            # Depth establishes which gap is valid; RGB then supplies the
            # sharper local book edge when stereo depth erodes near occlusions.
            if rgb_edge_profile is not None and rgb_edge_refine > 0:
                left_guess = x0 + left_local
                left_edge_start = max(x0, left_guess - rgb_edge_refine)
                left_edge_end = min(x1 - 1, left_guess + rgb_edge_refine)
                left_scores = rgb_edge_profile[left_edge_start : left_edge_end + 1]
                if left_scores.size:
                    best_left = left_edge_start + int(np.argmax(left_scores))
                    if rgb_edge_profile[best_left] >= minimum_rgb_edge:
                        left_local = best_left - x0

                right_guess = x0 + right_local
                right_edge_start = max(x0, right_guess - rgb_edge_refine)
                right_edge_end = min(x1 - 1, right_guess + rgb_edge_refine)
                right_scores = rgb_edge_profile[right_edge_start : right_edge_end + 1]
                if right_scores.size:
                    best_right = right_edge_start + int(np.argmax(right_scores))
                    if rgb_edge_profile[best_right] >= minimum_rgb_edge:
                        right_local = best_right - x0

            if right_local <= left_local:
                continue

            left_pixel = x0 + left_local
            right_pixel = x0 + right_local
            left_line, right_line = self._fit_boundary_lines(
                opening_mask,
                rgb,
                x0,
                y0,
                left_pixel,
                right_pixel,
            )
            left_u = float(np.polyval(left_line, center_v))
            right_u = float(np.polyval(right_line, center_v))
            if right_u <= left_u:
                continue
            center_u = 0.5 * (left_u + right_u)
            left_pixel = int(round(left_u))
            right_pixel = int(round(right_u))
            center_pixel = int(round(center_u))

            sample_u = np.array([left_u, right_u, center_u], dtype=np.float64)
            sample_v = np.full(3, center_v, dtype=np.float64)
            sample_depth = self._plane_depth(normal, offset, sample_u, sample_v, fx, fy, cx, cy)
            if not np.all(np.isfinite(sample_depth)):
                continue
            points = np.column_stack(
                (
                    (sample_u - cx) * sample_depth / fx,
                    (sample_v - cy) * sample_depth / fy,
                    sample_depth,
                )
            )
            left_point, right_point, center_point = points

            center_line = 0.5 * (left_line + right_line)
            vertical_v = np.array([float(y0), float(y1 - 1)], dtype=np.float64)
            vertical_u = np.polyval(center_line, vertical_v)
            vertical_depth = self._plane_depth(normal, offset, vertical_u, vertical_v, fx, fy, cx, cy)
            if not np.all(np.isfinite(vertical_depth)):
                continue
            vertical_points = np.column_stack(
                (
                    (vertical_u - cx) * vertical_depth / fx,
                    (vertical_v - cy) * vertical_depth / fy,
                    vertical_depth,
                )
            )
            vertical_up = _normalise(vertical_points[0] - vertical_points[1])
            lateral_axis = _normalise(np.cross(vertical_up, normal))
            slot_width = abs(float(np.dot(right_point - left_point, lateral_axis)))
            if minimum_width <= slot_width <= maximum_width:
                selected = (
                    left_local,
                    right_local,
                    left_pixel,
                    right_pixel,
                    center_pixel,
                    left_line,
                    right_line,
                    left_point,
                    right_point,
                    center_point,
                    vertical_up,
                    slot_width,
                )
                break

        if selected is None:
            return None
        (
            left_local,
            right_local,
            left_pixel,
            right_pixel,
            center_pixel,
            left_line,
            right_line,
            left_point,
            right_point,
            center_point,
            vertical_up,
            slot_width,
        ) = selected

        inside_strength = float(np.mean(smooth_profile[left_local : right_local + 1]))
        side_span = max(right_local - left_local + 1, 8)
        left_side = smooth_profile[max(0, left_local - side_span) : left_local]
        right_side = smooth_profile[right_local + 1 : min(smooth_profile.size, right_local + 1 + side_span)]
        outside_values = np.concatenate((left_side, right_side))
        outside_strength = float(np.mean(outside_values)) if outside_values.size else 1.0
        edge_contrast = float(np.clip(inside_strength - outside_strength, 0.0, 1.0))
        width_score = float(
            np.clip(
                min(slot_width - minimum_width, maximum_width - slot_width)
                / max(0.25 * (maximum_width - minimum_width), 1.0e-6),
                0.0,
                1.0,
            )
        )
        confidence = (
            0.25 * np.clip(plane_inlier_ratio / 0.75, 0.0, 1.0)
            + 0.30 * np.clip(inside_strength, 0.0, 1.0)
            + 0.20 * edge_contrast
            + 0.15 * np.clip(valid_coverage / 0.85, 0.0, 1.0)
            + 0.10 * width_score
        )

        return SlotDetection(
            left_pixel=left_pixel,
            right_pixel=right_pixel,
            center_pixel=center_pixel,
            left_line=left_line,
            right_line=right_line,
            left_point=left_point,
            right_point=right_point,
            center_point=center_point,
            plane_normal=normal,
            vertical_axis=vertical_up,
            slot_width=slot_width,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            profile=smooth_profile,
            opening_mask=opening_mask,
            plane_inlier_ratio=plane_inlier_ratio,
        )

    def _filtered_detection(self, current: SlotDetection) -> SlotDetection:
        history = list(self.detection_history)
        center_point = np.median(np.stack([item.center_point for item in history]), axis=0)
        left_point = np.median(np.stack([item.left_point for item in history]), axis=0)
        right_point = np.median(np.stack([item.right_point for item in history]), axis=0)
        normal = _normalise(np.mean(np.stack([item.plane_normal for item in history]), axis=0))
        vertical_axis = _normalise(np.mean(np.stack([item.vertical_axis for item in history]), axis=0))
        left_line = np.median(np.stack([item.left_line for item in history]), axis=0)
        right_line = np.median(np.stack([item.right_line for item in history]), axis=0)
        slot_width = float(np.median([item.slot_width for item in history]))

        temporal_score = 1.0
        if len(history) > 1:
            centres = np.stack([item.center_point for item in history])
            centre_jitter = float(np.linalg.norm(np.std(centres, axis=0)))
            width_jitter = float(np.std([item.slot_width for item in history]))
            temporal_score = math.exp(-centre_jitter / 0.004 - width_jitter / 0.003)
        confidence = float(np.clip(0.85 * current.confidence + 0.15 * temporal_score, 0.0, 1.0))

        return SlotDetection(
            left_pixel=int(round(np.median([item.left_pixel for item in history]))),
            right_pixel=int(round(np.median([item.right_pixel for item in history]))),
            center_pixel=int(round(np.median([item.center_pixel for item in history]))),
            left_line=left_line,
            right_line=right_line,
            left_point=left_point,
            right_point=right_point,
            center_point=center_point,
            plane_normal=normal,
            vertical_axis=vertical_axis,
            slot_width=slot_width,
            confidence=confidence,
            profile=current.profile,
            opening_mask=current.opening_mask,
            plane_inlier_ratio=current.plane_inlier_ratio,
        )

    def _publish_detection(self, detection: SlotDetection, source_message: Image):
        header = source_message.header
        pose = PoseStamped()
        pose.header = header
        pose.pose.position.x = float(detection.center_point[0])
        pose.pose.position.y = float(detection.center_point[1])
        pose.pose.position.z = float(detection.center_point[2])

        insertion_axis = _normalise(detection.plane_normal)
        up_axis = detection.vertical_axis.copy()
        up_axis -= insertion_axis * float(np.dot(up_axis, insertion_axis))
        up_axis = _normalise(up_axis)
        lateral_axis = _normalise(np.cross(up_axis, insertion_axis))
        rotation = np.column_stack((insertion_axis, lateral_axis, up_axis))
        qx, qy, qz, qw = _rotation_matrix_to_quaternion(rotation)
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        left = PointStamped()
        left.header = header
        left.point.x, left.point.y, left.point.z = (float(value) for value in detection.left_point)
        right = PointStamped()
        right.header = header
        right.point.x, right.point.y, right.point.z = (float(value) for value in detection.right_point)

        self.pose_publisher.publish(pose)
        self.left_publisher.publish(left)
        self.right_publisher.publish(right)
        self.width_publisher.publish(Float32(data=float(detection.slot_width)))
        self.confidence_publisher.publish(Float32(data=float(detection.confidence)))

    def _make_debug_image(self, rgb: np.ndarray, detection: SlotDetection | None) -> np.ndarray:
        debug = rgb.copy()
        height, width = debug.shape[:2]
        x0 = int(round(float(self.get_parameter("roi_x_min").value) * width))
        x1 = int(round(float(self.get_parameter("roi_x_max").value) * width))
        y0 = int(round(float(self.get_parameter("roi_y_min").value) * height))
        y1 = int(round(float(self.get_parameter("roi_y_max").value) * height))
        cv2.rectangle(debug, (x0, y0), (x1, y1), (255, 180, 0), 2)

        if detection is None:
            cv2.putText(
                debug,
                "NO VALID SLOT",
                (16, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            return debug

        mask = np.zeros_like(debug)
        mask_roi = detection.opening_mask.astype(np.uint8) * 150
        mask[y0:y1, x0:x1, 2] = mask_roi
        debug = cv2.addWeighted(debug, 1.0, mask, 0.35, 0.0)

        def image_line(coefficients: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
            top_x = int(round(float(np.polyval(coefficients, y0))))
            bottom_x = int(round(float(np.polyval(coefficients, y1))))
            return (
                (int(np.clip(top_x, 0, width - 1)), y0),
                (int(np.clip(bottom_x, 0, width - 1)), y1),
            )

        left_top, left_bottom = image_line(detection.left_line)
        right_top, right_bottom = image_line(detection.right_line)
        center_top, center_bottom = image_line(0.5 * (detection.left_line + detection.right_line))
        cv2.line(debug, left_top, left_bottom, (0, 255, 0), 3)
        cv2.line(debug, right_top, right_bottom, (0, 255, 0), 3)
        cv2.line(debug, center_top, center_bottom, (255, 255, 0), 2)
        cv2.putText(
            debug,
            f"slot={detection.slot_width * 1000.0:.1f} mm  confidence={detection.confidence:.2f}",
            (16, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (20, 255, 20),
            2,
            cv2.LINE_AA,
        )
        return debug

    def _publish_debug(self, debug: np.ndarray, source_message: Image):
        message = self.bridge.cv2_to_imgmsg(debug, encoding="bgr8")
        message.header = source_message.header
        self.debug_publisher.publish(message)

    def _log_status(self, text: str):
        now = time.monotonic()
        if now - self.last_log_time >= 1.0:
            self.get_logger().info(text)
            self.last_log_time = now


def main(args=None):
    rclpy.init(args=args)
    node = RgbdSlotDetector()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
