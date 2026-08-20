#!/usr/bin/env python3
"""Estimate a fixed end-effector-to-book transform from a recorded ArUco view."""

from __future__ import annotations

import csv
from datetime import datetime
import json
import math
from pathlib import Path
import time

import cv2
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point, TransformStamped
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
import tf2_ros
from visualization_msgs.msg import Marker, MarkerArray
import yaml

from .marker_book_calibration import (
    CalibrationSample,
    MarkerBookCalibrationAccumulator,
    compose_eef_book_transform,
    make_book_marker_transform,
)
from .book_frame_audit import (
    apply_book_axis_correction,
    book_axis_correction_transform,
    book_frame_audit_report,
    expected_policy_book_rotation_in_eef,
)
from .policy_observation_math import (
    invert_transform,
    make_transform,
    matrix_to_quaternion_xyzw,
)


def _stamp_nanoseconds(message) -> int:
    return int(message.header.stamp.sec) * 1_000_000_000 + int(
        message.header.stamp.nanosec
    )


def _transform_message_to_matrix(message) -> np.ndarray:
    value = message.transform
    return make_transform(
        [value.translation.x, value.translation.y, value.translation.z],
        [value.rotation.x, value.rotation.y, value.rotation.z, value.rotation.w],
    )


def _marker_object_points(marker_size_m: float) -> np.ndarray:
    half = 0.5 * float(marker_size_m)
    return np.array(
        [
            [-half, +half, 0.0],
            [+half, +half, 0.0],
            [+half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def _load_mount(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Marker mount YAML does not exist: {path}")
    with path.open("r", encoding="utf-8") as stream:
        mount = yaml.safe_load(stream)
    if not isinstance(mount, dict):
        raise ValueError("Marker mount YAML must contain a mapping.")

    required = (
        "dictionary",
        "marker_id",
        "marker_black_size_m",
        "book_size_xyz_m",
        "marker_center_in_book_m",
    )
    missing = [key for key in required if key not in mount]
    if missing:
        raise ValueError(f"Marker mount YAML is missing: {', '.join(missing)}")

    center = mount["marker_center_in_book_m"]
    if not isinstance(center, dict) or any(axis not in center for axis in "xyz"):
        raise ValueError("marker_center_in_book_m must contain x, y, and z.")
    mount["marker_center_in_book_xyz_m"] = [
        float(center["x"]),
        float(center["y"]),
        float(center["z"]),
    ]
    mount["marker_id"] = int(mount["marker_id"])
    mount["marker_black_size_m"] = float(mount["marker_black_size_m"])
    mount["book_size_xyz_m"] = [float(value) for value in mount["book_size_xyz_m"]]
    if mount["marker_black_size_m"] <= 0.0:
        raise ValueError("marker_black_size_m must be positive.")
    if len(mount["book_size_xyz_m"]) != 3 or min(mount["book_size_xyz_m"]) <= 0.0:
        raise ValueError("book_size_xyz_m must contain three positive dimensions.")
    return mount


def _dictionary_from_name(name: str):
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"OpenCV ArUco dictionary is unavailable: {name}")
    return cv2.aruco.getPredefinedDictionary(int(getattr(cv2.aruco, name)))


class MarkerBookCalibrationNode(Node):
    """Subscriber-only marker calibration node intended for rosbag replay."""

    def __init__(self):
        super().__init__("marker_book_calibration")
        self._declare_parameters()

        mount_path = Path(str(self.get_parameter("mount_yaml").value)).expanduser()
        self.mount_path = mount_path.resolve()
        self.mount = _load_mount(self.mount_path)
        self.marker_id = self.mount["marker_id"]
        self.marker_size_m = self.mount["marker_black_size_m"]
        self.book_size_xyz_m = np.asarray(
            self.mount["book_size_xyz_m"], dtype=np.float64
        )
        self.frame_audit_enabled = bool(
            self.get_parameter("enable_frame_audit").value
        )
        self.transform_old_book_policy_book = book_axis_correction_transform(
            self.get_parameter("frame_audit_correction_quaternion_xyzw").value
        )
        self.expected_policy_book_rotation_eef = (
            expected_policy_book_rotation_in_eef(
                self.get_parameter(
                    "frame_audit_expected_quaternion_eef_policy_book_xyzw"
                ).value
            )
        )
        rotation_book_marker = self.mount.get("rotation_book_marker")
        if rotation_book_marker is None:
            self.transform_book_marker = make_book_marker_transform(
                self.mount["marker_center_in_book_xyz_m"]
            )
        else:
            self.transform_book_marker = make_book_marker_transform(
                self.mount["marker_center_in_book_xyz_m"],
                rotation_book_marker,
            )

        self.dictionary = _dictionary_from_name(str(self.mount["dictionary"]))
        if hasattr(cv2.aruco, "DetectorParameters_create"):
            self.detector_parameters = cv2.aruco.DetectorParameters_create()
        else:
            self.detector_parameters = cv2.aruco.DetectorParameters()
        self.detector_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.object_points = _marker_object_points(self.marker_size_m)
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=60.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.detected_tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        visualization_qos = QoSProfile(depth=2)
        visualization_qos.reliability = ReliabilityPolicy.RELIABLE
        visualization_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.book_visualization_publisher = self.create_publisher(
            MarkerArray,
            str(self.get_parameter("visualization_topic").value),
            visualization_qos,
        )

        self.camera_info = None
        self.latest_depth = None
        self.latest_depth_stamp_ns = None
        self.transform_eef_camera = None
        self.camera_frame = None
        self.frame_index = 0
        self.last_image_wall_time = None
        self.last_processed_stamp_ns = None
        self.completed = False
        self.report_written = False
        self.counters = {
            "rgb_frames": 0,
            "marker_not_found": 0,
            "pose_rejected": 0,
            "depth_rejected": 0,
            "tf_unavailable": 0,
            "accepted_samples": 0,
        }
        self.accumulator = MarkerBookCalibrationAccumulator(
            maximum_translation_deviation_m=float(
                self.get_parameter("maximum_translation_deviation_m").value
            ),
            maximum_rotation_deviation_deg=float(
                self.get_parameter("maximum_rotation_deviation_deg").value
            ),
        )

        self.create_subscription(
            CameraInfo,
            str(self.get_parameter("camera_info_topic").value),
            self._camera_info_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter("depth_topic").value),
            self._depth_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter("image_topic").value),
            self._image_callback,
            qos_profile_sensor_data,
        )
        self.create_timer(0.50, self._idle_timer_callback)

        self.get_logger().info(
            "Marker-to-book calibration started in BAG-ONLY, READ-ONLY mode."
        )
        self.get_logger().info(
            f"Using {self.mount['dictionary']} id={self.marker_id}, "
            f"black size={self.marker_size_m * 1000.0:.1f} mm"
        )
        self.get_logger().info(
            "No action, IK, trajectory, gripper, or robot-command interface is created."
        )
        if self.frame_audit_enabled:
            self.get_logger().warning(
                "Book-frame audit candidates are enabled for visualization only; "
                "selection_authorized=false."
            )

    def _declare_parameters(self):
        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter(
            "depth_topic", "/camera/aligned_depth_to_color/image_raw"
        )
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("mount_yaml", "")
        self.declare_parameter(
            "output_dir", "/tmp/bookshelf_marker_book_calibration"
        )
        self.declare_parameter("eef_frame", "link_eef")
        self.declare_parameter("camera_frame", "")
        self.declare_parameter("target_samples", 250)
        self.declare_parameter("enable_frame_audit", False)
        self.declare_parameter(
            "frame_audit_correction_quaternion_xyzw",
            [0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)],
        )
        self.declare_parameter(
            "frame_audit_expected_quaternion_eef_policy_book_xyzw",
            [math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0],
        )
        self.declare_parameter("minimum_samples", 30)
        self.declare_parameter("minimum_inlier_fraction", 0.70)
        self.declare_parameter("maximum_reprojection_error_px", 3.0)
        self.declare_parameter("maximum_depth_error_m", 0.030)
        self.declare_parameter("sync_tolerance_s", 0.050)
        self.declare_parameter("tf_lookup_timeout_s", 0.10)
        self.declare_parameter("maximum_translation_deviation_m", 0.010)
        self.declare_parameter("maximum_rotation_deviation_deg", 5.0)
        self.declare_parameter("debug_stride", 50)
        self.declare_parameter("idle_finalize_s", 2.0)
        self.declare_parameter(
            "visualization_topic", "/bookshelf_policy/book_boxes"
        )
        self.declare_parameter("detected_marker_frame", "calibration_aruco0_marker")
        self.declare_parameter("detected_book_frame", "calibration_detected_book")
        self.declare_parameter(
            "frame_audit_candidate_frame", "calibration_policy_book_candidate"
        )
        # Long enough to bridge ordinary message jitter while still removing a
        # stale visualization promptly after real detection loss.
        self.declare_parameter("visualization_lifetime_s", 1.0)

    def _camera_info_callback(self, message: CameraInfo):
        self.camera_info = message

    def _depth_callback(self, message: Image):
        try:
            depth = self.bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")
        except CvBridgeError as error:
            self.get_logger().warning(f"Could not convert depth image: {error}")
            return
        depth = np.asarray(depth)
        if message.encoding.upper() in ("16UC1", "MONO16"):
            depth = depth.astype(np.float32) * 0.001
        else:
            depth = depth.astype(np.float32)
        self.latest_depth = depth
        self.latest_depth_stamp_ns = _stamp_nanoseconds(message)

    def _image_callback(self, message: Image):
        self.last_image_wall_time = time.monotonic()
        stamp_ns = _stamp_nanoseconds(message)
        if stamp_ns == self.last_processed_stamp_ns:
            return
        self.last_processed_stamp_ns = stamp_ns
        self.frame_index += 1
        self.counters["rgb_frames"] += 1

        if self.camera_info is None:
            return
        try:
            image = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        except CvBridgeError as error:
            self.get_logger().warning(f"Could not convert RGB image: {error}")
            return

        camera_matrix = np.asarray(self.camera_info.k, dtype=np.float64).reshape(3, 3)
        distortion = np.asarray(self.camera_info.d, dtype=np.float64)
        corners, ids, _ = cv2.aruco.detectMarkers(
            image,
            self.dictionary,
            parameters=self.detector_parameters,
        )
        marker_index = self._find_marker_index(ids)
        if marker_index is None:
            self.counters["marker_not_found"] += 1
            return

        marker_corners = np.asarray(corners[marker_index], dtype=np.float64).reshape(4, 2)
        pose = self._estimate_pose(marker_corners, camera_matrix, distortion)
        if pose is None:
            self.counters["pose_rejected"] += 1
            return
        transform_camera_marker, reprojection_error_px = pose
        maximum_reprojection = float(
            self.get_parameter("maximum_reprojection_error_px").value
        )
        if reprojection_error_px > maximum_reprojection:
            self.counters["pose_rejected"] += 1
            return

        measured_depth_m, depth_error_m = self._measure_depth(
            marker_corners,
            transform_camera_marker[2, 3],
            stamp_ns,
        )
        maximum_depth_error = float(self.get_parameter("maximum_depth_error_m").value)
        if math.isfinite(depth_error_m) and depth_error_m > maximum_depth_error:
            self.counters["depth_rejected"] += 1
            return

        camera_frame = str(self.get_parameter("camera_frame").value).strip()
        if not camera_frame:
            camera_frame = str(message.header.frame_id or self.camera_info.header.frame_id)
        try:
            transform_eef_camera_message = self.tf_buffer.lookup_transform(
                str(self.get_parameter("eef_frame").value),
                camera_frame,
                Time(),
                timeout=Duration(
                    seconds=float(self.get_parameter("tf_lookup_timeout_s").value)
                ),
            )
        except Exception as error:
            self.counters["tf_unavailable"] += 1
            if self.counters["tf_unavailable"] <= 3:
                self.get_logger().warning(
                    f"Waiting for TF {self.get_parameter('eef_frame').value} <- "
                    f"{camera_frame}: {error}"
                )
            return

        transform_eef_camera = _transform_message_to_matrix(
            transform_eef_camera_message
        )
        self.transform_eef_camera = transform_eef_camera
        self.camera_frame = camera_frame
        transform_eef_book = compose_eef_book_transform(
            transform_eef_camera,
            transform_camera_marker,
            self.transform_book_marker,
        )
        self._publish_rviz_visualization(
            message.header.stamp,
            camera_frame,
            transform_camera_marker,
            transform_eef_book,
        )
        if self.completed:
            return
        sample = CalibrationSample(
            frame_index=self.frame_index,
            stamp_ns=stamp_ns,
            reprojection_error_px=float(reprojection_error_px),
            marker_depth_m=float(measured_depth_m),
            depth_error_m=float(depth_error_m),
            transform_camera_marker=transform_camera_marker,
            transform_eef_book=transform_eef_book,
        )
        self.accumulator.add(sample)
        self.counters["accepted_samples"] += 1

        debug_stride = max(int(self.get_parameter("debug_stride").value), 1)
        if len(self.accumulator.samples) == 1 or len(self.accumulator.samples) % debug_stride == 0:
            self._save_debug_image(
                image,
                marker_corners,
                transform_camera_marker,
                camera_matrix,
                distortion,
            )

        count = len(self.accumulator.samples)
        if count % 25 == 0:
            self.get_logger().info(
                f"accepted={count}, reprojection={reprojection_error_px:.2f} px, "
                f"depth_error={depth_error_m * 1000.0:.1f} mm"
                if math.isfinite(depth_error_m)
                else f"accepted={count}, reprojection={reprojection_error_px:.2f} px, depth unavailable"
            )
        target = max(int(self.get_parameter("target_samples").value), 1)
        if count >= target:
            self._finish("target sample count reached")

    def _publish_rviz_visualization(
        self,
        stamp,
        camera_frame: str,
        transform_camera_marker: np.ndarray,
        transform_eef_book: np.ndarray,
    ):
        marker_frame = str(self.get_parameter("detected_marker_frame").value)
        book_frame = str(self.get_parameter("detected_book_frame").value)
        transform_marker_book = invert_transform(self.transform_book_marker)
        transforms = [
            self._transform_message(
                stamp,
                camera_frame,
                marker_frame,
                transform_camera_marker,
            ),
            self._transform_message(
                stamp,
                marker_frame,
                book_frame,
                transform_marker_book,
            ),
        ]
        candidate_frame = str(
            self.get_parameter("frame_audit_candidate_frame").value
        )
        if self.frame_audit_enabled:
            transform_eef_policy_book, _ = self._diagnostic_policy_book_transform(
                transform_eef_book
            )
            transforms.append(
                self._transform_message(
                    stamp,
                    str(self.get_parameter("eef_frame").value),
                    candidate_frame,
                    transform_eef_policy_book,
                )
            )
        self.detected_tf_broadcaster.sendTransform(transforms)

        lifetime_s = max(
            float(self.get_parameter("visualization_lifetime_s").value), 0.0
        )
        lifetime_sec = int(lifetime_s)
        lifetime_nanosec = int((lifetime_s % 1.0) * 1.0e9)

        book_marker = Marker()
        # Keep the marker in the same detected-book frame as the diagnostic TF.
        # A zero timestamp asks RViz for the latest transform, avoiding a race
        # between the TF and MarkerArray subscriptions.
        book_marker.header.frame_id = book_frame
        book_marker.ns = "bookshelf_books"
        book_marker.id = 0
        book_marker.type = Marker.CUBE
        book_marker.action = Marker.ADD
        book_marker.pose.orientation.w = 1.0
        book_marker.scale.x = float(self.book_size_xyz_m[0])
        book_marker.scale.y = float(self.book_size_xyz_m[1])
        book_marker.scale.z = float(self.book_size_xyz_m[2])
        book_marker.color.r = 0.0
        book_marker.color.g = 0.85
        book_marker.color.b = 1.0
        book_marker.color.a = 0.45
        book_marker.frame_locked = True
        book_marker.lifetime.sec = lifetime_sec
        book_marker.lifetime.nanosec = lifetime_nanosec

        detected_marker = Marker()
        detected_marker.header.frame_id = marker_frame
        detected_marker.ns = "bookshelf_books"
        detected_marker.id = 1
        detected_marker.type = Marker.CUBE
        detected_marker.action = Marker.ADD
        detected_marker.pose.orientation.w = 1.0
        detected_marker.scale.x = float(self.marker_size_m)
        detected_marker.scale.y = float(self.marker_size_m)
        detected_marker.scale.z = 0.001
        detected_marker.color.r = 1.0
        detected_marker.color.g = 0.1
        detected_marker.color.b = 0.8
        detected_marker.color.a = 0.85
        detected_marker.frame_locked = True
        detected_marker.lifetime.sec = lifetime_sec
        detected_marker.lifetime.nanosec = lifetime_nanosec
        markers = [book_marker, detected_marker]
        if self.frame_audit_enabled:
            candidate_book = self._book_cube_marker(
                candidate_frame,
                "bookshelf_book_frame_audit",
                10,
                (1.0, 0.75, 0.0, 0.42),
                lifetime_sec,
                lifetime_nanosec,
            )
            markers.append(candidate_book)
            markers.extend(
                self._axis_markers(
                    candidate_frame,
                    "policy_book",
                    30,
                    lifetime_sec,
                    lifetime_nanosec,
                )
            )
        self.book_visualization_publisher.publish(MarkerArray(markers=markers))

    def _book_cube_marker(
        self, frame_id, namespace, marker_id, rgba, lifetime_sec, lifetime_nanosec
    ):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = float(self.book_size_xyz_m[0])
        marker.scale.y = float(self.book_size_xyz_m[1])
        marker.scale.z = float(self.book_size_xyz_m[2])
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = rgba
        marker.frame_locked = True
        marker.lifetime.sec = lifetime_sec
        marker.lifetime.nanosec = lifetime_nanosec
        return marker

    @staticmethod
    def _axis_markers(
        frame_id, label, first_id, lifetime_sec, lifetime_nanosec
    ):
        markers = []
        axes = (
            ("X depth", (1.0, 0.0, 0.0), (1.0, 0.1, 0.1, 0.95)),
            ("Y thickness", (0.0, 1.0, 0.0), (0.1, 1.0, 0.1, 0.95)),
            ("Z up", (0.0, 0.0, 1.0), (0.2, 0.4, 1.0, 0.95)),
        )
        axis_length_m = 0.10
        for offset, (axis_name, endpoint, rgba) in enumerate(axes):
            endpoint = tuple(axis_length_m * value for value in endpoint)
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.ns = f"bookshelf_book_frame_audit_{label}"
            marker.id = first_id + offset
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.points = [
                Point(x=0.0, y=0.0, z=0.0),
                Point(x=endpoint[0], y=endpoint[1], z=endpoint[2]),
            ]
            marker.scale.x = 0.004
            marker.scale.y = 0.008
            marker.scale.z = 0.012
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = rgba
            marker.frame_locked = True
            marker.lifetime.sec = lifetime_sec
            marker.lifetime.nanosec = lifetime_nanosec
            markers.append(marker)

            text = Marker()
            text.header.frame_id = frame_id
            text.ns = f"bookshelf_book_frame_audit_{label}_labels"
            text.id = first_id + 10 + offset
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position = Point(
                x=endpoint[0], y=endpoint[1], z=endpoint[2]
            )
            text.pose.orientation.w = 1.0
            text.scale.z = 0.018
            text.color.r, text.color.g, text.color.b, text.color.a = rgba
            text.text = f"{label}: {axis_name}"
            text.frame_locked = True
            text.lifetime.sec = lifetime_sec
            text.lifetime.nanosec = lifetime_nanosec
            markers.append(text)
        return markers

    @staticmethod
    def _transform_message(stamp, parent: str, child: str, transform: np.ndarray):
        message = TransformStamped()
        message.header.stamp = stamp
        message.header.frame_id = parent
        message.child_frame_id = child
        message.transform.translation.x = float(transform[0, 3])
        message.transform.translation.y = float(transform[1, 3])
        message.transform.translation.z = float(transform[2, 3])
        quaternion = matrix_to_quaternion_xyzw(transform[:3, :3])
        message.transform.rotation.x = float(quaternion[0])
        message.transform.rotation.y = float(quaternion[1])
        message.transform.rotation.z = float(quaternion[2])
        message.transform.rotation.w = float(quaternion[3])
        return message

    def _find_marker_index(self, ids):
        if ids is None:
            return None
        matches = np.flatnonzero(np.asarray(ids).reshape(-1) == self.marker_id)
        return int(matches[0]) if matches.size else None

    def _estimate_pose(self, corners, camera_matrix, distortion):
        image_points = np.ascontiguousarray(
            np.asarray(corners, dtype=np.float64).reshape(4, 2)
        )
        result = cv2.solvePnPGeneric(
            self.object_points,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not result or not bool(result[0]):
            return None
        rvecs, tvecs = result[1], result[2]
        best = None
        for rvec, tvec in zip(rvecs, tvecs):
            rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
            tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
            if not np.all(np.isfinite(tvec)) or float(tvec[2]) <= 0.0:
                continue
            # IPPE provides the two planar-square pose branches, but its raw
            # analytic candidates can retain several pixels of reprojection
            # error at close range. Refine each branch before applying the
            # existing quality threshold; this preserves ambiguity handling
            # without rejecting a clearly detected marker on solver error.
            try:
                rvec, tvec = cv2.solvePnPRefineLM(
                    self.object_points,
                    image_points,
                    camera_matrix,
                    distortion,
                    rvec,
                    tvec,
                )
            except cv2.error:
                continue
            if not np.all(np.isfinite(tvec)) or float(tvec[2]) <= 0.0:
                continue
            projected, _ = cv2.projectPoints(
                self.object_points, rvec, tvec, camera_matrix, distortion
            )
            residual = projected.reshape(4, 2) - image_points
            error = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
            if best is None or error < best[0]:
                rotation, _ = cv2.Rodrigues(rvec)
                transform = np.eye(4, dtype=np.float64)
                transform[:3, :3] = rotation
                transform[:3, 3] = tvec.reshape(3)
                best = (error, transform)
        if best is None:
            return None
        return best[1], best[0]

    def _measure_depth(self, corners, marker_pose_depth_m, rgb_stamp_ns):
        if self.latest_depth is None or self.latest_depth_stamp_ns is None:
            return float("nan"), float("nan")
        tolerance_ns = int(float(self.get_parameter("sync_tolerance_s").value) * 1.0e9)
        if abs(self.latest_depth_stamp_ns - rgb_stamp_ns) > tolerance_ns:
            return float("nan"), float("nan")
        center = np.mean(corners, axis=0)
        column = int(round(float(center[0])))
        row = int(round(float(center[1])))
        row0, row1 = max(row - 3, 0), min(row + 4, self.latest_depth.shape[0])
        col0, col1 = max(column - 3, 0), min(column + 4, self.latest_depth.shape[1])
        patch = self.latest_depth[row0:row1, col0:col1]
        valid = patch[np.isfinite(patch) & (patch > 0.0)]
        if valid.size == 0:
            return float("nan"), float("nan")
        measured = float(np.median(valid))
        return measured, abs(measured - float(marker_pose_depth_m))

    def _save_debug_image(
        self,
        image,
        marker_corners,
        transform_camera_marker,
        camera_matrix,
        distortion,
    ):
        debug = image.copy()
        polygon = np.round(marker_corners).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(debug, [polygon], True, (0, 255, 0), 2, cv2.LINE_AA)
        rotation_vector, _ = cv2.Rodrigues(transform_camera_marker[:3, :3])
        translation_vector = transform_camera_marker[:3, 3].reshape(3, 1)
        cv2.drawFrameAxes(
            debug,
            camera_matrix,
            distortion,
            rotation_vector,
            translation_vector,
            min(self.marker_size_m * 0.75, 0.04),
            2,
        )

        transform_camera_book = transform_camera_marker @ invert_transform(
            self.transform_book_marker
        )
        self._draw_book_cuboid(
            debug,
            transform_camera_book,
            camera_matrix,
            distortion,
            color_bgr=(255, 255, 0),
        )
        cv2.putText(
            debug,
            "live policy book frame (cyan)",
            (20, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
        if self.frame_audit_enabled:
            transform_eef_book = self.transform_eef_camera @ transform_camera_book
            transform_eef_policy_book, preferred_source = (
                self._diagnostic_policy_book_transform(transform_eef_book)
            )
            transform_camera_candidate = (
                invert_transform(self.transform_eef_camera)
                @ transform_eef_policy_book
            )
            self._draw_book_cuboid(
                debug,
                transform_camera_candidate,
                camera_matrix,
                distortion,
                color_bgr=(0, 255, 255),
            )
            candidate_rotation_vector, _ = cv2.Rodrigues(
                transform_camera_candidate[:3, :3]
            )
            cv2.drawFrameAxes(
                debug,
                camera_matrix,
                distortion,
                candidate_rotation_vector,
                transform_camera_candidate[:3, 3].reshape(3, 1),
                0.08,
                2,
            )
            cv2.putText(
                debug,
                f"policy book frame (yellow; {preferred_source} axes)",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            debug,
            f"accepted sample {len(self.accumulator.samples)}",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        output_dir = self._output_dir() / "debug"
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(output_dir / f"sample_{len(self.accumulator.samples):04d}.png"),
            debug,
        )

    def _diagnostic_policy_book_transform(self, transform_eef_book):
        audit = book_frame_audit_report(
            transform_eef_book,
            transform_old_book_policy_book=self.transform_old_book_policy_book,
            expected_rotation_eef_policy_book=self.expected_policy_book_rotation_eef,
        )
        if audit["candidate_preferred"]:
            return (
                apply_book_axis_correction(
                    transform_eef_book,
                    self.transform_old_book_policy_book,
                ),
                "corrected",
            )
        return np.asarray(transform_eef_book, dtype=np.float64), "saved"

    def _draw_book_cuboid(
        self,
        image,
        transform_camera_book,
        camera_matrix,
        distortion,
        *,
        color_bgr,
    ):
        half = 0.5 * self.book_size_xyz_m
        points = np.array(
            [
                [x * half[0], y * half[1], z * half[2]]
                for x in (-1.0, 1.0)
                for y in (-1.0, 1.0)
                for z in (-1.0, 1.0)
            ],
            dtype=np.float64,
        )
        camera_points = (
            transform_camera_book[:3, :3] @ points.T
        ).T + transform_camera_book[:3, 3]
        if np.any(camera_points[:, 2] <= 0.0):
            return
        rotation_vector, _ = cv2.Rodrigues(transform_camera_book[:3, :3])
        projected, _ = cv2.projectPoints(
            points,
            rotation_vector,
            transform_camera_book[:3, 3].reshape(3, 1),
            camera_matrix,
            distortion,
        )
        projected = np.round(projected.reshape(8, 2)).astype(np.int32)
        edges = (
            (0, 1), (0, 2), (0, 4),
            (1, 3), (1, 5),
            (2, 3), (2, 6),
            (3, 7),
            (4, 5), (4, 6),
            (5, 7), (6, 7),
        )
        for first, second in edges:
            cv2.line(
                image,
                tuple(projected[first]),
                tuple(projected[second]),
                color_bgr,
                2,
                cv2.LINE_AA,
            )

    def _idle_timer_callback(self):
        if self.completed or self.last_image_wall_time is None:
            return
        idle_s = float(self.get_parameter("idle_finalize_s").value)
        if time.monotonic() - self.last_image_wall_time >= idle_s:
            self._finish("image stream became idle")

    def _finish(self, reason: str):
        if self.completed:
            return
        self.completed = True
        self._write_report(reason)

    def _output_dir(self) -> Path:
        return Path(str(self.get_parameter("output_dir").value)).expanduser()

    def _write_report(self, reason: str):
        if self.report_written:
            return
        self.report_written = True
        output_dir = self._output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        result = self.accumulator.result() if self.accumulator.samples else None
        inlier_mask = (
            np.asarray(result["inlier_mask"], dtype=bool)
            if result is not None
            else np.zeros(0, dtype=bool)
        )
        self._write_samples(output_dir / "marker_book_samples.csv", inlier_mask)

        minimum_samples = max(int(self.get_parameter("minimum_samples").value), 1)
        minimum_inlier_fraction = float(
            self.get_parameter("minimum_inlier_fraction").value
        )
        calibration_valid = bool(
            result is not None
            and int(result["inlier_samples"]) >= minimum_samples
            and float(result["inlier_fraction"]) >= minimum_inlier_fraction
        )
        frame_audit = None
        if self.frame_audit_enabled and result is not None:
            frame_audit = book_frame_audit_report(
                result["transform_eef_book"],
                transform_old_book_policy_book=(
                    self.transform_old_book_policy_book
                ),
                expected_rotation_eef_policy_book=(
                    self.expected_policy_book_rotation_eef
                ),
            )
        summary = {
            "schema_version": 1,
            "generated_at": datetime.now().astimezone().isoformat(),
            "completion_reason": reason,
            "hardware_commanded": False,
            "read_only": True,
            "calibration_valid": calibration_valid,
            "minimum_inlier_samples": minimum_samples,
            "minimum_inlier_fraction": minimum_inlier_fraction,
            "frames": self.counters,
            "software": {"opencv_version": cv2.__version__},
            "camera": self._camera_provenance(),
            "transform_eef_camera": self._transform_payload(
                self.transform_eef_camera
            ),
            "mount_yaml": str(self.mount_path),
            "mount": self.mount,
            "frame_convention": {
                "book": "+X depth/insertion, +Y thickness/lateral, +Z up",
                "marker_mapping": (
                    "+X marker = -Y book; +Y marker = +Z book; "
                    "+Z marker = -X book"
                ),
                "transform_output": "T_eef_book (book pose expressed in link_eef)",
            },
            "result": self._serialise_result(result),
            "frame_audit": frame_audit,
            "limitations": [
                "The result is valid only while the book remains rigidly fixed in the recorded grasp.",
                "The marker mounting measurements and axis mapping are treated as exact inputs.",
                "The projected cuboid should be inspected before using the transform downstream.",
                "Frame-audit candidates are never selected or promoted automatically.",
            ],
        }
        summary_path = output_dir / "marker_book_calibration_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        if frame_audit is not None:
            audit_path = output_dir / "book_frame_audit_report.json"
            audit_path.write_text(
                json.dumps(frame_audit, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self.get_logger().info(f"Book-frame audit written to {audit_path}")

        if calibration_valid:
            self._write_adapter_yaml(output_dir / "eef_book_calibration.yaml", result)
            self.get_logger().info(
                f"VALID calibration written to {output_dir / 'eef_book_calibration.yaml'}"
            )
        else:
            self.get_logger().error(
                f"Calibration is not valid: need {minimum_samples} inliers; "
                f"got {0 if result is None else result['inlier_samples']} "
                f"with fraction {0.0 if result is None else result['inlier_fraction']:.3f}."
            )
        self.get_logger().info(f"Calibration report written to {summary_path}")

    def _write_samples(self, path: Path, inlier_mask: np.ndarray):
        fields = [
            "frame_index", "stamp_ns", "inlier", "reprojection_error_px",
            "marker_depth_m", "depth_error_m", "camera_marker_x", "camera_marker_y",
            "camera_marker_z", "eef_book_x", "eef_book_y", "eef_book_z",
            "eef_book_qx", "eef_book_qy", "eef_book_qz", "eef_book_qw",
        ]
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for index, sample in enumerate(self.accumulator.samples):
                quaternion = matrix_to_quaternion_xyzw(
                    sample.transform_eef_book[:3, :3]
                )
                writer.writerow(
                    {
                        "frame_index": sample.frame_index,
                        "stamp_ns": sample.stamp_ns,
                        "inlier": bool(inlier_mask[index]) if index < len(inlier_mask) else False,
                        "reprojection_error_px": sample.reprojection_error_px,
                        "marker_depth_m": sample.marker_depth_m,
                        "depth_error_m": sample.depth_error_m,
                        "camera_marker_x": sample.transform_camera_marker[0, 3],
                        "camera_marker_y": sample.transform_camera_marker[1, 3],
                        "camera_marker_z": sample.transform_camera_marker[2, 3],
                        "eef_book_x": sample.transform_eef_book[0, 3],
                        "eef_book_y": sample.transform_eef_book[1, 3],
                        "eef_book_z": sample.transform_eef_book[2, 3],
                        "eef_book_qx": quaternion[0],
                        "eef_book_qy": quaternion[1],
                        "eef_book_qz": quaternion[2],
                        "eef_book_qw": quaternion[3],
                    }
                )

    def _serialise_result(self, result):
        if result is None:
            return None
        return {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in result.items()
            if key != "inlier_mask"
        }

    def _camera_provenance(self):
        if self.camera_info is None:
            return None
        return {
            "frame_id": self.camera_frame or self.camera_info.header.frame_id,
            "width": int(self.camera_info.width),
            "height": int(self.camera_info.height),
            "distortion_model": str(self.camera_info.distortion_model),
            "k": [float(value) for value in self.camera_info.k],
            "d": [float(value) for value in self.camera_info.d],
        }

    def _transform_payload(self, transform):
        if transform is None:
            return None
        quaternion = matrix_to_quaternion_xyzw(transform[:3, :3])
        return {
            "translation_xyz_m": [float(value) for value in transform[:3, 3]],
            "quaternion_xyzw": [float(value) for value in quaternion],
            "matrix": transform.tolist(),
        }

    def _write_adapter_yaml(self, path: Path, result):
        payload = {
            "policy_observation_adapter": {
                "ros__parameters": {
                    "book_pose_source": "marker",
                    "latch_eef_book_from_marker": False,
                    "use_configured_eef_book_transform": True,
                    "eef_book_translation_xyz": [
                        float(value) for value in result["translation_xyz_m"]
                    ],
                    "eef_book_quaternion_xyzw": [
                        float(value) for value in result["quaternion_xyzw"]
                    ],
                    "eef_book_transform_status": (
                        "measured_aruco_original_id0_static_grasp"
                    ),
                }
            }
        }
        with path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(payload, stream, sort_keys=False)


def main(args=None):
    rclpy.init(args=args)
    node = MarkerBookCalibrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if not node.report_written:
            node._write_report("clean interruption")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
