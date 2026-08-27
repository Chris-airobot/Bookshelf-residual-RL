#!/usr/bin/env python3
"""Single-detector dual-ArUco calibration and book-pose reconstruction."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster
import yaml

from .dual_aruco_math import (
    RobustTransformAccumulator, derive_secondary_book, load_reference_book_transform,
    load_secondary_book_transform, marker_object_points, quaternion_angle_deg)
from .geometry import invert_transform, make_transform, matrix_to_quaternion_xyzw


def _dictionary(name):
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Unknown OpenCV ArUco dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(int(getattr(cv2.aruco, name)))


def _estimate_pose(corners, object_points, camera_matrix, distortion):
    results = cv2.solvePnPGeneric(
        object_points, np.asarray(corners, dtype=np.float64).reshape(4, 2),
        camera_matrix, distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    candidates = []
    for rvec, tvec in zip(results[1], results[2]):
        rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
        if hasattr(cv2, "solvePnPRefineLM"):
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points, np.asarray(corners).reshape(4, 2), camera_matrix,
                distortion, rvec, tvec)
        if not np.all(np.isfinite(tvec)) or tvec[2, 0] <= 0.0:
            continue
        projected, _ = cv2.projectPoints(
            object_points, rvec, tvec, camera_matrix, distortion)
        rms = float(np.sqrt(np.mean(np.sum(
            (projected.reshape(4, 2) - np.asarray(corners).reshape(4, 2)) ** 2,
            axis=1))))
        rotation, _ = cv2.Rodrigues(rvec)
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = tvec.reshape(3)
        candidates.append((rms, transform, rvec, tvec))
    return min(candidates, key=lambda value: value[0]) if candidates else None


class DualArucoBookNode(Node):
    def __init__(self):
        super().__init__("dual_aruco_book")
        self._declare_parameters()
        self.mode = str(self.get_parameter("mode").value)
        if self.mode not in ("calibrate", "runtime"):
            raise ValueError("mode must be calibrate or runtime")
        self.reference_id = int(self.get_parameter("reference_marker_id").value)
        self.secondary_id = int(self.get_parameter("secondary_marker_id").value)
        if self.reference_id == self.secondary_id:
            raise ValueError("Reference and secondary marker IDs must differ")
        self.sizes = {
            self.reference_id: float(self.get_parameter("reference_marker_size_m").value),
            self.secondary_id: float(self.get_parameter("secondary_marker_size_m").value),
        }
        self.objects = {key: marker_object_points(value) for key, value in self.sizes.items()}
        self.dictionary_name = str(self.get_parameter("dictionary").value)
        self.dictionary = _dictionary(self.dictionary_name)
        self.detector_parameters = (
            cv2.aruco.DetectorParameters_create() if
            hasattr(cv2.aruco, "DetectorParameters_create") else
            cv2.aruco.DetectorParameters())
        self.detector_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.bridge = CvBridge()
        self.camera_info = None
        self.tf_broadcaster = TransformBroadcaster(self)
        self.debug_publisher = self.create_publisher(
            Image, str(self.get_parameter("debug_image_topic").value), 2)
        self.diagnostic_publisher = self.create_publisher(
            String, str(self.get_parameter("diagnostics_topic").value), 10)
        self.create_subscription(CameraInfo, str(self.get_parameter("camera_info_topic").value),
                                 self._camera_info, qos_profile_sensor_data)
        self.create_subscription(Image, str(self.get_parameter("image_topic").value),
                                 self._image, qos_profile_sensor_data)
        reference_path = Path(str(self.get_parameter("reference_mount_yaml").value)).expanduser()
        self.reference_mount, self.transform_reference_book = load_reference_book_transform(reference_path)
        if self.reference_mount["dictionary"] != self.dictionary_name:
            raise ValueError("Reference mount dictionary does not match detector dictionary")
        if int(self.reference_mount["marker_id"]) != self.reference_id:
            raise ValueError("Reference mount marker ID does not match configured ID")
        if not math.isclose(float(self.reference_mount["marker_black_size_m"]),
                            self.sizes[self.reference_id], abs_tol=1.0e-9):
            raise ValueError("Reference mount marker size does not match configured size")
        self.transform_secondary_book = None
        self.accumulator = RobustTransformAccumulator(
            self.get_parameter("maximum_translation_deviation_m").value,
            self.get_parameter("maximum_rotation_deviation_deg").value)
        self.saved = False
        secondary_path = Path(str(self.get_parameter("secondary_mount_yaml").value)).expanduser()
        self.secondary_path = secondary_path
        if self.mode == "runtime":
            data, self.transform_secondary_book = load_secondary_book_transform(secondary_path)
            if int(data["marker_id"]) != self.secondary_id:
                raise ValueError("Secondary mount marker ID does not match configured ID")
            if data["dictionary"] != self.dictionary_name:
                raise ValueError("Secondary mount dictionary does not match detector dictionary")
            if not math.isclose(float(data["marker_black_size_m"]),
                                self.sizes[self.secondary_id], abs_tol=1.0e-9):
                raise ValueError("Secondary mount marker size does not match configured size")
        self.get_logger().info(
            f"dual ArUco {self.mode}: reference id={self.reference_id} size={self.sizes[self.reference_id]:.3f} m; "
            f"secondary id={self.secondary_id} size={self.sizes[self.secondary_id]:.3f} m")

    def _declare_parameters(self):
        from ament_index_python.packages import get_package_share_directory
        share = Path(get_package_share_directory("bookshelf_simple_experiment_ros"))
        self.declare_parameter("mode", "runtime")
        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("camera_frame", "")
        self.declare_parameter("dictionary", "DICT_ARUCO_ORIGINAL")
        self.declare_parameter("reference_marker_id", 0)
        self.declare_parameter("reference_marker_size_m", 0.039)
        self.declare_parameter("secondary_marker_id", 10)
        self.declare_parameter("secondary_marker_size_m", 0.039)
        self.declare_parameter("reference_mount_yaml", str(share / "config/reference_marker0_book_mount.yaml"))
        self.declare_parameter("secondary_mount_yaml", "~/BookshelfFiles/experiment_configs/simple_dual_aruco/secondary_marker_book_mount.yaml")
        self.declare_parameter("target_samples", 200)
        self.declare_parameter("minimum_visualization_samples", 5)
        self.declare_parameter("maximum_reprojection_error_px", 3.0)
        self.declare_parameter("maximum_translation_deviation_m", 0.010)
        self.declare_parameter("maximum_rotation_deviation_deg", 5.0)
        self.declare_parameter("debug_image_topic", "/bookshelf_simple/dual_aruco/debug_image")
        self.declare_parameter("diagnostics_topic", "/bookshelf_simple/dual_aruco/diagnostics")
        self.declare_parameter("reference_marker_frame", "bookshelf_reference_marker_detected")
        self.declare_parameter("secondary_marker_frame", "bookshelf_secondary_marker_detected")
        self.declare_parameter("book_from_reference_frame", "bookshelf_book_from_reference")
        self.declare_parameter("book_from_secondary_frame", "bookshelf_book_from_secondary")
        self.declare_parameter("book_frame", "bookshelf_book")

    def _camera_info(self, message):
        self.camera_info = message

    def _image(self, message):
        if self.camera_info is None:
            return
        image = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        corners, ids, _ = cv2.aruco.detectMarkers(
            image, self.dictionary, parameters=self.detector_parameters)
        camera_matrix = np.asarray(self.camera_info.k, dtype=np.float64).reshape(3, 3)
        distortion = np.asarray(self.camera_info.d, dtype=np.float64)
        poses = {}
        pose_vectors = {}
        if ids is not None:
            for index, marker_id_value in enumerate(ids.flatten()):
                marker_id = int(marker_id_value)
                if marker_id not in self.objects:
                    continue
                pose = _estimate_pose(corners[index], self.objects[marker_id], camera_matrix, distortion)
                if pose and pose[0] <= float(self.get_parameter("maximum_reprojection_error_px").value):
                    poses[marker_id] = pose[1]
                    pose_vectors[marker_id] = (pose[2], pose[3])
        camera_frame = str(self.get_parameter("camera_frame").value).strip()
        camera_frame = camera_frame or message.header.frame_id or self.camera_info.header.frame_id
        if self.mode == "calibrate" and self.reference_id in poses and self.secondary_id in poses and not self.saved:
            transform_reference_secondary = invert_transform(poses[self.reference_id]) @ poses[self.secondary_id]
            self.accumulator.add(transform_reference_secondary)
            if len(self.accumulator.transforms) >= int(self.get_parameter("minimum_visualization_samples").value):
                estimate = self.accumulator.result()["transform"]
                self.transform_secondary_book = derive_secondary_book(estimate, self.transform_reference_book)
            if len(self.accumulator.transforms) >= int(self.get_parameter("target_samples").value):
                self._save_calibration()
        books = {}
        if self.reference_id in poses:
            books["reference"] = poses[self.reference_id] @ self.transform_reference_book
        if self.secondary_id in poses and self.transform_secondary_book is not None:
            books["secondary"] = poses[self.secondary_id] @ self.transform_secondary_book
        self._publish_transforms(message.header.stamp, camera_frame, poses, books)
        diagnostic = {"mode": self.mode, "visible_ids": sorted(poses),
                      "samples": len(self.accumulator.transforms)}
        if set(books) == {"reference", "secondary"}:
            relative = invert_transform(books["reference"]) @ books["secondary"]
            diagnostic["book_pose_difference"] = {
                "translation_mm": float(np.linalg.norm(relative[:3, 3]) * 1000.0),
                "rotation_deg": quaternion_angle_deg(
                    [0, 0, 0, 1], matrix_to_quaternion_xyzw(relative[:3, :3]))}
        self.diagnostic_publisher.publish(String(data=json.dumps(diagnostic, separators=(",", ":"))))
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(image, corners, ids)
            for marker_id, (rvec, tvec) in pose_vectors.items():
                cv2.drawFrameAxes(image, camera_matrix, distortion, rvec, tvec,
                                  0.5 * self.sizes[marker_id])
        debug = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
        debug.header = message.header
        self.debug_publisher.publish(debug)

    def _publish_transforms(self, stamp, parent, poses, books):
        messages = []
        frames = {self.reference_id: str(self.get_parameter("reference_marker_frame").value),
                  self.secondary_id: str(self.get_parameter("secondary_marker_frame").value)}
        for marker_id, transform in poses.items():
            messages.append(self._transform_message(stamp, parent, frames[marker_id], transform))
        if "reference" in books:
            messages.append(self._transform_message(stamp, parent,
                str(self.get_parameter("book_from_reference_frame").value), books["reference"]))
        if "secondary" in books:
            messages.append(self._transform_message(stamp, parent,
                str(self.get_parameter("book_from_secondary_frame").value), books["secondary"]))
        canonical = books.get("reference", books.get("secondary"))
        if canonical is not None:
            messages.append(self._transform_message(stamp, parent,
                str(self.get_parameter("book_frame").value), canonical))
        if messages:
            self.tf_broadcaster.sendTransform(messages)

    @staticmethod
    def _transform_message(stamp, parent, child, transform):
        message = TransformStamped()
        message.header.stamp = stamp
        message.header.frame_id = parent
        message.child_frame_id = child
        message.transform.translation.x, message.transform.translation.y, message.transform.translation.z = map(float, transform[:3, 3])
        quaternion = matrix_to_quaternion_xyzw(transform[:3, :3])
        message.transform.rotation.x, message.transform.rotation.y, message.transform.rotation.z, message.transform.rotation.w = map(float, quaternion)
        return message

    def _save_calibration(self):
        result = self.accumulator.result()
        transform_reference_secondary = result["transform"]
        transform_secondary_book = derive_secondary_book(
            transform_reference_secondary, self.transform_reference_book)
        def value(transform):
            return {"translation_xyz_m": [float(v) for v in transform[:3, 3]],
                    "quaternion_xyzw": [float(v) for v in matrix_to_quaternion_xyzw(transform[:3, :3])]}
        document = {
            "dictionary": self.dictionary_name,
            "marker_id": self.secondary_id,
            "marker_black_size_m": self.sizes[self.secondary_id],
            "transform_convention": "T_A_B maps coordinates from frame B into frame A",
            "transform_secondary_book": {"direction": "T_secondary_book", **value(transform_secondary_book)},
            "transform_reference_secondary": {
                "direction": "T_reference_secondary = inverse(T_camera_reference) @ T_camera_secondary",
                **value(transform_reference_secondary)},
            "reference_marker_id": self.reference_id,
            "reference_marker_black_size_m": self.sizes[self.reference_id],
            "input_sample_count": result["input_samples"],
            "accepted_sample_count": result["accepted_samples"],
            "translation_variation_m": result["translation_variation_m"],
            "rotation_variation_deg": result["rotation_variation_deg"],
        }
        self.secondary_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.secondary_path.with_suffix(self.secondary_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(document, stream, sort_keys=False)
        temporary.replace(self.secondary_path)
        self.transform_secondary_book = transform_secondary_book
        self.saved = True
        self.get_logger().info(
            f"Saved dual-marker calibration ({result['accepted_samples']} inliers) to {self.secondary_path}")


def main(args=None):
    rclpy.init(args=args)
    node = DualArucoBookNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
