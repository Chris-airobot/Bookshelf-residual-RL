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
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
import tf2_ros
import yaml

from .marker_book_calibration import (
    CalibrationSample,
    MarkerBookCalibrationAccumulator,
    compose_eef_book_transform,
    make_book_marker_transform,
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
        self.detector_parameters = cv2.aruco.DetectorParameters_create()
        self.detector_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.object_points = _marker_object_points(self.marker_size_m)
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=60.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

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
        if self.completed:
            return
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

    def _find_marker_index(self, ids):
        if ids is None:
            return None
        matches = np.flatnonzero(np.asarray(ids).reshape(-1) == self.marker_id)
        return int(matches[0]) if matches.size else None

    def _estimate_pose(self, corners, camera_matrix, distortion):
        result = cv2.solvePnPGeneric(
            self.object_points,
            np.asarray(corners, dtype=np.float64),
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
            projected, _ = cv2.projectPoints(
                self.object_points, rvec, tvec, camera_matrix, distortion
            )
            residual = projected.reshape(4, 2) - np.asarray(corners).reshape(4, 2)
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

    def _draw_book_cuboid(self, image, transform_camera_book, camera_matrix, distortion):
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
                (255, 255, 0),
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
                "marker_mapping": "+X marker = +X book; +Y marker = +Z book; +Z marker = -Y book",
                "transform_output": "T_eef_book (book pose expressed in link_eef)",
            },
            "result": self._serialise_result(result),
            "limitations": [
                "The result is valid only while the book remains rigidly fixed in the recorded grasp.",
                "The marker mounting measurements and axis mapping are treated as exact inputs.",
                "The projected cuboid should be inspected before using the transform downstream.",
            ],
        }
        summary_path = output_dir / "marker_book_calibration_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )

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
                    "book_pose_source": "eef_fixed",
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
