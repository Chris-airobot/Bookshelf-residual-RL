#!/usr/bin/env python3
"""Publish a static xArm joint pose and coarse physical scene for RViz only."""

from __future__ import annotations

import json

from geometry_msgs.msg import Point
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from .offline_scene_visualization import (
    build_offline_scene_geometry,
    shelf_bottom_height_m,
    shelf_front_plane_error_m,
    table_top_height_m,
    validated_joint_state,
)
from .policy_observation_math import matrix_to_quaternion_xyzw


def _apply_transform_pose(marker: Marker, transform: np.ndarray) -> None:
    marker.pose.position.x = float(transform[0, 3])
    marker.pose.position.y = float(transform[1, 3])
    marker.pose.position.z = float(transform[2, 3])
    quaternion = matrix_to_quaternion_xyzw(transform[:3, :3])
    marker.pose.orientation.x = float(quaternion[0])
    marker.pose.orientation.y = float(quaternion[1])
    marker.pose.orientation.z = float(quaternion[2])
    marker.pose.orientation.w = float(quaternion[3])


def _set_color(marker: Marker, rgba) -> None:
    marker.color.r = float(rgba[0])
    marker.color.g = float(rgba[1])
    marker.color.b = float(rgba[2])
    marker.color.a = float(rgba[3])


class OfflineSceneVisualizerNode(Node):
    """Publisher-only visualizer with no motion or planning interfaces."""

    def __init__(self):
        super().__init__("offline_scene_visualizer")
        self._declare_parameters()
        if not bool(self.get_parameter("visualization_only").value):
            raise ValueError("visualization_only must remain true")

        self.base_frame = str(self.get_parameter("base_frame").value)
        self.tcp_frame = str(self.get_parameter("tcp_frame").value)
        self.publish_joint_states = bool(
            self.get_parameter("publish_joint_states").value
        )
        self.joint_names, self.joint_positions = validated_joint_state(
            self.get_parameter("joint_names").value,
            self.get_parameter("joint_positions").value,
        )
        self.geometry = build_offline_scene_geometry(
            slot_translation_xyz=self.get_parameter("slot_translation_xyz").value,
            slot_quaternion_xyzw=self.get_parameter("slot_quaternion_xyzw").value,
            slot_width_m=self.get_parameter("slot_width_m").value,
            slot_visual_height_m=self.get_parameter("slot_visual_height_m").value,
            shelf_size_xyz=self.get_parameter("shelf_size_xyz").value,
            shelf_center_offset_slot_xyz=self.get_parameter(
                "shelf_center_offset_slot_xyz"
            ).value,
            shelf_bottom_height_base_m=self.get_parameter(
                "shelf_bottom_height_base_m"
            ).value,
            table_size_xyz=self.get_parameter("table_size_xyz").value,
            table_center_base_xyz=self.get_parameter("table_center_base_xyz").value,
            table_quaternion_base_xyzw=self.get_parameter(
                "table_quaternion_base_xyzw"
            ).value,
            held_book_size_xyz=self.get_parameter("held_book_size_xyz").value,
            preinsert_book_center_slot_xyz=self.get_parameter(
                "preinsert_book_center_slot_xyz"
            ).value,
        )
        self.held_book_center_tcp_xyz = self._vector(
            "held_book_center_tcp_xyz", 3
        )
        self.held_book_quaternion_tcp_xyzw = self._vector(
            "held_book_quaternion_tcp_xyzw", 4
        )

        marker_qos = QoSProfile(depth=1)
        marker_qos.reliability = ReliabilityPolicy.RELIABLE
        marker_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.marker_publisher = self.create_publisher(
            MarkerArray,
            str(self.get_parameter("marker_topic").value),
            marker_qos,
        )
        self.joint_state_publisher = None
        if self.publish_joint_states:
            self.joint_state_publisher = self.create_publisher(
                JointState,
                str(self.get_parameter("joint_state_topic").value),
                10,
            )
        self.status_publisher = self.create_publisher(
            String,
            str(self.get_parameter("status_topic").value),
            marker_qos,
        )

        rate = max(float(self.get_parameter("publish_rate_hz").value), 1.0)
        self.timer = self.create_timer(1.0 / rate, self._publish)
        self.get_logger().warning(
            "OFFLINE VISUALIZATION ONLY: no hardware, planner, controller, "
            "trajectory, gripper-command, or execution interface exists."
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("visualization_only", True)
        self.declare_parameter("publish_joint_states", True)
        self.declare_parameter("base_frame", "link_base")
        self.declare_parameter("tcp_frame", "link_tcp")
        self.declare_parameter(
            "joint_names",
            [
                "joint1",
                "joint2",
                "joint3",
                "joint4",
                "joint5",
                "joint6",
                "joint7",
                "drive_joint",
            ],
        )
        self.declare_parameter("joint_positions", [0.0] * 8)
        self.declare_parameter("slot_translation_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter("slot_quaternion_xyzw", [0.0, 0.0, 0.0, 1.0])
        self.declare_parameter("slot_width_m", 0.04)
        self.declare_parameter("slot_visual_height_m", 0.25)
        self.declare_parameter("shelf_size_xyz", [0.30, 0.95, 0.40])
        self.declare_parameter("shelf_center_offset_slot_xyz", [0.15, 0.0, 0.0])
        self.declare_parameter("shelf_bottom_height_base_m", 0.015)
        self.declare_parameter("table_size_xyz", [1.50, 0.60, 0.05])
        self.declare_parameter("table_center_base_xyz", [0.75, 0.0, -0.025])
        self.declare_parameter(
            "table_quaternion_base_xyzw", [0.0, 0.0, 0.0, 1.0]
        )
        self.declare_parameter("held_book_size_xyz", [0.156, 0.034, 0.236])
        self.declare_parameter("held_book_center_tcp_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter(
            "held_book_quaternion_tcp_xyzw", [0.0, 0.0, 0.0, 1.0]
        )
        self.declare_parameter(
            "preinsert_book_center_slot_xyz", [-0.108, 0.0, 0.006]
        )
        self.declare_parameter("marker_topic", "/bookshelf_offline_scene/markers")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("status_topic", "/bookshelf_offline_scene/status")
        self.declare_parameter("publish_rate_hz", 10.0)

    def _vector(self, name: str, length: int) -> tuple[float, ...]:
        values = np.asarray(self.get_parameter(name).value, dtype=np.float64)
        if values.shape != (length,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be a finite {length}D vector")
        return tuple(float(value) for value in values)

    def _base_marker(self, marker_id: int, marker_type: int, namespace: str) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.base_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        return marker

    def _cube(self, marker_id, namespace, transform, size, color) -> Marker:
        marker = self._base_marker(marker_id, Marker.CUBE, namespace)
        _apply_transform_pose(marker, transform)
        marker.scale.x, marker.scale.y, marker.scale.z = [float(v) for v in size]
        _set_color(marker, color)
        return marker

    def _text(self, marker_id, text, position, color) -> Marker:
        marker = self._base_marker(marker_id, Marker.TEXT_VIEW_FACING, "labels")
        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = float(position[2])
        marker.scale.z = 0.045
        marker.text = str(text)
        _set_color(marker, color)
        return marker

    def _slot_markers(self) -> list[Marker]:
        transform = self.geometry.transform_base_slot
        arrow = self._base_marker(10, Marker.ARROW, "approved_slot")
        start = transform[:3, 3]
        end = start + transform[:3, 0] * 0.18
        arrow.points = [
            Point(x=float(value[0]), y=float(value[1]), z=float(value[2]))
            for value in (start, end)
        ]
        arrow.scale.x = 0.018
        arrow.scale.y = 0.032
        arrow.scale.z = 0.045
        _set_color(arrow, (0.10, 0.95, 0.25, 1.0))

        opening = self._base_marker(11, Marker.LINE_LIST, "approved_slot")
        half_y = 0.5 * self.geometry.slot_width_m
        half_z = 0.5 * self.geometry.slot_visual_height_m
        corners_slot = np.array(
            [
                [0.0, -half_y, -half_z, 1.0],
                [0.0, +half_y, -half_z, 1.0],
                [0.0, +half_y, +half_z, 1.0],
                [0.0, -half_y, +half_z, 1.0],
            ],
            dtype=np.float64,
        )
        corners_base = (transform @ corners_slot.T).T[:, :3]
        for first, second in ((0, 1), (1, 2), (2, 3), (3, 0)):
            for index in (first, second):
                corner = corners_base[index]
                opening.points.append(
                    Point(
                        x=float(corner[0]),
                        y=float(corner[1]),
                        z=float(corner[2]),
                    )
                )
        opening.scale.x = 0.008
        _set_color(opening, (0.15, 1.0, 0.35, 1.0))

        label_position = start + np.array([0.0, 0.0, 0.18])
        label = self._text(
            12,
            f"approved slot | width {self.geometry.slot_width_m * 1000.0:.1f} mm",
            label_position,
            (0.2, 1.0, 0.4, 1.0),
        )
        return [arrow, opening, label]

    def _held_book_marker(self) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.tcp_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "held_book"
        marker.id = 20
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = self.held_book_center_tcp_xyz[0]
        marker.pose.position.y = self.held_book_center_tcp_xyz[1]
        marker.pose.position.z = self.held_book_center_tcp_xyz[2]
        marker.pose.orientation.x = self.held_book_quaternion_tcp_xyzw[0]
        marker.pose.orientation.y = self.held_book_quaternion_tcp_xyzw[1]
        marker.pose.orientation.z = self.held_book_quaternion_tcp_xyzw[2]
        marker.pose.orientation.w = self.held_book_quaternion_tcp_xyzw[3]
        marker.scale.x, marker.scale.y, marker.scale.z = self.geometry.held_book_size_xyz
        _set_color(marker, (0.20, 0.55, 1.0, 0.60))
        return marker

    def _markers(self) -> MarkerArray:
        markers = MarkerArray()
        markers.markers.append(
            self._cube(
                1,
                "coarse_scene",
                self.geometry.transform_base_table,
                self.geometry.table_size_xyz,
                (0.38, 0.42, 0.48, 0.65),
            )
        )
        markers.markers.append(
            self._cube(
                2,
                "coarse_scene",
                self.geometry.transform_base_shelf,
                self.geometry.shelf_size_xyz,
                (0.92, 0.34, 0.18, 0.34),
            )
        )
        shelf_position = self.geometry.transform_base_shelf[:3, 3]
        markers.markers.append(
            self._text(
                3,
                "coarse bookshelf keep-out",
                shelf_position + np.array([0.0, 0.0, 0.25]),
                (1.0, 0.55, 0.25, 1.0),
            )
        )
        markers.markers.extend(self._slot_markers())
        markers.markers.append(self._held_book_marker())

        target_transform = (
            self.geometry.transform_base_slot
            @ self.geometry.transform_slot_preinsert_book
        )
        markers.markers.append(
            self._cube(
                30,
                "preinsert_reference",
                target_transform,
                self.geometry.held_book_size_xyz,
                (0.15, 0.95, 0.55, 0.24),
            )
        )
        markers.markers.append(
            self._text(
                31,
                "pre-insertion book reference",
                target_transform[:3, 3] + np.array([0.0, 0.0, 0.17]),
                (0.25, 1.0, 0.65, 1.0),
            )
        )
        return markers

    def _publish(self) -> None:
        now = self.get_clock().now().to_msg()
        joint_state = JointState()
        joint_state.header.stamp = now
        joint_state.name = list(self.joint_names)
        joint_state.position = list(self.joint_positions)
        if self.joint_state_publisher is not None:
            self.joint_state_publisher.publish(joint_state)
        self.marker_publisher.publish(self._markers())

        status = {
            "visualization_only": True,
            "hardware_commanded": False,
            "planning_requested": False,
            "execution_authorized": False,
            "publishes_joint_states": self.publish_joint_states,
            "base_frame": self.base_frame,
            "tcp_frame": self.tcp_frame,
            "shelf_front_plane_error_m": shelf_front_plane_error_m(self.geometry),
            "shelf_bottom_height_m": shelf_bottom_height_m(self.geometry),
            "table_top_height_m": table_top_height_m(self.geometry),
            "joint_names": list(self.joint_names),
            "joint_positions": list(self.joint_positions),
        }
        message = String()
        message.data = json.dumps(status, sort_keys=True)
        self.status_publisher.publish(message)


def main(args=None):
    rclpy.init(args=args)
    node = OfflineSceneVisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
