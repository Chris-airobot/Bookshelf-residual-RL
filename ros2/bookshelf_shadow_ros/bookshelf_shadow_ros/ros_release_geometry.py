"""Pure helpers for a read-only ROS xArm release-geometry snapshot."""

from __future__ import annotations

import itertools
import math
from pathlib import Path
import struct

import numpy as np

from .policy_observation_math import invert_transform, matrix_to_quaternion_xyzw


XARM_GRIPPER_COLLISION_MESHES = {
    "xarm_gripper_base_link": "gripper/" + "xarm/base_link.stl",
    "left_outer_knuckle": "gripper/" + "xarm/left_outer_knuckle.stl",
    "left_finger": "gripper/" + "xarm/left_finger.stl",
    "left_inner_knuckle": "gripper/" + "xarm/left_inner_knuckle.stl",
    "right_outer_knuckle": "gripper/" + "xarm/right_outer_knuckle.stl",
    "right_finger": "gripper/" + "xarm/right_finger.stl",
    "right_inner_knuckle": "gripper/" + "xarm/right_inner_knuckle.stl",
}


def pose_dict(transform) -> dict:
    """Return the common diagnostic pose representation from a 4x4 transform."""

    transform = validated_transform(transform)
    quaternion_xyzw = matrix_to_quaternion_xyzw(transform[:3, :3])
    return {
        "position_xyz_m": transform[:3, 3].astype(float).tolist(),
        "quaternion_wxyz": [
            float(quaternion_xyzw[3]),
            float(quaternion_xyzw[0]),
            float(quaternion_xyzw[1]),
            float(quaternion_xyzw[2]),
        ],
    }


def validated_transform(transform) -> np.ndarray:
    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (4, 4) or not np.all(np.isfinite(value)):
        raise ValueError("transform must be a finite 4x4 matrix")
    if not np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-8):
        raise ValueError("transform has an invalid homogeneous row")
    return value


def relative_pose(parent, child) -> dict:
    return pose_dict(invert_transform(validated_transform(parent)) @ validated_transform(child))


def cuboid_corners(transform, size_xyz) -> np.ndarray:
    transform = validated_transform(transform)
    size = np.asarray(size_xyz, dtype=np.float64)
    if size.shape != (3,) or not np.all(np.isfinite(size)) or np.any(size <= 0.0):
        raise ValueError("cuboid size must contain three positive finite values")
    local = np.asarray(
        list(itertools.product(*[(-0.5 * value, 0.5 * value) for value in size])),
        dtype=np.float64,
    )
    return local @ transform[:3, :3].T + transform[:3, 3]


def aabb_corners(minimum, maximum) -> np.ndarray:
    minimum = np.asarray(minimum, dtype=np.float64)
    maximum = np.asarray(maximum, dtype=np.float64)
    if minimum.shape != (3,) or maximum.shape != (3,):
        raise ValueError("AABB limits must be three-dimensional")
    if not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(maximum)):
        raise ValueError("AABB limits must be finite")
    if np.any(maximum < minimum):
        raise ValueError("AABB maximum must not be below its minimum")
    return np.asarray(list(itertools.product(*zip(minimum, maximum))), dtype=np.float64)


def transformed_aabb(minimum, maximum, transform) -> tuple[np.ndarray, np.ndarray]:
    transform = validated_transform(transform)
    corners = aabb_corners(minimum, maximum)
    transformed = corners @ transform[:3, :3].T + transform[:3, 3]
    return transformed.min(axis=0), transformed.max(axis=0)


def aabb_distance(minimum_a, maximum_a, minimum_b, maximum_b) -> float:
    minimum_a = np.asarray(minimum_a, dtype=np.float64)
    maximum_a = np.asarray(maximum_a, dtype=np.float64)
    minimum_b = np.asarray(minimum_b, dtype=np.float64)
    maximum_b = np.asarray(maximum_b, dtype=np.float64)
    separation = np.maximum(np.maximum(minimum_b - maximum_a, minimum_a - maximum_b), 0.0)
    return float(np.linalg.norm(separation))


def stl_local_bounds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read binary or ASCII STL vertices without adding a mesh dependency."""

    path = Path(path)
    data = path.read_bytes()
    vertices = []
    if len(data) >= 84:
        triangle_count = struct.unpack_from("<I", data, 80)[0]
        expected_size = 84 + 50 * triangle_count
        if triangle_count > 0 and expected_size <= len(data):
            for index in range(triangle_count):
                values = struct.unpack_from("<12f", data, 84 + 50 * index)
                vertices.extend(
                    (values[3:6], values[6:9], values[9:12])
                )
    if not vertices:
        for raw_line in data.decode("ascii", errors="ignore").splitlines():
            fields = raw_line.strip().split()
            if len(fields) == 4 and fields[0].lower() == "vertex":
                vertices.append(tuple(float(value) for value in fields[1:]))
    if not vertices:
        raise ValueError(f"STL contains no readable vertices: {path}")
    array = np.asarray(vertices, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"STL contains non-finite vertices: {path}")
    return array.min(axis=0), array.max(axis=0)


def rectangular_slot_obstacles(
    *, slot_depth_m: float, slot_width_m: float, slot_height_m: float
) -> dict[str, dict[str, list[float]]]:
    """Build four finite AABB proxies around the approved slot opening."""

    values = np.asarray([slot_depth_m, slot_width_m, slot_height_m], dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("slot depth, width, and height must be positive and finite")
    depth, width, height = values
    lateral_extent = max(0.50, 3.0 * width)
    vertical_extent = max(0.40, 2.0 * height)
    return {
        "left_side": {
            "minimum_xyz_m": [0.0, -lateral_extent, -vertical_extent],
            "maximum_xyz_m": [depth, -0.5 * width, vertical_extent],
        },
        "right_side": {
            "minimum_xyz_m": [0.0, 0.5 * width, -vertical_extent],
            "maximum_xyz_m": [depth, lateral_extent, vertical_extent],
        },
        "bottom_deck": {
            "minimum_xyz_m": [0.0, -0.5 * width, -vertical_extent],
            "maximum_xyz_m": [depth, 0.5 * width, -0.5 * height],
        },
        "top_deck": {
            "minimum_xyz_m": [0.0, -0.5 * width, 0.5 * height],
            "maximum_xyz_m": [depth, 0.5 * width, vertical_extent],
        },
    }


def gripper_collision_snapshot(
    *,
    transform_base_slot,
    link_transforms_base: dict[str, np.ndarray],
    mesh_bounds: dict[str, tuple[np.ndarray, np.ndarray]],
    slot_depth_m: float,
    slot_width_m: float,
    slot_height_m: float,
) -> dict:
    """Compare xArm collision-mesh AABBs with a rectangular slot proxy."""

    transform_slot_base = invert_transform(validated_transform(transform_base_slot))
    obstacles = rectangular_slot_obstacles(
        slot_depth_m=slot_depth_m,
        slot_width_m=slot_width_m,
        slot_height_m=slot_height_m,
    )
    bodies = []
    closest = None
    for name, transform_base_link in sorted(link_transforms_base.items()):
        if name not in mesh_bounds:
            continue
        local_minimum, local_maximum = mesh_bounds[name]
        slot_minimum, slot_maximum = transformed_aabb(
            local_minimum,
            local_maximum,
            transform_slot_base @ validated_transform(transform_base_link),
        )
        distances = {}
        for obstacle_name, obstacle in obstacles.items():
            distance = aabb_distance(
                slot_minimum,
                slot_maximum,
                obstacle["minimum_xyz_m"],
                obstacle["maximum_xyz_m"],
            )
            distances[obstacle_name] = distance
            candidate = {
                "body": name,
                "obstacle": obstacle_name,
                "distance_m": distance,
            }
            if closest is None or distance < closest["distance_m"]:
                closest = candidate
        bodies.append(
            {
                "name": name,
                "local_collision_envelope": {
                    "minimum_xyz_m": np.asarray(local_minimum, dtype=float).tolist(),
                    "maximum_xyz_m": np.asarray(local_maximum, dtype=float).tolist(),
                },
                "slot_frame_envelope": {
                    "minimum_xyz_m": slot_minimum.astype(float).tolist(),
                    "maximum_xyz_m": slot_maximum.astype(float).tolist(),
                },
                "opening_margins": {
                    "mouth_to_body_nearest_x_m": float(-slot_maximum[0]),
                    "body_to_back_x_m": float(slot_depth_m - slot_maximum[0]),
                    "left_channel_margin_m": float(slot_minimum[1] + 0.5 * slot_width_m),
                    "right_channel_margin_m": float(0.5 * slot_width_m - slot_maximum[1]),
                    "deck_margin_m": float(slot_minimum[2] + 0.5 * slot_height_m),
                    "top_margin_m": float(0.5 * slot_height_m - slot_maximum[2]),
                },
                "shelf_obstacle_aabb_distances": distances,
            }
        )
    if not bodies:
        raise ValueError("no xArm gripper collision bodies were available")
    return {
        "method": (
            "Conservative AABB separation using official xarm_description collision "
            "STL bounds, live TF link poses, and an approved-config rectangular slot "
            "opening proxy. Zero means envelope overlap, not proven mesh contact."
        ),
        "bodies": bodies,
        "closest_body_obstacle_pair": closest,
    }


def book_snapshot(transform_base_book, transform_base_slot, book_size_xyz, slot_depth_m) -> dict:
    transform_base_book = validated_transform(transform_base_book)
    transform_base_slot = validated_transform(transform_base_slot)
    transform_slot_book = invert_transform(transform_base_slot) @ transform_base_book
    corners_slot = cuboid_corners(transform_slot_book, book_size_xyz)
    rear_x = float(corners_slot[:, 0].min())
    front_x = float(corners_slot[:, 0].max())
    return {
        "coordinate_frame": "slot",
        "pose": pose_dict(transform_slot_book),
        "pose_base": pose_dict(transform_base_book),
        "corners_xyz_m": corners_slot.astype(float).tolist(),
        "rear_x_m": rear_x,
        "front_x_m": front_x,
        "leading_edge_penetration_from_mouth_m": front_x,
        "trailing_edge_depth_from_mouth_m": rear_x,
        "front_to_back_remaining_m": float(slot_depth_m) - front_x,
    }


def quaternion_angle_deg(quaternion_wxyz) -> float:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion must contain four finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-12:
        raise ValueError("quaternion norm is zero")
    return math.degrees(2.0 * math.acos(min(1.0, abs(float(quaternion[0] / norm)))))
