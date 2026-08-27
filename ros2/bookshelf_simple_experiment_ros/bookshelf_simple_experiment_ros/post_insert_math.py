"""Verified pure geometry and nominal PUSH helpers for the simple episode."""

from dataclasses import dataclass
import math

import numpy as np


def _vector(value, expected_size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (expected_size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape ({expected_size},)")
    return array


def retreat_progress(start_xyz, current_xyz, direction_xyz) -> float:
    """Return signed travel from start along a normalized retreat direction."""

    start = _vector(start_xyz, 3, "retreat start")
    current = _vector(current_xyz, 3, "retreat current")
    direction = _vector(direction_xyz, 3, "retreat direction")
    magnitude = float(np.linalg.norm(direction))
    if magnitude <= 0.0:
        raise ValueError("retreat direction must be nonzero")
    return float(np.dot(current - start, direction / magnitude))


def simulated_book_push_distance(
    push_progress_m: float,
    contact_distance_m: float,
    requested_book_distance_m: float,
) -> float:
    """Return fake book travel after the closed gripper reaches contact."""

    values = np.asarray(
        [push_progress_m, contact_distance_m, requested_book_distance_m],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("push distances must be finite and nonnegative")
    return float(
        np.clip(
            float(push_progress_m) - float(contact_distance_m),
            0.0,
            float(requested_book_distance_m),
        )
    )


def oriented_box_contact_gap(
    contact_point_xyz,
    box_transform,
    box_size_xyz,
    approach_direction_xyz,
) -> float:
    """Return the signed gap from a point to an oriented box's near face."""

    point = _vector(contact_point_xyz, 3, "contact point").astype(np.float64)
    transform = np.asarray(box_transform, dtype=np.float64)
    size = _vector(box_size_xyz, 3, "box size").astype(np.float64)
    direction = _vector(
        approach_direction_xyz, 3, "approach direction"
    ).astype(np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("box transform must be a finite 4x4 matrix")
    if np.any(size <= 0.0):
        raise ValueError("box dimensions must be positive")
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 0.0:
        raise ValueError("approach direction must be nonzero")
    direction /= direction_norm
    local_direction = transform[:3, :3].T @ direction
    support_radius = float(np.dot(np.abs(local_direction), size * 0.5))
    near_face_position = float(np.dot(transform[:3, 3], direction)) - support_radius
    return near_face_position - float(np.dot(point, direction))


@dataclass(frozen=True)
class NominalPushConfig:
    push_dx: float = 0.0008
    lateral_gain: float = 0.35
    height_gain: float = 0.30
    yaw_gain: float = 0.20
    pitch_gain: float = 0.08
    push_z_fraction_from_bottom: float = 0.20
    book_size: tuple[float, float, float] = (0.156, 0.034, 0.236)
    dy_limit: float = 0.0005
    dz_limit: float = 0.0010
    dyaw_limit: float = math.radians(0.35)
    dpitch_limit: float = math.radians(0.25)


def _book_vertical_half_extent(raw: np.ndarray, book_size) -> float:
    depth, thickness, height = _vector(book_size, 3, "book_size")
    yaw = float(raw[5])
    up_x = float(raw[10])
    up_y = float(raw[11])
    up_horizontal_sq = up_x * up_x + up_y * up_y
    if up_horizontal_sq > 1.0 + 1.0e-5:
        raise ValueError("book up-axis components are inconsistent")
    up_z = math.sqrt(max(1.0 - up_horizontal_sq, 0.0))
    up = np.array([up_x, up_y, up_z], dtype=np.float64)
    horizontal_depth = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
    projection = float(np.dot(horizontal_depth, up[:2]))
    denominator = math.sqrt(up_z * up_z + projection * projection)
    if denominator <= 1.0e-8:
        raise ValueError("book orientation cannot recover a depth axis")
    depth_axis = np.array(
        [
            horizontal_depth[0] * up_z / denominator,
            horizontal_depth[1] * up_z / denominator,
            -projection / denominator,
        ],
        dtype=np.float64,
    )
    thickness_axis = np.cross(up, depth_axis)
    return 0.5 * float(
        depth * abs(depth_axis[2])
        + thickness * abs(thickness_axis[2])
        + height * abs(up[2])
    )


def compute_push_nominal_delta(
    raw_metrics,
    config: NominalPushConfig = NominalPushConfig(),
) -> np.ndarray:
    """Reproduce the simulator's nominal PUSH controller from raw 12D metrics."""

    raw = _vector(raw_metrics, 12, "raw_metrics")
    if not math.isclose(float(raw[0]), 1.0, abs_tol=1.0e-4):
        raise ValueError("nominal PUSH requires mode observation 1.0")
    lat_err = float(raw[3])
    yaw_err = float(raw[5])
    tool_to_book_y = float(raw[7])
    tool_to_book_z = float(raw[8])
    tilt_x = float(raw[10])
    vertical_half_extent = _book_vertical_half_extent(raw, config.book_size)
    desired_tool_z_from_book = (
        2.0 * config.push_z_fraction_from_bottom - 1.0
    ) * vertical_half_extent
    return np.array(
        [
            config.push_dx,
            np.clip(
                config.lateral_gain * (lat_err - tool_to_book_y),
                -config.dy_limit,
                config.dy_limit,
            ),
            np.clip(
                config.height_gain * (desired_tool_z_from_book - tool_to_book_z),
                -config.dz_limit,
                config.dz_limit,
            ),
            np.clip(-config.yaw_gain * yaw_err, -config.dyaw_limit, config.dyaw_limit),
            np.clip(-config.pitch_gain * tilt_x, -config.dpitch_limit, config.dpitch_limit),
        ],
        dtype=np.float32,
    )
