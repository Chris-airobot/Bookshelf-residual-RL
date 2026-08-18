"""Pure geometry and validation for the offline physical-scene viewer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .policy_observation_math import make_transform


@dataclass(frozen=True)
class OfflineSceneGeometry:
    """Transforms and dimensions needed by the RViz-only scene publisher."""

    transform_base_slot: np.ndarray
    transform_base_shelf: np.ndarray
    transform_base_table: np.ndarray
    transform_slot_preinsert_book: np.ndarray
    shelf_size_xyz: tuple[float, float, float]
    table_size_xyz: tuple[float, float, float]
    held_book_size_xyz: tuple[float, float, float]
    slot_width_m: float
    slot_visual_height_m: float
    slot_support_anchored: bool


def positive_vector(values, *, length: int, label: str) -> tuple[float, ...]:
    """Return a finite, strictly positive vector of the requested length."""

    result = finite_vector(values, length=length, label=label)
    if any(value <= 0.0 for value in result):
        raise ValueError(f"{label} must contain only positive values")
    return result


def finite_vector(values, *, length: int, label: str) -> tuple[float, ...]:
    """Return one finite numeric vector with a stable validation error."""

    result = np.asarray(values, dtype=np.float64)
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be a finite {length}D vector")
    return tuple(float(value) for value in result)


def validated_joint_state(names, positions) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """Validate a complete, uniquely named visualization joint state."""

    joint_names = tuple(str(value).strip() for value in names)
    joint_positions = finite_vector(
        positions, length=len(joint_names), label="joint_positions"
    )
    if not joint_names or any(not value for value in joint_names):
        raise ValueError("joint_names must contain non-empty names")
    if len(set(joint_names)) != len(joint_names):
        raise ValueError("joint_names must be unique")
    return joint_names, joint_positions


def build_offline_scene_geometry(
    *,
    slot_translation_xyz,
    slot_quaternion_xyzw,
    slot_width_m: float,
    slot_visual_height_m: float,
    shelf_size_xyz,
    shelf_center_offset_slot_xyz,
    shelf_bottom_height_base_m: float,
    table_size_xyz,
    table_center_base_xyz,
    table_quaternion_base_xyzw,
    held_book_size_xyz,
    preinsert_book_center_slot_xyz,
    anchor_slot_to_shelf_support_height: bool = False,
) -> OfflineSceneGeometry:
    """Compose the coarse physical boxes in their explicit source frames."""

    slot_translation = finite_vector(
        slot_translation_xyz, length=3, label="slot_translation_xyz"
    )
    slot_quaternion = finite_vector(
        slot_quaternion_xyzw, length=4, label="slot_quaternion_xyzw"
    )
    shelf_size = positive_vector(
        shelf_size_xyz, length=3, label="shelf_size_xyz"
    )
    shelf_offset = finite_vector(
        shelf_center_offset_slot_xyz,
        length=3,
        label="shelf_center_offset_slot_xyz",
    )
    shelf_bottom_height = float(shelf_bottom_height_base_m)
    if not np.isfinite(shelf_bottom_height):
        raise ValueError("shelf_bottom_height_base_m must be finite")
    table_size = positive_vector(
        table_size_xyz, length=3, label="table_size_xyz"
    )
    table_center = finite_vector(
        table_center_base_xyz, length=3, label="table_center_base_xyz"
    )
    table_quaternion = finite_vector(
        table_quaternion_base_xyzw,
        length=4,
        label="table_quaternion_base_xyzw",
    )
    held_book_size = positive_vector(
        held_book_size_xyz, length=3, label="held_book_size_xyz"
    )
    preinsert_center = finite_vector(
        preinsert_book_center_slot_xyz,
        length=3,
        label="preinsert_book_center_slot_xyz",
    )
    slot_width = float(slot_width_m)
    slot_height = float(slot_visual_height_m)
    if not np.isfinite(slot_width) or slot_width <= 0.0:
        raise ValueError("slot_width_m must be positive")
    if not np.isfinite(slot_height) or slot_height <= 0.0:
        raise ValueError("slot_visual_height_m must be positive")

    transform_base_slot = make_transform(slot_translation, slot_quaternion)
    slot_support_anchored = bool(anchor_slot_to_shelf_support_height)
    if slot_support_anchored:
        up_axis = transform_base_slot[:3, 2]
        up_axis = up_axis / float(np.linalg.norm(up_axis))
        if float(up_axis[2]) < 0.0:
            up_axis = -up_axis
        if float(up_axis[2]) < 0.5:
            raise ValueError("slot up axis is not sufficiently vertical")
        lower_edge_z = float(
            transform_base_slot[2, 3] - 0.5 * slot_height * up_axis[2]
        )
        transform_base_slot[2, 3] += shelf_bottom_height - lower_edge_z
    shelf_heading_xy = transform_base_slot[:2, 0]
    heading_norm = float(np.linalg.norm(shelf_heading_xy))
    if heading_norm < 1e-9:
        raise ValueError("slot +X axis cannot define a level shelf heading")
    shelf_x_axis = np.array(
        [shelf_heading_xy[0] / heading_norm, shelf_heading_xy[1] / heading_norm, 0.0]
    )
    shelf_y_axis = np.array([-shelf_x_axis[1], shelf_x_axis[0], 0.0])
    shelf_rotation = np.column_stack(
        (shelf_x_axis, shelf_y_axis, np.array([0.0, 0.0, 1.0]))
    )
    shelf_center = np.asarray(slot_translation, dtype=np.float64)
    shelf_center[:2] += (
        shelf_rotation @ np.array([shelf_offset[0], shelf_offset[1], 0.0])
    )[:2]
    shelf_center[2] = shelf_bottom_height + 0.5 * shelf_size[2]
    transform_base_shelf = np.eye(4, dtype=np.float64)
    transform_base_shelf[:3, :3] = shelf_rotation
    transform_base_shelf[:3, 3] = shelf_center
    transform_base_table = make_transform(table_center, table_quaternion)
    transform_slot_preinsert_book = make_transform(preinsert_center)

    return OfflineSceneGeometry(
        transform_base_slot=transform_base_slot,
        transform_base_shelf=transform_base_shelf,
        transform_base_table=transform_base_table,
        transform_slot_preinsert_book=transform_slot_preinsert_book,
        shelf_size_xyz=shelf_size,
        table_size_xyz=table_size,
        held_book_size_xyz=held_book_size,
        slot_width_m=slot_width,
        slot_visual_height_m=slot_height,
        slot_support_anchored=slot_support_anchored,
    )


def shelf_front_plane_error_m(geometry: OfflineSceneGeometry) -> float:
    """Return the shelf-front offset from the approved slot-mouth plane."""

    slot_center_shelf = (
        np.linalg.inv(geometry.transform_base_shelf)
        @ geometry.transform_base_slot[:, 3]
    )
    return float(slot_center_shelf[0] + 0.5 * geometry.shelf_size_xyz[0])


def shelf_bottom_height_m(geometry: OfflineSceneGeometry) -> float:
    """Return the bottom face of the level coarse shelf box."""

    return float(
        geometry.transform_base_shelf[2, 3] - 0.5 * geometry.shelf_size_xyz[2]
    )


def table_top_height_m(geometry: OfflineSceneGeometry) -> float:
    """Return the top face of the axis-aligned coarse table box."""

    return float(
        geometry.transform_base_table[2, 3] + 0.5 * geometry.table_size_xyz[2]
    )
