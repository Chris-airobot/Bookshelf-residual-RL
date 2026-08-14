"""Pure geometry and fail-closed mode gates for the bookshelf planning scene."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .policy_tool_control_math import make_transform, validated_transform


GLOBAL_APPROACH = "global_approach"
LOCAL_INSERTION = "local_insertion"


def ros_uint8_constant(value) -> int:
    """Normalize ROS uint8 constants emitted as integers or one-byte values."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        encoded = bytes(value)
        if len(encoded) != 1:
            raise ValueError("ROS uint8 byte constants must contain exactly one byte")
        return encoded[0]
    converted = int(value)
    if not 0 <= converted <= 255:
        raise ValueError(f"ROS uint8 constant is outside [0, 255]: {converted}")
    return converted
SCENE_MODES = (GLOBAL_APPROACH, LOCAL_INSERTION)


@dataclass(frozen=True)
class BoxSpec:
    """A box pose and dimensions expressed in one named frame."""

    frame_id: str
    size_xyz: tuple[float, float, float]
    transform_frame_box: np.ndarray


def validated_box_size(size_xyz, label: str) -> tuple[float, float, float]:
    values = np.asarray(size_xyz, dtype=np.float64)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{label} must be a finite 3D vector")
    if np.any(values <= 0.0):
        raise ValueError(f"{label} dimensions must be positive")
    return tuple(float(value) for value in values)


def shelf_box_from_slot(
    transform_base_slot,
    *,
    base_frame: str,
    size_xyz,
    center_offset_slot_xyz,
    level_with_base: bool = False,
    bottom_height_base_m: float | None = None,
) -> BoxSpec:
    """Place the coarse bookshelf keep-out box relative to the slot mouth."""

    transform_base_slot = validated_transform(transform_base_slot)
    size = validated_box_size(size_xyz, "shelf_box_size_xyz")
    offset = np.asarray(center_offset_slot_xyz, dtype=np.float64)
    if offset.shape != (3,) or not np.all(np.isfinite(offset)):
        raise ValueError("shelf_box_center_offset_slot_xyz must be a finite 3D vector")

    if level_with_base:
        if bottom_height_base_m is None:
            raise ValueError("shelf_bottom_height_base_m must be finite")
        bottom_height = float(bottom_height_base_m)
        if not math.isfinite(bottom_height):
            raise ValueError("shelf_bottom_height_base_m must be finite")
        heading_xy = transform_base_slot[:2, 0]
        heading_norm = float(np.linalg.norm(heading_xy))
        if heading_norm < 1.0e-9:
            raise ValueError("slot +X axis cannot define a level shelf heading")
        shelf_x_axis = np.array(
            [heading_xy[0] / heading_norm, heading_xy[1] / heading_norm, 0.0],
            dtype=np.float64,
        )
        shelf_y_axis = np.array(
            [-shelf_x_axis[1], shelf_x_axis[0], 0.0], dtype=np.float64
        )
        shelf_rotation = np.column_stack(
            (shelf_x_axis, shelf_y_axis, np.array([0.0, 0.0, 1.0]))
        )
        shelf_center = transform_base_slot[:3, 3].copy()
        shelf_center[:2] += (shelf_rotation @ offset)[:2]
        shelf_center[2] = bottom_height + 0.5 * size[2] + offset[2]
        transform_base_shelf = np.eye(4, dtype=np.float64)
        transform_base_shelf[:3, :3] = shelf_rotation
        transform_base_shelf[:3, 3] = shelf_center
    else:
        if bottom_height_base_m is not None:
            raise ValueError(
                "bottom_height_base_m requires level_with_base=True"
            )
        transform_base_shelf = transform_base_slot @ make_transform(offset)

    return BoxSpec(
        frame_id=str(base_frame),
        size_xyz=size,
        transform_frame_box=transform_base_shelf,
    )


def configured_box(
    *,
    frame_id: str,
    size_xyz,
    center_xyz,
    quaternion_xyzw,
    label: str,
) -> BoxSpec:
    """Create a validated box directly from configured frame coordinates."""

    size = validated_box_size(size_xyz, f"{label}_size_xyz")
    return BoxSpec(
        frame_id=str(frame_id),
        size_xyz=size,
        transform_frame_box=make_transform(center_xyz, quaternion_xyzw),
    )


def shelf_front_plane_error_m(
    size_xyz,
    center_offset_slot_xyz,
) -> float:
    """Return how far the box front face is from slot-frame X=0."""

    size = validated_box_size(size_xyz, "shelf_box_size_xyz")
    offset = np.asarray(center_offset_slot_xyz, dtype=np.float64)
    if offset.shape != (3,) or not np.all(np.isfinite(offset)):
        raise ValueError("shelf_box_center_offset_slot_xyz must be a finite 3D vector")
    return float(offset[0] - 0.5 * size[0])


def local_handoff_error(
    *,
    hardware_measurements_confirmed: bool,
    allow_local_insertion: bool,
    activation_ready: bool,
    activation_fresh: bool,
    global_scene_applied: bool,
    shelf_front_plane_error: float,
    maximum_front_plane_error_m: float,
) -> str | None:
    """Reject a global-to-local scene transition unless every gate is open."""

    if not hardware_measurements_confirmed:
        return "hardware measurements are not confirmed"
    if not allow_local_insertion:
        return "local insertion handoff is disabled"
    if not global_scene_applied:
        return "global approach scene has not been applied"
    if not activation_ready:
        return "policy activation is not ready"
    if not activation_fresh:
        return "policy activation status is missing or stale"
    error = float(shelf_front_plane_error)
    tolerance = float(maximum_front_plane_error_m)
    if not math.isfinite(error) or not math.isfinite(tolerance) or tolerance < 0.0:
        return "shelf front-plane validation parameters are invalid"
    if abs(error) > tolerance:
        return (
            "shelf box front face does not coincide with the slot mouth: "
            f"error={error:.6f} m"
        )
    return None


def scene_status_error(
    status,
    *,
    required_mode: str = LOCAL_INSERTION,
) -> str | None:
    """Validate the runtime scene status consumed by guarded planning."""

    if not isinstance(status, dict):
        return "planning scene status is unavailable"
    if status.get("mode") != required_mode:
        return (
            f"planning scene mode is {status.get('mode')!r}, "
            f"expected {required_mode!r}"
        )
    if not bool(status.get("scene_applied")):
        return "planning scene has not been applied"
    if not bool(status.get("hardware_measurements_confirmed")):
        return "planning scene hardware measurements are unconfirmed"
    objects = status.get("objects")
    if not isinstance(objects, dict):
        return "planning scene object status is unavailable"
    if bool(objects.get("bookshelf_keepout")):
        return "bookshelf keep-out box is still active during local insertion"
    if not bool(objects.get("table")):
        return "table collision object is not active"
    if not bool(objects.get("held_book")):
        return "held-book collision object is not active"
    return None


def global_scene_status_error(status) -> str | None:
    """Require the measured coarse scene used for the global approach plan."""

    if not isinstance(status, dict):
        return "planning scene status is unavailable"
    if status.get("mode") != GLOBAL_APPROACH:
        return (
            f"planning scene mode is {status.get('mode')!r}, "
            f"expected {GLOBAL_APPROACH!r}"
        )
    if not bool(status.get("scene_applied")):
        return "planning scene has not been applied"
    if not bool(status.get("hardware_measurements_confirmed")):
        return "planning scene hardware measurements are unconfirmed"
    objects = status.get("objects")
    if not isinstance(objects, dict):
        return "planning scene object status is unavailable"
    if not bool(objects.get("bookshelf_keepout")):
        return "bookshelf keep-out box is not active during global approach"
    if not bool(objects.get("table")):
        return "table collision object is not active"
    if not bool(objects.get("held_book")):
        return "held-book collision object is not active"
    return None
