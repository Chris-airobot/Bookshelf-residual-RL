"""Pure geometry and fail-closed mode gates for the bookshelf planning scene."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .policy_tool_control_math import make_transform, validated_transform


GLOBAL_APPROACH = "global_approach"
LOCAL_INSERTION = "local_insertion"
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
) -> BoxSpec:
    """Place the coarse bookshelf keep-out box relative to the slot mouth."""

    transform_base_slot = validated_transform(transform_base_slot)
    size = validated_box_size(size_xyz, "shelf_box_size_xyz")
    offset = np.asarray(center_offset_slot_xyz, dtype=np.float64)
    if offset.shape != (3,) or not np.all(np.isfinite(offset)):
        raise ValueError("shelf_box_center_offset_slot_xyz must be a finite 3D vector")
    return BoxSpec(
        frame_id=str(base_frame),
        size_xyz=size,
        transform_frame_box=transform_base_slot @ make_transform(offset),
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
