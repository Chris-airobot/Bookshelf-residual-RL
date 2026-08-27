"""Exact pure geometry for the trained 12-D bookshelf observation.

Source-equivalent to the verified shadow observation helper, while this
isolated package retains no runtime dependency on the deployment stack.
"""

from dataclasses import dataclass
import math

import numpy as np


OBSERVATION_LABELS = (
    "mode",
    "rear_to_mouth",
    "front_to_back",
    "lat_err",
    "z_err",
    "yaw_err",
    "tool_to_book_x",
    "tool_to_book_y",
    "tool_to_book_z",
    "gripper_open",
    "tilt_x",
    "tilt_y",
)


@dataclass(frozen=True)
class ObservationScales:
    rear_to_mouth: float = 0.08
    front_to_back: float = 0.08
    lateral: float = 0.05
    vertical: float = 0.05
    yaw: float = math.radians(30.0)
    tool_to_book: float = 0.25


def _validated_transform(transform) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected transform shape (4, 4), got {transform.shape}.")
    if not np.all(np.isfinite(transform)):
        raise ValueError("Transform contains non-finite values.")
    return transform


def invert_transform(transform) -> np.ndarray:
    transform = _validated_transform(transform)
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -(inverse[:3, :3] @ transform[:3, 3])
    return inverse


def _book_corners_in_slot(transform_slot_book: np.ndarray, book_size) -> np.ndarray:
    depth, thickness, height = np.asarray(book_size, dtype=np.float64)
    if min(depth, thickness, height) <= 0.0:
        raise ValueError("All book dimensions must be positive.")
    half = 0.5 * np.array([depth, thickness, height], dtype=np.float64)
    signs = np.array(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, +1.0],
            [-1.0, +1.0, -1.0],
            [-1.0, +1.0, +1.0],
            [+1.0, -1.0, -1.0],
            [+1.0, -1.0, +1.0],
            [+1.0, +1.0, -1.0],
            [+1.0, +1.0, +1.0],
        ],
        dtype=np.float64,
    )
    local_corners = signs * half
    return (
        transform_slot_book[:3, :3] @ local_corners.T
    ).T + transform_slot_book[:3, 3]


def wrap_to_pi(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def compute_policy_observation(
    transform_slot_book,
    transform_slot_tool,
    *,
    book_size=(0.156, 0.034, 0.236),
    slot_depth=0.20,
    mode_observation=0.0,
    gripper_open=0.0,
    scales=ObservationScales(),
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact unscaled metrics and simulator-scaled observation."""

    transform_slot_book = _validated_transform(transform_slot_book)
    transform_slot_tool = _validated_transform(transform_slot_tool)
    if slot_depth <= 0.0:
        raise ValueError("slot_depth must be positive.")

    corners = _book_corners_in_slot(transform_slot_book, book_size)
    rear_x = float(np.min(corners[:, 0]))
    front_x = float(np.max(corners[:, 0]))
    book_position = transform_slot_book[:3, 3]
    tool_position = transform_slot_tool[:3, 3]
    book_depth_axis = transform_slot_book[:3, 0]
    book_up_axis = transform_slot_book[:3, 2]
    yaw_error = wrap_to_pi(
        math.atan2(float(book_depth_axis[1]), float(book_depth_axis[0]))
    )

    raw = np.array(
        [
            float(mode_observation),
            rear_x,
            float(slot_depth) - front_x,
            -float(book_position[1]),
            float(book_position[2]),
            yaw_error,
            float(tool_position[0] - book_position[0]),
            float(tool_position[1] - book_position[1]),
            float(tool_position[2] - book_position[2]),
            float(np.clip(gripper_open, 0.0, 1.0)),
            float(np.clip(book_up_axis[0], -1.0, 1.0)),
            float(np.clip(book_up_axis[1], -1.0, 1.0)),
        ],
        dtype=np.float64,
    )
    divisors = np.array(
        [
            1.0,
            scales.rear_to_mouth,
            scales.front_to_back,
            scales.lateral,
            scales.vertical,
            scales.yaw,
            scales.tool_to_book,
            scales.tool_to_book,
            scales.tool_to_book,
            1.0,
            1.0,
            1.0,
        ],
        dtype=np.float64,
    )
    if np.any(divisors <= 0.0):
        raise ValueError("Observation scales must be positive.")
    observation = np.clip(raw / divisors, -1.0, 1.0)
    observation[0] = float(np.clip(mode_observation, 0.0, 1.0))
    observation[9] = float(np.clip(gripper_open, 0.0, 1.0))
    return raw.astype(np.float32), observation.astype(np.float32)
