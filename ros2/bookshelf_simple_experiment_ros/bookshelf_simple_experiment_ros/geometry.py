"""Pure rigid-transform math for the simple pre-insertion workflow."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

REVIEWED_EEF_BOOK_TRANSLATION_XYZ = (
    0.006189808263520789,
    0.004397635899244547,
    0.18076520526773382,
)
REVIEWED_EEF_BOOK_QUATERNION_XYZW = (
    0.7170947434170492,
    0.01281329455160485,
    0.6961397093730864,
    0.03162994594249451,
)


@dataclass(frozen=True)
class PreinsertTarget:
    transform_slot_book: np.ndarray
    transform_base_book: np.ndarray
    transform_base_eef: np.ndarray
    transform_base_tcp: np.ndarray


def _validated_transform(value) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("expected a finite 4x4 transform")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-9):
        raise ValueError("invalid homogeneous transform")
    return transform


def quaternion_xyzw_to_matrix(quaternion) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("expected a finite xyzw quaternion")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1.0e-12:
        raise ValueError("zero quaternion")
    x, y, z, w = quaternion / norm
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def matrix_to_quaternion_xyzw(matrix) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("expected a finite 3x3 rotation")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        result = [(matrix[2,1]-matrix[1,2])/scale,
                  (matrix[0,2]-matrix[2,0])/scale,
                  (matrix[1,0]-matrix[0,1])/scale, 0.25*scale]
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(max(1+matrix[0,0]-matrix[1,1]-matrix[2,2], 0.0))*2
            result = [0.25*scale, (matrix[0,1]+matrix[1,0])/scale,
                      (matrix[0,2]+matrix[2,0])/scale, (matrix[2,1]-matrix[1,2])/scale]
        elif index == 1:
            scale = math.sqrt(max(1+matrix[1,1]-matrix[0,0]-matrix[2,2], 0.0))*2
            result = [(matrix[0,1]+matrix[1,0])/scale, 0.25*scale,
                      (matrix[1,2]+matrix[2,1])/scale, (matrix[0,2]-matrix[2,0])/scale]
        else:
            scale = math.sqrt(max(1+matrix[2,2]-matrix[0,0]-matrix[1,1], 0.0))*2
            result = [(matrix[0,2]+matrix[2,0])/scale,
                      (matrix[1,2]+matrix[2,1])/scale, 0.25*scale,
                      (matrix[1,0]-matrix[0,1])/scale]
    result = np.asarray(result, dtype=np.float64)
    return result / np.linalg.norm(result)


def make_transform(translation, quaternion_xyzw=None) -> np.ndarray:
    translation = np.asarray(translation, dtype=np.float64)
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise ValueError("expected a finite xyz translation")
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = translation
    if quaternion_xyzw is not None:
        result[:3, :3] = quaternion_xyzw_to_matrix(quaternion_xyzw)
    return result


def invert_transform(transform) -> np.ndarray:
    transform = _validated_transform(transform)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -(result[:3, :3] @ transform[:3, 3])
    return result


def reviewed_eef_book_transform() -> np.ndarray:
    """Return the approved transform whose convention is T_eef_book."""
    return make_transform(
        REVIEWED_EEF_BOOK_TRANSLATION_XYZ,
        REVIEWED_EEF_BOOK_QUATERNION_XYZW,
    )


def compute_preinsert_target(
    transform_base_slot,
    transform_eef_book,
    transform_eef_tcp,
    *,
    book_depth_m: float = 0.156,
    standoff_m: float = 0.030,
    vertical_offset_m: float = 0.006,
) -> PreinsertTarget:
    """Place the book front face before the mouth; slot +X enters shelf."""
    if book_depth_m <= 0.0 or standoff_m < 0.0:
        raise ValueError("book depth must be positive and standoff non-negative")
    base_slot = _validated_transform(transform_base_slot)
    eef_book = _validated_transform(transform_eef_book)
    eef_tcp = _validated_transform(transform_eef_tcp)
    slot_book = make_transform(
        [-(0.5 * book_depth_m + standoff_m), 0.0, vertical_offset_m]
    )
    base_book = base_slot @ slot_book
    base_eef = base_book @ invert_transform(eef_book)
    base_tcp = base_eef @ eef_tcp
    return PreinsertTarget(slot_book, base_book, base_eef, base_tcp)
