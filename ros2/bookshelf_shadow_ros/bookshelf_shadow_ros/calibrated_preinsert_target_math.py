"""Pure geometry for a calibrated, read-only pre-insertion target report."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .policy_observation_math import (
    OBSERVATION_LABELS,
    ObservationScales,
    compute_policy_observation,
    invert_transform,
    make_transform,
    matrix_to_quaternion_xyzw,
)


EXPECTED_PREINSERT_CLIPPED_LABELS = frozenset(
    ("rear_to_mouth", "front_to_back")
)


@dataclass(frozen=True)
class PreinsertTargetSpec:
    """Geometry shared with the trained residual insertion environment."""

    book_size: tuple[float, float, float] = (0.156, 0.034, 0.236)
    slot_depth: float = 0.20
    standoff: float = 0.030
    vertical_offset: float = 0.006
    gripper_open: float = 0.0
    observation_scales: ObservationScales = ObservationScales()


@dataclass(frozen=True)
class CalibratedPreinsertTarget:
    """All transforms and policy diagnostics for one static target."""

    transform_slot_book_target: np.ndarray
    transform_base_book_target: np.ndarray
    transform_base_eef_target: np.ndarray
    transform_base_policy_tool_target: np.ndarray
    transform_slot_eef_target: np.ndarray
    transform_slot_policy_tool_target: np.ndarray
    raw_metrics: np.ndarray
    observation_12d: np.ndarray
    clipped_labels: tuple[str, ...]
    expected_clipped_labels: tuple[str, ...]
    unexpected_clipped_labels: tuple[str, ...]


def compute_calibrated_preinsert_target(
    transform_base_slot,
    transform_eef_book,
    *,
    transform_eef_policy_tool=None,
    spec=PreinsertTargetSpec(),
) -> CalibratedPreinsertTarget:
    """Solve the EEF pose that places the calibrated book at pre-insertion.

    The slot origin is the desired book centre at the shelf mouth. A positive
    slot X points into the shelf, so a book whose front face is ``standoff``
    before the mouth has centre X ``-(depth / 2 + standoff)``.
    """

    transform_base_slot = _validated_transform(transform_base_slot)
    transform_eef_book = _validated_transform(transform_eef_book)
    transform_eef_policy_tool = (
        np.eye(4, dtype=np.float64)
        if transform_eef_policy_tool is None
        else _validated_transform(transform_eef_policy_tool)
    )
    _validate_spec(spec)

    book_depth = float(spec.book_size[0])
    transform_slot_book_target = make_transform(
        [-(0.5 * book_depth + spec.standoff), 0.0, spec.vertical_offset]
    )
    transform_base_book_target = transform_base_slot @ transform_slot_book_target
    transform_base_eef_target = (
        transform_base_book_target @ invert_transform(transform_eef_book)
    )
    transform_slot_eef_target = (
        invert_transform(transform_base_slot) @ transform_base_eef_target
    )
    transform_base_policy_tool_target = (
        transform_base_eef_target @ transform_eef_policy_tool
    )
    transform_slot_policy_tool_target = (
        invert_transform(transform_base_slot) @ transform_base_policy_tool_target
    )

    raw_metrics, observation_12d = compute_policy_observation(
        transform_slot_book_target,
        transform_slot_policy_tool_target,
        book_size=spec.book_size,
        slot_depth=spec.slot_depth,
        mode_observation=0.0,
        gripper_open=spec.gripper_open,
        scales=spec.observation_scales,
    )
    clipped = _clipped_labels(observation_12d)
    expected = tuple(
        label for label in clipped if label in EXPECTED_PREINSERT_CLIPPED_LABELS
    )
    unexpected = tuple(
        label for label in clipped if label not in EXPECTED_PREINSERT_CLIPPED_LABELS
    )
    return CalibratedPreinsertTarget(
        transform_slot_book_target=transform_slot_book_target,
        transform_base_book_target=transform_base_book_target,
        transform_base_eef_target=transform_base_eef_target,
        transform_base_policy_tool_target=transform_base_policy_tool_target,
        transform_slot_eef_target=transform_slot_eef_target,
        transform_slot_policy_tool_target=transform_slot_policy_tool_target,
        raw_metrics=raw_metrics,
        observation_12d=observation_12d,
        clipped_labels=clipped,
        expected_clipped_labels=expected,
        unexpected_clipped_labels=unexpected,
    )


def compute_preserved_tcp_orientation_preinsert_target(
    transform_base_slot,
    transform_eef_book,
    transform_base_eef_current,
    transform_base_tcp_current,
    *,
    transform_eef_policy_tool=None,
    spec=PreinsertTargetSpec(),
) -> tuple[CalibratedPreinsertTarget, dict]:
    """Place the book centre at pre-insertion while preserving TCP rotation.

    The current ``link_eef -> link_tcp`` and calibrated ``link_eef -> book``
    transforms determine the book offset from TCP. The target TCP keeps its
    current base-frame rotation exactly; only its translation changes.
    """

    transform_base_slot = _validated_transform(transform_base_slot)
    transform_eef_book = _validated_transform(transform_eef_book)
    transform_base_eef_current = _validated_transform(transform_base_eef_current)
    transform_base_tcp_current = _validated_transform(transform_base_tcp_current)
    transform_eef_policy_tool = (
        np.eye(4, dtype=np.float64)
        if transform_eef_policy_tool is None
        else _validated_transform(transform_eef_policy_tool)
    )
    _validate_spec(spec)

    transform_eef_tcp = (
        invert_transform(transform_base_eef_current) @ transform_base_tcp_current
    )
    transform_tcp_book = invert_transform(transform_eef_tcp) @ transform_eef_book

    book_depth = float(spec.book_size[0])
    transform_slot_book_reference = make_transform(
        [-(0.5 * book_depth + spec.standoff), 0.0, spec.vertical_offset]
    )
    transform_base_book_reference = (
        transform_base_slot @ transform_slot_book_reference
    )

    transform_base_tcp_target = np.eye(4, dtype=np.float64)
    transform_base_tcp_target[:3, :3] = transform_base_tcp_current[:3, :3]
    transform_base_tcp_target[:3, 3] = (
        transform_base_book_reference[:3, 3]
        - transform_base_tcp_target[:3, :3] @ transform_tcp_book[:3, 3]
    )
    transform_base_eef_target = (
        transform_base_tcp_target @ invert_transform(transform_eef_tcp)
    )
    transform_base_book_target = transform_base_eef_target @ transform_eef_book
    transform_slot_base = invert_transform(transform_base_slot)
    transform_slot_book_target = transform_slot_base @ transform_base_book_target
    transform_base_policy_tool_target = (
        transform_base_eef_target @ transform_eef_policy_tool
    )
    transform_slot_eef_target = transform_slot_base @ transform_base_eef_target
    transform_slot_policy_tool_target = (
        transform_slot_base @ transform_base_policy_tool_target
    )

    raw_metrics, observation_12d = compute_policy_observation(
        transform_slot_book_target,
        transform_slot_policy_tool_target,
        book_size=spec.book_size,
        slot_depth=spec.slot_depth,
        mode_observation=0.0,
        gripper_open=spec.gripper_open,
        scales=spec.observation_scales,
    )
    clipped = _clipped_labels(observation_12d)
    expected = tuple(
        label for label in clipped if label in EXPECTED_PREINSERT_CLIPPED_LABELS
    )
    unexpected = tuple(
        label for label in clipped if label not in EXPECTED_PREINSERT_CLIPPED_LABELS
    )
    target = CalibratedPreinsertTarget(
        transform_slot_book_target=transform_slot_book_target,
        transform_base_book_target=transform_base_book_target,
        transform_base_eef_target=transform_base_eef_target,
        transform_base_policy_tool_target=transform_base_policy_tool_target,
        transform_slot_eef_target=transform_slot_eef_target,
        transform_slot_policy_tool_target=transform_slot_policy_tool_target,
        raw_metrics=raw_metrics,
        observation_12d=observation_12d,
        clipped_labels=clipped,
        expected_clipped_labels=expected,
        unexpected_clipped_labels=unexpected,
    )
    orientation_error = (
        transform_slot_book_reference[:3, :3].T
        @ transform_slot_book_target[:3, :3]
    )
    diagnostics = {
        "transform_eef_tcp": transform_eef_tcp,
        "transform_tcp_book": transform_tcp_book,
        "transform_base_tcp_current": transform_base_tcp_current,
        "transform_base_tcp_target": transform_base_tcp_target,
        "transform_slot_book_reference": transform_slot_book_reference,
        "tcp_orientation_change_deg": rotation_angle_deg(
            transform_base_tcp_current[:3, :3].T
            @ transform_base_tcp_target[:3, :3]
        ),
        "book_orientation_error_deg": rotation_angle_deg(orientation_error),
        "book_center_error_m": float(
            np.linalg.norm(
                transform_base_book_target[:3, 3]
                - transform_base_book_reference[:3, 3]
            )
        ),
    }
    return target, diagnostics


def compare_current_eef_to_target(
    transform_base_eef_current,
    transform_base_slot,
    transform_eef_book,
    target: CalibratedPreinsertTarget,
    *,
    transform_eef_policy_tool=None,
    spec=PreinsertTargetSpec(),
) -> dict:
    """Compare a measured EEF pose with the calibrated target in base/slot frames."""

    transform_base_eef_current = _validated_transform(transform_base_eef_current)
    transform_base_slot = _validated_transform(transform_base_slot)
    transform_eef_book = _validated_transform(transform_eef_book)
    transform_eef_policy_tool = (
        np.eye(4, dtype=np.float64)
        if transform_eef_policy_tool is None
        else _validated_transform(transform_eef_policy_tool)
    )
    transform_base_book_current = transform_base_eef_current @ transform_eef_book
    transform_base_policy_tool_current = (
        transform_base_eef_current @ transform_eef_policy_tool
    )
    transform_slot_base = invert_transform(transform_base_slot)
    transform_slot_book_current = transform_slot_base @ transform_base_book_current
    transform_slot_policy_tool_current = (
        transform_slot_base @ transform_base_policy_tool_current
    )
    raw_metrics, observation_12d = compute_policy_observation(
        transform_slot_book_current,
        transform_slot_policy_tool_current,
        book_size=spec.book_size,
        slot_depth=spec.slot_depth,
        mode_observation=0.0,
        gripper_open=spec.gripper_open,
        scales=spec.observation_scales,
    )

    transform_current_target = (
        invert_transform(transform_base_eef_current)
        @ target.transform_base_eef_target
    )
    base_translation_delta = (
        target.transform_base_eef_target[:3, 3]
        - transform_base_eef_current[:3, 3]
    )
    clipped = _clipped_labels(observation_12d)
    return {
        "transform_base_eef_current": transform_base_eef_current,
        "transform_base_book_current": transform_base_book_current,
        "transform_base_policy_tool_current": transform_base_policy_tool_current,
        "transform_slot_book_current": transform_slot_book_current,
        "transform_current_eef_to_target_eef": transform_current_target,
        "target_minus_current_translation_base_m": base_translation_delta,
        "target_minus_current_translation_norm_m": float(
            np.linalg.norm(base_translation_delta)
        ),
        "target_minus_current_rotation_deg": rotation_angle_deg(
            transform_current_target[:3, :3]
        ),
        "raw_metrics": raw_metrics,
        "observation_12d": observation_12d,
        "clipped_labels": clipped,
    }


def calibration_sensitivity(
    transform_base_slot,
    transform_eef_book,
    *,
    transform_eef_policy_tool=None,
    spec=PreinsertTargetSpec(),
    samples=2000,
    translation_uncertainty_m=0.002,
    rotation_uncertainty_deg=2.0,
    seed=42,
) -> dict:
    """Propagate bounded EEF-to-book calibration errors through the target.

    The commanded EEF target is solved with the nominal calibration. Each
    sample then treats the true rigid grasp as nominal calibration followed by
    a random local translation and axis-angle perturbation.
    """

    samples = int(samples)
    translation_uncertainty_m = float(translation_uncertainty_m)
    rotation_uncertainty_deg = float(rotation_uncertainty_deg)
    if samples <= 0:
        raise ValueError("samples must be positive.")
    if translation_uncertainty_m < 0.0 or rotation_uncertainty_deg < 0.0:
        raise ValueError("Calibration uncertainties must be non-negative.")

    target = compute_calibrated_preinsert_target(
        transform_base_slot,
        transform_eef_book,
        transform_eef_policy_tool=transform_eef_policy_tool,
        spec=spec,
    )
    transform_slot_base = invert_transform(transform_base_slot)
    desired = target.transform_slot_book_target
    rng = np.random.default_rng(int(seed))
    translation_errors = np.empty((samples, 3), dtype=np.float64)
    translation_norms = np.empty(samples, dtype=np.float64)
    rotation_errors = np.empty(samples, dtype=np.float64)

    for index in range(samples):
        local_translation = rng.uniform(
            -translation_uncertainty_m,
            translation_uncertainty_m,
            size=3,
        )
        local_rotation = _random_bounded_axis_angle(
            rng, math.radians(rotation_uncertainty_deg)
        )
        perturbation = np.eye(4, dtype=np.float64)
        perturbation[:3, :3] = local_rotation
        perturbation[:3, 3] = local_translation
        actual_eef_book = transform_eef_book @ perturbation
        actual_base_book = target.transform_base_eef_target @ actual_eef_book
        actual_slot_book = transform_slot_base @ actual_base_book
        desired_actual = invert_transform(desired) @ actual_slot_book
        translation_errors[index] = desired_actual[:3, 3]
        translation_norms[index] = np.linalg.norm(desired_actual[:3, 3])
        rotation_errors[index] = rotation_angle_deg(desired_actual[:3, :3])

    return {
        "samples": samples,
        "seed": int(seed),
        "translation_uncertainty_per_axis_m": translation_uncertainty_m,
        "rotation_uncertainty_deg": rotation_uncertainty_deg,
        "translation_error_xyz_m": {
            axis: _statistics(translation_errors[:, axis_index], absolute=True)
            for axis_index, axis in enumerate(("x", "y", "z"))
        },
        "translation_error_norm_m": _statistics(translation_norms),
        "rotation_error_deg": _statistics(rotation_errors),
    }


def transform_to_dict(transform) -> dict:
    """Return a JSON-ready translation/quaternion representation."""

    transform = _validated_transform(transform)
    quaternion = matrix_to_quaternion_xyzw(transform[:3, :3])
    return {
        "translation_xyz_m": [float(value) for value in transform[:3, 3]],
        "quaternion_xyzw": [float(value) for value in quaternion],
        "matrix": [[float(value) for value in row] for row in transform],
    }


def labelled_values(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(OBSERVATION_LABELS),):
        raise ValueError("Expected one value for every observation label.")
    return {
        label: float(value) for label, value in zip(OBSERVATION_LABELS, values)
    }


def rotation_angle_deg(rotation) -> float:
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("Rotation must have shape (3, 3).")
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    if cosine >= 1.0 - 1.0e-12:
        return 0.0
    return math.degrees(math.acos(cosine))


def _clipped_labels(observation, tolerance=1.0e-6) -> tuple[str, ...]:
    observation = np.asarray(observation, dtype=np.float64)
    return tuple(
        label
        for label, value in zip(OBSERVATION_LABELS, observation)
        if abs(float(value)) >= 1.0 - tolerance
    )


def _random_bounded_axis_angle(rng, maximum_angle_rad: float) -> np.ndarray:
    if maximum_angle_rad == 0.0:
        return np.eye(3, dtype=np.float64)
    axis = rng.normal(size=3)
    norm = float(np.linalg.norm(axis))
    while norm < 1.0e-12:
        axis = rng.normal(size=3)
        norm = float(np.linalg.norm(axis))
    axis /= norm
    angle = float(rng.uniform(-maximum_angle_rad, maximum_angle_rad))
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    return (
        np.eye(3, dtype=np.float64)
        + math.sin(angle) * cross
        + (1.0 - math.cos(angle)) * (cross @ cross)
    )


def _statistics(values, *, absolute=False) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if absolute:
        values = np.abs(values)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95.0)),
        "max": float(np.max(values)),
    }


def _validate_spec(spec: PreinsertTargetSpec):
    book_size = np.asarray(spec.book_size, dtype=np.float64)
    if book_size.shape != (3,) or np.any(book_size <= 0.0):
        raise ValueError("book_size must contain three positive dimensions.")
    if spec.slot_depth <= 0.0:
        raise ValueError("slot_depth must be positive.")
    if spec.standoff < 0.0:
        raise ValueError("standoff must be non-negative.")


def _validated_transform(transform) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("Transform must be a finite 4x4 matrix.")
    return transform
