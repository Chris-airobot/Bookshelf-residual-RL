"""Pure, read-only comparison helpers for the static-slot environment check."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class SlotCheckTolerances:
    maximum_translation_error_m: float
    maximum_rotation_error_deg: float
    maximum_width_error_m: float
    minimum_confidence: float

    def __post_init__(self):
        if self.maximum_translation_error_m < 0.0:
            raise ValueError("maximum_translation_error_m must be non-negative")
        if self.maximum_rotation_error_deg < 0.0:
            raise ValueError("maximum_rotation_error_deg must be non-negative")
        if self.maximum_width_error_m < 0.0:
            raise ValueError("maximum_width_error_m must be non-negative")
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be in [0, 1]")


def compare_slot_measurement(
    reference_transform,
    measured_transform,
    *,
    reference_width_m: float,
    measured_width_m: float,
    confidence: float,
    tolerances: SlotCheckTolerances,
) -> dict:
    """Compare one live slot estimate with the immutable configured reference."""

    reference = _validated_transform(reference_transform)
    measured = _validated_transform(measured_transform)
    reference_width_m = float(reference_width_m)
    measured_width_m = float(measured_width_m)
    confidence = float(confidence)
    scalar_values = (reference_width_m, measured_width_m, confidence)
    if not all(math.isfinite(value) for value in scalar_values):
        raise ValueError("Slot width and confidence values must be finite")
    if reference_width_m <= 0.0 or measured_width_m <= 0.0:
        raise ValueError("Slot widths must be positive")

    translation_error_m = float(
        np.linalg.norm(measured[:3, 3] - reference[:3, 3])
    )
    relative_rotation = reference[:3, :3].T @ measured[:3, :3]
    cosine = float(np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0))
    rotation_error_deg = math.degrees(math.acos(cosine))
    width_error_m = abs(measured_width_m - reference_width_m)

    failures = []
    epsilon = 1.0e-12
    if confidence + epsilon < tolerances.minimum_confidence:
        failures.append("confidence")
    if translation_error_m > tolerances.maximum_translation_error_m + epsilon:
        failures.append("translation")
    if rotation_error_deg > tolerances.maximum_rotation_error_deg + epsilon:
        failures.append("rotation")
    if width_error_m > tolerances.maximum_width_error_m + epsilon:
        failures.append("width")

    return {
        "matches": not failures,
        "failed_checks": failures,
        "translation_error_m": translation_error_m,
        "rotation_error_deg": rotation_error_deg,
        "width_error_m": width_error_m,
        "measured_width_m": measured_width_m,
        "confidence": confidence,
    }


class ConsecutiveMatchGate:
    """Require uninterrupted agreement; a mismatch immediately fails closed."""

    def __init__(self, required_matches: int):
        self.required_matches = int(required_matches)
        if self.required_matches < 1:
            raise ValueError("required_matches must be at least one")
        self.matching_samples = 0

    @property
    def passed(self) -> bool:
        return self.matching_samples >= self.required_matches

    def update(self, matches: bool) -> bool:
        if matches:
            self.matching_samples = min(
                self.matching_samples + 1, self.required_matches
            )
        else:
            self.matching_samples = 0
        return self.passed

    def reset(self) -> None:
        self.matching_samples = 0


def _validated_transform(value) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform, got {transform.shape}")
    if not np.all(np.isfinite(transform)):
        raise ValueError("Transform contains non-finite values")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6):
        raise ValueError("Transform rotation is not orthonormal")
    if np.linalg.det(rotation) < 0.999999:
        raise ValueError("Transform rotation must be right-handed")
    return transform
