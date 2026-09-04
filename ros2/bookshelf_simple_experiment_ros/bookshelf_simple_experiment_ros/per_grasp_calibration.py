"""Robust episode-fixed EEF-to-book calibration for the real experiment."""

from __future__ import annotations

import math

import numpy as np


class FreshMarkerSampleGate:
    """Count only fresh, uniquely timestamped marker observations."""

    def __init__(self, maximum_age_s):
        maximum_age = float(maximum_age_s)
        if not math.isfinite(maximum_age) or maximum_age <= 0.0:
            raise ValueError("marker maximum age must be finite and positive")
        self.maximum_age_s = maximum_age
        self.reset()

    def reset(self):
        self.total_reads_attempted = 0
        self.duplicate_samples_rejected = 0
        self.stale_samples_rejected = 0
        self.lookup_samples_rejected = 0
        self._accepted_stamps_ns = set()
        self._accepted_ages_s = []

    def reject_lookup(self):
        self.total_reads_attempted += 1
        self.lookup_samples_rejected += 1

    def accept(self, stamp_ns, now_ns):
        """Return True only once for a nonzero marker stamp within the age limit."""
        self.total_reads_attempted += 1
        stamp = int(stamp_ns)
        age_s = (int(now_ns) - stamp) * 1.0e-9
        if (
            stamp <= 0
            or not math.isfinite(age_s)
            or not 0.0 <= age_s <= self.maximum_age_s
        ):
            self.stale_samples_rejected += 1
            return False
        if stamp in self._accepted_stamps_ns:
            self.duplicate_samples_rejected += 1
            return False
        self._accepted_stamps_ns.add(stamp)
        self._accepted_ages_s.append(age_s)
        return True

    @property
    def accepted_count(self):
        return len(self._accepted_stamps_ns)

    def require_minimum(self, minimum_samples):
        minimum = int(minimum_samples)
        if self.accepted_count < minimum:
            raise ValueError(
                "insufficient fresh unique marker samples: "
                f"{self.accepted_count}/{minimum} required"
            )

    def diagnostics(self, now_ns=None):
        newest_age_s = None
        oldest_age_s = None
        if self._accepted_stamps_ns and now_ns is not None:
            newest_age_s = (int(now_ns) - max(self._accepted_stamps_ns)) * 1.0e-9
            oldest_age_s = (int(now_ns) - min(self._accepted_stamps_ns)) * 1.0e-9
        return {
            "total_reads_attempted": int(self.total_reads_attempted),
            "unique_fresh_samples": int(self.accepted_count),
            "duplicate_samples_rejected": int(self.duplicate_samples_rejected),
            "stale_samples_rejected": int(self.stale_samples_rejected),
            "lookup_samples_rejected": int(self.lookup_samples_rejected),
            "newest_accepted_sample_age_s": newest_age_s,
            "oldest_accepted_sample_age_s": oldest_age_s,
            "minimum_marker_age_at_read_s": (
                float(min(self._accepted_ages_s)) if self._accepted_ages_s else None
            ),
            "maximum_marker_age_at_read_s": (
                float(max(self._accepted_ages_s)) if self._accepted_ages_s else None
            ),
            "marker_max_age_s": float(self.maximum_age_s),
        }


def select_eef_book_transform(per_grasp, fixed):
    """Select the frozen transform, or the explicit reviewed fallback."""
    return np.asarray(fixed if per_grasp is None else per_grasp, dtype=np.float64)


def semantic_held_gripper_observation(measured_value, held_value):
    """Map physical aperture to the simulation held semantic for INSERT only."""
    measured = float(measured_value)
    held = float(held_value)
    if not math.isfinite(measured):
        raise ValueError("measured gripper aperture must be finite")
    if not math.isfinite(held) or not 0.0 <= held <= 1.0:
        raise ValueError("held gripper semantic must lie in [0, 1]")
    return held


def rotation_angle_rad(rotation):
    value = (float(np.trace(rotation)) - 1.0) * 0.5
    return math.acos(float(np.clip(value, -1.0, 1.0)))


def robust_average_transforms(
    transforms, translation_outlier_m=0.005, orientation_outlier_rad=math.radians(5.0)
):
    """Return a robust rigid-transform mean and calibration diagnostics."""

    values = np.asarray(transforms, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (4, 4) or len(values) == 0:
        raise ValueError("calibration requires one or more 4x4 transforms")
    if not np.all(np.isfinite(values)):
        raise ValueError("calibration transforms must be finite")

    center_t = np.median(values[:, :3, 3], axis=0)
    rotation_sum = np.sum(values[:, :3, :3], axis=0)
    u, _, vt = np.linalg.svd(rotation_sum)
    center_r = u @ vt
    if np.linalg.det(center_r) < 0.0:
        u[:, -1] *= -1.0
        center_r = u @ vt

    translation_error = np.linalg.norm(values[:, :3, 3] - center_t, axis=1)
    orientation_error = np.asarray([
        rotation_angle_rad(center_r.T @ value[:3, :3]) for value in values
    ])
    keep = ((translation_error <= float(translation_outlier_m))
            & (orientation_error <= float(orientation_outlier_rad)))
    accepted = values[keep]
    if len(accepted) == 0:
        raise ValueError("all per-grasp calibration samples were rejected")

    mean_t = np.mean(accepted[:, :3, 3], axis=0)
    u, _, vt = np.linalg.svd(np.sum(accepted[:, :3, :3], axis=0))
    mean_r = u @ vt
    if np.linalg.det(mean_r) < 0.0:
        u[:, -1] *= -1.0
        mean_r = u @ vt
    translation_residual = accepted[:, :3, 3] - mean_t
    orientation_residual = np.asarray([
        rotation_angle_rad(mean_r.T @ value[:3, :3]) for value in accepted
    ])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = mean_r
    transform[:3, 3] = mean_t
    return transform, {
        "sample_count": int(len(values)),
        "accepted_count": int(len(accepted)),
        "rejected_count": int(len(values) - len(accepted)),
        "translation_std_m": np.std(translation_residual, axis=0).tolist(),
        "translation_rms_m": float(np.sqrt(np.mean(np.sum(translation_residual**2, axis=1)))),
        "orientation_std_rad": float(np.std(orientation_residual)),
        "orientation_rms_rad": float(np.sqrt(np.mean(orientation_residual**2))),
        "orientation_max_rad": float(np.max(orientation_residual)),
    }
