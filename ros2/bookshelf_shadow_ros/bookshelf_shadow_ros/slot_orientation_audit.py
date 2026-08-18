"""Pure orientation statistics for read-only slot detector audits."""

import math

import numpy as np

from .policy_observation_math import (
    normalise_quaternion_xyzw,
    quaternion_xyzw_to_matrix,
)


def _rotation_error_deg(left_quaternion, right_quaternion) -> float:
    left = normalise_quaternion_xyzw(left_quaternion)
    right = normalise_quaternion_xyzw(right_quaternion)
    cosine = float(np.clip(abs(np.dot(left, right)), 0.0, 1.0))
    return math.degrees(2.0 * math.acos(cosine))


def _scalar_statistics(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "p95": float(np.percentile(values, 95.0)),
    }


def _mean_quaternion(quaternions) -> np.ndarray:
    quaternions = np.asarray(quaternions, dtype=np.float64)
    reference = quaternions[0]
    aligned = np.where(
        (quaternions @ reference)[:, None] < 0.0,
        -quaternions,
        quaternions,
    )
    return normalise_quaternion_xyzw(np.mean(aligned, axis=0))


def _axis_metrics(quaternion) -> dict:
    rotation = quaternion_xyzw_to_matrix(quaternion)
    insertion_axis = rotation[:, 0]
    up_axis = rotation[:, 2]
    up_tilt_deg = math.degrees(
        math.acos(float(np.clip(up_axis[2], -1.0, 1.0)))
    )
    insertion_tilt_deg = math.degrees(
        math.asin(float(np.clip(abs(insertion_axis[2]), 0.0, 1.0)))
    )
    return {
        "insertion_axis_base": insertion_axis.astype(float).tolist(),
        "up_axis_base": up_axis.astype(float).tolist(),
        "up_tilt_from_base_vertical_deg": up_tilt_deg,
        "insertion_tilt_from_base_horizontal_deg": insertion_tilt_deg,
    }


class SlotOrientationAuditAccumulator:
    """Separate stable orientation bias from frame-to-frame detector noise."""

    def __init__(
        self,
        *,
        minimum_confidence: float = 0.60,
        stable_spread_p95_deg: float = 1.0,
        meaningful_disagreement_deg: float = 2.0,
    ):
        self.minimum_confidence = float(minimum_confidence)
        self.stable_spread_p95_deg = float(stable_spread_p95_deg)
        self.meaningful_disagreement_deg = float(meaningful_disagreement_deg)
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be in [0, 1]")
        if self.stable_spread_p95_deg < 0.0:
            raise ValueError("stable_spread_p95_deg must be non-negative")
        if self.meaningful_disagreement_deg < 0.0:
            raise ValueError("meaningful_disagreement_deg must be non-negative")
        self.rows = []

    def add(self, live_quaternion_xyzw, reference_quaternion_xyzw, confidence):
        try:
            confidence = float(confidence)
            live = normalise_quaternion_xyzw(live_quaternion_xyzw)
            reference = normalise_quaternion_xyzw(reference_quaternion_xyzw)
            valid = math.isfinite(confidence) and confidence >= self.minimum_confidence
        except (TypeError, ValueError):
            confidence = math.nan
            live = None
            reference = None
            valid = False
        self.rows.append(
            {
                "valid": bool(valid),
                "confidence": confidence,
                "live_quaternion_xyzw": live,
                "reference_quaternion_xyzw": reference,
            }
        )
        return bool(valid)

    def summary(self) -> dict:
        valid_rows = [row for row in self.rows if row["valid"]]
        result = {
            "samples": len(self.rows),
            "accepted_samples": len(valid_rows),
            "rejected_samples": len(self.rows) - len(valid_rows),
            "minimum_confidence": self.minimum_confidence,
        }
        if not valid_rows:
            result["classification"] = "insufficient_valid_samples"
            return result

        live = np.stack([row["live_quaternion_xyzw"] for row in valid_rows])
        reference = np.stack(
            [row["reference_quaternion_xyzw"] for row in valid_rows]
        )
        mean_live = _mean_quaternion(live)
        mean_reference = _mean_quaternion(reference)
        spread = [_rotation_error_deg(item, mean_live) for item in live]
        disagreement = [
            _rotation_error_deg(live_item, reference_item)
            for live_item, reference_item in zip(live, reference)
        ]
        spread_statistics = _scalar_statistics(spread)
        disagreement_statistics = _scalar_statistics(disagreement)
        stable = spread_statistics["p95"] <= self.stable_spread_p95_deg
        meaningfully_different = (
            disagreement_statistics["mean"] >= self.meaningful_disagreement_deg
        )
        if stable and meaningfully_different:
            classification = "stable_systematic_orientation_difference"
        elif not stable:
            classification = "temporally_variable_orientation_detection"
        else:
            classification = "stable_orientation_agreement"

        result.update(
            {
                "classification": classification,
                "temporally_stable": stable,
                "meaningful_reference_disagreement": meaningfully_different,
                "confidence": _scalar_statistics(
                    [row["confidence"] for row in valid_rows]
                ),
                "mean_live_quaternion_xyzw": mean_live.astype(float).tolist(),
                "mean_reference_quaternion_xyzw": mean_reference.astype(float).tolist(),
                "live_orientation_spread_deg": spread_statistics,
                "live_to_reference_rotation_error_deg": disagreement_statistics,
                "mean_live_axes": _axis_metrics(mean_live),
                "mean_reference_axes": _axis_metrics(mean_reference),
                "thresholds": {
                    "stable_spread_p95_deg": self.stable_spread_p95_deg,
                    "meaningful_disagreement_deg": self.meaningful_disagreement_deg,
                },
            }
        )
        return result
