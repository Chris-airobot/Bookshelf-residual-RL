"""Pure fail-closed handoff logic for activating the local insertion policy."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from .policy_observation_math import OBSERVATION_LABELS


@dataclass(frozen=True)
class PolicyActivationLimits:
    maximum_abs_normalized_observation: float = 5.0
    minimum_rear_to_mouth_m: float = -0.26
    maximum_rear_to_mouth_m: float = -0.10
    minimum_front_to_back_m: float = 0.12
    maximum_front_to_back_m: float = 0.32
    maximum_abs_lateral_error_m: float = 0.025
    maximum_abs_vertical_error_m: float = 0.030
    maximum_abs_yaw_error_rad: float = math.radians(20.0)
    maximum_gripper_open: float = 0.25
    required_mode: float = 0.0
    mode_tolerance: float = 1.0e-6


@dataclass(frozen=True)
class ActivationEnvelope:
    lower: np.ndarray
    upper: np.ndarray
    source: str
    metadata: dict


@dataclass(frozen=True)
class PolicyActivationEvaluation:
    ready: bool
    reasons: tuple[str, ...]
    normalized_outliers: dict[str, float]
    envelope_outliers: dict[str, dict[str, float]]
    geometry: dict[str, float]


@dataclass(frozen=True)
class PolicyActivationDecision:
    ready: bool
    instantaneous_ready: bool
    consecutive_ready_samples: int
    required_stable_samples: int
    evaluation: PolicyActivationEvaluation


def activation_allows_policy_calculation(
    checks_passed: bool,
    block_on_activation_checks: bool,
) -> bool:
    """Return whether diagnostic activation checks may block calculation."""

    return bool(checks_passed) or not bool(block_on_activation_checks)


def load_activation_envelope(path) -> ActivationEnvelope:
    path = Path(path).expanduser().resolve()
    document = json.loads(path.read_text(encoding="utf-8"))
    if int(document.get("schema_version", 0)) != 1:
        raise ValueError("Activation envelope schema_version must be 1.")
    if tuple(document.get("labels", ())) != tuple(OBSERVATION_LABELS):
        raise ValueError("Activation envelope labels do not match the 12D policy.")
    lower = _vector(document.get("lower"), "activation envelope lower")
    upper = _vector(document.get("upper"), "activation envelope upper")
    if np.any(lower >= upper):
        raise ValueError("Activation envelope lower bounds must be below upper bounds.")
    return ActivationEnvelope(
        lower=lower,
        upper=upper,
        source=str(document.get("source", path)),
        metadata=dict(document.get("metadata", {})),
    )


def build_activation_envelope(
    normalized_observations,
    *,
    lower_percentile=0.5,
    upper_percentile=99.5,
    margin=0.50,
    maximum_abs_bound=5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build conservative per-channel bounds from simulator-local observations."""

    values = np.asarray(normalized_observations, dtype=np.float64)
    width = len(OBSERVATION_LABELS)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != width:
        raise ValueError(
            f"normalized_observations must have shape (N, {width}) with N >= 2."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("normalized_observations contains non-finite values.")
    lower_percentile = float(lower_percentile)
    upper_percentile = float(upper_percentile)
    margin = float(margin)
    maximum_abs_bound = float(maximum_abs_bound)
    if not 0.0 <= lower_percentile < upper_percentile <= 100.0:
        raise ValueError("Activation-envelope percentiles are invalid.")
    if margin < 0.0 or maximum_abs_bound <= 0.0:
        raise ValueError("Envelope margin and absolute bound are invalid.")

    lower = np.percentile(values, lower_percentile, axis=0) - margin
    upper = np.percentile(values, upper_percentile, axis=0) + margin
    lower = np.maximum(lower, -maximum_abs_bound)
    upper = np.minimum(upper, maximum_abs_bound)
    if np.any(lower >= upper):
        labels = [
            label
            for label, low, high in zip(OBSERVATION_LABELS, lower, upper)
            if low >= high
        ]
        raise ValueError(f"Derived activation envelope collapsed for {labels}.")
    return lower, upper


def evaluate_policy_activation(
    observation,
    normalized_observation,
    raw_metrics,
    *,
    limits=PolicyActivationLimits(),
    envelope: ActivationEnvelope | None = None,
    require_envelope=False,
) -> PolicyActivationEvaluation:
    observation = _vector(observation, "observation")
    normalized = _vector(normalized_observation, "normalized_observation")
    raw = _vector(raw_metrics, "raw_metrics")
    _validate_limits(limits)

    values = dict(zip(OBSERVATION_LABELS, raw))
    reasons = []
    normalized_outliers = {
        label: float(value)
        for label, value in zip(OBSERVATION_LABELS, normalized)
        if abs(float(value)) > limits.maximum_abs_normalized_observation
    }
    if normalized_outliers:
        reasons.append(
            "normalized observation exceeds configured magnitude: "
            f"{sorted(normalized_outliers)}"
        )

    envelope_outliers = {}
    if envelope is None:
        if require_envelope:
            reasons.append("simulator activation envelope is required but unavailable")
    else:
        for index, label in enumerate(OBSERVATION_LABELS):
            value = float(normalized[index])
            lower = float(envelope.lower[index])
            upper = float(envelope.upper[index])
            if value < lower or value > upper:
                envelope_outliers[label] = {
                    "value": value,
                    "lower": lower,
                    "upper": upper,
                }
        if envelope_outliers:
            reasons.append(
                "normalized observation is outside simulator envelope: "
                f"{sorted(envelope_outliers)}"
            )

    geometry_checks = (
        (
            limits.minimum_rear_to_mouth_m
            <= values["rear_to_mouth"]
            <= limits.maximum_rear_to_mouth_m,
            "rear_to_mouth is outside the local-policy approach range",
        ),
        (
            limits.minimum_front_to_back_m
            <= values["front_to_back"]
            <= limits.maximum_front_to_back_m,
            "front_to_back is outside the local-policy approach range",
        ),
        (
            abs(values["lat_err"]) <= limits.maximum_abs_lateral_error_m,
            "lateral error exceeds the local-policy limit",
        ),
        (
            abs(values["z_err"]) <= limits.maximum_abs_vertical_error_m,
            "vertical error exceeds the local-policy limit",
        ),
        (
            abs(values["yaw_err"]) <= limits.maximum_abs_yaw_error_rad,
            "yaw error exceeds the local-policy limit",
        ),
        (
            values["gripper_open"] <= limits.maximum_gripper_open,
            "gripper is not sufficiently closed for local insertion",
        ),
        (
            abs(values["mode"] - limits.required_mode) <= limits.mode_tolerance,
            "policy mode is not the configured insertion mode",
        ),
    )
    reasons.extend(message for passed, message in geometry_checks if not passed)
    geometry = {
        name: float(values[name])
        for name in (
            "mode",
            "rear_to_mouth",
            "front_to_back",
            "lat_err",
            "z_err",
            "yaw_err",
            "gripper_open",
        )
    }
    return PolicyActivationEvaluation(
        ready=not reasons,
        reasons=tuple(reasons),
        normalized_outliers=normalized_outliers,
        envelope_outliers=envelope_outliers,
        geometry=geometry,
    )


class PolicyActivationTracker:
    """Require consecutive acceptable samples before enabling local inference."""

    def __init__(self, required_stable_samples=10):
        self.required_stable_samples = int(required_stable_samples)
        if self.required_stable_samples < 1:
            raise ValueError("required_stable_samples must be at least one.")
        self.consecutive_ready_samples = 0

    def reset(self):
        self.consecutive_ready_samples = 0

    def update(self, evaluation: PolicyActivationEvaluation) -> PolicyActivationDecision:
        if evaluation.ready:
            self.consecutive_ready_samples += 1
        else:
            self.reset()
        return PolicyActivationDecision(
            ready=(
                evaluation.ready
                and self.consecutive_ready_samples >= self.required_stable_samples
            ),
            instantaneous_ready=evaluation.ready,
            consecutive_ready_samples=self.consecutive_ready_samples,
            required_stable_samples=self.required_stable_samples,
            evaluation=evaluation,
        )


def activation_decision_dict(decision: PolicyActivationDecision) -> dict:
    evaluation = decision.evaluation
    return {
        "ready": decision.ready,
        "instantaneous_ready": decision.instantaneous_ready,
        "consecutive_ready_samples": decision.consecutive_ready_samples,
        "required_stable_samples": decision.required_stable_samples,
        "reasons": list(evaluation.reasons),
        "normalized_outliers": evaluation.normalized_outliers,
        "envelope_outliers": evaluation.envelope_outliers,
        "geometry": evaluation.geometry,
    }


def _vector(value, name):
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    if value.shape != (len(OBSERVATION_LABELS),) or not np.all(np.isfinite(value)):
        raise ValueError(
            f"{name} must be a finite vector with shape ({len(OBSERVATION_LABELS)},)."
        )
    return value


def _validate_limits(limits):
    finite = (
        limits.maximum_abs_normalized_observation,
        limits.minimum_rear_to_mouth_m,
        limits.maximum_rear_to_mouth_m,
        limits.minimum_front_to_back_m,
        limits.maximum_front_to_back_m,
        limits.maximum_abs_lateral_error_m,
        limits.maximum_abs_vertical_error_m,
        limits.maximum_abs_yaw_error_rad,
        limits.maximum_gripper_open,
        limits.required_mode,
        limits.mode_tolerance,
    )
    if not all(math.isfinite(float(value)) for value in finite):
        raise ValueError("Policy activation limits must be finite.")
    if limits.maximum_abs_normalized_observation <= 0.0:
        raise ValueError("maximum_abs_normalized_observation must be positive.")
    if limits.minimum_rear_to_mouth_m >= limits.maximum_rear_to_mouth_m:
        raise ValueError("rear_to_mouth activation range is invalid.")
    if limits.minimum_front_to_back_m >= limits.maximum_front_to_back_m:
        raise ValueError("front_to_back activation range is invalid.")
    if min(
        limits.maximum_abs_lateral_error_m,
        limits.maximum_abs_vertical_error_m,
        limits.maximum_abs_yaw_error_rad,
        limits.mode_tolerance,
    ) < 0.0:
        raise ValueError("Policy activation tolerances cannot be negative.")
    if not 0.0 <= limits.maximum_gripper_open <= 1.0:
        raise ValueError("maximum_gripper_open must be within [0, 1].")
