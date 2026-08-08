"""Pure row-by-row preflight checks for a recorded policy shadow audit."""

from __future__ import annotations

import csv
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .policy_activation import (
    PolicyActivationLimits,
    PolicyActivationTracker,
    evaluate_policy_activation,
)
from .policy_observation_math import OBSERVATION_LABELS


def audit_recorded_activation_csv(
    csv_path,
    *,
    envelope,
    limits=PolicyActivationLimits(),
    stable_samples=10,
) -> dict:
    csv_path = Path(csv_path).expanduser().resolve()
    tracker = PolicyActivationTracker(stable_samples)
    reason_counts = Counter()
    normalized_outlier_counts = Counter()
    envelope_outlier_counts = Counter()
    samples = 0
    instantaneous_ready_samples = 0
    ready_samples = 0
    maximum_streak = 0
    first_ready_sample = None

    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {
            f"{prefix}_{label}"
            for prefix in ("raw", "obs", "normalized")
            for label in OBSERVATION_LABELS
        }
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"Policy-stream CSV is missing columns: {missing}")
        for row in reader:
            raw = _row_vector(row, "raw")
            observation = _row_vector(row, "obs")
            normalized = _row_vector(row, "normalized")
            evaluation = evaluate_policy_activation(
                observation,
                normalized,
                raw,
                limits=limits,
                envelope=envelope,
                require_envelope=True,
            )
            decision = tracker.update(evaluation)
            samples += 1
            instantaneous_ready_samples += int(evaluation.ready)
            ready_samples += int(decision.ready)
            maximum_streak = max(maximum_streak, decision.consecutive_ready_samples)
            if decision.ready and first_ready_sample is None:
                first_ready_sample = samples - 1
            reason_counts.update(evaluation.reasons)
            normalized_outlier_counts.update(evaluation.normalized_outliers.keys())
            envelope_outlier_counts.update(evaluation.envelope_outliers.keys())

    return {
        "samples": samples,
        "instantaneous_ready_samples": instantaneous_ready_samples,
        "instantaneous_ready_fraction": (
            instantaneous_ready_samples / samples if samples else 0.0
        ),
        "activation_ready_samples": ready_samples,
        "activation_ready_fraction": ready_samples / samples if samples else 0.0,
        "maximum_consecutive_ready_samples": maximum_streak,
        "required_stable_samples": int(stable_samples),
        "first_ready_sample": first_ready_sample,
        "final_activation_ready": bool(
            samples and tracker.consecutive_ready_samples >= stable_samples
        ),
        "reason_counts": dict(sorted(reason_counts.items())),
        "normalized_outlier_counts": dict(sorted(normalized_outlier_counts.items())),
        "envelope_outlier_counts": dict(sorted(envelope_outlier_counts.items())),
    }


def audit_simulator_activation_samples(
    observations,
    normalized_observations,
    raw_metrics,
    *,
    envelope,
    limits=PolicyActivationLimits(),
    stable_samples=10,
) -> dict:
    """Check independent simulator states and one repeated representative state."""

    observations = _sample_matrix(observations, "observations")
    normalized = _sample_matrix(
        normalized_observations,
        "normalized_observations",
    )
    raw = _sample_matrix(raw_metrics, "raw_metrics")
    if not (observations.shape == normalized.shape == raw.shape):
        raise ValueError("Simulator activation arrays must have matching shapes.")

    evaluations = []
    reason_counts = Counter()
    normalized_outlier_counts = Counter()
    envelope_outlier_counts = Counter()
    ready_indices = []
    for index, (observation, normalized_value, raw_value) in enumerate(
        zip(observations, normalized, raw)
    ):
        evaluation = evaluate_policy_activation(
            observation,
            normalized_value,
            raw_value,
            limits=limits,
            envelope=envelope,
            require_envelope=True,
        )
        evaluations.append(evaluation)
        if evaluation.ready:
            ready_indices.append(index)
        reason_counts.update(evaluation.reasons)
        normalized_outlier_counts.update(evaluation.normalized_outliers.keys())
        envelope_outlier_counts.update(evaluation.envelope_outliers.keys())

    representative_index = None
    repeated_stability_passed = False
    if ready_indices:
        ready_normalized = normalized[ready_indices]
        median = np.median(ready_normalized, axis=0)
        relative_index = int(
            np.argmin(np.linalg.norm(ready_normalized - median[None, :], axis=1))
        )
        representative_index = int(ready_indices[relative_index])
        tracker = PolicyActivationTracker(stable_samples)
        decision = None
        for _ in range(stable_samples):
            decision = tracker.update(evaluations[representative_index])
        repeated_stability_passed = bool(decision and decision.ready)

    sample_count = int(observations.shape[0])
    ready_count = len(ready_indices)
    return {
        "samples": sample_count,
        "instantaneous_ready_samples": ready_count,
        "instantaneous_ready_fraction": (
            ready_count / sample_count if sample_count else 0.0
        ),
        "representative_sample_index": representative_index,
        "repeated_stability_samples": int(stable_samples),
        "repeated_stability_passed": repeated_stability_passed,
        "sample_order_is_temporal": False,
        "reason_counts": dict(sorted(reason_counts.items())),
        "normalized_outlier_counts": dict(sorted(normalized_outlier_counts.items())),
        "envelope_outlier_counts": dict(sorted(envelope_outlier_counts.items())),
    }


def load_policy_stream_summary(path) -> tuple[dict, dict]:
    path = Path(path).expanduser().resolve()
    document = json.loads(path.read_text(encoding="utf-8"))
    return document, document.get("policy_stream", document)


def _row_vector(row, prefix):
    values = np.asarray(
        [float(row[f"{prefix}_{label}"]) for label in OBSERVATION_LABELS],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{prefix} observation contains non-finite values")
    return values


def _sample_matrix(values, name):
    values = np.asarray(values, dtype=np.float64)
    width = len(OBSERVATION_LABELS)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != width:
        raise ValueError(f"{name} must have shape (N, {width}) with N > 0.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values.")
    return values
