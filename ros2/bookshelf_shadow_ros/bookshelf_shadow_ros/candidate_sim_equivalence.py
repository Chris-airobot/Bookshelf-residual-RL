"""Pure summaries for simulator-to-real policy observation equivalence."""

from __future__ import annotations

import numpy as np

from .policy_observation_math import OBSERVATION_LABELS
from .policy_shadow_math import POLICY_ACTION_LABELS


def compare_candidate_to_simulator_preinsert(
    simulator_observations,
    simulator_normalized_observations,
    simulator_actor_means,
    simulator_actions,
    candidate_observation,
    candidate_normalized_observation,
    candidate_actor_mean,
    candidate_action,
    *,
    observation_envelope_tolerance=1.0e-4,
    normalized_envelope_tolerance=1.0e-3,
    action_tolerance=1.0e-6,
) -> dict:
    """Compare one real candidate against a simulator-derived distribution."""

    sim_observation = _matrix(
        simulator_observations, len(OBSERVATION_LABELS), "simulator_observations"
    )
    sim_normalized = _matrix(
        simulator_normalized_observations,
        len(OBSERVATION_LABELS),
        "simulator_normalized_observations",
    )
    sim_actor = _matrix(
        simulator_actor_means, len(POLICY_ACTION_LABELS), "simulator_actor_means"
    )
    sim_action = _matrix(
        simulator_actions, len(POLICY_ACTION_LABELS), "simulator_actions"
    )
    candidate_observation = _vector(
        candidate_observation, len(OBSERVATION_LABELS), "candidate_observation"
    )
    candidate_normalized = _vector(
        candidate_normalized_observation,
        len(OBSERVATION_LABELS),
        "candidate_normalized_observation",
    )
    candidate_actor = _vector(
        candidate_actor_mean, len(POLICY_ACTION_LABELS), "candidate_actor_mean"
    )
    candidate_action = _vector(
        candidate_action, len(POLICY_ACTION_LABELS), "candidate_action"
    )

    observation = _distribution_comparison(
        sim_observation,
        candidate_observation,
        OBSERVATION_LABELS,
        observation_envelope_tolerance,
    )
    normalized = _distribution_comparison(
        sim_normalized,
        candidate_normalized,
        OBSERVATION_LABELS,
        normalized_envelope_tolerance,
    )
    actor = _distribution_comparison(
        sim_actor,
        candidate_actor,
        POLICY_ACTION_LABELS,
        float("inf"),
    )
    action_median = np.median(sim_action, axis=0)
    action_error = np.abs(candidate_action - action_median)
    action_match = action_error <= float(action_tolerance)

    simulator_saturation_fraction = np.mean(
        np.abs(sim_action) >= 1.0 - 1.0e-6,
        axis=0,
    )
    candidate_saturated = np.abs(candidate_action) >= 1.0 - 1.0e-6
    dominant_simulator_saturation = simulator_saturation_fraction >= 0.95
    saturation_match = candidate_saturated == dominant_simulator_saturation

    passed = bool(
        observation["inside_envelope"]
        and normalized["inside_envelope"]
        and np.all(action_match)
        and np.all(saturation_match)
    )
    return {
        "equivalence_passed": passed,
        "sample_count": int(sim_observation.shape[0]),
        "observation": observation,
        "normalized_observation": normalized,
        "actor_mean": actor,
        "action": {
            "simulator_median": _labelled(action_median, POLICY_ACTION_LABELS),
            "candidate": _labelled(candidate_action, POLICY_ACTION_LABELS),
            "absolute_error": _labelled(action_error, POLICY_ACTION_LABELS),
            "tolerance": float(action_tolerance),
            "matching_labels": [
                label
                for label, match in zip(POLICY_ACTION_LABELS, action_match)
                if bool(match)
            ],
            "mismatching_labels": [
                label
                for label, match in zip(POLICY_ACTION_LABELS, action_match)
                if not bool(match)
            ],
            "simulator_saturation_fraction": _labelled(
                simulator_saturation_fraction, POLICY_ACTION_LABELS
            ),
            "candidate_saturated_labels": [
                label
                for label, saturated in zip(
                    POLICY_ACTION_LABELS, candidate_saturated
                )
                if bool(saturated)
            ],
            "saturation_mismatching_labels": [
                label
                for label, match in zip(POLICY_ACTION_LABELS, saturation_match)
                if not bool(match)
            ],
        },
    }


def labelled_distribution(values, labels) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(labels):
        raise ValueError("Distribution shape does not match labels.")
    return {
        label: {
            "min": float(np.min(values[:, index])),
            "mean": float(np.mean(values[:, index])),
            "median": float(np.median(values[:, index])),
            "max": float(np.max(values[:, index])),
        }
        for index, label in enumerate(labels)
    }


def _distribution_comparison(values, candidate, labels, tolerance) -> dict:
    minimum = np.min(values, axis=0)
    maximum = np.max(values, axis=0)
    median = np.median(values, axis=0)
    error = np.abs(candidate - median)
    inside = (candidate >= minimum - tolerance) & (
        candidate <= maximum + tolerance
    )
    return {
        "inside_envelope": bool(np.all(inside)),
        "envelope_tolerance": float(tolerance),
        "simulator_distribution": labelled_distribution(values, labels),
        "candidate": _labelled(candidate, labels),
        "absolute_error_from_simulator_median": _labelled(error, labels),
        "outside_envelope_labels": [
            label for label, matches in zip(labels, inside) if not bool(matches)
        ],
        "maximum_absolute_error_from_median": float(np.max(error)),
    }


def _matrix(values, width: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] != width:
        raise ValueError(f"{name} must have shape (N, {width}) with N > 0.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return array


def _vector(values, width: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.shape != (width,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector with shape ({width},).")
    return array


def _labelled(values, labels) -> dict[str, float]:
    return {
        label: float(value)
        for label, value in zip(labels, np.asarray(values).reshape(-1))
    }
