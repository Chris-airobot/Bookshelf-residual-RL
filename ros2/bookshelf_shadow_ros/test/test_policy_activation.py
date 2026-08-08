import json
import math

import numpy as np

from bookshelf_shadow_ros.policy_activation import (
    PolicyActivationLimits,
    PolicyActivationTracker,
    activation_decision_dict,
    build_activation_envelope,
    evaluate_policy_activation,
    load_activation_envelope,
)
from bookshelf_shadow_ros.policy_observation_math import OBSERVATION_LABELS


def _preinsert_values():
    raw = np.array(
        [0.0, -0.186, 0.230, 0.0, 0.006, 0.0, -0.032, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    observation = np.array(
        [0.0, -1.0, 1.0, 0.0, 0.12, 0.0, -0.128, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    normalized = np.array(
        [-0.92, -0.76, 0.45, 0.25, 0.45, 0.47, 0.71, -0.21, 0.16, -0.29, -0.41, -0.09]
    )
    return observation, normalized, raw


def test_preinsert_state_becomes_ready_after_stable_samples():
    observation, normalized, raw = _preinsert_values()
    evaluation = evaluate_policy_activation(observation, normalized, raw)
    assert evaluation.ready

    tracker = PolicyActivationTracker(required_stable_samples=3)
    assert not tracker.update(evaluation).ready
    assert not tracker.update(evaluation).ready
    decision = tracker.update(evaluation)
    assert decision.ready
    assert activation_decision_dict(decision)["consecutive_ready_samples"] == 3


def test_recorded_far_pose_is_rejected_by_distribution_and_geometry():
    observation = np.array(
        [0.0, -1.0, 1.0, 0.85765, -1.0, -1.0, -0.49793, -0.01931, -0.03795, 0.39176, 0.09719, 0.02217]
    )
    normalized = np.array(
        [-0.922, -0.757, 0.454, -10.0, -9.909, -10.0, 1.351, 1.451, 0.162, 2.854, 0.112, -0.101]
    )
    raw = np.array(
        [0.0, -0.27819, 0.41717, 0.04288, -0.06330, -1.59809, -0.12448, -0.00483, -0.00949, 0.39176, 0.09719, 0.02217]
    )

    result = evaluate_policy_activation(observation, normalized, raw)
    assert not result.ready
    assert set(result.normalized_outliers) == {"lat_err", "z_err", "yaw_err"}
    assert any("rear_to_mouth" in reason for reason in result.reasons)
    assert any("front_to_back" in reason for reason in result.reasons)
    assert any("gripper" in reason for reason in result.reasons)


def test_required_envelope_fails_closed_when_missing():
    observation, normalized, raw = _preinsert_values()
    result = evaluate_policy_activation(
        observation,
        normalized,
        raw,
        require_envelope=True,
    )
    assert not result.ready
    assert "required but unavailable" in result.reasons[0]


def test_envelope_loader_and_outlier_detection(tmp_path):
    lower = np.full(12, -2.0)
    upper = np.full(12, 2.0)
    path = tmp_path / "envelope.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "labels": list(OBSERVATION_LABELS),
                "lower": lower.tolist(),
                "upper": upper.tolist(),
                "source": "simulator_test",
            }
        ),
        encoding="utf-8",
    )
    envelope = load_activation_envelope(path)
    observation, normalized, raw = _preinsert_values()
    normalized[5] = math.pi
    result = evaluate_policy_activation(
        observation,
        normalized,
        raw,
        envelope=envelope,
    )
    assert not result.ready
    assert set(result.envelope_outliers) == {"yaw_err"}


def test_tracker_resets_after_one_bad_sample():
    observation, normalized, raw = _preinsert_values()
    good = evaluate_policy_activation(observation, normalized, raw)
    bad_normalized = np.array(normalized, copy=True)
    bad_normalized[3] = 8.0
    bad = evaluate_policy_activation(observation, bad_normalized, raw)
    tracker = PolicyActivationTracker(required_stable_samples=2)
    assert not tracker.update(good).ready
    assert not tracker.update(bad).ready
    assert not tracker.update(good).ready
    assert tracker.update(good).ready


def test_build_activation_envelope_adds_margin_and_absolute_limit():
    values = np.vstack(
        [
            np.linspace(-0.5, 0.5, len(OBSERVATION_LABELS)),
            np.linspace(-0.4, 0.6, len(OBSERVATION_LABELS)),
            np.linspace(-0.6, 0.4, len(OBSERVATION_LABELS)),
        ]
    )
    lower, upper = build_activation_envelope(
        values,
        lower_percentile=0.0,
        upper_percentile=100.0,
        margin=0.25,
        maximum_abs_bound=0.70,
    )
    assert lower.shape == (12,)
    assert upper.shape == (12,)
    assert np.all(lower >= -0.70)
    assert np.all(upper <= 0.70)
    assert np.all(lower < upper)
