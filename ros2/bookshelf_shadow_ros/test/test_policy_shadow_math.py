import json
import math

import numpy as np
import pytest

from bookshelf_shadow_ros.policy_shadow_math import (
    NumpyActorBundle,
    ResidualMotionConfig,
    combine_motion_delta,
    compute_insert_nominal_delta,
    scale_residual_action,
    validate_shadow_inputs,
)


def test_nominal_insert_waits_for_alignment_then_moves_forward():
    raw = np.zeros(12, dtype=np.float32)
    raw[1] = -0.050
    raw[3] = 0.010

    unaligned = compute_insert_nominal_delta(raw)
    np.testing.assert_allclose(unaligned[0], 0.0)
    np.testing.assert_allclose(unaligned[1], 0.0015)

    raw[3] = 0.001
    aligned = compute_insert_nominal_delta(raw)
    np.testing.assert_allclose(aligned[0], 0.0010)
    np.testing.assert_allclose(aligned[1], 0.00025)

    raw[1] = -0.020
    near_mouth = compute_insert_nominal_delta(raw)
    np.testing.assert_allclose(near_mouth[0], 0.0007)


def test_nominal_insert_rejects_non_insert_mode():
    raw = np.zeros(12, dtype=np.float32)
    raw[0] = 1.0
    with pytest.raises(ValueError, match="INSERT mode only"):
        compute_insert_nominal_delta(raw)


def test_residual_scaling_and_final_limits_match_environment():
    config = ResidualMotionConfig()
    action = np.array([2.0, -2.0, 0.5, 0.5, -0.5, 0.75], dtype=np.float32)
    residual = scale_residual_action(action, config)
    np.testing.assert_allclose(
        residual,
        [
            0.0020,
            -0.0010,
            0.00075,
            0.5 * math.radians(0.35),
            -0.5 * math.radians(0.30),
        ],
    )

    final = combine_motion_delta(
        [0.02, 0.02, 0.02, 0.02, 0.02],
        residual,
        config,
    )
    np.testing.assert_allclose(final, np.asarray(config.final_limits, dtype=np.float32))


def test_numpy_actor_bundle_normalizes_and_runs_relu_actor(tmp_path):
    path = tmp_path / "bundle.npz"
    policy_0_weight = np.zeros((256, 12), dtype=np.float32)
    policy_0_weight[0, 0] = 2.0
    policy_1_weight = np.zeros((256, 256), dtype=np.float32)
    policy_1_weight[0, 0] = 3.0
    action_weight = np.zeros((6, 256), dtype=np.float32)
    action_weight[0, 0] = 4.0

    np.savez_compressed(
        path,
        schema_version=np.array(1),
        observation_size=np.array(12),
        action_size=np.array(6),
        activation=np.array("relu"),
        obs_mean=np.zeros(12, dtype=np.float32),
        obs_var=np.ones(12, dtype=np.float32),
        obs_epsilon=np.array(1.0e-8),
        obs_clip=np.array(10.0),
        action_low=np.full(6, -100.0, dtype=np.float32),
        action_high=np.full(6, 100.0, dtype=np.float32),
        policy_0_weight=policy_0_weight,
        policy_0_bias=np.zeros(256, dtype=np.float32),
        policy_1_weight=policy_1_weight,
        policy_1_bias=np.zeros(256, dtype=np.float32),
        action_weight=action_weight,
        action_bias=np.zeros(6, dtype=np.float32),
        metadata_json=np.array(json.dumps({"test": True})),
    )

    bundle = NumpyActorBundle(path)
    observation = np.zeros(12, dtype=np.float32)
    observation[0] = 0.5
    normalized, actor_mean, action = bundle.predict(observation)

    np.testing.assert_allclose(normalized[0], 0.5)
    np.testing.assert_allclose(actor_mean[0], 12.0)
    np.testing.assert_allclose(action[0], 1.0)
    assert bundle.metadata == {"test": True}


@pytest.mark.parametrize(
    ("index", "value", "motion_index", "expected_sign"),
    [
        (3, +0.004, 1, +1),
        (3, -0.004, 1, -1),
        (4, +0.008, 2, -1),
        (4, +0.004, 2, +1),
        (5, math.radians(+3.0), 3, -1),
        (5, math.radians(-3.0), 3, +1),
        (10, +0.05, 4, -1),
        (10, -0.05, 4, +1),
    ],
)
def test_nominal_insert_correction_directions(index, value, motion_index, expected_sign):
    raw = np.zeros(12, dtype=np.float32)
    raw[1] = -0.050
    raw[4] = 0.006
    raw[index] = value

    delta = compute_insert_nominal_delta(raw)
    assert np.sign(delta[motion_index]) == expected_sign


@pytest.mark.parametrize(
    ("index", "value"),
    [
        (3, 0.0061),
        (4, 0.0101),
        (5, math.radians(6.1)),
        (10, 0.101),
    ],
)
def test_nominal_insert_forward_gate_blocks_each_large_alignment_error(index, value):
    raw = np.zeros(12, dtype=np.float32)
    raw[1] = -0.050
    raw[4] = 0.006
    raw[index] = value
    assert compute_insert_nominal_delta(raw)[0] == pytest.approx(0.0)


def test_shadow_input_validation_accepts_only_fresh_paired_finite_vectors():
    observation = np.zeros(12, dtype=np.float32)
    raw = np.zeros(12, dtype=np.float32)
    assert (
        validate_shadow_inputs(
            observation,
            raw,
            observation_valid=True,
            valid_age_s=0.01,
            observation_age_s=0.01,
            raw_metrics_age_s=0.01,
            pair_skew_s=0.01,
        )
        is None
    )


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({"observation_valid": False}, "upstream observation_valid is false"),
        ({"valid_age_s": 0.51}, "observation_valid message is stale"),
        ({"observation_age_s": 0.51}, "12D observation is missing or stale"),
        ({"raw_metrics_age_s": 0.51}, "raw metrics are missing or stale"),
        ({"pair_skew_s": 0.11}, "skew is 0.110 s"),
    ],
)
def test_shadow_input_validation_rejects_invalid_timing(updates, expected):
    arguments = {
        "observation_valid": True,
        "valid_age_s": 0.01,
        "observation_age_s": 0.01,
        "raw_metrics_age_s": 0.01,
        "pair_skew_s": 0.01,
    }
    arguments.update(updates)
    error = validate_shadow_inputs(
        np.zeros(12, dtype=np.float32),
        np.zeros(12, dtype=np.float32),
        **arguments,
    )
    assert expected in error


def test_shadow_input_validation_rejects_nan_and_wrong_shape():
    raw = np.zeros(12, dtype=np.float32)
    common = {
        "observation_valid": True,
        "valid_age_s": 0.01,
        "observation_age_s": 0.01,
        "raw_metrics_age_s": 0.01,
        "pair_skew_s": 0.01,
    }
    assert "expected 12D" in validate_shadow_inputs(np.zeros(11), raw, **common)
    observation = np.zeros(12, dtype=np.float32)
    observation[2] = np.nan
    assert "non-finite" in validate_shadow_inputs(observation, raw, **common)


def test_numpy_actor_is_deterministic(tmp_path):
    path = tmp_path / "bundle.npz"
    rng = np.random.default_rng(7)
    np.savez_compressed(
        path,
        schema_version=np.array(1),
        observation_size=np.array(12),
        action_size=np.array(6),
        activation=np.array("relu"),
        obs_mean=np.zeros(12, dtype=np.float32),
        obs_var=np.ones(12, dtype=np.float32),
        obs_epsilon=np.array(1.0e-8),
        obs_clip=np.array(10.0),
        action_low=np.full(6, -1.0, dtype=np.float32),
        action_high=np.full(6, 1.0, dtype=np.float32),
        policy_0_weight=rng.normal(size=(256, 12)).astype(np.float32),
        policy_0_bias=rng.normal(size=256).astype(np.float32),
        policy_1_weight=rng.normal(size=(256, 256)).astype(np.float32),
        policy_1_bias=rng.normal(size=256).astype(np.float32),
        action_weight=rng.normal(size=(6, 256)).astype(np.float32),
        action_bias=rng.normal(size=6).astype(np.float32),
        metadata_json=np.array(json.dumps({"test": True})),
    )
    bundle = NumpyActorBundle(path)
    observation = rng.normal(size=12).astype(np.float32)
    first = bundle.predict(observation)
    for _ in range(20):
        repeated = bundle.predict(observation)
        for expected, actual in zip(first, repeated):
            np.testing.assert_array_equal(expected, actual)
