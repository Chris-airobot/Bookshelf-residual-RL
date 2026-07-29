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
