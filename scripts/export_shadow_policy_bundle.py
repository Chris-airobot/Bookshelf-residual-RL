#!/usr/bin/env python3
"""Export the trained SB3 actor and VecNormalize state to a portable NumPy bundle."""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np


OBSERVATION_SIZE = 12
ACTION_SIZE = 6


class _IgnoredRandomGenerator:
    """Discard pickle-only RNG state that is irrelevant for deterministic inference."""

    def __setstate__(self, state):
        self.state = state


def _ignored_bit_generator_ctor(*_args, **_kwargs):
    return _IgnoredRandomGenerator()


def _ignored_generator_ctor(*_args, **_kwargs):
    return _IgnoredRandomGenerator()


def _load_vecnormalize(path: Path):
    """Load stats while tolerating NumPy 2.x RNG pickles under NumPy 1.x."""

    import numpy.random._pickle as random_pickle

    original_generator_ctor = random_pickle.__generator_ctor
    original_bit_generator_ctor = random_pickle.__bit_generator_ctor
    sitecustomize_module = sys.modules.get("sitecustomize")
    original_compat_ctor = None
    if sitecustomize_module is not None and hasattr(
        sitecustomize_module, "_compat_bit_generator_ctor"
    ):
        original_compat_ctor = sitecustomize_module._compat_bit_generator_ctor
        sitecustomize_module._compat_bit_generator_ctor = _ignored_bit_generator_ctor

    random_pickle.__generator_ctor = _ignored_generator_ctor
    random_pickle.__bit_generator_ctor = _ignored_bit_generator_ctor
    try:
        with path.open("rb") as stream:
            return pickle.load(stream)
    finally:
        random_pickle.__generator_ctor = original_generator_ctor
        random_pickle.__bit_generator_ctor = original_bit_generator_ctor
        if original_compat_ctor is not None:
            sitecustomize_module._compat_bit_generator_ctor = original_compat_ctor


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _linear_parameters(module):
    return (
        module.weight.detach().cpu().numpy().astype(np.float32),
        module.bias.detach().cpu().numpy().astype(np.float32),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--vecnormalize", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    vecnormalize_path = args.vecnormalize.expanduser().resolve()
    output = args.output.expanduser().resolve()
    for path in (checkpoint, vecnormalize_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    import torch
    from gymnasium.spaces import Box
    from stable_baselines3 import PPO

    custom_objects = {
        "observation_space": Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float32,
        ),
        "action_space": Box(
            low=-100.0,
            high=100.0,
            shape=(ACTION_SIZE,),
            dtype=np.float32,
        ),
        "ep_info_buffer": None,
        "ep_success_buffer": None,
    }
    model = PPO.load(
        checkpoint,
        device="cpu",
        print_system_info=False,
        custom_objects=custom_objects,
    )
    vecnormalize = _load_vecnormalize(vecnormalize_path)

    if model.observation_space.shape != (OBSERVATION_SIZE,):
        raise ValueError(f"Expected 12D model observation, got {model.observation_space.shape}.")
    if model.action_space.shape != (ACTION_SIZE,):
        raise ValueError(f"Expected 6D model action, got {model.action_space.shape}.")
    if type(model.policy.features_extractor).__name__ != "FlattenExtractor":
        raise ValueError("Expected the trained FlattenExtractor.")

    policy_modules = list(model.policy.mlp_extractor.policy_net)
    if (
        len(policy_modules) != 4
        or not isinstance(policy_modules[0], torch.nn.Linear)
        or not isinstance(policy_modules[1], torch.nn.ReLU)
        or not isinstance(policy_modules[2], torch.nn.Linear)
        or not isinstance(policy_modules[3], torch.nn.ReLU)
    ):
        raise ValueError(f"Unexpected policy network: {model.policy.mlp_extractor.policy_net}")
    if not isinstance(model.policy.action_net, torch.nn.Linear):
        raise ValueError(f"Unexpected action head: {model.policy.action_net}")

    policy_0_weight, policy_0_bias = _linear_parameters(policy_modules[0])
    policy_1_weight, policy_1_bias = _linear_parameters(policy_modules[2])
    action_weight, action_bias = _linear_parameters(model.policy.action_net)

    obs_mean = np.asarray(vecnormalize.obs_rms.mean, dtype=np.float32)
    obs_var = np.asarray(vecnormalize.obs_rms.var, dtype=np.float32)
    if obs_mean.shape != (OBSERVATION_SIZE,) or obs_var.shape != (OBSERVATION_SIZE,):
        raise ValueError("VecNormalize statistics do not match the 12D policy.")
    if not bool(vecnormalize.norm_obs):
        raise ValueError("The trained run did not enable observation normalization.")

    repo_root = Path(__file__).resolve().parents[1]
    package_root = repo_root / "ros2" / "bookshelf_shadow_ros"
    sys.path.insert(0, str(package_root))
    from bookshelf_shadow_ros.policy_shadow_math import NumpyActorBundle

    metadata = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "vecnormalize": str(vecnormalize_path),
        "vecnormalize_sha256": _sha256(vecnormalize_path),
        "algorithm": "PPO",
        "policy": "MlpPolicy",
        "feature_extractor": "FlattenExtractor",
        "actor_architecture": [12, 256, 256, 6],
        "activation": "ReLU",
        "deterministic_only": True,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        schema_version=np.array(NumpyActorBundle.SCHEMA_VERSION, dtype=np.int64),
        observation_size=np.array(OBSERVATION_SIZE, dtype=np.int64),
        action_size=np.array(ACTION_SIZE, dtype=np.int64),
        activation=np.array("relu"),
        obs_mean=obs_mean,
        obs_var=obs_var,
        obs_epsilon=np.array(float(vecnormalize.epsilon), dtype=np.float64),
        obs_clip=np.array(float(vecnormalize.clip_obs), dtype=np.float64),
        action_low=np.asarray(model.action_space.low, dtype=np.float32),
        action_high=np.asarray(model.action_space.high, dtype=np.float32),
        policy_0_weight=policy_0_weight,
        policy_0_bias=policy_0_bias,
        policy_1_weight=policy_1_weight,
        policy_1_bias=policy_1_bias,
        action_weight=action_weight,
        action_bias=action_bias,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )

    bundle = NumpyActorBundle(output)
    rng = np.random.default_rng(42)
    test_observations = rng.uniform(-1.0, 1.0, size=(32, OBSERVATION_SIZE)).astype(np.float32)
    normalized = np.clip(
        (test_observations - obs_mean) / np.sqrt(obs_var + float(vecnormalize.epsilon)),
        -float(vecnormalize.clip_obs),
        float(vecnormalize.clip_obs),
    ).astype(np.float32)
    reference_actions, _ = model.predict(normalized, deterministic=True)
    bundle_actions = np.stack([bundle.predict(observation)[2] for observation in test_observations])
    reference_actions = np.clip(reference_actions, -1.0, 1.0)
    maximum_error = float(np.max(np.abs(reference_actions - bundle_actions)))
    if maximum_error > 1.0e-5:
        output.unlink(missing_ok=True)
        raise RuntimeError(
            f"Portable actor verification failed: max action error {maximum_error:.3e}."
        )

    print(f"Exported verified shadow policy bundle: {output}")
    print(f"Bundle sha256: {bundle.sha256}")
    print(f"Maximum SB3-vs-NumPy action error: {maximum_error:.3e}")


if __name__ == "__main__":
    main()
