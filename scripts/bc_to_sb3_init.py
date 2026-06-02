"""Convert a BC checkpoint into an SB3-compatible .zip so PPO can be
warm-started with the pre-trained actor weights.

Only the actor (policy_net + action_net) is copied from BC.
The value network and log_std are left at SB3's default random init —
PPO will train those from scratch.

No Isaac Sim dependency — pure PyTorch + stable-baselines3.

Usage:
    python scripts/bc_to_sb3_init.py
    python scripts/bc_to_sb3_init.py --bc_ckpt data/bc/bc_policy_best.pt \
                                      --out     data/bc/bc_init_sb3.zip
"""

import argparse
import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


# ---------------------------------------------------------------------------
# Minimal fake env — only used to build the SB3 model
# ---------------------------------------------------------------------------

class _FakeEnv(gym.Env):
    def __init__(self, obs_dim: int, action_dim: int):
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            -100.0, 100.0, shape=(action_dim,), dtype=np.float32
        )

    def reset(self, **kwargs):
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, False, False, {}


# ---------------------------------------------------------------------------
# Weight mapping
#
# BC BCPolicy.net is a flat Sequential:
#   net.0  Linear(10, 256)    <- layer 0
#   net.1  ReLU
#   net.2  Linear(256, 256)   <- layer 2
#   net.3  ReLU
#   net.4  Linear(256, 5)     <- output
#
# SB3 MlpPolicy with net_arch=[256,256], activation=ReLU:
#   mlp_extractor.policy_net.0  Linear(10, 256)
#   mlp_extractor.policy_net.1  ReLU
#   mlp_extractor.policy_net.2  Linear(256, 256)
#   mlp_extractor.policy_net.3  ReLU
#   action_net                  Linear(256, 5)
# ---------------------------------------------------------------------------

_BC_TO_SB3 = {
    "net.0.weight": "mlp_extractor.policy_net.0.weight",
    "net.0.bias":   "mlp_extractor.policy_net.0.bias",
    "net.2.weight": "mlp_extractor.policy_net.2.weight",
    "net.2.bias":   "mlp_extractor.policy_net.2.bias",
    "net.4.weight": "action_net.weight",
    "net.4.bias":   "action_net.bias",
}


def convert(bc_ckpt_path: str, out_path: str):
    # --- Load BC weights ---
    bc_ckpt = torch.load(bc_ckpt_path, map_location="cpu", weights_only=False)
    bc_state = bc_ckpt["model_state_dict"]
    hidden = bc_ckpt.get("hidden", 256)
    obs_dim = int(bc_ckpt.get("obs_dim", bc_state["net.0.weight"].shape[1]))
    action_dim = int(bc_ckpt.get("action_dim", bc_state["net.4.weight"].shape[0]))

    print(f"BC checkpoint  : {bc_ckpt_path}")
    print(f"  epoch        : {bc_ckpt.get('epoch', '?')}")
    print(f"  val_loss     : {bc_ckpt.get('val_loss', '?'):.4f}")
    print(f"  hidden size  : {hidden}")
    print(f"  obs_dim      : {obs_dim}")
    print(f"  action_dim   : {action_dim}")

    if hidden != 256:
        raise ValueError(
            f"BC hidden size is {hidden}, but sb3_ppo_cfg.yaml uses 256. "
            "Update net_arch in the yaml to match before running this script."
        )

    # --- Build a fresh SB3 PPO model with matching arch ---
    env = DummyVecEnv([lambda: _FakeEnv(obs_dim, action_dim)])
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs={
            "net_arch": [256, 256],
            "activation_fn": nn.ReLU,
            "squash_output": False,
        },
        verbose=0,
    )

    # --- Copy BC actor weights into SB3 policy ---
    sb3_state = model.policy.state_dict()
    n_copied = 0
    for bc_key, sb3_key in _BC_TO_SB3.items():
        if bc_key not in bc_state:
            raise KeyError(f"BC checkpoint missing expected key: {bc_key}")
        if sb3_key not in sb3_state:
            raise KeyError(f"SB3 policy missing expected key: {sb3_key}")
        bc_tensor = bc_state[bc_key]
        sb3_tensor = sb3_state[sb3_key]
        if bc_tensor.shape != sb3_tensor.shape:
            raise ValueError(
                f"Shape mismatch for {bc_key} -> {sb3_key}: "
                f"{bc_tensor.shape} vs {sb3_tensor.shape}"
            )
        sb3_state[sb3_key] = bc_tensor.clone()
        n_copied += 1
        print(f"  copied  {bc_key:20s} -> {sb3_key}  {tuple(bc_tensor.shape)}")

    model.policy.load_state_dict(sb3_state)
    print(f"\n{n_copied} weight tensors copied from BC to SB3 actor.")

    # SB3 default log_std=0 → std=1.0, which is huge relative to BC action
    # magnitudes (~[-1, 1]). This makes the initial PPO rollouts look random
    # even though the actor mean is correct. Start with a small std so the
    # policy begins near-deterministic (like BC eval) and PPO can fine-tune.
    log_std_init = -2.0  # std ≈ 0.14
    with torch.no_grad():
        model.policy.log_std.fill_(log_std_init)
    print(f"log_std initialised to {log_std_init} (std ≈ {torch.exp(torch.tensor(log_std_init)):.3f}).")

    # --- Save obs normalisation stats for VecNormalize warm-start ---
    base_path = out_path.removesuffix(".zip")
    stats_path = base_path + "_obs_stats.npz"
    if "obs_mean" in bc_ckpt and "obs_std" in bc_ckpt:
        obs_mean = bc_ckpt["obs_mean"].numpy()
        obs_std = bc_ckpt["obs_std"].numpy()
        np.savez(stats_path, mean=obs_mean, std=obs_std)
        print(f"Obs stats saved to: {stats_path}")
    else:
        print("WARNING: BC checkpoint has no obs_mean/obs_std. "
              "Re-run bc_train.py to include normalisation stats.")

    # --- Save as SB3 .zip ---
    model.save(out_path)
    zip_path = out_path if out_path.endswith(".zip") else out_path + ".zip"
    print(f"\nSaved SB3 init checkpoint to: {zip_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc_ckpt", type=str, default="data/bc/bc_policy_best.pt")
    parser.add_argument("--out",     type=str, default="data/bc/bc_init_sb3")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(args.bc_ckpt, args.out)
