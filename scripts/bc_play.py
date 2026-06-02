"""Run a trained BC policy checkpoint in the Isaac Lab sim.

Loads the MLP weights from data/bc/bc_policy_best.pt (or --checkpoint),
runs episodes, and prints per-episode outcomes so you can judge whether
BC learned anything useful before moving to RL fine-tuning.

Usage:
    ~/isaacsim/python.sh scripts/bc_play.py --task Bookshelf-Direct-v4
    ~/isaacsim/python.sh scripts/bc_play.py --task Bookshelf-Direct-v4 \\
        --checkpoint data/bc/bc_policy_best.pt --num_episodes 20
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import pathlib
import sys

# Ensure the bookshelf package is importable regardless of install state.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_BOOKSHELF_SRC = _REPO_ROOT / "source" / "bookshelf"
if str(_BOOKSHELF_SRC) not in sys.path:
    sys.path.insert(0, str(_BOOKSHELF_SRC))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="BC policy evaluation for Bookshelf-v4.")
parser.add_argument("--task", type=str, default="Bookshelf-Direct-v4")
parser.add_argument("--checkpoint", type=str, default="data/bc/bc_policy_best.pt")
parser.add_argument("--num_episodes", type=int, default=10,
                    help="Number of episodes to run before exiting (0 = run forever).")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.num_envs = 1  # always single env for inspection

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import torch.nn as nn

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import bookshelf.tasks  # noqa: F401


# ---------------------------------------------------------------------------
# Policy (must match bc_train.py exactly)
# ---------------------------------------------------------------------------

class BCPolicy(nn.Module):
    """Small MLP policy."""

    def __init__(self, obs_dim: int = 10, action_dim: int = 5, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def load_policy(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    policy = BCPolicy(
        obs_dim=ckpt.get("obs_dim", 10),
        action_dim=ckpt.get("action_dim", 5),
        hidden=ckpt.get("hidden", 256),
    )
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.to(device).eval()
    obs_mean = ckpt.get("obs_mean", None)
    obs_std = ckpt.get("obs_std", None)
    if obs_mean is not None and obs_std is not None:
        obs_mean = obs_mean.to(device)
        obs_std = obs_std.to(device)
    print(f"[BC] Loaded checkpoint from {checkpoint_path}  "
          f"(epoch {ckpt.get('epoch', '?')}, val_loss {ckpt.get('val_loss', '?'):.4f})")
    return policy, obs_mean, obs_std


# ---------------------------------------------------------------------------
# Episode tracker
# ---------------------------------------------------------------------------

class EpisodeStats:
    def __init__(self):
        self.ep = 0
        self.results = []

    def record(self, steps: int, total_reward: float, terminated: bool, truncated: bool,
               reached_push: bool, success: bool):
        self.ep += 1
        outcome = "SUCCESS" if success else ("DROP/TERM" if terminated else "TIMEOUT")
        push_str = " [reached PUSH]" if reached_push else ""
        print(
            f"  Ep {self.ep:>3} | {steps:>4} steps | reward {total_reward:>8.2f} | "
            f"{outcome}{push_str}"
        )
        self.results.append(dict(
            success=success, terminated=terminated, truncated=truncated,
            steps=steps, reward=total_reward, reached_push=reached_push,
        ))

    def summary(self):
        n = len(self.results)
        if n == 0:
            return
        n_success = sum(r["success"] for r in self.results)
        n_push = sum(r["reached_push"] for r in self.results)
        avg_reward = sum(r["reward"] for r in self.results) / n
        print("\n" + "=" * 60)
        print(f"  Episodes run   : {n}")
        print(f"  Successes      : {n_success} / {n}  ({100*n_success/n:.0f}%)")
        print(f"  Reached PUSH   : {n_push} / {n}  ({100*n_push/n:.0f}%)")
        print(f"  Avg reward     : {avg_reward:.2f}")
        print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device(args_cli.device if hasattr(args_cli, "device") else "cuda")

    # --- Load policy ---
    policy, obs_mean, obs_std = load_policy(args_cli.checkpoint, device)

    # --- Build env ---
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=str(device),
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    print(f"[INFO] Obs space : {env.observation_space}")
    print(f"[INFO] Action space: {env.action_space}")

    stats = EpisodeStats()
    with torch.no_grad():
        obs_dict, _ = env.reset()

    # obs_dict is a dict with key "policy".
    obs = obs_dict["policy"]

    ep_steps = 0
    ep_reward = 0.0
    ep_reached_push = False

    # mode_obs_push = 1.0 from cfg -- obs[0] == 1.0 means PUSH phase
    PUSH_MODE_VALUE = 1.0
    PUSH_MODE_TOL = 0.1
    # Success: large positive reward at termination (success_bonus=100, drop=-20)
    SUCCESS_REWARD_THRESH = 50.0

    print(f"\nRunning {args_cli.num_episodes or 'unlimited'} episodes...\n")

    while simulation_app.is_running():
        # Use no_grad, not inference_mode: inference_mode marks Isaac Lab's
        # internal buffers as inference tensors, which causes env.reset() to
        # fail when it tries to write to them outside inference_mode.
        with torch.no_grad():
            obs_in = obs
            if obs_mean is not None and obs_std is not None:
                obs_in = (obs_in - obs_mean) / obs_std
            action = policy(obs_in)

            obs_dict, reward, terminated, truncated, _ = env.step(action)
            obs = obs_dict["policy"]

        ep_steps += 1
        ep_reward += reward.item()

        # Check if we've entered PUSH phase (obs[0] == 1.0)
        if abs(obs[0, 0].item() - PUSH_MODE_VALUE) < PUSH_MODE_TOL:
            ep_reached_push = True

        done = terminated.item() or truncated.item()
        if done:
            success = terminated.item() and (reward.item() > SUCCESS_REWARD_THRESH)
            stats.record(ep_steps, ep_reward, terminated.item(), truncated.item(),
                         ep_reached_push, success)

            # Reset for next episode
            ep_steps = 0
            ep_reward = 0.0
            ep_reached_push = False

            if args_cli.num_episodes > 0 and stats.ep >= args_cli.num_episodes:
                break

            with torch.no_grad():
                obs_dict, _ = env.reset()
            obs = obs_dict["policy"]

    stats.summary()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
