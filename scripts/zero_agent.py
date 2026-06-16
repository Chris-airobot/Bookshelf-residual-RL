# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--disable_bookshelf",
    action="store_true",
    default=False,
    help="If the task supports it, remove the bookshelf obstacles for free-space debugging.",
)
parser.add_argument(
    "--nominal_release_assist",
    action="store_true",
    default=False,
    help="If supported, let the nominal release condition trigger scripted release for nominal-only testing.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import bookshelf.tasks  # noqa: F401


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    if args_cli.disable_bookshelf:
        if hasattr(env_cfg, "debug_omit_bookshelf_obstacles"):
            env_cfg.debug_omit_bookshelf_obstacles = True
        else:
            print("[WARN]: --disable_bookshelf was set, but this task config does not support it.")
    if args_cli.nominal_release_assist:
        if hasattr(env_cfg, "enable_nominal_release_assist"):
            env_cfg.enable_nominal_release_assist = True
            env_cfg.nominal_release_assist_until_frac = 1.0
        elif hasattr(env_cfg, "debug_disable_nominal_release"):
            env_cfg.debug_disable_nominal_release = False
        else:
            print("[WARN]: --nominal_release_assist was set, but this task config does not support it.")
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute zero actions
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            # apply actions
            env.step(actions)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
