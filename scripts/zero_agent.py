# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import math

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
parser.add_argument(
    "--ideal",
    action="store_true",
    default=False,
    help="Use position-only control, fix the book to the tool, and disable reset randomization.",
)
parser.add_argument(
    "--light_randomization",
    action="store_true",
    default=False,
    help="Use small realistic reset noise without fixing the book to the tool.",
)
parser.add_argument(
    "--slot_clearance",
    type=float,
    default=None,
    help="Use a fixed slot clearance in meters and disable the clearance curriculum.",
)
parser.add_argument(
    "--missing_index",
    type=int,
    default=None,
    help="Fix the missing slot index; omit it to keep random slot selection.",
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
    if args_cli.ideal:
        if hasattr(env_cfg, "enable_residual_reset_curriculum"):
            env_cfg.enable_residual_reset_curriculum = False
        if hasattr(env_cfg, "debug_position_only_target_ee"):
            env_cfg.debug_position_only_target_ee = True
        if hasattr(env_cfg, "debug_hold_book_fixed_to_tool"):
            env_cfg.debug_hold_book_fixed_to_tool = True
        for name in (
            "reset_arm_joint_pos_noise",
            "book_grasp_x_jitter",
            "book_grasp_y_jitter",
            "book_grasp_z_jitter",
            "book_grasp_yaw_jitter",
        ):
            if hasattr(env_cfg, name):
                setattr(env_cfg, name, 0.0)
    elif args_cli.light_randomization:
        if hasattr(env_cfg, "enable_residual_reset_curriculum"):
            env_cfg.enable_residual_reset_curriculum = False
        env_cfg.reset_arm_joint_pos_noise = math.radians(1.0)
        env_cfg.book_grasp_x_jitter = 0.002
        env_cfg.book_grasp_y_jitter = 0.002
        env_cfg.book_grasp_z_jitter = 0.002
        env_cfg.book_grasp_yaw_jitter = math.radians(2.0)
    if args_cli.slot_clearance is not None:
        if hasattr(env_cfg, "enable_residual_clearance_curriculum"):
            env_cfg.enable_residual_clearance_curriculum = False
        env_cfg.slot_lateral_clearance_min = args_cli.slot_clearance
        env_cfg.slot_lateral_clearance_max = args_cli.slot_clearance
    if args_cli.missing_index is not None:
        env_cfg.forced_missing_book_index = args_cli.missing_index
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
