# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bookshelf v6 - stage 1 insertion base controller.

Spawns the robot at the pre-insertion pose (book front at slot mouth) and runs
a proportional Cartesian controller that drives the gripper/hand frame to the
50%-inserted target pose.

Stages:
  INSERT  – this script drives the EE from the slot mouth to the target depth.
             The gripper stays closed; no release is attempted.
"""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Bookshelf v6 base-controller-only debug runner.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Bookshelf-Direct-v6", help="Name of the task.")
parser.add_argument(
    "--spawn_inside_fraction",
    type=float,
    default=0.0,
    help="Insertion fraction used for spawning: 0.0 = book front at slot mouth (pre-insertion).",
)
parser.add_argument(
    "--target_fraction",
    type=float,
    default=0.5,
    help="Insertion fraction the controller drives toward: 0.5 = book half inside slot.",
)
parser.add_argument(
    "--disable_bookshelf",
    action="store_true",
    default=False,
    help="Remove bookshelf obstacles for free-space debugging.",
)
parser.add_argument(
    "--position_only",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Ignore yaw/pitch orientation error (position only).",
)
parser.add_argument("--kp", type=float, default=1.0, help="Proportional gain applied to normalised action error.")
parser.add_argument(
    "--hold_tolerance",
    type=float,
    default=0.0002,
    help="Hold the arm with zero action once the tool is within this position error in meters.",
)
parser.add_argument(
    "--preinsert_hold_seconds",
    type=float,
    default=0.0,
    help="Seconds to hold after reaching the debug target before episode reset.",
)
parser.add_argument("--status_interval", type=int, default=120, help="Print status every N env steps; 0 disables.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils import math as math_utils

import bookshelf.tasks  # noqa: F401


def _set_if_present(cfg, name: str, value) -> None:
    if hasattr(cfg, name):
        setattr(cfg, name, value)


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    _set_if_present(env_cfg, "debug_use_curobo_planner", False)
    _set_if_present(env_cfg, "debug_use_lula_rrt_planner", False)
    _set_if_present(env_cfg, "debug_done_on_preinsert_reached", True)
    _set_if_present(env_cfg, "debug_preinsert_hold_seconds", args_cli.preinsert_hold_seconds)
    _set_if_present(env_cfg, "debug_spawn_at_target_tool_pose", True)
    _set_if_present(env_cfg, "debug_spawn_with_curobo", False)
    _set_if_present(env_cfg, "debug_spawn_inside_fraction", args_cli.spawn_inside_fraction)
    _set_if_present(env_cfg, "debug_freeze_tool_to_book_transform", True)
    _set_if_present(env_cfg, "debug_omit_bookshelf_obstacles", args_cli.disable_bookshelf)
    _set_if_present(env_cfg, "debug_position_only_target_ee", args_cli.position_only)
    _set_if_present(env_cfg, "debug_disable_nominal_release", True)
    _set_if_present(env_cfg, "enable_nominal_controller", False)
    _set_if_present(env_cfg, "forced_missing_book_index", 5)
    # Keep this debug runner on the action-driven path.  In position-only mode,
    # v6 holds the reset EE quaternion internally and only the XYZ action moves.
    _set_if_present(env_cfg, "debug_use_full_target_ee_quat", False)

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    print(
        "[Bookshelf v6 base] "
        "stage=INSERT, "
        f"spawn_fraction={args_cli.spawn_inside_fraction:.2f}, "
        f"target_fraction={args_cli.target_fraction:.2f}, "
        f"bookshelf={'off' if args_cli.disable_bookshelf else 'on'}, "
        f"position_only={args_cli.position_only}, "
        f"kp={args_cli.kp:.3f}"
    )

    env.reset()
    target_tool_pos_env_fixed, target_tool_quat_fixed = unwrapped._planned_tool_release_pose_quat(
        inside_fraction=args_cli.target_fraction
    )
    step = 0
    kp = float(args_cli.kp)

    while simulation_app.is_running():
        with torch.inference_mode():
            # --- target tool pose for the desired insertion depth ---
            target_tool_pos_env = target_tool_pos_env_fixed
            target_tool_quat = target_tool_quat_fixed

            # --- current tool state ---
            current_tool_pos_env = unwrapped._ee_tool_pos_env()
            _, current_ee_quat_b = unwrapped._ee_pose_in_base()
            _, current_pitch, current_yaw = math_utils.euler_xyz_from_quat(current_ee_quat_b)

            target_tool_pos_w = target_tool_pos_env + unwrapped.scene.env_origins
            _, target_tool_quat_b = math_utils.subtract_frame_transforms(
                unwrapped.robot.data.root_pos_w,
                unwrapped.robot.data.root_quat_w,
                target_tool_pos_w,
                target_tool_quat,
            )
            _, target_pitch, target_yaw = math_utils.euler_xyz_from_quat(target_tool_quat_b)

            # --- position P-controller ---
            # Each axis is normalised by its own action scale so the action
            # saturates to ±1 when the error exceeds the per-step limit.
            # The env then applies its own final_d*_limit clamp, so actual
            # step size is bounded by the env config (e.g. 8 mm/step in x).
            pos_err = target_tool_pos_env - current_tool_pos_env
            actions = torch.zeros(env.action_space.shape, device=unwrapped.device)
            pos_err_norm = torch.linalg.norm(pos_err, dim=-1)
            active = pos_err_norm >= float(args_cli.hold_tolerance)
            actions[:, 0] = torch.where(
                active, torch.clamp(kp * pos_err[:, 0] / float(unwrapped.cfg.dx_action_scale), -1.0, 1.0), 0.0
            )
            actions[:, 1] = torch.where(
                active, torch.clamp(kp * pos_err[:, 1] / float(unwrapped.cfg.dy_action_scale), -1.0, 1.0), 0.0
            )
            actions[:, 2] = torch.where(
                active, torch.clamp(kp * pos_err[:, 2] / float(unwrapped.cfg.dz_action_scale), -1.0, 1.0), 0.0
            )

            # --- orientation P-controller ---
            yaw_err = _wrap_to_pi(target_yaw - current_yaw)
            pitch_err = _wrap_to_pi(target_pitch - current_pitch)
            if not args_cli.position_only:
                actions[:, 3] = torch.clamp(kp * yaw_err / float(unwrapped.cfg.dyaw_action_scale), -1.0, 1.0)
                actions[:, 4] = torch.clamp(kp * pitch_err / float(unwrapped.cfg.dpitch_action_scale), -1.0, 1.0)
                actions[:, 3:5] = torch.where(active.unsqueeze(-1), actions[:, 3:5], torch.zeros_like(actions[:, 3:5]))

            env.step(actions)

            if args_cli.status_interval > 0 and step % args_cli.status_interval == 0:
                mode = getattr(unwrapped, "_mode", None)
                mode0 = int(mode[0].item()) if mode is not None and mode.numel() > 0 else -1
                pos_err0 = float(torch.linalg.norm(pos_err[0]).item())
                yaw_err0 = float(torch.rad2deg(torch.abs(yaw_err[0])).item())
                pitch_err0 = float(torch.rad2deg(torch.abs(pitch_err[0])).item())
                print(
                    f"[Bookshelf v6 base] step={step} mode0={mode0} "
                    f"ee_pos_err={pos_err0:.4f} m  yaw_err={yaw_err0:.2f} deg  pitch_err={pitch_err0:.2f} deg"
                )

            step += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
