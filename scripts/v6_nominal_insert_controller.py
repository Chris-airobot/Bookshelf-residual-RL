# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone nominal geometric insertion controller for Bookshelf v6.

This script intentionally keeps the controller logic here instead of using the
v6 nominal controller.  It sends final local delta actions directly:

    [dx, dy, dz, dyaw, dpitch, release]

where the deltas are computed from book/slot geometry.
"""

import argparse
import math

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Standalone Bookshelf v6 nominal insertion controller.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Bookshelf-Direct-v6", help="Name of the task.")
parser.add_argument(
    "--disable_bookshelf",
    action="store_true",
    default=False,
    help="Remove bookshelf obstacles for free-space debugging.",
)
parser.add_argument(
    "--missing_index",
    type=int,
    default=-1,
    help="Missing shelf-book slot index. Use -1 for random slot each reset.",
)
parser.add_argument("--seed", type=int, default=0, help="Seed for deterministic debug resets.")
parser.add_argument(
    "--randomize_reset",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Enable arm/grasp/row randomization. Default off for nominal-controller debugging.",
)
parser.add_argument(
    "--slot_clearance",
    type=float,
    default=0.020,
    help="Extra lateral slot clearance in meters. Larger values make nominal-controller debugging easier.",
)
parser.add_argument(
    "--spawn_at_preinsert",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Start the robot/book at the planned pre-insertion pose for second-stage insertion debugging.",
)
parser.add_argument(
    "--spawn_inside_fraction",
    type=float,
    default=0.0,
    help="Initial insertion fraction when --spawn_at_preinsert is enabled.",
)
parser.add_argument(
    "--preinsert_standoff",
    type=float,
    default=0.030,
    help="Extra pre-insertion distance outside the shelf mouth, meters. Use 0.0 for the old spawn pose.",
)
parser.add_argument("--episode_length_s", type=float, default=60.0, help="Episode length for this debug script.")
parser.add_argument("--dx", type=float, default=0.0008, help="Nominal forward insertion delta per env step, meters.")
parser.add_argument("--k_lat", type=float, default=0.35, help="Lateral correction gain.")
parser.add_argument("--k_z", type=float, default=0.30, help="Height correction gain.")
parser.add_argument("--k_yaw", type=float, default=0.45, help="Yaw correction gain.")
parser.add_argument("--k_tilt", type=float, default=0.20, help="Pitch/tilt correction gain.")
parser.add_argument(
    "--position_only",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Hold EE orientation fixed while debugging XYZ insertion. Disable later to test yaw/pitch actions.",
)
parser.add_argument(
    "--hold_book_fixed_to_tool",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Kinematically glue the book to the tool during insertion; disabled automatically when scripted release fires.",
)
parser.add_argument("--dy_limit", type=float, default=0.0015, help="Max lateral delta per env step, meters.")
parser.add_argument("--dz_limit", type=float, default=0.0015, help="Max vertical delta per env step, meters.")
parser.add_argument("--dyaw_limit_deg", type=float, default=0.35, help="Max yaw delta per env step, degrees.")
parser.add_argument("--dpitch_limit_deg", type=float, default=0.25, help="Max pitch delta per env step, degrees.")
parser.add_argument("--lat_gate", type=float, default=0.010, help="Reduce forward motion above this lateral error.")
parser.add_argument("--z_gate", type=float, default=0.015, help="Reduce forward motion above this height error.")
parser.add_argument("--yaw_gate_deg", type=float, default=8.0, help="Reduce forward motion above this yaw error.")
parser.add_argument(
    "--unaligned_dx_scale",
    type=float,
    default=0.2,
    help="Scale dx by this factor while lateral/height/yaw errors are above gates.",
)
parser.add_argument(
    "--scripted_release",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Let the nominal controller trigger the env scripted release for nominal-only evaluation.",
)

parser.add_argument("--release_fraction", type=float, default=0.50, help="Book inside fraction needed for release.")
parser.add_argument("--release_lat_tol", type=float, default=0.006, help="Release lateral tolerance, meters.")
parser.add_argument("--release_z_tol", type=float, default=0.012, help="Release height tolerance, meters.")
parser.add_argument("--release_yaw_tol_deg", type=float, default=6.0, help="Release yaw tolerance, degrees.")
parser.add_argument("--release_tilt_tol", type=float, default=0.12, help="Release tilt-x tolerance.")
parser.add_argument("--push_dx", type=float, default=0.0008, help="Nominal forward push delta per env step, meters.")
parser.add_argument("--push_k_y", type=float, default=0.35, help="Push-stage gain for holding the latched tool Y.")
parser.add_argument("--push_k_z", type=float, default=0.30, help="Push-stage gain for moving toward the lower tool Z.")
parser.add_argument("--push_dy_limit", type=float, default=0.0005, help="Max push-stage lateral delta per env step, meters.")
parser.add_argument("--push_dz_limit", type=float, default=0.0010, help="Max push-stage vertical delta per env step, meters.")
parser.add_argument(
    "--push_z_fraction_from_bottom",
    type=float,
    default=0.20,
    help="Push-stage tool Z target as a fraction of current book height measured from the book bottom.",
)
parser.add_argument("--status_interval", type=int, default=60, help="Print status every N env steps; 0 disables.")

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

_MODE_PUSH = 2


def _set_if_present(cfg, name: str, value) -> None:
    if hasattr(cfg, name):
        setattr(cfg, name, value)


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _yaw_from_quat_wxyz(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _clamp_abs(x: torch.Tensor, limit: float) -> torch.Tensor:
    return torch.clamp(x, -float(limit), float(limit))


def _action_from_delta(unwrapped, delta: torch.Tensor) -> torch.Tensor:
    actions = torch.zeros(unwrapped.num_envs, int(unwrapped.cfg.action_space), device=unwrapped.device)
    actions[:, 0] = torch.clamp(delta[:, 0] / float(unwrapped.cfg.dx_action_scale), -1.0, 1.0)
    actions[:, 1] = torch.clamp(delta[:, 1] / float(unwrapped.cfg.dy_action_scale), -1.0, 1.0)
    actions[:, 2] = torch.clamp(delta[:, 2] / float(unwrapped.cfg.dz_action_scale), -1.0, 1.0)
    actions[:, 3] = torch.clamp(delta[:, 3] / float(unwrapped.cfg.dyaw_action_scale), -1.0, 1.0)
    actions[:, 4] = torch.clamp(delta[:, 4] / float(unwrapped.cfg.dpitch_action_scale), -1.0, 1.0)
    return actions


def _book_tilt_x(unwrapped) -> torch.Tensor:
    quat = unwrapped.book.data.root_link_quat_w
    spine_l = torch.zeros_like(quat[..., 1:4])
    spine_l[..., 1] = 1.0
    spine_w = math_utils.quat_apply(quat, spine_l)
    return torch.clamp(spine_w[:, 0], -1.0, 1.0)


def _compute_nominal_quantities(unwrapped) -> dict[str, torch.Tensor]:
    corners = unwrapped._book_corners_env()
    front_x = corners[..., 0].max(dim=-1).values
    rear_x = corners[..., 0].min(dim=-1).values
    book_depth = torch.clamp(front_x - rear_x, min=1.0e-4)
    mouth_x = float(unwrapped._geom_mouth_x)

    book_pos_env = unwrapped.book.data.root_link_pos_w - unwrapped.scene.env_origins
    tool_pos_env = unwrapped._ee_tool_pos_env()
    slot_y = unwrapped._slot_center_y()
    z_target = float(unwrapped.cfg.shelf_top_z + unwrapped.cfg.shelf_thickness + 0.5 * unwrapped.cfg.book_size[1])
    book_bottom_z = corners[..., 2].min(dim=-1).values
    book_top_z = corners[..., 2].max(dim=-1).values

    yaw = _yaw_from_quat_wxyz(unwrapped.book.data.root_link_quat_w)
    tilt_x = _book_tilt_x(unwrapped)

    return {
        "inside_fraction": torch.clamp((front_x - mouth_x) / book_depth, 0.0, 1.0),
        "front_to_mouth": front_x - mouth_x,
        "rear_to_mouth": rear_x - mouth_x,
        "book_x": book_pos_env[:, 0],
        "book_y": book_pos_env[:, 1],
        "book_z": book_pos_env[:, 2],
        "book_bottom_z": book_bottom_z,
        "book_top_z": book_top_z,
        "tool_x": tool_pos_env[:, 0],
        "tool_y": tool_pos_env[:, 1],
        "tool_z": tool_pos_env[:, 2],
        "slot_y": slot_y,
        "lat_err": slot_y - book_pos_env[:, 1],
        "z_err": book_pos_env[:, 2] - z_target,
        "yaw_err": _wrap_to_pi(yaw),
        "tilt_x": tilt_x,
        "front_x": front_x,
        "rear_x": rear_x,
    }


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed

    # This script owns the nominal controller.  Disable v6's built-in nominal and planner/debug paths.
    _set_if_present(env_cfg, "enable_nominal_controller", False)
    _set_if_present(env_cfg, "episode_length_s", args_cli.episode_length_s)
    _set_if_present(env_cfg, "debug_disable_nominal_release", True)
    _set_if_present(env_cfg, "debug_use_curobo_planner", False)
    _set_if_present(env_cfg, "debug_use_lula_rrt_planner", False)
    _set_if_present(env_cfg, "debug_spawn_at_target_tool_pose", args_cli.spawn_at_preinsert)
    spawn_inside_fraction = float(args_cli.spawn_inside_fraction)
    if args_cli.spawn_at_preinsert:
        spawn_inside_fraction -= float(args_cli.preinsert_standoff) / float(env_cfg.book_size[0])
    _set_if_present(env_cfg, "debug_spawn_inside_fraction", spawn_inside_fraction)
    _set_if_present(env_cfg, "debug_spawn_with_curobo", False)
    _set_if_present(env_cfg, "debug_done_on_preinsert_reached", False)
    _set_if_present(env_cfg, "debug_position_only_target_ee", args_cli.position_only)
    _set_if_present(env_cfg, "debug_use_full_target_ee_quat", False)
    _set_if_present(env_cfg, "show_target_book_marker", False)
    _set_if_present(env_cfg, "show_target_ee_marker", False)
    _set_if_present(env_cfg, "show_current_ee_marker", False)
    _set_if_present(env_cfg, "debug_hold_book_fixed_to_tool", args_cli.hold_book_fixed_to_tool)
    _set_if_present(env_cfg, "debug_omit_bookshelf_obstacles", args_cli.disable_bookshelf)
    _set_if_present(env_cfg, "forced_missing_book_index", args_cli.missing_index)
    _set_if_present(env_cfg, "slot_lateral_clearance_min", args_cli.slot_clearance)
    _set_if_present(env_cfg, "slot_lateral_clearance_max", args_cli.slot_clearance)
    if not args_cli.randomize_reset:
        _set_if_present(env_cfg, "reset_arm_joint_pos_noise", 0.0)
        _set_if_present(env_cfg, "book_grasp_x_jitter", 0.0)
        _set_if_present(env_cfg, "book_grasp_y_jitter", 0.0)
        _set_if_present(env_cfg, "book_grasp_z_jitter", 0.0)
        _set_if_present(env_cfg, "book_grasp_yaw_jitter", 0.0)
        _set_if_present(env_cfg, "side_book_merge_probability", 0.0)

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    print(
        "[v6 nominal standalone] "
        f"slot_clearance={args_cli.slot_clearance:.3f} m, "
        f"missing_index={args_cli.missing_index}, "
        f"spawn_at_preinsert={args_cli.spawn_at_preinsert}, "
        f"preinsert_standoff={args_cli.preinsert_standoff:.3f} m, "
        f"spawn_inside_fraction={spawn_inside_fraction:.3f}, "
        f"position_only={args_cli.position_only}, "
        f"hold_book_fixed_to_tool={args_cli.hold_book_fixed_to_tool}, "
        f"randomize_reset={args_cli.randomize_reset}, "
        f"bookshelf={'off' if args_cli.disable_bookshelf else 'on'}, "
        f"scripted_release={args_cli.scripted_release}"
    )

    env.reset()
    step = 0
    release_latched = torch.zeros(unwrapped.num_envs, dtype=torch.bool, device=unwrapped.device)
    push_anchor_latched = torch.zeros(unwrapped.num_envs, dtype=torch.bool, device=unwrapped.device)
    push_y_anchor = torch.zeros(unwrapped.num_envs, device=unwrapped.device, dtype=torch.float32)
    push_z_anchor = torch.zeros(unwrapped.num_envs, device=unwrapped.device, dtype=torch.float32)

    dyaw_limit = math.radians(args_cli.dyaw_limit_deg)
    dpitch_limit = math.radians(args_cli.dpitch_limit_deg)
    yaw_gate = math.radians(args_cli.yaw_gate_deg)
    release_yaw_tol = math.radians(args_cli.release_yaw_tol_deg)

    while simulation_app.is_running():
        with torch.inference_mode():
            q = _compute_nominal_quantities(unwrapped)
            mode = getattr(unwrapped, "_mode", None)
            push_mode = (
                mode == _MODE_PUSH
                if mode is not None
                else torch.zeros(unwrapped.num_envs, dtype=torch.bool, device=unwrapped.device)
            )

            aligned = (
                (torch.abs(q["lat_err"]) < float(args_cli.lat_gate))
                & (torch.abs(q["z_err"]) < float(args_cli.z_gate))
                & (torch.abs(q["yaw_err"]) < yaw_gate)
            )

            dx = torch.full((unwrapped.num_envs,), float(args_cli.dx), device=unwrapped.device, dtype=torch.float32)
            dx = torch.where(aligned, dx, dx * float(args_cli.unaligned_dx_scale))
            dy = _clamp_abs(float(args_cli.k_lat) * q["lat_err"], args_cli.dy_limit)
            dz = _clamp_abs(-float(args_cli.k_z) * q["z_err"], args_cli.dz_limit)
            dyaw = _clamp_abs(-float(args_cli.k_yaw) * q["yaw_err"], dyaw_limit)
            dpitch = _clamp_abs(-float(args_cli.k_tilt) * q["tilt_x"], dpitch_limit)
            if args_cli.position_only:
                dyaw = torch.zeros_like(dyaw)
                dpitch = torch.zeros_like(dpitch)

            push_dx = torch.full(
                (unwrapped.num_envs,), float(args_cli.push_dx), device=unwrapped.device, dtype=torch.float32
            )
            push_dy = _clamp_abs(float(args_cli.push_k_y) * (push_y_anchor - q["tool_y"]), args_cli.push_dy_limit)
            push_dz = _clamp_abs(float(args_cli.push_k_z) * (push_z_anchor - q["tool_z"]), args_cli.push_dz_limit)
            push_dyaw = torch.zeros_like(push_dx)
            push_dpitch = torch.zeros_like(push_dx)

            insert_delta = torch.stack((dx, dy, dz, dyaw, dpitch), dim=-1)
            push_delta = torch.stack((push_dx, push_dy, push_dz, push_dyaw, push_dpitch), dim=-1)
            delta = torch.where(push_mode.unsqueeze(-1), push_delta, insert_delta)
            actions = _action_from_delta(unwrapped, delta)

            release_ready = (
                (q["inside_fraction"] >= float(args_cli.release_fraction))
                & (torch.abs(q["lat_err"]) < float(args_cli.release_lat_tol))
                & (torch.abs(q["z_err"]) < float(args_cli.release_z_tol))
                & (torch.abs(q["yaw_err"]) < release_yaw_tol)
                & (torch.abs(q["tilt_x"]) < float(args_cli.release_tilt_tol))
            )
            if args_cli.scripted_release:
                newly_released = release_ready & ~release_latched
                if torch.any(newly_released):
                    push_y_anchor = torch.where(newly_released, q["tool_y"], push_y_anchor)
                    book_height = torch.clamp(q["book_top_z"] - q["book_bottom_z"], min=1.0e-4)
                    push_z_from_book = q["book_bottom_z"] + float(args_cli.push_z_fraction_from_bottom) * book_height
                    push_z_anchor = torch.where(
                        newly_released, push_z_from_book, push_z_anchor
                    )
                    push_anchor_latched |= newly_released
                release_latched |= release_ready
                actions[:, -1] = torch.where(release_latched, torch.ones_like(actions[:, -1]), actions[:, -1])
                if args_cli.hold_book_fixed_to_tool:
                    unwrapped.cfg.debug_hold_book_fixed_to_tool = not bool(torch.any(release_latched).item())

            need_push_anchor = push_mode & ~push_anchor_latched
            if torch.any(need_push_anchor):
                push_y_anchor = torch.where(need_push_anchor, q["tool_y"], push_y_anchor)
                book_height = torch.clamp(q["book_top_z"] - q["book_bottom_z"], min=1.0e-4)
                push_z_from_book = q["book_bottom_z"] + float(args_cli.push_z_fraction_from_bottom) * book_height
                push_z_anchor = torch.where(need_push_anchor, push_z_from_book, push_z_anchor)
                push_anchor_latched |= need_push_anchor

            _, _, terminated, truncated, _ = env.step(actions)
            done = terminated | truncated
            if torch.any(done):
                release_latched = torch.where(done, torch.zeros_like(release_latched), release_latched)
                push_anchor_latched = torch.where(done, torch.zeros_like(push_anchor_latched), push_anchor_latched)
                push_y_anchor = torch.where(done, torch.zeros_like(push_y_anchor), push_y_anchor)
                push_z_anchor = torch.where(done, torch.zeros_like(push_z_anchor), push_z_anchor)
                if args_cli.hold_book_fixed_to_tool:
                    unwrapped.cfg.debug_hold_book_fixed_to_tool = True

            if args_cli.status_interval > 0 and step % args_cli.status_interval == 0:
                mode0 = int(mode[0].item()) if mode is not None and mode.numel() > 0 else -1
                print(
                    "[v6 nominal standalone] "
                    f"step={step} mode0={mode0} "
                    f"inside={float(q['inside_fraction'][0].item()):.3f} "
                    f"front_mouth={float(q['front_to_mouth'][0].item()):+.4f} "
                    f"rear_mouth={float(q['rear_to_mouth'][0].item()):+.4f} "
                    f"book_x={float(q['book_x'][0].item()):+.4f} "
                    f"tool_x={float(q['tool_x'][0].item()):+.4f} "
                    f"lat={float(q['lat_err'][0].item()):+.4f} "
                    f"tool_y={float(q['tool_y'][0].item()):+.4f} "
                    f"tool_z={float(q['tool_z'][0].item()):+.4f} "
                    f"z={float(q['z_err'][0].item()):+.4f} "
                    f"yaw={float(torch.rad2deg(q['yaw_err'][0]).item()):+.2f}deg "
                    f"tilt_x={float(q['tilt_x'][0].item()):+.3f} "
                    f"push_y_anchor={float(push_y_anchor[0].item()):+.4f} "
                    f"push_z_anchor={float(push_z_anchor[0].item()):+.4f} "
                    f"dx={float(delta[0, 0].item()):+.4f} "
                    f"dy={float(delta[0, 1].item()):+.4f} "
                    f"dz={float(delta[0, 2].item()):+.4f} "
                    f"release_ready={bool(release_ready[0].item())}"
                )

            step += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
