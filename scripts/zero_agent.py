# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
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
    "--freeze_nominal_controller",
    action="store_true",
    default=False,
    help="Hold a residual task at its reset pose for visual inspection.",
)
parser.add_argument(
    "--xarm_reachable_grasp_demo",
    action="store_true",
    default=False,
    help=(
        "Spawn the xArm at its measured reachable joint pose, support the book "
        "while the gripper closes, then release it for a dynamic grasp test."
    ),
)
parser.add_argument(
    "--xarm_panda_reset_grasp_demo",
    action="store_true",
    default=False,
    help=(
        "Spawn the xArm at its measured reachable joint pose, snap the book "
        "to the measured finger midpoint exactly as the Panda task did, hold "
        "that pose briefly while the gripper settles, then release it."
    ),
)
parser.add_argument(
    "--xarm_forward_backward_demo",
    action="store_true",
    default=False,
    help=(
        "Run the Panda-style dynamic xArm grasp, then move the tool smoothly "
        "forward along environment +X and back to its starting pose."
    ),
)
parser.add_argument(
    "--xarm_nominal_controller_demo",
    action="store_true",
    default=False,
    help=(
        "Use the verified dynamic xArm grasp and centered bookshelf, then "
        "hand control to the nominal insertion controller with zero residual."
    ),
)
parser.add_argument(
    "--xarm_target_pose_demo",
    action="store_true",
    default=False,
    help=(
        "Spawn and hold only the xArm at its configured target joint pose; "
        "remove the book and bookshelf and bypass IK, policy, and grasp logic."
    ),
)
parser.add_argument(
    "--xarm_grasp_hold_gap_mm",
    type=float,
    default=None,
    help=(
        "Inner finger-pad gap in millimetres for the xArm grasp demo. "
        "The task default is 32 mm for the 34 mm book."
    ),
)
parser.add_argument(
    "--xarm_nominal_push_step_mm",
    type=float,
    default=0.8,
    help=(
        "Forward increment in millimetres accumulated by the bounded xArm "
        "nominal-controller PUSH target each step."
    ),
)
parser.add_argument(
    "--xarm_nominal_push_target_lead_mm",
    type=float,
    default=10.0,
    help=(
        "Maximum distance in millimetres that the retained PUSH target may "
        "lead the measured xArm tool pose."
    ),
)
parser.add_argument(
    "--xarm_nominal_episode_length_s",
    type=float,
    default=25.0,
    help="Episode duration in seconds for the xArm nominal-controller demo.",
)
parser.add_argument(
    "--xarm_nominal_push_vertical_target_lead_mm",
    type=float,
    default=10.0,
    help=(
        "Maximum vertical target lead in millimetres during PUSH lowering. "
        "This is independent of the forward PUSH target lead."
    ),
)
parser.add_argument(
    "--xarm_nominal_push_vertical_step_mm",
    type=float,
    default=1.0,
    help=(
        "Maximum vertical increment in millimetres added to the retained "
        "target during the xArm nominal-controller PUSH stage."
    ),
)
parser.add_argument(
    "--xarm_nominal_retreat_mm",
    type=float,
    default=120.0,
    help=(
        "Straight-line retreat distance in millimetres after releasing the "
        "book and before closing the gripper for PUSH."
    ),
)
parser.add_argument(
    "--xarm_nominal_push_recovery_step_mm",
    type=float,
    default=6.0,
    help=(
        "Current-relative lateral target offset in millimetres while the xArm "
        "PUSH stage is recovering alignment with the released book spine."
    ),
)
parser.add_argument(
    "--xarm_shelf_closer_mm",
    type=float,
    default=None,
    help=(
        "Move the complete bookshelf task toward the xArm by this distance "
        "in millimetres. This shifts the shelf, side books, slot, and book "
        "targets together. xArm bookshelf demos default to 50 mm; pass 0 to "
        "restore the original distance."
    ),
)
parser.add_argument(
    "--xarm_motion_distance_mm",
    type=float,
    default=50.0,
    help="Forward travel in millimetres for --xarm_forward_backward_demo.",
)
parser.add_argument(
    "--xarm_motion_half_period_steps",
    type=int,
    default=180,
    help=(
        "Simulation steps used for the forward leg and again for the return "
        "leg in --xarm_forward_backward_demo."
    ),
)
parser.add_argument(
    "--debug_no_resets",
    action="store_true",
    default=False,
    help="Disable episode resets so a failed grasp remains visible for inspection.",
)
parser.add_argument(
    "--debug_grasp_interval",
    type=int,
    default=0,
    help="Print one grasp diagnostic every N simulation steps; zero disables it.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="Exit cleanly after this many environment steps; zero runs until closed.",
)
parser.add_argument(
    "--episodes",
    type=int,
    default=0,
    help=(
        "Exit after this many completed episodes and print each terminal result; "
        "zero runs until closed."
    ),
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
parser.add_argument(
    "--all_missing_indices",
    action="store_true",
    default=False,
    help=(
        "Run one episode for every missing slot index in order, resetting the "
        "environment without restarting Isaac Sim."
    ),
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


_EPISODE_FAILURE_NAMES = {
    0: "none",
    1: "success",
    2: "book_dropped",
    3: "timeout",
    4: "push_mode_not_reached",
    5: "insertion_depth",
    6: "lateral_alignment",
    7: "vertical_alignment",
    8: "yaw_alignment",
    9: "book_not_upright",
    10: "book_unstable",
    11: "out_of_bounds",
    12: "book_fell",
}


def _first_env_value(value, env_index: int = 0):
    """Convert one vectorized environment metric to a JSON scalar."""
    if isinstance(value, torch.Tensor):
        value = value.detach().flatten()[env_index].item()
    elif hasattr(value, "reshape") and hasattr(value, "item"):
        value = value.reshape(-1)[env_index].item()
    elif isinstance(value, (list, tuple)):
        value = value[env_index]
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return value.item() if hasattr(value, "item") else str(value)


def _episode_result(metrics: dict, episode: int, terminated, truncated) -> dict:
    """Build one compact, explicit terminal report for environment zero."""
    result = {
        "episode": int(episode),
        "terminated": bool(_first_env_value(terminated)),
        "truncated": bool(_first_env_value(truncated)),
    }
    for key, value in metrics.items():
        if key.startswith("episode_metric_"):
            result[key.removeprefix("episode_metric_")] = _first_env_value(value)

    failure_code = int(result.get("failure_code", 0))
    result["failure_name"] = _EPISODE_FAILURE_NAMES.get(
        failure_code, f"unknown_{failure_code}"
    )
    result["success"] = bool(result.get("success", failure_code == 1))
    return result


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    if hasattr(env_cfg, "enable_reset_acceptance_gate"):
        env_cfg.enable_reset_acceptance_gate = False
    if args_cli.all_missing_indices and args_cli.missing_index is not None:
        raise ValueError(
            "--all_missing_indices and --missing_index cannot be used together"
        )
    selected_xarm_demos = sum(
        bool(value)
        for value in (
            args_cli.xarm_target_pose_demo,
            args_cli.xarm_reachable_grasp_demo,
            args_cli.xarm_panda_reset_grasp_demo,
            args_cli.xarm_forward_backward_demo,
            args_cli.xarm_nominal_controller_demo,
        )
    )
    if selected_xarm_demos > 1:
        raise ValueError(
            "--xarm_target_pose_demo, --xarm_reachable_grasp_demo, "
            "--xarm_panda_reset_grasp_demo, and "
            "--xarm_forward_backward_demo, and "
            "--xarm_nominal_controller_demo are mutually exclusive"
        )
    if (
        args_cli.xarm_grasp_hold_gap_mm is not None
        and not (
            args_cli.xarm_reachable_grasp_demo
            or args_cli.xarm_panda_reset_grasp_demo
            or args_cli.xarm_forward_backward_demo
            or args_cli.xarm_nominal_controller_demo
        )
    ):
        raise ValueError(
            "--xarm_grasp_hold_gap_mm requires an xArm grasp demo"
        )
    if args_cli.xarm_shelf_closer_mm is not None and args_cli.xarm_shelf_closer_mm < 0.0:
        raise ValueError("--xarm_shelf_closer_mm must be non-negative")
    if args_cli.xarm_shelf_closer_mm is not None and args_cli.xarm_shelf_closer_mm > 0.0 and not (
        args_cli.xarm_forward_backward_demo
        or args_cli.xarm_nominal_controller_demo
    ):
        raise ValueError(
            "--xarm_shelf_closer_mm requires an xArm demo with the bookshelf"
        )
    if args_cli.xarm_target_pose_demo:
        required = (
            "reset_to_slot_relative_tool_pose",
            "debug_robot_target_pose_only",
            "debug_freeze_nominal_controller",
        )
        missing = [name for name in required if not hasattr(env_cfg, name)]
        if missing:
            raise ValueError(
                "--xarm_target_pose_demo is unsupported by this task; "
                f"missing config fields: {missing}"
            )

        env_cfg.reset_to_slot_relative_tool_pose = False
        env_cfg.debug_robot_target_pose_only = True
        env_cfg.debug_reachable_grasp_sequence = False
        env_cfg.debug_hold_book_fixed_to_tool = False
        env_cfg.debug_freeze_nominal_controller = True
        env_cfg.debug_disable_episode_resets = True
        env_cfg.debug_omit_bookshelf_obstacles = True
        env_cfg.enable_residual_reset_curriculum = False
        env_cfg.reset_arm_joint_pos_noise = 0.0
        for name in (
            "show_robot_base_reference_marker",
            "show_target_book_marker",
            "show_target_ee_marker",
            "show_current_ee_marker",
            "show_reachable_grasp_target_frame",
        ):
            if hasattr(env_cfg, name):
                setattr(env_cfg, name, False)
        print(
            "[XARM_TARGET_POSE_DEMO] Robot only: spawning and holding the "
            "configured xArm joint pose with no book, shelf, IK, or policy.",
            flush=True,
        )
    if (
        args_cli.xarm_reachable_grasp_demo
        or args_cli.xarm_panda_reset_grasp_demo
        or args_cli.xarm_forward_backward_demo
        or args_cli.xarm_nominal_controller_demo
    ):
        required = (
            "reset_to_slot_relative_tool_pose",
            "debug_hold_book_fixed_to_tool",
            "debug_freeze_nominal_controller",
            "debug_reachable_grasp_sequence",
        )
        missing = [name for name in required if not hasattr(env_cfg, name)]
        if missing:
            raise ValueError(
                "--xarm_reachable_grasp_demo is unsupported by this task; "
                f"missing config fields: {missing}"
            )

        env_cfg.reset_to_slot_relative_tool_pose = False
        env_cfg.debug_hold_book_fixed_to_tool = False
        env_cfg.debug_robot_target_pose_only = True
        env_cfg.debug_spawn_book_with_collision_clearance = bool(
            args_cli.xarm_reachable_grasp_demo
        )
        env_cfg.debug_spawn_book_panda_style = bool(
            args_cli.xarm_panda_reset_grasp_demo
            or args_cli.xarm_forward_backward_demo
            or args_cli.xarm_nominal_controller_demo
        )
        env_cfg.debug_robot_forward_backward_demo = bool(
            args_cli.xarm_forward_backward_demo
        )
        env_cfg.debug_robot_nominal_controller_demo = bool(
            args_cli.xarm_nominal_controller_demo
        )
        env_cfg.debug_reachable_grasp_sequence = False
        env_cfg.debug_freeze_nominal_controller = not bool(
            args_cli.xarm_nominal_controller_demo
        )
        env_cfg.debug_disable_episode_resets = True
        if args_cli.xarm_grasp_hold_gap_mm is not None:
            from bookshelf.tasks.direct.bookshelf.xarm7_asset_cfg import (
                xarm7_gripper_joint_for_pad_gap,
            )

            hold_gap_m = 0.001 * float(args_cli.xarm_grasp_hold_gap_mm)
            hold_joint_pos = xarm7_gripper_joint_for_pad_gap(hold_gap_m)
            env_cfg.gripper_closed_joint_pos = hold_joint_pos
            print(
                "[XARM_GRASP_DEMO] custom hold gap: "
                f"{args_cli.xarm_grasp_hold_gap_mm:.3f} mm -> "
                f"drive_joint={hold_joint_pos:.6f} rad; PUSH still closes fully",
                flush=True,
            )
        if hasattr(env_cfg, "debug_omit_bookshelf_obstacles"):
            env_cfg.debug_omit_bookshelf_obstacles = True
            if (
                args_cli.xarm_forward_backward_demo
                or args_cli.xarm_nominal_controller_demo
            ):
                env_cfg.debug_omit_bookshelf_obstacles = False
        if (
            args_cli.xarm_forward_backward_demo
            or args_cli.xarm_nominal_controller_demo
        ):
            shelf_closer_mm = (
                50.0
                if args_cli.xarm_shelf_closer_mm is None
                else float(args_cli.xarm_shelf_closer_mm)
            )
            shelf_shift_m = 0.001 * shelf_closer_mm
            env_cfg.slot_x_open = float(env_cfg.slot_x_open) - shelf_shift_m
            env_cfg.slot_x_back = float(env_cfg.slot_x_back) - shelf_shift_m
            # Keep the original ten-position row expected by the scenario
            # buffers. Only shift the default index-4 diagnostic opening to
            # y=0. Explicit edge-slot tests retain the physical symmetric row.
            row_book_count = int(env_cfg.row_book_count)
            selected_missing_index = (
                0
                if args_cli.all_missing_indices
                else 4 if args_cli.missing_index is None else int(args_cli.missing_index)
            )
            if not 0 <= selected_missing_index < row_book_count:
                raise ValueError(
                    "--missing_index must be between 0 and "
                    f"{row_book_count - 1}, got {selected_missing_index}"
                )
            if hasattr(env_cfg, "forced_missing_book_index"):
                env_cfg.forced_missing_book_index = selected_missing_index
            if args_cli.all_missing_indices:
                env_cfg.debug_forced_missing_book_index_sequence = tuple(
                    range(row_book_count)
                )
                if args_cli.episodes not in (0, row_book_count):
                    raise ValueError(
                        "--all_missing_indices runs exactly one episode per slot; "
                        f"--episodes must be 0 or {row_book_count}"
                    )
                args_cli.episodes = row_book_count
            if hasattr(env_cfg, "side_book_merge_probability"):
                env_cfg.side_book_merge_probability = 0.0
            row_pitch_m = float(env_cfg.neighbor_book_size[2])
            row_y_offset_m = (
                0.5 * row_pitch_m
                if args_cli.missing_index is None and not args_cli.all_missing_indices
                else 0.0
            )
            env_cfg.debug_row_layout_y_offset_m = row_y_offset_m
            selected_slot_center_y_m = (
                selected_missing_index - 0.5 * (row_book_count - 1)
            ) * row_pitch_m + row_y_offset_m
            print(
                "[XARM_FORWARD_BACKWARD_DEMO] bookshelf restored with a "
                "deterministic opening "
                f"(row_book_count={row_book_count}, "
                f"missing={selected_missing_index}, "
                f"slot_center_y_m={selected_slot_center_y_m:.6f}, "
                f"row_y_offset_m={row_y_offset_m:.6f}, "
                f"closer_m={shelf_shift_m:.6f})",
                flush=True,
            )
            if args_cli.all_missing_indices:
                print(
                    "[XARM_SLOT_SEQUENCE] one Isaac process will reset through "
                    f"slots 0..{row_book_count - 1}",
                    flush=True,
                )
        if hasattr(env_cfg, "enable_residual_reset_curriculum"):
            env_cfg.enable_residual_reset_curriculum = False
        if hasattr(env_cfg, "reset_arm_joint_pos_noise"):
            env_cfg.reset_arm_joint_pos_noise = 0.0
        for name in (
            "book_grasp_x_jitter",
            "book_grasp_y_jitter",
            "book_grasp_z_jitter",
            "book_grasp_yaw_jitter",
        ):
            if hasattr(env_cfg, name):
                setattr(env_cfg, name, 0.0)
        if hasattr(env_cfg, "book_grasp_translation_jitter_min"):
            env_cfg.book_grasp_translation_jitter_min = (0.0, 0.0, 0.0)
        if hasattr(env_cfg, "book_grasp_translation_jitter_max"):
            env_cfg.book_grasp_translation_jitter_max = (0.0, 0.0, 0.0)
        for name in (
            "show_robot_base_reference_marker",
            "show_target_book_marker",
            "show_target_ee_marker",
            "show_current_ee_marker",
        ):
            if hasattr(env_cfg, name):
                setattr(env_cfg, name, False)
        if hasattr(env_cfg, "show_reachable_grasp_target_frame"):
            env_cfg.show_reachable_grasp_target_frame = True
        if hasattr(env_cfg, "reachable_grasp_target_frame_source"):
            env_cfg.reachable_grasp_target_frame_source = "sequence_target"
        if (
            args_cli.xarm_panda_reset_grasp_demo
            or args_cli.xarm_forward_backward_demo
            or args_cli.xarm_nominal_controller_demo
        ):
            env_cfg.debug_robot_target_gripper_ramp_steps = 1
            env_cfg.debug_robot_target_gripper_settle_steps = 10
            print(
                "[XARM_PANDA_RESET_GRASP_DEMO] The xArm is spawned at the "
                "verified target pose; the book is snapped to the measured "
                "finger midpoint using the Panda reset convention.",
                flush=True,
            )
            print(
                "[XARM_PANDA_RESET_GRASP_DEMO] immediate close command; "
                "placement-support settle: 10 steps",
                flush=True,
            )
            if args_cli.xarm_forward_backward_demo:
                if args_cli.xarm_motion_distance_mm <= 0.0:
                    raise ValueError("--xarm_motion_distance_mm must be positive")
                if args_cli.xarm_motion_half_period_steps <= 0:
                    raise ValueError(
                        "--xarm_motion_half_period_steps must be positive"
                    )
                env_cfg.debug_robot_forward_backward_distance_m = (
                    0.001 * float(args_cli.xarm_motion_distance_mm)
                )
                env_cfg.debug_robot_forward_backward_half_period_steps = int(
                    args_cli.xarm_motion_half_period_steps
                )
                print(
                    "[XARM_FORWARD_BACKWARD_DEMO] after the dynamic grasp "
                    f"settles, move +X {args_cli.xarm_motion_distance_mm:.1f} mm "
                    "and return continuously; "
                    f"half_period_steps={args_cli.xarm_motion_half_period_steps}",
                    flush=True,
                )
            elif args_cli.xarm_nominal_controller_demo:
                if args_cli.xarm_nominal_episode_length_s <= 0.0:
                    raise ValueError(
                        "--xarm_nominal_episode_length_s must be positive"
                    )
                if args_cli.xarm_nominal_retreat_mm <= 0.0:
                    raise ValueError("--xarm_nominal_retreat_mm must be positive")
                if args_cli.xarm_nominal_push_step_mm <= 0.0:
                    raise ValueError("--xarm_nominal_push_step_mm must be positive")
                if args_cli.xarm_nominal_push_target_lead_mm <= 0.0:
                    raise ValueError(
                        "--xarm_nominal_push_target_lead_mm must be positive"
                    )
                if args_cli.xarm_nominal_push_vertical_target_lead_mm <= 0.0:
                    raise ValueError(
                        "--xarm_nominal_push_vertical_target_lead_mm must be positive"
                    )
                if args_cli.xarm_nominal_push_vertical_step_mm <= 0.0:
                    raise ValueError(
                        "--xarm_nominal_push_vertical_step_mm must be positive"
                    )
                if args_cli.xarm_nominal_push_recovery_step_mm <= 0.0:
                    raise ValueError(
                        "--xarm_nominal_push_recovery_step_mm must be positive"
                    )
                env_cfg.enable_nominal_controller = True
                env_cfg.episode_length_s = float(
                    args_cli.xarm_nominal_episode_length_s
                )
                env_cfg.enable_nominal_release_assist = True
                env_cfg.nominal_release_assist_until_frac = 1.0
                env_cfg.nominal_push_dx = (
                    0.001 * float(args_cli.xarm_nominal_push_step_mm)
                )
                env_cfg.nominal_push_dz_limit = (
                    0.001 * float(args_cli.xarm_nominal_push_vertical_step_mm)
                )
                env_cfg.debug_use_full_target_ee_quat = False
                env_cfg.debug_use_base_frame_quat_deltas = False
                env_cfg.debug_position_only_target_ee = True
                # Solve at the offset xArm tool point instead of assuming the
                # 172 mm link7-to-tool offset follows wrist rotation exactly.
                env_cfg.debug_pose_ik_rotation_weight = 1.0
                # Retain the working Cartesian INSERT and fixed-retreat paths.
                # PUSH retains a bounded Cartesian target so the arm has enough
                # error to move under load without allowing target runaway.
                env_cfg.debug_integrate_position_target_ee = True
                env_cfg.debug_scripted_current_relative_target = False
                env_cfg.debug_scripted_fixed_retreat_path = True
                env_cfg.debug_scripted_fixed_retreat_total_dx = (
                    -0.001 * float(args_cli.xarm_nominal_retreat_mm)
                )
                env_cfg.debug_nominal_push_current_relative_target = False
                env_cfg.debug_nominal_push_reuse_insert_forward = True
                env_cfg.debug_nominal_push_lower_before_forward = True
                env_cfg.debug_nominal_push_align_to_book_center = True
                env_cfg.debug_nominal_push_hold_y_only = True
                env_cfg.debug_nominal_push_lock_y_to_entry = True
                env_cfg.debug_nominal_push_max_target_lead_m = (
                    0.001 * float(args_cli.xarm_nominal_push_target_lead_mm)
                )
                env_cfg.debug_nominal_push_max_vertical_target_lead_m = (
                    0.001
                    * float(args_cli.xarm_nominal_push_vertical_target_lead_mm)
                )
                # Do not add pause or recovery gates around the proven PUSH motion.
                env_cfg.debug_nominal_push_tracking_pause_enabled = False
                env_cfg.debug_nominal_push_spine_tracking_enabled = False
                env_cfg.debug_nominal_push_spine_recovery_step_m = (
                    0.001 * float(args_cli.xarm_nominal_push_recovery_step_mm)
                )
                if args_cli.ideal:
                    # The old ideal case keeps the measured book-to-tool
                    # transform rigid. Hand off immediately so the dynamic
                    # grasp cannot drift sideways before nominal control.
                    env_cfg.debug_robot_nominal_handoff_wait_steps = 0
                env_cfg.debug_spawn_inside_fraction = 0.0
                env_cfg.debug_print_residual_components = True
                env_cfg.debug_print_residual_interval = 30
                env_cfg.debug_disable_episode_resets = False
                print(
                    "[XARM_NOMINAL_CONTROLLER_DEMO] use the original "
                    "book-relative pre-insertion calculation with the xArm; "
                    "then run nominal insert/release/retreat/push with the "
                    "measured pre-insertion orientation held fixed, retained "
                    "Cartesian INSERT, fixed straight-line retreat, and a "
                    "book-centered lowering before a bounded retained-target "
                    "Panda PUSH along +X only while the gripper is fully closed, "
                    "and an exactly zero "
                    "policy residual; "
                    f"retreat={args_cli.xarm_nominal_retreat_mm:.1f} mm, "
                    f"push_step={args_cli.xarm_nominal_push_step_mm:.1f} mm, "
                    "push_target_lead="
                    f"{args_cli.xarm_nominal_push_target_lead_mm:.1f} mm, "
                    "episode_length="
                    f"{args_cli.xarm_nominal_episode_length_s:.1f} s, "
                    "vertical_target_lead="
                    f"{args_cli.xarm_nominal_push_vertical_target_lead_mm:.1f} mm, "
                    "vertical_step_limit="
                    f"{args_cli.xarm_nominal_push_vertical_step_mm:.1f} mm, "
                    "lateral_recovery_step="
                    f"{args_cli.xarm_nominal_push_recovery_step_mm:.1f} mm",
                    flush=True,
                )
        else:
            print(
                "[XARM_GRASP_DEMO] The verified xArm target pose and book-matched "
                "gripper state are held unchanged; the book is placed at the "
                "measured finger pads with palm clearance.",
                flush=True,
            )
            print(
                "[XARM_GRASP_DEMO] smooth gripper closure ramp: "
                f"{int(env_cfg.debug_robot_target_gripper_ramp_steps)} steps; "
                "placement-support settle: "
                f"{int(env_cfg.debug_robot_target_gripper_settle_steps)} steps",
                flush=True,
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
        if (
            args_cli.xarm_nominal_controller_demo
            and hasattr(env_cfg, "debug_use_base_frame_quat_deltas")
        ):
            env_cfg.debug_use_full_target_ee_quat = False
            env_cfg.debug_use_base_frame_quat_deltas = False
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
    if args_cli.freeze_nominal_controller:
        if hasattr(env_cfg, "debug_freeze_nominal_controller"):
            env_cfg.debug_freeze_nominal_controller = True
        else:
            print("[WARN]: --freeze_nominal_controller was set, but this task config does not support it.")
    if args_cli.debug_no_resets:
        if hasattr(env_cfg, "debug_disable_episode_resets"):
            env_cfg.debug_disable_episode_resets = True
            print("[WARN]: Episode resets are disabled for this debug run.")
        else:
            print("[WARN]: --debug_no_resets was set, but this task config does not support it.")
    if args_cli.debug_grasp_interval < 0:
        raise ValueError("--debug_grasp_interval must be non-negative")
    if args_cli.max_steps < 0:
        raise ValueError("--max_steps must be non-negative")
    if args_cli.episodes < 0:
        raise ValueError("--episodes must be non-negative")
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
    step_count = 0
    completed_episodes = 0
    if args_cli.debug_grasp_interval > 0:
        print(
            "[GRASP_DEBUG] "
            + json.dumps(env.unwrapped.debug_grasp_snapshot(env_index=0), sort_keys=True),
            flush=True,
        )
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute zero actions
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            # apply actions
            _, _, terminated, truncated, info = env.step(actions)
            step_count += 1
            done = torch.logical_or(
                torch.as_tensor(terminated), torch.as_tensor(truncated)
            )
            if bool(done.flatten()[0].item()):
                completed_episodes += 1
                metrics = info if isinstance(info, dict) else {}
                if not any(key.startswith("episode_metric_") for key in metrics):
                    metrics = getattr(env.unwrapped, "extras", {})
                result = _episode_result(
                    metrics,
                    completed_episodes,
                    terminated,
                    truncated,
                )
                print(
                    "[ZERO_AGENT_EPISODE] "
                    + json.dumps(result, sort_keys=True),
                    flush=True,
                )
                if args_cli.episodes > 0 and completed_episodes >= args_cli.episodes:
                    print(
                        f"[ZERO_AGENT] completed episodes={completed_episodes}; closing",
                        flush=True,
                    )
                    break
            if args_cli.debug_grasp_interval > 0 and step_count % args_cli.debug_grasp_interval == 0:
                print(
                    "[GRASP_DEBUG] "
                    + json.dumps(env.unwrapped.debug_grasp_snapshot(env_index=0), sort_keys=True),
                    flush=True,
                )
            if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
                print(
                    f"[ZERO_AGENT] reached max_steps={args_cli.max_steps}; closing",
                    flush=True,
                )
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
