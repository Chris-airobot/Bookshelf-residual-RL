# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import atexit
import json
import math
import select
import sys
import termios
import tty

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
    "--xarm_action_probe",
    nargs=7,
    type=float,
    metavar=("DX", "DY", "DZ", "BASE_Z", "BASE_X", "BASE_Y", "RELEASE"),
    help=(
        "Apply one complete normalized xArm policy action vector while the "
        "nominal controller is frozen. Components must be in [-1, 1]."
    ),
)
parser.add_argument(
    "--xarm_keyboard_action_probe",
    action="store_true",
    default=False,
    help=(
        "Interactively command xArm residual axes with keys 1-6; shifted "
        "keys command negative directions, and key 7 requests release."
    ),
)
parser.add_argument(
    "--xarm_action_probe_steps",
    type=int,
    default=30,
    help="Control steps for which --xarm_action_probe is held; default 30.",
)
parser.add_argument(
    "--xarm_action_probe_wait_steps",
    type=int,
    default=120,
    help="Settling steps before the xArm action probe starts; default 120.",
)
parser.add_argument(
    "--xarm_action_probe_observe_steps",
    type=int,
    default=180,
    help="Zero-action settling steps after the xArm action probe; default 180.",
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


def _xarm_action_probe_result(start: dict, final: dict) -> dict:
    """Summarize one keyboard command using measured EE and book poses."""

    def pose_change(position_key: str, quaternion_key: str) -> dict:
        start_pos = start[position_key]
        final_pos = final[position_key]
        translation = [float(b - a) for a, b in zip(start_pos, final_pos)]
        start_quat = start[quaternion_key]
        final_quat = final[quaternion_key]
        quat_dot = abs(sum(a * b for a, b in zip(start_quat, final_quat)))
        quat_dot = min(1.0, max(-1.0, quat_dot))
        return {
            "before_position_xyz_m": start_pos,
            "after_position_xyz_m": final_pos,
            "translation_xyz_m": translation,
            "translation_norm_m": math.sqrt(sum(v * v for v in translation)),
            "before_quaternion_wxyz": start_quat,
            "after_quaternion_wxyz": final_quat,
            "rotation_deg": math.degrees(2.0 * math.acos(quat_dot)),
        }

    return {
        "end_effector": pose_change(
            "tool_position_env_m",
            "tool_quaternion_wxyz",
        ),
        "book": pose_change(
            "book_position_env_m",
            "book_quaternion_wxyz",
        ),
        "final_arm_max_target_error_rad": final.get("arm_max_target_error_rad"),
    }


class _TerminalKeyReader:
    """Read individual terminal keys without blocking the simulator."""

    def __init__(self):
        if not sys.stdin.isatty():
            raise RuntimeError(
                "--xarm_keyboard_action_probe requires an interactive terminal"
            )
        self._fd = sys.stdin.fileno()
        self._settings = termios.tcgetattr(self._fd)
        self._closed = False
        tty.setcbreak(self._fd)
        atexit.register(self.close)

    def read_available(self) -> list[str]:
        keys = []
        while select.select([sys.stdin], [], [], 0.0)[0]:
            keys.append(sys.stdin.read(1))
        return keys

    def close(self) -> None:
        if self._closed:
            return
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._settings)
        self._closed = True


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
    action_probe_requested = (
        args_cli.xarm_action_probe is not None
        or args_cli.xarm_keyboard_action_probe
    )
    selected_xarm_demos = sum(
        bool(value)
        for value in (
            args_cli.xarm_target_pose_demo,
            args_cli.xarm_reachable_grasp_demo,
            args_cli.xarm_panda_reset_grasp_demo,
            args_cli.xarm_forward_backward_demo,
            args_cli.xarm_nominal_controller_demo,
            action_probe_requested,
        )
    )
    if selected_xarm_demos > 1:
        raise ValueError(
            "--xarm_target_pose_demo, --xarm_reachable_grasp_demo, "
            "--xarm_panda_reset_grasp_demo, and "
            "--xarm_forward_backward_demo, and "
            "--xarm_nominal_controller_demo, and "
            "--xarm_action_probe/--xarm_keyboard_action_probe are mutually exclusive"
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
    if action_probe_requested:
        if args_cli.xarm_action_probe is not None:
            if any(not math.isfinite(value) for value in args_cli.xarm_action_probe):
                raise ValueError("--xarm_action_probe values must be finite")
            if any(abs(value) > 1.0 for value in args_cli.xarm_action_probe):
                raise ValueError("--xarm_action_probe values must be within [-1, 1]")
        if args_cli.xarm_action_probe_steps <= 0:
            raise ValueError("--xarm_action_probe_steps must be positive")
        if args_cli.xarm_action_probe_wait_steps < 0:
            raise ValueError("--xarm_action_probe_wait_steps must be non-negative")
        if args_cli.xarm_action_probe_observe_steps <= 0:
            raise ValueError("--xarm_action_probe_observe_steps must be positive")
        if not hasattr(env_cfg, "debug_pose_ik_rotation_weight"):
            raise ValueError("--xarm_action_probe requires a residual bookshelf task")

        env_cfg.scene.num_envs = 1
        env_cfg.enable_residual_reset_curriculum = False
        env_cfg.enable_residual_action_scale_curriculum = False
        env_cfg.reset_arm_joint_pos_noise = 0.0
        env_cfg.book_grasp_x_jitter = 0.0
        env_cfg.book_grasp_y_jitter = 0.0
        env_cfg.book_grasp_z_jitter = 0.0
        env_cfg.book_grasp_yaw_jitter = 0.0
        env_cfg.book_grasp_translation_jitter_min = (0.0, 0.0, 0.0)
        env_cfg.book_grasp_translation_jitter_max = (0.0, 0.0, 0.0)
        env_cfg.debug_freeze_nominal_controller = True
        env_cfg.debug_disable_episode_resets = True
        # Use the normal dynamic xArm grasp in the complete bookshelf scene:
        # support the book during gripper settling, then let physics take over.
        # Never teleport the book along with the tool during motion.
        env_cfg.debug_hold_book_fixed_to_tool = False
        env_cfg.debug_omit_target_book = False
        env_cfg.debug_omit_bookshelf_obstacles = False
        env_cfg.debug_use_full_target_ee_quat = False
        # Explicit base-frame quaternion axes remain independent at the
        # xArm's near-vertical wrist orientation: key 4 is Z and key 5 is X.
        env_cfg.debug_use_base_frame_quat_deltas = True
        env_cfg.debug_action5_as_base_x = True
        env_cfg.debug_position_only_target_ee = False
        env_cfg.debug_pose_ik_rotation_weight = None
        # A key pulse represents a retained Cartesian displacement. Advance
        # that target once per control step and keep the other pose components
        # fixed while the arm converges.
        env_cfg.debug_integrate_position_target_ee = True
        env_cfg.show_robot_base_reference_marker = False
        env_cfg.show_target_book_marker = False
        env_cfg.show_reachable_grasp_target_frame = False
        env_cfg.show_current_ee_marker = True
        env_cfg.show_target_ee_marker = True
        env_cfg.target_ee_marker_source = "controller_target"
        # Keyboard output is event-based: one key line and one before/after
        # result. Periodic full grasp dumps make that impossible to read.
        args_cli.debug_grasp_interval = 0

        if args_cli.xarm_action_probe is not None:
            probe_action = [float(value) for value in args_cli.xarm_action_probe]
            per_step_physical = [
                probe_action[0] * float(env_cfg.dx_action_scale),
                probe_action[1] * float(env_cfg.dy_action_scale),
                probe_action[2] * float(env_cfg.dz_action_scale),
                math.degrees(probe_action[3] * float(env_cfg.dyaw_action_scale)),
                math.degrees(probe_action[4] * float(env_cfg.dpitch_action_scale)),
                math.degrees(
                    probe_action[5]
                    * float(env_cfg.dbase_y_rotation_action_scale)
                ),
                probe_action[6],
            ]
            print(
                "[XARM_ACTION_PROBE] nominal controller frozen; normalized="
                f"{probe_action}; per_step=[dx={per_step_physical[0]:+.6f}m, "
                f"dy={per_step_physical[1]:+.6f}m, "
                f"dz={per_step_physical[2]:+.6f}m, "
                f"dyaw={per_step_physical[3]:+.6f}deg, "
                f"dpitch={per_step_physical[4]:+.6f}deg, "
                f"dbase_y={per_step_physical[5]:+.6f}deg, "
                f"release={per_step_physical[6]:+.3f}]; "
                f"hold_steps={args_cli.xarm_action_probe_steps}",
                flush=True,
            )
        else:
            print(
                "[XARM_KEYBOARD_ACTION] controls: "
                "1=+X !=-X 2=+Y @=-Y 3=+Z #=-Z "
                "4=+base-Z $=-base-Z 5=+base-X %=-base-X "
                "6=+base-Y ^=-base-Y 7=release/open (use last); "
                "0=stop p=print pose q=quit; "
                f"pulse_steps={args_cli.xarm_action_probe_steps}",
                flush=True,
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
                "verified target pose; the book is placed from the configured "
                "grasp calibration before support is released.",
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
                # Use the same measured-pose-relative DLS IK path as Panda.
                env_cfg.debug_pose_ik_rotation_weight = None
                env_cfg.debug_integrate_position_target_ee = False
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

    if action_probe_requested and int(env.action_space.shape[-1]) != 7:
        raise ValueError(
            "xArm action probes require six Cartesian actions plus release; "
            f"got action shape {env.action_space.shape}"
        )

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    step_count = 0
    completed_episodes = 0
    action_probe_start = None
    action_probe_finished = False
    keyboard_reader = None
    keyboard_action = torch.zeros(7, device=env.unwrapped.device)
    keyboard_action_label = ""
    keyboard_action_steps_left = 0
    keyboard_observe_steps_left = 0
    keyboard_probe_start = None
    keyboard_ready_printed = False
    if args_cli.xarm_keyboard_action_probe:
        keyboard_reader = _TerminalKeyReader()
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
            if args_cli.xarm_keyboard_action_probe:
                if step_count >= int(args_cli.xarm_action_probe_wait_steps):
                    if not keyboard_ready_printed:
                        env.unwrapped.debug_hold_current_robot_pose(env_index=0)
                        print(
                            "[XARM_KEYBOARD_READY] press 1-7; Shift reverses "
                            "directions 1-6; use 7 last",
                            flush=True,
                        )
                        keyboard_ready_printed = True

                    key_actions = {
                        "1": ("+X", (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
                        "!": ("-X", (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
                        "2": ("+Y", (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
                        "@": ("-Y", (0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
                        "3": ("+Z", (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),
                        "#": ("-Z", (0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0)),
                        "4": ("+base-Z", (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)),
                        "$": ("-base-Z", (0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0)),
                        "5": ("+base-X", (0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)),
                        "%": ("-base-X", (0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0)),
                        "6": ("+base-Y", (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)),
                        "^": ("-base-Y", (0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0)),
                        "7": ("release/open", (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
                    }
                    quit_requested = False
                    for key in keyboard_reader.read_available():
                        if key == "q":
                            quit_requested = True
                            break
                        if key == "p":
                            print(
                                "[XARM_KEYBOARD_POSE] "
                                + json.dumps(
                                    env.unwrapped.debug_grasp_snapshot(env_index=0),
                                    sort_keys=True,
                                ),
                                flush=True,
                            )
                            continue
                        if key == "0" or key == " ":
                            keyboard_action_steps_left = 0
                            keyboard_observe_steps_left = 0
                            keyboard_probe_start = None
                            held_pose = env.unwrapped.debug_hold_current_robot_pose(
                                env_index=0
                            )
                            print(
                                "[XARM_KEYBOARD_ACTION] stopped at measured pose "
                                f"tool_xyz={held_pose['tool_position_env_m']}",
                                flush=True,
                            )
                            continue
                        if key not in key_actions:
                            continue
                        if keyboard_action_steps_left > 0:
                            print(
                                f"[XARM_KEYBOARD_ACTION] key={key!r} ignored; "
                                "the short command pulse is still active",
                                flush=True,
                            )
                            continue
                        if keyboard_probe_start is not None:
                            keyboard_probe_final = (
                                env.unwrapped.debug_grasp_snapshot(env_index=0)
                            )
                            result = _xarm_action_probe_result(
                                keyboard_probe_start,
                                keyboard_probe_final,
                            )
                            result["command"] = keyboard_action_label
                            print(
                                "[XARM_KEYBOARD_RESULT] "
                                + json.dumps(result, sort_keys=True),
                                flush=True,
                            )
                            keyboard_probe_start = None
                            keyboard_observe_steps_left = 0
                        keyboard_action_label, values = key_actions[key]
                        keyboard_probe_start = (
                            env.unwrapped.debug_grasp_snapshot(env_index=0)
                        )
                        keyboard_action = torch.tensor(
                            values,
                            device=env.unwrapped.device,
                            dtype=actions.dtype,
                        )
                        keyboard_action_steps_left = int(
                            args_cli.xarm_action_probe_steps
                        )
                        keyboard_observe_steps_left = 0
                        print(
                            "[XARM_KEYBOARD_ACTION] "
                            f"key={key!r} command={keyboard_action_label} "
                            f"normalized={list(values)} "
                            f"steps={keyboard_action_steps_left}",
                            flush=True,
                        )
                    if quit_requested:
                        print("[XARM_KEYBOARD_ACTION] quitting", flush=True)
                        break

                if keyboard_action_steps_left > 0:
                    actions[0, :] = keyboard_action
                    keyboard_action_steps_left -= 1
                    if keyboard_action_steps_left == 0:
                        keyboard_observe_steps_left = int(
                            args_cli.xarm_action_probe_observe_steps
                        )
                elif keyboard_observe_steps_left > 0:
                    keyboard_observe_steps_left -= 1
            if args_cli.xarm_action_probe is not None:
                probe_start_step = int(args_cli.xarm_action_probe_wait_steps)
                probe_stop_step = probe_start_step + int(args_cli.xarm_action_probe_steps)
                if step_count == probe_start_step:
                    action_probe_start = env.unwrapped.debug_grasp_snapshot(env_index=0)
                    print(
                        "[XARM_ACTION_PROBE_START] "
                        + json.dumps(action_probe_start, sort_keys=True),
                        flush=True,
                    )
                if probe_start_step <= step_count < probe_stop_step:
                    actions[0, :] = torch.tensor(
                        args_cli.xarm_action_probe,
                        device=env.unwrapped.device,
                        dtype=actions.dtype,
                    )
            # apply actions
            _, _, terminated, truncated, info = env.step(actions)
            step_count += 1
            if (
                args_cli.xarm_keyboard_action_probe
                and keyboard_probe_start is not None
                and keyboard_action_steps_left == 0
                and keyboard_observe_steps_left == 0
            ):
                keyboard_probe_final = env.unwrapped.debug_grasp_snapshot(env_index=0)
                result = _xarm_action_probe_result(
                    keyboard_probe_start, keyboard_probe_final
                )
                result["command"] = keyboard_action_label
                print(
                    "[XARM_KEYBOARD_RESULT] "
                    + json.dumps(result, sort_keys=True),
                    flush=True,
                )
                keyboard_probe_start = None
            if args_cli.xarm_action_probe is not None:
                probe_finish_step = (
                    int(args_cli.xarm_action_probe_wait_steps)
                    + int(args_cli.xarm_action_probe_steps)
                    + int(args_cli.xarm_action_probe_observe_steps)
                )
                if step_count >= probe_finish_step and not action_probe_finished:
                    if action_probe_start is None:
                        raise RuntimeError("xArm action probe did not capture its start pose")
                    action_probe_final = env.unwrapped.debug_grasp_snapshot(env_index=0)
                    print(
                        "[XARM_ACTION_PROBE_RESULT] "
                        + json.dumps(
                            _xarm_action_probe_result(
                                action_probe_start, action_probe_final
                            ),
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    action_probe_finished = True
                    break
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

    if keyboard_reader is not None:
        keyboard_reader.close()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
