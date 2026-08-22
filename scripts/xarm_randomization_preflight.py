#!/usr/bin/env python3
"""Replay deterministic xArm grasp-randomization tiers in one Isaac session."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from isaaclab.app import AppLauncher

from xarm_randomization_preflight_core import (
    DEFAULT_PROFILES,
    profile_documents,
    summarize_preflight_rows,
)


parser = argparse.ArgumentParser(
    description=(
        "Hold deterministic randomized xArm grasps, record physical drift, and "
        "select the hardest stable pre-training profile."
    )
)
parser.add_argument(
    "--task",
    default="Bookshelf-XArm7-Residual-Direct-v0",
    help="Registered Isaac Lab task.",
)
parser.add_argument(
    "--repetitions_per_slot",
    type=int,
    default=10,
    help="Samples per profile and slot. Three profiles use 30 parallel envs.",
)
parser.add_argument(
    "--hold_seconds",
    type=float,
    default=2.0,
    help="Dynamic hold duration after each randomized reset.",
)
parser.add_argument("--seed", type=int, default=20260822)
parser.add_argument("--side_book_merge_probability", type=float, default=0.35)
parser.add_argument("--minimum_pass_rate", type=float, default=0.95)
parser.add_argument(
    "--phase",
    choices=("grasp_only", "shelf_standoff"),
    default="grasp_only",
    help=(
        "grasp_only disables shelf collisions and hides neighboring books; "
        "shelf_standoff keeps them and starts the held book farther from the shelf."
    ),
)
parser.add_argument(
    "--shelf_standoff_mm",
    type=float,
    default=30.0,
    help="Additional rearward TCP offset used only by shelf_standoff.",
)
parser.add_argument(
    "--gripper_settle_steps",
    type=int,
    default=60,
    help="Physics steps that hold the requested book pose while the gripper closes.",
)
parser.add_argument("--maximum_initial_placement_error_mm", type=float, default=1.0)
parser.add_argument("--maximum_initial_rotation_error_deg", type=float, default=1.0)
parser.add_argument("--maximum_translation_drift_mm", type=float, default=3.0)
parser.add_argument("--maximum_rotation_drift_deg", type=float, default=3.0)
parser.add_argument(
    "--ground_contact_height_mm",
    type=float,
    default=2.0,
    help=(
        "Report a large relative grasp displacement as book_dropped when the "
        "book's lowest point also reaches this height above the ground."
    ),
)
parser.add_argument("--maximum_arm_target_error_deg", type=float, default=8.0)
parser.add_argument(
    "--impulse_linear_speed_mps",
    type=float,
    default=0.5,
    help="Report-only threshold for a suspicious contact impulse.",
)
parser.add_argument(
    "--impulse_angular_speed_radps",
    type=float,
    default=5.0,
    help="Report-only threshold for a suspicious contact impulse.",
)
parser.add_argument(
    "--output_dir",
    type=Path,
    default=None,
    help="Output directory. Defaults to /tmp/bookshelf_xarm_randomization_preflight_<time>.",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable Fabric and use USD I/O operations.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

_active_output_dir: Path | None = None

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.utils import math as math_utils
from isaaclab_tasks.utils import parse_env_cfg

import bookshelf.tasks  # noqa: F401
from bookshelf.tasks.direct.bookshelf.frozen_scenario_bank import (
    FROZEN_SCENARIO_FIELDS,
    frozen_scenarios_sha256,
    generate_frozen_scenario_bank,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_revision(repository: Path) -> dict[str, str | None]:
    def run(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(repository), *arguments],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() or None

    return {
        "repository": str(repository.resolve()),
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_short": run("status", "--short"),
    }


def _output_directory() -> Path:
    if args_cli.output_dir is not None:
        return args_cli.output_dir.expanduser().resolve()
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path(f"/tmp/bookshelf_xarm_randomization_preflight_{stamp}")


def _validate_arguments() -> None:
    if args_cli.task != "Bookshelf-XArm7-Residual-Direct-v0":
        raise ValueError("This preflight is restricted to Bookshelf-XArm7-Residual-Direct-v0.")
    if args_cli.repetitions_per_slot <= 0:
        raise ValueError("--repetitions_per_slot must be positive")
    if args_cli.hold_seconds <= 0.0:
        raise ValueError("--hold_seconds must be positive")
    if not 0.0 <= args_cli.side_book_merge_probability <= 1.0:
        raise ValueError("--side_book_merge_probability must be in [0, 1]")
    if not 0.0 <= args_cli.minimum_pass_rate <= 1.0:
        raise ValueError("--minimum_pass_rate must be in [0, 1]")
    if args_cli.gripper_settle_steps <= 0:
        raise ValueError("--gripper_settle_steps must be positive")
    if not math.isfinite(args_cli.shelf_standoff_mm) or args_cli.shelf_standoff_mm < 0.0:
        raise ValueError("--shelf_standoff_mm must be finite and nonnegative")
    for name in (
        "maximum_initial_placement_error_mm",
        "maximum_initial_rotation_error_deg",
        "maximum_translation_drift_mm",
        "maximum_rotation_drift_deg",
        "maximum_arm_target_error_deg",
        "impulse_linear_speed_mps",
        "impulse_angular_speed_radps",
    ):
        if not math.isfinite(float(getattr(args_cli, name))) or float(getattr(args_cli, name)) <= 0.0:
            raise ValueError(f"--{name} must be positive and finite")
    if (
        not math.isfinite(args_cli.ground_contact_height_mm)
        or args_cli.ground_contact_height_mm < 0.0
    ):
        raise ValueError("--ground_contact_height_mm must be finite and nonnegative")


def _build_bank(output_dir: Path, slot_pitch_m: float) -> tuple[Path, dict[int, dict]]:
    scenarios = []
    scenario_metadata: dict[int, dict] = {}
    profile_docs = profile_documents(DEFAULT_PROFILES)
    global_id = 0

    for repetition in range(args_cli.repetitions_per_slot):
        for profile_index, profile in enumerate(DEFAULT_PROFILES):
            profile.validate()
            x_limit, y_limit, z_limit = profile.grasp_translation_abs_m
            chunk = generate_frozen_scenario_bank(
                scenario_count=10,
                seed=args_cli.seed + 1000 * repetition + 100 * profile_index,
                slot_clearance_min=profile.slot_clearance_range_m[0],
                slot_clearance_max=profile.slot_clearance_range_m[1],
                slot_pitch=slot_pitch_m,
                row_book_count=10,
                side_book_merge_probability=args_cli.side_book_merge_probability,
                arm_joint_noise=profile.arm_joint_noise_abs_rad,
                grasp_x_jitter=x_limit,
                grasp_y_jitter=y_limit,
                grasp_z_jitter=z_limit,
                grasp_yaw_jitter=profile.grasp_yaw_abs_rad,
                missing_book_indices=range(10),
            )
            for scenario in chunk["scenarios"]:
                scenario = dict(scenario)
                scenario["scenario_id"] = global_id
                scenarios.append(scenario)
                scenario_metadata[global_id] = {
                    "profile": profile.name,
                    "profile_index": profile_index,
                    "repetition": repetition,
                    "missing_book_index": scenario["missing_book_index"],
                }
                global_id += 1

    bank = {
        "schema_version": 1,
        "kind": "bookshelf_frozen_evaluation_scenario_bank",
        "generated_at": _utc_now(),
        "scenario_count": len(scenarios),
        "scenario_sha256": frozen_scenarios_sha256(scenarios),
        "scenario_fields": FROZEN_SCENARIO_FIELDS,
        "source": {
            "kind": "xarm_randomization_preflight",
            "seed": args_cli.seed,
            "repetitions_per_slot": args_cli.repetitions_per_slot,
            "side_book_merge_probability": args_cli.side_book_merge_probability,
            "profiles": profile_docs,
            "scenario_metadata": {
                str(key): value for key, value in scenario_metadata.items()
            },
        },
        "scenarios": scenarios,
    }
    bank_path = output_dir / "frozen_preflight_scenarios.json"
    bank_path.write_text(json.dumps(bank, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return bank_path, scenario_metadata


def _tensor_cpu(snapshot: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    return snapshot[name].detach().cpu()


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError("preflight produced no sample rows")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    global _active_output_dir

    _validate_arguments()
    output_dir = _output_directory()
    output_dir.mkdir(parents=True, exist_ok=False)
    _active_output_dir = output_dir

    profiles = profile_documents(DEFAULT_PROFILES)
    profile_order = [profile["name"] for profile in profiles]
    num_envs = 10 * len(profiles)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    slot_pitch_m = float(env_cfg.neighbor_book_size[2])
    bank_path, scenario_metadata = _build_bank(output_dir, slot_pitch_m)

    env_cfg.evaluation_scenario_bank = str(bank_path)
    env_cfg.enable_residual_reset_curriculum = False
    env_cfg.enable_residual_clearance_curriculum = False
    env_cfg.enable_reset_acceptance_gate = False
    env_cfg.debug_freeze_nominal_controller = True
    env_cfg.debug_disable_nominal_release = True
    env_cfg.debug_disable_episode_resets = True
    env_cfg.debug_hold_book_fixed_to_tool = False
    env_cfg.debug_omit_bookshelf_obstacles = args_cli.phase == "grasp_only"
    env_cfg.reset_warmup_steps = args_cli.gripper_settle_steps
    if args_cli.phase == "shelf_standoff":
        reset_offset = list(env_cfg.reset_tool_offset_slot_xyz)
        reset_offset[0] -= 0.001 * args_cli.shelf_standoff_mm
        env_cfg.reset_tool_offset_slot_xyz = tuple(reset_offset)
    env_cfg.forced_missing_book_index = -1
    env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), args_cli.hold_seconds + 1.0)
    for marker_name in (
        "show_robot_base_reference_marker",
        "show_target_book_marker",
        "show_target_ee_marker",
        "show_current_ee_marker",
        "show_reachable_grasp_target_frame",
    ):
        if hasattr(env_cfg, marker_name):
            setattr(env_cfg, marker_name, False)

    run_configuration = {
        "schema_version": 1,
        "kind": "bookshelf_xarm_randomization_preflight",
        "generated_at": _utc_now(),
        "task": args_cli.task,
        "seed": args_cli.seed,
        "num_envs": num_envs,
        "repetitions_per_slot": args_cli.repetitions_per_slot,
        "hold_seconds": args_cli.hold_seconds,
        "phase": args_cli.phase,
        "gripper_settle_steps": args_cli.gripper_settle_steps,
        "shelf_standoff_mm": (
            args_cli.shelf_standoff_mm
            if args_cli.phase == "shelf_standoff"
            else 0.0
        ),
        "side_book_merge_probability": args_cli.side_book_merge_probability,
        "profiles": profiles,
        "thresholds": {
            "minimum_pass_rate": args_cli.minimum_pass_rate,
            "maximum_initial_placement_error_mm": (
                args_cli.maximum_initial_placement_error_mm
            ),
            "maximum_initial_rotation_error_deg": (
                args_cli.maximum_initial_rotation_error_deg
            ),
            "maximum_translation_drift_mm": args_cli.maximum_translation_drift_mm,
            "maximum_rotation_drift_deg": args_cli.maximum_rotation_drift_deg,
            "ground_contact_height_mm": args_cli.ground_contact_height_mm,
            "maximum_arm_target_error_deg": args_cli.maximum_arm_target_error_deg,
            "impulse_linear_speed_mps": args_cli.impulse_linear_speed_mps,
            "impulse_angular_speed_radps": args_cli.impulse_angular_speed_radps,
        },
        "scene_randomization": {
            "applied_to_physics": args_cli.phase == "shelf_standoff",
            "missing_slots": list(range(10)),
            "slot_pitch_m": slot_pitch_m,
            "neighbor_book_height_variants": True,
            "single_and_double_width_neighbor_books": True,
            "side_book_merge_probability": args_cli.side_book_merge_probability,
        },
        "physics_randomization": {
            "applied": False,
            "reason": "This preflight isolates reset geometry before mass and friction randomization.",
        },
        "repository": _git_revision(Path.cwd()),
        "frozen_scenario_bank": str(bank_path),
    }
    (output_dir / "requested_values.json").write_text(
        json.dumps(run_configuration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    step_dt = float(getattr(unwrapped, "step_dt", 1.0 / 60.0))
    hold_steps = max(1, int(round(args_cli.hold_seconds / step_dt)))
    zero_actions = torch.zeros(env.action_space.shape, device=unwrapped.device)
    rows: list[dict] = []

    try:
        for repetition in range(args_cli.repetitions_per_slot):
            env.reset()
            initial = unwrapped.debug_grasp_batch_snapshot()
            initial_pos = initial["book_position_in_grasp_frame_m"].clone()
            initial_quat = initial["book_quaternion_in_grasp_frame_wxyz"].clone()
            expected_initial_pos = initial[
                "expected_book_position_in_grasp_frame_m"
            ].clone()
            expected_initial_quat = initial[
                "expected_book_quaternion_in_grasp_frame_wxyz"
            ].clone()
            initial_placement_error = torch.linalg.norm(
                initial_pos - expected_initial_pos,
                dim=-1,
            )
            initial_rotation_error = math_utils.quat_error_magnitude(
                expected_initial_quat,
                initial_quat,
            )
            max_translation_drift = torch.zeros(num_envs, device=unwrapped.device)
            max_rotation_drift = torch.zeros(num_envs, device=unwrapped.device)
            max_linear_speed = initial["book_linear_speed_mps"].clone()
            max_angular_speed = initial["book_angular_speed_radps"].clone()
            max_arm_error = initial["arm_max_target_error_rad"].clone()
            initial_book_center_z = initial["book_position_env_m"][:, 2].clone()
            min_book_center_z = initial_book_center_z.clone()
            min_lowest_z = initial["book_lowest_z_env_m"].clone()

            for _ in range(hold_steps):
                env.step(zero_actions)
                current = unwrapped.debug_grasp_batch_snapshot()
                translation_drift = torch.linalg.norm(
                    current["book_position_in_grasp_frame_m"] - initial_pos,
                    dim=-1,
                )
                rotation_drift = math_utils.quat_error_magnitude(
                    initial_quat,
                    current["book_quaternion_in_grasp_frame_wxyz"],
                )
                max_translation_drift = torch.maximum(max_translation_drift, translation_drift)
                max_rotation_drift = torch.maximum(max_rotation_drift, rotation_drift)
                max_linear_speed = torch.maximum(max_linear_speed, current["book_linear_speed_mps"])
                max_angular_speed = torch.maximum(max_angular_speed, current["book_angular_speed_radps"])
                max_arm_error = torch.maximum(max_arm_error, current["arm_max_target_error_rad"])
                min_book_center_z = torch.minimum(
                    min_book_center_z,
                    current["book_position_env_m"][:, 2],
                )
                min_lowest_z = torch.minimum(min_lowest_z, current["book_lowest_z_env_m"])

            final = unwrapped.debug_grasp_batch_snapshot()
            scenario_ids = _tensor_cpu(initial, "scenario_bank_index").to(dtype=torch.long)
            requested_jitter = _tensor_cpu(initial, "grasp_jitter")
            requested_joint_noise = _tensor_cpu(initial, "joint_noise_rad")
            applied_joint_noise = _tensor_cpu(initial, "applied_joint_noise_rad")
            initial_rel_pos = _tensor_cpu(initial, "book_position_in_grasp_frame_m")
            initial_rel_quat = _tensor_cpu(
                initial, "book_quaternion_in_grasp_frame_wxyz"
            )
            expected_initial_rel_pos = _tensor_cpu(
                initial, "expected_book_position_in_grasp_frame_m"
            )
            expected_initial_rel_quat = _tensor_cpu(
                initial, "expected_book_quaternion_in_grasp_frame_wxyz"
            )
            initial_placement_error_cpu = initial_placement_error.detach().cpu()
            initial_rotation_error_cpu = initial_rotation_error.detach().cpu()
            final_rel_pos = _tensor_cpu(final, "book_position_in_grasp_frame_m")
            final_rel_quat = _tensor_cpu(final, "book_quaternion_in_grasp_frame_wxyz")
            slot_center_y = _tensor_cpu(initial, "slot_center_y_m")
            clearance = _tensor_cpu(initial, "slot_clearance_m")
            row_wide_mask = _tensor_cpu(initial, "row_wide_mask").to(dtype=torch.long)
            missing_index = _tensor_cpu(initial, "missing_book_index").to(dtype=torch.long)
            finger_distance = _tensor_cpu(final, "finger_origin_distance_m")
            book_half_extent = _tensor_cpu(final, "book_half_extent_across_gripper_m")

            for env_index in range(num_envs):
                scenario_id = int(scenario_ids[env_index].item())
                if scenario_id < 0 or scenario_id not in scenario_metadata:
                    raise RuntimeError(
                        f"environment {env_index} has no frozen scenario assignment"
                    )
                metadata = scenario_metadata[scenario_id]
                reasons = []
                translation_drift_mm = 1000.0 * float(max_translation_drift[env_index].item())
                rotation_drift_deg = math.degrees(float(max_rotation_drift[env_index].item()))
                initial_placement_error_mm = 1000.0 * float(
                    initial_placement_error_cpu[env_index].item()
                )
                initial_rotation_error_deg = math.degrees(
                    float(initial_rotation_error_cpu[env_index].item())
                )
                downward_drift_mm = 1000.0 * max(
                    0.0,
                    float(
                        (
                            initial_book_center_z[env_index]
                            - min_book_center_z[env_index]
                        ).item()
                    ),
                )
                arm_error_deg = math.degrees(float(max_arm_error[env_index].item()))
                lowest_z_m = float(min_lowest_z[env_index].item())
                finite = all(
                    math.isfinite(value)
                    for value in (
                        translation_drift_mm,
                        rotation_drift_deg,
                        initial_placement_error_mm,
                        initial_rotation_error_deg,
                        downward_drift_mm,
                        arm_error_deg,
                        lowest_z_m,
                    )
                )
                if not finite:
                    reasons.append("non_finite")
                excessive_relative_translation = (
                    translation_drift_mm > args_cli.maximum_translation_drift_mm
                )
                reached_ground = (
                    1000.0 * lowest_z_m <= args_cli.ground_contact_height_mm
                )
                if excessive_relative_translation:
                    reasons.append(
                        "book_dropped" if reached_ground else "translation_drift"
                    )
                if rotation_drift_deg > args_cli.maximum_rotation_drift_deg:
                    reasons.append("rotation_drift")
                if arm_error_deg > args_cli.maximum_arm_target_error_deg:
                    reasons.append("arm_tracking")

                linear_speed = float(max_linear_speed[env_index].item())
                angular_speed = float(max_angular_speed[env_index].item())
                impulse_suspect = (
                    linear_speed > args_cli.impulse_linear_speed_mps
                    or angular_speed > args_cli.impulse_angular_speed_radps
                )
                jitter = requested_jitter[env_index]
                joint_noise = requested_joint_noise[env_index]
                applied_noise = applied_joint_noise[env_index]
                row = {
                    "scenario_id": scenario_id,
                    "profile": metadata["profile"],
                    "repetition": metadata["repetition"],
                    "env_index": env_index,
                    "phase": args_cli.phase,
                    "missing_book_index": int(missing_index[env_index].item()),
                    "slot_center_y_mm": 1000.0
                    * float(slot_center_y[env_index].item()),
                    "slot_clearance_mm": 1000.0 * float(clearance[env_index].item()),
                    "row_wide_mask": int(row_wide_mask[env_index].item()),
                    "grasp_jitter_x_mm": 1000.0 * float(jitter[0].item()),
                    "grasp_jitter_y_mm": 1000.0 * float(jitter[1].item()),
                    "grasp_jitter_z_depth_mm": 1000.0 * float(jitter[2].item()),
                    "grasp_jitter_yaw_deg": math.degrees(float(jitter[3].item())),
                    "arm_joint_noise_max_abs_deg": math.degrees(
                        float(torch.abs(joint_noise).max().item())
                    ),
                    "applied_arm_joint_noise_max_abs_deg": math.degrees(
                        float(torch.abs(applied_noise).max().item())
                    ),
                    **{
                        f"arm_joint_{joint_index + 1}_noise_deg": math.degrees(
                            float(joint_noise[joint_index].item())
                        )
                        for joint_index in range(int(joint_noise.numel()))
                    },
                    **{
                        f"applied_arm_joint_{joint_index + 1}_noise_deg": (
                            math.degrees(float(applied_noise[joint_index].item()))
                        )
                        for joint_index in range(int(applied_noise.numel()))
                    },
                    "initial_book_grasp_x_mm": 1000.0 * float(initial_rel_pos[env_index, 0].item()),
                    "initial_book_grasp_y_mm": 1000.0 * float(initial_rel_pos[env_index, 1].item()),
                    "initial_book_grasp_z_mm": 1000.0 * float(initial_rel_pos[env_index, 2].item()),
                    "initial_book_grasp_qw": float(initial_rel_quat[env_index, 0].item()),
                    "initial_book_grasp_qx": float(initial_rel_quat[env_index, 1].item()),
                    "initial_book_grasp_qy": float(initial_rel_quat[env_index, 2].item()),
                    "initial_book_grasp_qz": float(initial_rel_quat[env_index, 3].item()),
                    "expected_initial_book_grasp_x_mm": 1000.0
                    * float(expected_initial_rel_pos[env_index, 0].item()),
                    "expected_initial_book_grasp_y_mm": 1000.0
                    * float(expected_initial_rel_pos[env_index, 1].item()),
                    "expected_initial_book_grasp_z_mm": 1000.0
                    * float(expected_initial_rel_pos[env_index, 2].item()),
                    "expected_initial_book_grasp_qw": float(
                        expected_initial_rel_quat[env_index, 0].item()
                    ),
                    "expected_initial_book_grasp_qx": float(
                        expected_initial_rel_quat[env_index, 1].item()
                    ),
                    "expected_initial_book_grasp_qy": float(
                        expected_initial_rel_quat[env_index, 2].item()
                    ),
                    "expected_initial_book_grasp_qz": float(
                        expected_initial_rel_quat[env_index, 3].item()
                    ),
                    "initial_placement_error_mm": initial_placement_error_mm,
                    "initial_rotation_error_deg": initial_rotation_error_deg,
                    "initial_transform_matches_request": (
                        initial_placement_error_mm
                        <= args_cli.maximum_initial_placement_error_mm
                        and initial_rotation_error_deg
                        <= args_cli.maximum_initial_rotation_error_deg
                    ),
                    "final_book_grasp_x_mm": 1000.0 * float(final_rel_pos[env_index, 0].item()),
                    "final_book_grasp_y_mm": 1000.0 * float(final_rel_pos[env_index, 1].item()),
                    "final_book_grasp_z_mm": 1000.0 * float(final_rel_pos[env_index, 2].item()),
                    "final_book_grasp_qw": float(final_rel_quat[env_index, 0].item()),
                    "final_book_grasp_qx": float(final_rel_quat[env_index, 1].item()),
                    "final_book_grasp_qy": float(final_rel_quat[env_index, 2].item()),
                    "final_book_grasp_qz": float(final_rel_quat[env_index, 3].item()),
                    "maximum_translation_drift_mm": translation_drift_mm,
                    "maximum_rotation_drift_deg": rotation_drift_deg,
                    "maximum_world_downward_motion_mm": downward_drift_mm,
                    "minimum_book_center_z_m": float(
                        min_book_center_z[env_index].item()
                    ),
                    "minimum_book_lowest_z_m": lowest_z_m,
                    "maximum_book_linear_speed_mps": linear_speed,
                    "maximum_book_angular_speed_radps": angular_speed,
                    "maximum_arm_target_error_deg": arm_error_deg,
                    "final_finger_origin_distance_mm": 1000.0 * float(finger_distance[env_index].item()),
                    "final_book_half_extent_across_gripper_mm": 1000.0 * float(book_half_extent[env_index].item()),
                    "contact_impulse_suspected": impulse_suspect,
                    "passed": not reasons,
                    "failure_reasons": ";".join(reasons),
                }
                rows.append(row)

            completed = (repetition + 1) * num_envs
            print(
                f"[PREFLIGHT] completed {completed}/{args_cli.repetitions_per_slot * num_envs} samples",
                flush=True,
            )
    finally:
        env.close()

    samples_path = output_dir / "samples.csv"
    _write_csv(samples_path, rows)
    aggregate = summarize_preflight_rows(
        rows,
        profile_order=profile_order,
        minimum_pass_rate=args_cli.minimum_pass_rate,
    )
    aggregate.update(
        {
            "schema_version": 1,
            "kind": "bookshelf_xarm_randomization_preflight_summary",
            "generated_at": _utc_now(),
            "phase": args_cli.phase,
            "samples_csv": str(samples_path),
            "requested_values": str(output_dir / "requested_values.json"),
            "frozen_scenario_bank": str(bank_path),
        }
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("===== XARM RANDOMIZATION PREFLIGHT =====")
    for profile_name in profile_order:
        result = aggregate["profiles"][profile_name]
        print(
            f"{profile_name:8s}: {result['passed']}/{result['samples']} "
            f"({100.0 * result['pass_rate']:.1f}%)"
        )
    print(f"Recommended profile: {aggregate['recommended_profile']}")
    print(f"Samples: {samples_path}")
    print(f"Summary: {summary_path}")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        traceback.print_exc()
        sys.stderr.flush()
        if _active_output_dir is not None:
            failure_path = _active_output_dir / "preflight_failure.json"
            failure_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "bookshelf_xarm_randomization_preflight_failure",
                        "generated_at": _utc_now(),
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"Failure report: {failure_path}", file=sys.stderr, flush=True)
        raise
    finally:
        simulation_app.close()
