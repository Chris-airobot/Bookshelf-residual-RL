"""Build reproducibility spec artifacts for bookshelf RL runs."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any


def _collect_cfg_attrs_by_suffix(cfg: Any, suffix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in dir(cfg):
        if name.startswith("_") or not name.endswith(suffix):
            continue
        try:
            out[name] = getattr(cfg, name)
        except Exception:
            continue
    return out


def _collect_cfg_attrs_by_prefix(cfg: Any, prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in dir(cfg):
        if name.startswith("_") or not name.startswith(prefix):
            continue
        try:
            out[name] = getattr(cfg, name)
        except Exception:
            continue
    return out


def _file_fingerprint(path: Path) -> dict[str, Any]:
    payload = {
        "path": str(path),
        "exists": path.exists(),
        "sha256": None,
    }
    if path.exists():
        data = path.read_bytes()
        payload["sha256"] = hashlib.sha256(data).hexdigest()
    return payload


def _extract_method_sources(path: Path, method_names: list[str]) -> dict[str, dict[str, Any]]:
    """Extract exact method source blocks from the env implementation file."""
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out

    src = path.read_text()
    lines = src.splitlines()
    tree = ast.parse(src)

    target_class = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "BookshelfEnv":
            target_class = node
            break
    if target_class is None:
        return out

    for node in target_class.body:
        if isinstance(node, ast.FunctionDef) and node.name in method_names and node.end_lineno is not None:
            start = max(1, int(node.lineno))
            end = max(start, int(node.end_lineno))
            block = "\n".join(lines[start - 1 : end])
            out[node.name] = {
                "start_line": start,
                "end_line": end,
                "source": block,
            }
    return out


def build_experiment_spec(
    env_cfg: Any,
    task_name: str,
    env_cfg_path: str,
    env_impl_path: str,
) -> dict[str, Any]:
    """Build a compact artifact describing task semantics for reproducibility."""
    action_scales = _collect_cfg_attrs_by_suffix(env_cfg, "_action_scale")
    obs_scales = _collect_cfg_attrs_by_suffix(env_cfg, "_obs_scale")
    reward_terms = _collect_cfg_attrs_by_prefix(env_cfg, "rew_")
    env_impl_file = Path(env_impl_path)
    method_defs = _extract_method_sources(
        env_impl_file,
        method_names=["_pre_physics_step", "_apply_action", "_get_observations", "_get_rewards", "_get_dones"],
    )

    termination_fields = {
        name: getattr(env_cfg, name)
        for name in (
            "success_steps",
            "min_steps_before_success",
            "success_rear_to_mouth_min",
            "success_rear_to_mouth_max",
            "success_front_clear_min",
            "success_front_clear_max",
            "success_z_thresh",
            "success_yaw_thresh",
            "enable_failure_terminations",
            "book_floor_lowest_z_thresh",
            "max_abs_xy",
            "fell_height_thresh",
            "upright_dot_thresh",
        )
        if hasattr(env_cfg, name)
    }

    action_info = {
        "action_space": getattr(env_cfg, "action_space", None),
        "decimation": getattr(env_cfg, "decimation", None),
        "phase_scales": {
            "phase_insert_arm_scale": getattr(env_cfg, "phase_insert_arm_scale", None),
            "phase_release_arm_scale": getattr(env_cfg, "phase_release_arm_scale", None),
            "phase_retreat_arm_scale": getattr(env_cfg, "phase_retreat_arm_scale", None),
            "phase_push_arm_scale": getattr(env_cfg, "phase_push_arm_scale", None),
        },
        "gripper": {
            "gripper_open_action_threshold": getattr(env_cfg, "gripper_open_action_threshold", None),
            "gripper_closed_joint_pos": getattr(env_cfg, "gripper_closed_joint_pos", None),
            "gripper_open_joint_pos": getattr(env_cfg, "gripper_open_joint_pos", None),
        },
        "scales": action_scales,
        "definition": {
            "notes": [
                "Residual action unpacking is in _pre_physics_step.",
                "Joint/gripper command application is in _apply_action.",
            ],
            "source_blocks": {
                "pre_physics_step": method_defs.get("_pre_physics_step"),
                "apply_action": method_defs.get("_apply_action"),
            },
        },
    }

    observation_info = {
        "observation_space": getattr(env_cfg, "observation_space", None),
        "scales": obs_scales,
        "schema": {
            "policy_vector_order": [
                "phase_s",
                "front_to_mouth_scaled",
                "lat_err_scaled",
                "yaw_err_scaled",
                "z_err_scaled",
                "front_to_back_scaled",
                "rear_to_mouth_scaled",
                "gripper_open",
                "supported_on_shelf",
                "release_ready",
                "hand_clearance_scaled",
                "tool_to_book_x_scaled",
                "tool_to_book_y_scaled",
                "tool_to_book_z_scaled",
                "tool_yaw_minus_book_yaw_scaled",
                "tool_pitch_minus_book_pitch_scaled",
            ],
            "source_block": method_defs.get("_get_observations"),
        },
    }

    reward_info = {
        "global": {
            "step_penalty": getattr(env_cfg, "step_penalty", None),
            "success_bonus": getattr(env_cfg, "success_bonus", None),
            "drop_penalty": getattr(env_cfg, "drop_penalty", None),
        },
        "terms": reward_terms,
        "definition": {
            "source_block": method_defs.get("_get_rewards"),
            "notes": [
                "Reward is phase-gated (INSERT/RELEASE/RETREAT/PUSH).",
                "Terminal bonuses/penalties are added after phase reward.",
            ],
        },
    }

    env_cfg_fp = _file_fingerprint(Path(env_cfg_path))
    env_impl_fp = _file_fingerprint(Path(env_impl_path))

    observation_terms = [
        {
            "name": "phase_s",
            "what_it_means": "Current task phase (insert/release/retreat/push) normalized to [0, 1].",
            "how_computed": "phase / 3.0",
            "params": {},
            "unit": "[0,1]",
            "source": "_get_observations",
        },
        {
            "name": "front_to_mouth_scaled",
            "what_it_means": "Front edge depth relative to slot mouth.",
            "how_computed": "clamp(front_to_mouth / front_to_mouth_obs_scale, -1, 1)",
            "params": {"front_to_mouth_obs_scale": getattr(env_cfg, "front_to_mouth_obs_scale", None)},
            "unit": "normalized[-1,1]",
            "source": "_get_observations",
        },
        {
            "name": "rear_to_mouth_scaled",
            "what_it_means": "Rear edge depth relative to slot mouth.",
            "how_computed": "clamp(rear_to_mouth / rear_to_mouth_obs_scale, -1, 1)",
            "params": {"rear_to_mouth_obs_scale": getattr(env_cfg, "rear_to_mouth_obs_scale", None)},
            "unit": "normalized[-1,1]",
            "source": "_get_observations",
        },
        {
            "name": "lat_err_scaled",
            "what_it_means": "Lateral alignment error to shelf slot center.",
            "how_computed": "clamp(lat_err / lat_err_obs_scale, -1, 1)",
            "params": {"lat_err_obs_scale": getattr(env_cfg, "lat_err_obs_scale", None)},
            "unit": "normalized[-1,1]",
            "source": "_get_observations",
        },
        {
            "name": "yaw_err_scaled",
            "what_it_means": "Yaw misalignment between book and slot.",
            "how_computed": "clamp(yaw_err / yaw_err_obs_scale, -1, 1)",
            "params": {"yaw_err_obs_scale": getattr(env_cfg, "yaw_err_obs_scale", None)},
            "unit": "normalized[-1,1]",
            "source": "_get_observations",
        },
        {
            "name": "z_err_scaled",
            "what_it_means": "Vertical alignment error to slot.",
            "how_computed": "clamp(z_err / z_err_obs_scale, -1, 1)",
            "params": {"z_err_obs_scale": getattr(env_cfg, "z_err_obs_scale", None)},
            "unit": "normalized[-1,1]",
            "source": "_get_observations",
        },
    ]

    action_terms = [
        {
            "name": "action_vector",
            "what_it_means": "Policy outputs residual arm motion and gripper command.",
            "how_computed": "a = [dx, dy, dz, dyaw, dpitch, gripper]",
            "params": {"action_space": getattr(env_cfg, "action_space", None)},
            "unit": "policy output",
            "source": "_pre_physics_step",
        },
        {
            "name": "residual_scaling",
            "what_it_means": "Residual actions are scaled to metric motions.",
            "how_computed": "dx=a0*dx_action_scale, dy=a1*dy_action_scale, dz=a2*dz_action_scale, ...",
            "params": {
                "dx_action_scale": getattr(env_cfg, "dx_action_scale", None),
                "dy_action_scale": getattr(env_cfg, "dy_action_scale", None),
                "dz_action_scale": getattr(env_cfg, "dz_action_scale", None),
                "dyaw_action_scale": getattr(env_cfg, "dyaw_action_scale", None),
                "dpitch_action_scale": getattr(env_cfg, "dpitch_action_scale", None),
            },
            "unit": "m or rad per env step",
            "source": "_pre_physics_step",
        },
        {
            "name": "gripper_logic",
            "what_it_means": "Gripper is phase-gated and opens in release based on threshold.",
            "how_computed": "if phase==RELEASE and gripper_action>=threshold then open else phase-default",
            "params": {"gripper_open_action_threshold": getattr(env_cfg, "gripper_open_action_threshold", None)},
            "unit": "joint position command",
            "source": "_apply_action",
        },
    ]

    reward_terms_explained = [
        {
            "name": "insert_phase_reward",
            "what_it_means": "Encourages insertion progress and alignment, penalizes stall/hover.",
            "how_computed": "rear_progress + entry_terms + bonuses - alignment_penalty - hover_penalty",
            "params": {"phase": "INSERT", "weights_prefix": "rew_insert_"},
            "unit": "reward",
            "source": "_get_rewards",
        },
        {
            "name": "release_phase_reward",
            "what_it_means": "Encourages opening with stable book behavior.",
            "how_computed": "open_progress - speed_penalty - alignment_penalty",
            "params": {"phase": "RELEASE", "weights_prefix": "rew_release_"},
            "unit": "reward",
            "source": "_get_rewards",
        },
        {
            "name": "retreat_phase_reward",
            "what_it_means": "Encourages hand clearance and discourages pulling the book back.",
            "how_computed": "clearance_delta - book_speed_penalty - rear_pull_penalty",
            "params": {"phase": "RETREAT", "weights_prefix": "rew_retreat_"},
            "unit": "reward",
            "source": "_get_rewards",
        },
        {
            "name": "push_phase_reward",
            "what_it_means": "Encourages final seating depth and alignment in slot.",
            "how_computed": "rear_progress + post_push - stall + support_bonus - alignment_penalty",
            "params": {"phase": "PUSH", "weights_prefix": "rew_push_"},
            "unit": "reward",
            "source": "_get_rewards",
        },
        {
            "name": "global_terminal_terms",
            "what_it_means": "Adds per-step penalty and terminal success/drop terms.",
            "how_computed": "rew_total = rew_phase + step_penalty + success_bonus + drop_penalty",
            "params": {
                "step_penalty": getattr(env_cfg, "step_penalty", None),
                "success_bonus": getattr(env_cfg, "success_bonus", None),
                "drop_penalty": getattr(env_cfg, "drop_penalty", None),
            },
            "unit": "reward",
            "source": "_get_rewards",
        },
    ]

    termination_terms = [
        {
            "name": "success",
            "what_it_means": "Book satisfies placement gates in PUSH for required hold steps.",
            "how_computed": "placement_ok AND push_phase for success_steps consecutive steps",
            "params": {
                "success_steps": getattr(env_cfg, "success_steps", None),
                "min_steps_before_success": getattr(env_cfg, "min_steps_before_success", None),
            },
            "unit": "bool",
            "source": "_get_dones",
        },
        {
            "name": "drop_failure",
            "what_it_means": "Book touches floor and is not supported on shelf.",
            "how_computed": "lowest_corner_z < book_floor_lowest_z_thresh AND not on_shelf",
            "params": {"book_floor_lowest_z_thresh": getattr(env_cfg, "book_floor_lowest_z_thresh", None)},
            "unit": "bool",
            "source": "_get_dones",
        },
        {
            "name": "time_out",
            "what_it_means": "Episode reaches max length.",
            "how_computed": "episode_length_buf >= max_episode_length - 1",
            "params": {},
            "unit": "bool",
            "source": "_get_dones",
        },
        {
            "name": "optional_debug_failures",
            "what_it_means": "Out-of-bounds/fall terminations when enabled.",
            "how_computed": "oob OR fell (only if enable_failure_terminations=True)",
            "params": {"enable_failure_terminations": getattr(env_cfg, "enable_failure_terminations", None)},
            "unit": "bool",
            "source": "_get_dones",
        },
    ]

    return {
        "task": task_name,
        "plain_english_summary": {
            "observation": "Observation is a 16D vector combining phase, depth/alignment errors, support/readiness flags, and tool-book relative pose.",
            "action": "Action is a 6D residual command for arm motion (dx,dy,dz,dyaw,dpitch) plus gripper command, then scaled and phase-gated.",
            "reward": "Reward is phase-based (insert/release/retreat/push) plus global step/success/drop terms.",
            "termination": "Episode ends on success hold, floor-drop failure, optional debug failures, or timeout.",
        },
        "reward": reward_info,
        "reward_terms_explained": reward_terms_explained,
        "observation": observation_info,
        "observation_terms_explained": observation_terms,
        "action": action_info,
        "action_terms_explained": action_terms,
        "termination": termination_fields,
        "termination_terms_explained": termination_terms,
        "termination_definition": {
            "source_block": method_defs.get("_get_dones"),
            "notes": [
                "Termination includes success hold logic, floor-drop, optional debug failure terminations, and timeout.",
            ],
        },
        "source_files": {
            "env_cfg": env_cfg_fp,
            "env_impl": env_impl_fp,
        },
    }
