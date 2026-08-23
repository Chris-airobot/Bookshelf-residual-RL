#!/usr/bin/env python3
"""Record synchronized book, tool, and gripper geometry at accepted release."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser(
    description="Run one policy until INSERT release is accepted and record the geometry."
)
parser.add_argument("--task", required=True, help="Registered Isaac Lab task name.")
parser.add_argument("--checkpoint", required=True, help="Stable-Baselines3 model.zip path.")
parser.add_argument("--output", required=True, help="Output JSON path.")
parser.add_argument(
    "--task_source_root",
    default=str(_REPO_ROOT / "source" / "bookshelf"),
    help="bookshelf Python source root; use the archived Panda checkout for an exact old run.",
)
parser.add_argument(
    "--agent", default="sb3_cfg_entry_point", help="Agent configuration registry entry point."
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_steps", type=int, default=5000)
parser.add_argument(
    "--progress_interval",
    type=int,
    default=100,
    help="Print mode, depth, and release-action progress every N steps; zero disables it.",
)
parser.add_argument(
    "--controller_trace",
    action="store_true",
    help="Print the environment's nominal/residual/final Cartesian components for movement diagnosis.",
)
parser.add_argument(
    "--slot_clearance",
    type=float,
    default=None,
    help="Optional fixed total extra lateral clearance in meters.",
)
parser.add_argument(
    "--ideal_reset",
    action="store_true",
    help="Disable reset/grasp jitter so Panda and xArm frame geometry can be compared directly.",
)
parser.add_argument(
    "--policy_action_adapter",
    choices=("auto", "none", "panda6_to_xarm7"),
    default="auto",
    help="Map an old six-action Panda policy to the seven-action xArm task when needed.",
)

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

task_source_root = Path(args_cli.task_source_root).expanduser().resolve()
if not task_source_root.is_dir():
    raise FileNotFoundError(f"bookshelf task source root does not exist: {task_source_root}")
sys.path.insert(0, str(task_source_root))
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import torch
from pxr import Usd, UsdGeom, UsdPhysics
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import bookshelf.tasks  # noqa: F401


# Frozen simulator convention used by the ROS policy adapter. Quaternion here is WXYZ.
BOOK_TO_POLICY_TOOL_TRANSLATION_M = np.asarray(
    [-0.03682894911272874, -0.0010947493841520109, 0.0007504753567338686],
    dtype=np.float64,
)
BOOK_TO_POLICY_TOOL_QUATERNION_WXYZ = np.asarray(
    [-0.020317111231816707, 0.7205555086260095, -0.021613755787561972, 0.6927624553486299],
    dtype=np.float64,
)

XARM_GRIPPER_BODY_CANDIDATES = (
    "xarm_gripper_base_link",
    "left_outer_knuckle",
    "left_finger",
    "left_inner_knuckle",
    "right_outer_knuckle",
    "right_finger",
    "right_inner_knuckle",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError(f"invalid quaternion: {q.tolist()}")
    return q / norm


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    q = _quat_normalize(q)
    return np.asarray([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = _quat_normalize(a)
    bw, bx, by, bz = _quat_normalize(b)
    return _quat_normalize(
        np.asarray(
            [
                aw * bw - ax * bx - ay * by - az * bz,
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
            ],
            dtype=np.float64,
        )
    )


def _quat_rotate(q: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    q = _quat_normalize(q)
    vectors = np.asarray(vectors, dtype=np.float64)
    qw = q[0]
    qv = q[1:]
    uv = np.cross(np.broadcast_to(qv, vectors.shape), vectors)
    uuv = np.cross(np.broadcast_to(qv, vectors.shape), uv)
    return vectors + 2.0 * (qw * uv + uuv)


def _compose_pose(
    parent_position: np.ndarray,
    parent_quaternion: np.ndarray,
    child_position_in_parent: np.ndarray,
    child_quaternion_in_parent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    position = parent_position + _quat_rotate(parent_quaternion, child_position_in_parent)
    quaternion = _quat_multiply(parent_quaternion, child_quaternion_in_parent)
    return position, quaternion


def _relative_pose(
    parent_position: np.ndarray,
    parent_quaternion: np.ndarray,
    child_position: np.ndarray,
    child_quaternion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    inverse = _quat_conjugate(parent_quaternion)
    position = _quat_rotate(inverse, child_position - parent_position)
    quaternion = _quat_multiply(inverse, child_quaternion)
    return position, quaternion


def _pose_dict(position: np.ndarray, quaternion: np.ndarray) -> dict:
    return {
        "position_xyz_m": np.asarray(position, dtype=float).tolist(),
        "quaternion_wxyz": _quat_normalize(quaternion).tolist(),
    }


def _box_corners(minimum: np.ndarray, maximum: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            [x, y, z]
            for x in (minimum[0], maximum[0])
            for y in (minimum[1], maximum[1])
            for z in (minimum[2], maximum[2])
        ],
        dtype=np.float64,
    )


def _range_to_arrays(value) -> tuple[np.ndarray, np.ndarray] | None:
    aligned = value.ComputeAlignedRange() if hasattr(value, "ComputeAlignedRange") else value
    if aligned.IsEmpty():
        return None
    return (
        np.asarray(aligned.GetMin(), dtype=np.float64),
        np.asarray(aligned.GetMax(), dtype=np.float64),
    )


def _nearest_rigid_body(prim: Usd.Prim, stop_path: str) -> Usd.Prim | None:
    current = prim
    while current and current.IsValid() and str(current.GetPath()).startswith(stop_path):
        if current.HasAPI(UsdPhysics.RigidBodyAPI):
            return current
        current = current.GetParent()
    return None


def _find_body_prim(stage: Usd.Stage, body_name: str) -> Usd.Prim:
    robot_root = stage.GetPrimAtPath("/World/envs/env_0/Robot")
    if not robot_root.IsValid():
        raise RuntimeError("robot prim /World/envs/env_0/Robot was not found")
    matches = [
        prim
        for prim in Usd.PrimRange(robot_root)
        if prim.GetName() == body_name and prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    if len(matches) != 1:
        paths = [str(prim.GetPath()) for prim in matches]
        raise RuntimeError(f"expected one rigid body prim named {body_name!r}, got {paths}")
    return matches[0]


def _body_collision_local_bounds(
    stage: Usd.Stage, body_name: str
) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    body_prim = _find_body_prim(stage, body_name)
    body_path = str(body_prim.GetPath())
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy, UsdGeom.Tokens.render],
        useExtentsHint=True,
    )
    bounds = []
    collision_paths = []
    for prim in Usd.PrimRange(body_prim):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        owner = _nearest_rigid_body(prim, body_path)
        if owner is None or owner.GetPath() != body_prim.GetPath():
            continue
        value = _range_to_arrays(cache.ComputeRelativeBound(prim, body_prim))
        if value is None:
            continue
        bounds.append(value)
        collision_paths.append(str(prim.GetPath()))
    source = "usd_collision_api_local_bounds_transformed_by_live_body_pose"
    if not bounds:
        # The archived Franka USD authors geometry beneath panda_hand without
        # discoverable CollisionAPI prims. Preserve the measurement with an
        # explicitly labeled, more conservative authored-body envelope.
        value = _range_to_arrays(cache.ComputeRelativeBound(body_prim, body_prim))
        if value is None:
            raise RuntimeError(
                f"no collision or authored geometry bounds found for rigid body "
                f"{body_name!r} at {body_path}"
            )
        bounds.append(value)
        collision_paths.append(body_path)
        source = "usd_authored_body_bounds_fallback_no_collision_api"
    minimum = np.min(np.stack([item[0] for item in bounds]), axis=0)
    maximum = np.max(np.stack([item[1] for item in bounds]), axis=0)
    return minimum, maximum, collision_paths, source


def _static_shelf_aabbs(stage: Usd.Stage, env_origin: np.ndarray) -> list[dict]:
    root = stage.GetPrimAtPath("/World/envs/env_0/Bookshelf")
    if not root.IsValid():
        raise RuntimeError("bookshelf prim /World/envs/env_0/Bookshelf was not found")
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy, UsdGeom.Tokens.render],
        useExtentsHint=True,
    )
    result = []
    for prim in root.GetChildren():
        value = _range_to_arrays(cache.ComputeWorldBound(prim))
        if value is None:
            continue
        minimum, maximum = value
        result.append(
            {
                "name": prim.GetName(),
                "prim_path": str(prim.GetPath()),
                "minimum_xyz_m": (minimum - env_origin).tolist(),
                "maximum_xyz_m": (maximum - env_origin).tolist(),
            }
        )
    if not result:
        raise RuntimeError("no static shelf obstacle bounds were found")
    return result


def _aabb_distance(a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray) -> float:
    separation = np.maximum(np.maximum(a_min - b_max, b_min - a_max), 0.0)
    return float(np.linalg.norm(separation))


def _slot_center_y(raw, env_index: int) -> float:
    if hasattr(raw, "_slot_center_y"):
        return float(_as_numpy(raw._slot_center_y())[env_index])
    return float(raw.cfg.slot_center_y)


def _slot_clearance(raw, env_index: int) -> float:
    if hasattr(raw, "_slot_lateral_clearance"):
        return float(_as_numpy(raw._slot_lateral_clearance())[env_index])
    if hasattr(raw, "_current_slot_lateral_clearance"):
        return float(raw._current_slot_lateral_clearance())
    return float(raw.cfg.slot_lateral_clearance)


def _role_body_names(raw) -> tuple[str, str, str]:
    """Return palm/left/right names across old Panda and current shared configs."""
    available = set(raw.robot.body_names)
    hand = str(getattr(raw.cfg, "robot_hand_body_name", "panda_hand"))
    left = str(getattr(raw.cfg, "robot_left_finger_body_name", "panda_leftfinger"))
    right = str(getattr(raw.cfg, "robot_right_finger_body_name", "panda_rightfinger"))
    missing = [name for name in (hand, left, right) if name not in available]
    if missing:
        raise RuntimeError(
            f"configured diagnostic body names are unavailable: {missing}; "
            f"available bodies are {sorted(available)}"
        )
    return hand, left, right


def _selected_gripper_bodies(raw) -> list[str]:
    available = set(raw.robot.body_names)
    configured = list(_role_body_names(raw))
    candidates = list(XARM_GRIPPER_BODY_CANDIDATES) if "xarm_gripper_base_link" in available else configured
    selected = []
    for name in candidates + configured:
        if name in available and name not in selected:
            selected.append(name)
    return selected


def _body_pose(raw, body_name: str, env_index: int, env_origin: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    ids, _ = raw.robot.find_bodies(body_name)
    if len(ids) != 1:
        raise RuntimeError(f"expected one live body named {body_name!r}, got {ids}")
    body_id = int(ids[0])
    position = _as_numpy(raw.robot.data.body_pos_w[env_index, body_id]).astype(np.float64) - env_origin
    quaternion = _as_numpy(raw.robot.data.body_quat_w[env_index, body_id]).astype(np.float64)
    return position, quaternion, body_id


def _collision_geometry_snapshot(raw, env_index: int, shelf_aabbs: list[dict], opening: dict) -> dict:
    stage = sim_utils.get_current_stage()
    env_origin = _as_numpy(raw.scene.env_origins[env_index]).astype(np.float64)
    bodies = []
    closest = None
    for name in _selected_gripper_bodies(raw):
        position, quaternion, _ = _body_pose(raw, name, env_index, env_origin)
        local_min, local_max, collision_paths, collision_source = _body_collision_local_bounds(
            stage, name
        )
        world_corners = _quat_rotate(quaternion, _box_corners(local_min, local_max)) + position
        world_min = world_corners.min(axis=0)
        world_max = world_corners.max(axis=0)
        obstacle_distances = []
        for obstacle in shelf_aabbs:
            distance = _aabb_distance(
                world_min,
                world_max,
                np.asarray(obstacle["minimum_xyz_m"], dtype=np.float64),
                np.asarray(obstacle["maximum_xyz_m"], dtype=np.float64),
            )
            entry = {"obstacle": obstacle["name"], "distance_m": distance}
            obstacle_distances.append(entry)
            if closest is None or distance < closest["distance_m"]:
                closest = {"body": name, **entry}
        slot_min_y = float(opening["minimum_y_m"])
        slot_max_y = float(opening["maximum_y_m"])
        body = {
            "name": name,
            "frame_pose": _pose_dict(position, quaternion),
            "collision_source": collision_source,
            "collision_prim_paths": collision_paths,
            "local_bounds": {
                "minimum_xyz_m": local_min.tolist(),
                "maximum_xyz_m": local_max.tolist(),
            },
            "world_envelope": {
                "minimum_xyz_m": world_min.tolist(),
                "maximum_xyz_m": world_max.tolist(),
            },
            "opening_margins": {
                "mouth_to_body_nearest_x_m": float(opening["mouth_x_m"] - world_max[0]),
                "body_to_back_x_m": float(opening["back_x_m"] - world_max[0]),
                "left_channel_margin_m": float(world_min[1] - slot_min_y),
                "right_channel_margin_m": float(slot_max_y - world_max[1]),
                "deck_margin_m": float(world_min[2] - opening["deck_z_m"]),
                "neighbor_top_reference_margin_m": float(opening["neighbor_top_z_m"] - world_max[2]),
            },
            "shelf_obstacle_aabb_distances": obstacle_distances,
        }
        bodies.append(body)
    return {
        "method": (
            "Conservative AABB separation using each rigid body's USD collision bounds, "
            "transformed by the synchronized live body pose. Zero can mean envelope overlap; "
            "this is not triangle-mesh signed distance."
        ),
        "bodies": bodies,
        "closest_body_obstacle_pair": closest,
    }


def _release_snapshot(
    raw,
    env_index: int,
    step: int,
    observation_before: np.ndarray,
    policy_action: np.ndarray,
    applied_action: np.ndarray,
    action_adapter: str,
    checkpoint: Path,
    vecnormalize: Path,
) -> dict:
    env_origin = _as_numpy(raw.scene.env_origins[env_index]).astype(np.float64)
    book_position = _as_numpy(raw.book.data.root_link_pos_w[env_index]).astype(np.float64) - env_origin
    book_quaternion = _as_numpy(raw.book.data.root_link_quat_w[env_index]).astype(np.float64)
    corners = _as_numpy(raw._book_corners_env()[env_index]).astype(np.float64)
    metrics = raw._compute_task_metrics()
    metric_values = {
        name: float(_as_numpy(value)[env_index])
        for name, value in metrics.items()
    }

    hand_name, left_name, right_name = _role_body_names(raw)
    hand_position, hand_quaternion, _ = _body_pose(raw, hand_name, env_index, env_origin)
    left_position, left_quaternion, _ = _body_pose(raw, left_name, env_index, env_origin)
    right_position, right_quaternion, _ = _body_pose(raw, right_name, env_index, env_origin)

    tcp_position = _as_numpy(raw._ee_tool_pos_env()[env_index]).astype(np.float64)
    tcp_quaternion = _as_numpy(raw.robot.data.body_quat_w[env_index, raw._ee_body_idx]).astype(np.float64)
    policy_position, policy_quaternion = _compose_pose(
        book_position,
        book_quaternion,
        BOOK_TO_POLICY_TOOL_TRANSLATION_M,
        BOOK_TO_POLICY_TOOL_QUATERNION_WXYZ,
    )
    book_to_tcp = _relative_pose(book_position, book_quaternion, tcp_position, tcp_quaternion)
    tcp_to_policy = _relative_pose(tcp_position, tcp_quaternion, policy_position, policy_quaternion)

    center_y = _slot_center_y(raw, env_index)
    clearance = _slot_clearance(raw, env_index)
    neighbor_size = getattr(raw.cfg, "neighbor_book_size", raw.cfg.book_size)
    slot_half_width = 0.5 * (float(neighbor_size[2]) + clearance)
    opening = {
        "mouth_x_m": float(raw._geom_mouth_x),
        "back_x_m": float(raw.cfg.slot_x_back),
        "center_y_m": center_y,
        "total_extra_lateral_clearance_m": clearance,
        "minimum_y_m": center_y - slot_half_width,
        "maximum_y_m": center_y + slot_half_width,
        "deck_z_m": float(raw.cfg.shelf_top_z + raw.cfg.shelf_thickness),
        "neighbor_top_z_m": float(
            raw.cfg.shelf_top_z + raw.cfg.shelf_thickness + float(neighbor_size[1])
        ),
    }
    stage = sim_utils.get_current_stage()
    shelf_aabbs = _static_shelf_aabbs(stage, env_origin)
    collision_geometry = _collision_geometry_snapshot(raw, env_index, shelf_aabbs, opening)

    front_x = float(corners[:, 0].max())
    rear_x = float(corners[:, 0].min())
    return {
        "schema_version": 1,
        "kind": "bookshelf_release_geometry_diagnostic",
        "task": args_cli.task,
        "task_source_root": str(task_source_root),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "vecnormalize": str(vecnormalize),
        "vecnormalize_sha256": _sha256(vecnormalize),
        "seed": int(args_cli.seed),
        "release": {
            "accepted": True,
            "global_diagnostic_step": int(step),
            "episode_step": int(_as_numpy(raw.episode_length_buf)[env_index]),
            "mode_after_step": int(_as_numpy(raw._mode)[env_index]),
            "normalized_observation_before_action": observation_before.tolist(),
            "policy_action": policy_action.tolist(),
            "applied_environment_action": applied_action.tolist(),
            "action_adapter": action_adapter,
        },
        "book": {
            "pose": _pose_dict(book_position, book_quaternion),
            "corners_xyz_m": corners.tolist(),
            "rear_x_m": rear_x,
            "front_x_m": front_x,
            "leading_edge_penetration_from_mouth_m": front_x - opening["mouth_x_m"],
            "trailing_edge_depth_from_mouth_m": rear_x - opening["mouth_x_m"],
            "front_to_back_remaining_m": opening["back_x_m"] - front_x,
            "task_metrics": metric_values,
        },
        "physical_frames": {
            "palm_body_name": hand_name,
            "palm": _pose_dict(hand_position, hand_quaternion),
            "left_finger_body_name": left_name,
            "left_finger": _pose_dict(left_position, left_quaternion),
            "right_finger_body_name": right_name,
            "right_finger": _pose_dict(right_position, right_quaternion),
            "tcp": _pose_dict(tcp_position, tcp_quaternion),
        },
        "virtual_policy_tool": {
            "pose": _pose_dict(policy_position, policy_quaternion),
            "book_to_policy_tool": _pose_dict(
                BOOK_TO_POLICY_TOOL_TRANSLATION_M,
                BOOK_TO_POLICY_TOOL_QUATERNION_WXYZ,
            ),
            "book_to_tcp": _pose_dict(*book_to_tcp),
            "tcp_to_policy_tool": _pose_dict(*tcp_to_policy),
        },
        "slot_opening": opening,
        "static_shelf_obstacle_envelopes": shelf_aabbs,
        "physical_gripper_to_shelf": collision_geometry,
    }


def _apply_ideal_reset(env_cfg) -> None:
    for name in (
        "reset_arm_joint_pos_noise",
        "book_grasp_x_jitter",
        "book_grasp_y_jitter",
        "book_grasp_z_jitter",
        "book_grasp_yaw_jitter",
    ):
        if hasattr(env_cfg, name):
            setattr(env_cfg, name, 0.0)
    for name in (
        "enable_residual_reset_curriculum",
        "enable_slot_clearance_curriculum",
        "enable_residual_clearance_curriculum",
    ):
        if hasattr(env_cfg, name):
            setattr(env_cfg, name, False)


def _adapt_action(policy_action: np.ndarray, environment_action_dim: int) -> tuple[np.ndarray, str]:
    policy_action = np.asarray(policy_action, dtype=np.float32)
    if policy_action.ndim != 2 or policy_action.shape[0] != 1:
        raise ValueError(f"expected one policy action row, got {policy_action.shape}")
    policy_dim = int(policy_action.shape[1])
    requested = args_cli.policy_action_adapter
    if requested == "auto":
        requested = "none" if policy_dim == environment_action_dim else "panda6_to_xarm7"
    if requested == "none":
        if policy_dim != environment_action_dim:
            raise ValueError(
                f"policy has {policy_dim} actions but environment has {environment_action_dim}; "
                "use --policy_action_adapter panda6_to_xarm7 when comparing the old Panda policy on xArm"
            )
        return np.clip(policy_action, -1.0, 1.0), "none_environment_clamp"
    if requested == "panda6_to_xarm7":
        if policy_dim != 6 or environment_action_dim != 7:
            raise ValueError(
                "panda6_to_xarm7 requires a six-action policy and seven-action environment, "
                f"got {policy_dim} and {environment_action_dim}"
            )
        adapted = np.zeros((1, 7), dtype=np.float32)
        adapted[:, 0:5] = policy_action[:, 0:5]
        adapted[:, 6] = policy_action[:, 5]
        return np.clip(adapted, -1.0, 1.0), "panda6_to_xarm7_zero_extra_rotation_environment_clamp"
    raise AssertionError(requested)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
    output = Path(args_cli.output).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    if args_cli.max_steps <= 0:
        raise ValueError("--max_steps must be positive")
    if args_cli.progress_interval < 0:
        raise ValueError("--progress_interval must be non-negative")

    env_cfg.scene.num_envs = 1
    env_cfg.seed = int(args_cli.seed)
    agent_cfg["seed"] = int(args_cli.seed)
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.log_dir = str(output.parent)
    if hasattr(env_cfg, "enable_reset_acceptance_gate"):
        env_cfg.enable_reset_acceptance_gate = False
    if hasattr(env_cfg, "enable_nominal_release_assist"):
        env_cfg.enable_nominal_release_assist = False
    if hasattr(env_cfg, "policy_release_guard_mode"):
        env_cfg.policy_release_guard_mode = "none"
    if args_cli.controller_trace and hasattr(env_cfg, "debug_print_residual_components"):
        env_cfg.debug_print_residual_components = True
        env_cfg.debug_print_residual_interval = max(1, int(args_cli.progress_interval or 1))
        env_cfg.debug_print_residual_env_index = 0
    if args_cli.ideal_reset:
        _apply_ideal_reset(env_cfg)
    if args_cli.slot_clearance is not None:
        for name in ("slot_lateral_clearance", "slot_lateral_clearance_min", "slot_lateral_clearance_max"):
            if hasattr(env_cfg, name):
                setattr(env_cfg, name, float(args_cli.slot_clearance))
        for name in ("enable_slot_clearance_curriculum", "enable_residual_clearance_curriculum"):
            if hasattr(env_cfg, name):
                setattr(env_cfg, name, False)

    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    raw = gym_env.unwrapped
    process_sb3_cfg(agent_cfg, raw.num_envs)
    if isinstance(raw, DirectMARLEnv):
        gym_env = multi_agent_to_single_agent(gym_env)
        raw = gym_env.unwrapped
    env = Sb3VecEnvWrapper(gym_env, fast_variant=True)
    # Preserve the real target environment dimension before VecNormalize.load().
    # SB3 restores the Panda checkpoint's serialized six-action metadata even
    # when its wrapped xArm environment correctly accepts seven actions.
    environment_action_dim = int(env.action_space.shape[0])

    vecnormalize = Path(
        str(checkpoint).replace("/model", "/model_vecnormalize").replace(".zip", ".pkl")
    )
    if not vecnormalize.is_file():
        raise FileNotFoundError(
            f"matching VecNormalize state does not exist: {vecnormalize}. "
            "The release action is not comparable without the training normalization."
        )
    env = VecNormalize.load(vecnormalize, env)
    env.training = False
    env.norm_reward = False
    agent = PPO.load(checkpoint, env=None, device=env_cfg.sim.device)

    obs = env.reset()
    captured = None
    episodes_completed = 0
    for step in range(1, int(args_cli.max_steps) + 1):
        mode_before = _as_numpy(raw._mode).copy()
        observation_before = np.asarray(obs[0], dtype=np.float64).copy()
        raw_observation_before = np.asarray(
            env.unnormalize_obs(np.asarray(obs).copy())[0], dtype=np.float64
        )
        book_position_before = _as_numpy(raw._book_pos_env()[0]).astype(np.float64)
        tcp_position_before = _as_numpy(raw._ee_tool_pos_env()[0]).astype(np.float64)
        with torch.inference_mode():
            policy_action, _ = agent.predict(obs, deterministic=True)
            applied_action, adapter = _adapt_action(policy_action, environment_action_dim)
            obs, _, dones, _ = env.step(applied_action)
        mode_after = _as_numpy(raw._mode).copy()
        book_position_after = _as_numpy(raw._book_pos_env()[0]).astype(np.float64)
        tcp_position_after = _as_numpy(raw._ee_tool_pos_env()[0]).astype(np.float64)
        accepted = np.flatnonzero((mode_before == 0) & (mode_after == 1))
        if accepted.size:
            env_index = int(accepted[0])
            captured = _release_snapshot(
                raw,
                env_index,
                step,
                observation_before,
                np.asarray(policy_action[env_index], dtype=np.float64),
                np.asarray(applied_action[env_index], dtype=np.float64),
                adapter,
                checkpoint,
                vecnormalize,
            )
            break
        episodes_completed += int(np.count_nonzero(dones))
        if args_cli.progress_interval and (
            step == 1 or step % int(args_cli.progress_interval) == 0
        ):
            metrics = raw._compute_task_metrics()
            rear_to_mouth = float(_as_numpy(metrics["rear_to_mouth"])[0])
            front_to_back = float(_as_numpy(metrics["front_to_back"])[0])
            print(
                "[RELEASE_GEOMETRY_PROGRESS] "
                + json.dumps(
                    {
                        "step": step,
                        "mode": int(mode_after[0]),
                        "raw_observation_12d": raw_observation_before.tolist(),
                        "raw_policy_action": np.asarray(policy_action[0], dtype=float).tolist(),
                        "effective_environment_action": np.asarray(
                            applied_action[0], dtype=float
                        ).tolist(),
                        "action_adapter": adapter,
                        "book_delta_xyz_mm": (
                            1000.0 * (book_position_after - book_position_before)
                        ).tolist(),
                        "tcp_delta_xyz_mm": (
                            1000.0 * (tcp_position_after - tcp_position_before)
                        ).tolist(),
                        "rear_to_mouth_mm": 1000.0 * rear_to_mouth,
                        "front_to_back_mm": 1000.0 * front_to_back,
                        "episodes_completed": episodes_completed,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if captured is None:
        captured = {
            "schema_version": 1,
            "kind": "bookshelf_release_geometry_diagnostic",
            "task": args_cli.task,
            "task_source_root": str(task_source_root),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "seed": int(args_cli.seed),
            "release": {
                "accepted": False,
                "reason": "no INSERT-to-SCRIPTED transition before max_steps",
                "max_steps": int(args_cli.max_steps),
                "episodes_completed": episodes_completed,
            },
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(captured, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if captured["release"]["accepted"]:
        book = captured["book"]
        closest = captured["physical_gripper_to_shelf"]["closest_body_obstacle_pair"]
        print(
            "[RELEASE_GEOMETRY] "
            f"task={args_cli.task} "
            f"trailing_depth_mm={1000.0 * book['trailing_edge_depth_from_mouth_m']:.3f} "
            f"front_to_back_mm={1000.0 * book['front_to_back_remaining_m']:.3f} "
            f"closest_envelope={closest}"
        )
    else:
        print(f"[RELEASE_GEOMETRY] no accepted release: {captured['release']}")
    print(f"[RELEASE_GEOMETRY] output={output}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
