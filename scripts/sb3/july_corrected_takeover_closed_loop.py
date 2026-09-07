#!/usr/bin/env python3
"""Minimal closed-loop July-policy check from the corrected real takeover state."""

import argparse
import json
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BOOKSHELF_SRC = _REPO_ROOT / "source" / "bookshelf"
if str(_BOOKSHELF_SRC) not in sys.path:
    sys.path.insert(0, str(_BOOKSHELF_SRC))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", default="logs/july_corrected_takeover/result.json")
parser.add_argument("--cases", type=int, default=1, choices=(1, 10))
parser.add_argument("--perturbed", action="store_true")
parser.add_argument("--seed", type=int, default=20260904)
parser.add_argument(
    "--takeover-states-json",
    help="Optional audit JSON; run each internally consistent corrected raw takeover state.",
)
parser.add_argument("--video", action="store_true")
parser.add_argument("--video-length", type=int, default=900)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils import math as math_utils
from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
import bookshelf.tasks  # noqa: F401


TASK = "Bookshelf-Residual-Direct-v0"
TARGET_RAW = np.array([
    0.0,
    -0.186211,
    0.230009,
    0.000703,
    0.005741,
    -0.000746,
    -0.038644,
    -0.004081,
    0.003309,
    0.009838,
    0.000750,
    -0.000086,
], dtype=np.float64)


def _constant_schedule(value):
    return lambda _: float(value)


def _custom_objects(env, agent_cfg):
    learning_rate = agent_cfg.get("learning_rate")
    clip_range = agent_cfg.get("clip_range")
    return {
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "learning_rate": learning_rate,
        "lr_schedule": learning_rate if callable(learning_rate) else _constant_schedule(learning_rate),
        "clip_range": clip_range if callable(clip_range) else _constant_schedule(clip_range),
        "clip_range_vf": None,
    }


def _book_rotation(raw, device):
    """Construct a proper book rotation from yaw and the observed upright axis."""
    up = torch.tensor(
        [raw[10], raw[11], math.sqrt(max(0.0, 1.0 - raw[10] ** 2 - raw[11] ** 2))],
        device=device,
        dtype=torch.float32,
    )
    up = torch.nn.functional.normalize(up, dim=0)
    depth = torch.tensor(
        [math.cos(raw[5]), math.sin(raw[5]), 0.0], device=device, dtype=torch.float32
    )
    depth = torch.nn.functional.normalize(depth - torch.dot(depth, up) * up, dim=0)
    thickness = torch.linalg.cross(depth, up)
    return torch.stack((depth, up, thickness), dim=-1)


def _raw_observation(raw_env):
    metrics = raw_env._compute_task_metrics()
    tilt_x, tilt_y = raw_env._book_upright_tilt_obs()
    tool_to_book = raw_env._ee_tool_pos_env() - raw_env._book_pos_env()
    return torch.stack(
        (
            raw_env._mode.float(),
            metrics["rear_to_mouth"],
            metrics["front_to_back"],
            metrics["lat_err"],
            metrics["z_err"],
            metrics["yaw_err"],
            tool_to_book[:, 0],
            tool_to_book[:, 1],
            tool_to_book[:, 2],
            metrics["gripper_open"],
            tilt_x,
            tilt_y,
        ),
        dim=-1,
    )


def _policy_observation(raw_env):
    return raw_env._get_observations()["policy"].detach().cpu().numpy()


def _set_takeover_state(raw_env, requested_raw):
    """Place the robot and rigid book; no observation values are overwritten."""
    env_ids = torch.tensor([0], device=raw_env.device, dtype=torch.long)
    dtype = torch.float32
    rotation = _book_rotation(requested_raw, raw_env.device)
    book_quat = math_utils.quat_from_matrix(rotation.unsqueeze(0))[0]

    corners_l = raw_env._book_corners_local.to(device=raw_env.device, dtype=dtype)
    rotated = math_utils.quat_apply(
        book_quat.view(1, 4).expand(corners_l.shape[0], 4), corners_l
    )
    center = torch.zeros(3, device=raw_env.device, dtype=dtype)
    center[0] = (
        float(raw_env._geom_mouth_x)
        + float(requested_raw[1])
        - rotated[:, 0].amin()
    )
    center[1] = raw_env._slot_center_y()[0] - float(requested_raw[3])
    z_target = (
        float(raw_env.cfg.shelf_top_z)
        + float(raw_env.cfg.shelf_thickness)
        + 0.5 * float(raw_env.cfg.book_size[1])
    )
    center[2] = z_target + float(requested_raw[4])
    tool_pos = center + torch.tensor(requested_raw[6:9], device=raw_env.device, dtype=dtype)

    # Preserve the reset's valid grasp orientation relation, while imposing the
    # requested book orientation and exact tool position through the existing IK.
    current_tool_quat = raw_env._ee_pose_in_base()[1][0]
    current_book_quat = raw_env.book.data.root_link_quat_w[0]
    book_rel_tool = math_utils.quat_mul(math_utils.quat_inv(current_tool_quat), current_book_quat)
    tool_quat = math_utils.quat_mul(book_quat, math_utils.quat_inv(book_rel_tool))

    target_pos = raw_env._ee_tool_pos_env().detach().clone()
    target_quat = raw_env._ee_pose_in_base()[1].detach().clone()
    target_pos[0] = tool_pos
    target_quat[0] = tool_quat
    for _ in range(max(1, int(raw_env.cfg.reset_tool_ik_iters))):
        desired = raw_env._compute_ik_joint_targets_from_tool_quat(target_pos, target_quat)
        joint_pos = raw_env.robot.data.joint_pos[env_ids].clone()
        joint_vel = raw_env.robot.data.joint_vel[env_ids].clone()
        limits = raw_env.robot.data.soft_joint_pos_limits[env_ids][:, raw_env._arm_joint_ids]
        arm = torch.maximum(torch.minimum(desired[env_ids], limits[..., 1]), limits[..., 0])
        joint_pos[:, raw_env._arm_joint_ids] = arm
        joint_vel[:, raw_env._arm_joint_ids] = 0.0
        if len(raw_env._gripper_command_joint_ids) > 0:
            closed = float(raw_env.cfg.gripper_closed_joint_pos)
            opened = float(raw_env.cfg.gripper_open_joint_pos)
            finger = closed + float(requested_raw[9]) * (opened - closed)
            joint_pos[:, raw_env._gripper_command_joint_ids] = finger
            joint_vel[:, raw_env._gripper_command_joint_ids] = 0.0
        raw_env.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        raw_env.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        raw_env.scene.write_data_to_sim()
        raw_env.sim.forward()
        raw_env.scene.update(dt=0.0)

    book_state = raw_env.book.data.root_state_w[env_ids].clone()
    book_state[:, 0:3] = center + raw_env.scene.env_origins[env_ids]
    book_state[:, 3:7] = book_quat
    book_state[:, 7:] = 0.0
    raw_env.book.write_root_state_to_sim(book_state, env_ids=env_ids)
    raw_env.scene.write_data_to_sim()
    raw_env.sim.forward()
    raw_env.scene.update(dt=0.0)

    raw_env._arm_hold_joint_pos[env_ids] = raw_env.robot.data.joint_pos[env_ids][
        :, raw_env._arm_joint_ids
    ].clone()
    raw_env._target_pos_env[env_ids] = raw_env._ee_tool_pos_env()[env_ids]
    _, actual_quat = raw_env._ee_pose_in_base()
    _, _, yaw = math_utils.euler_xyz_from_quat(actual_quat[env_ids])
    raw_env._target_yaw[env_ids] = yaw
    raw_env._debug_position_only_ee_quat_b[env_ids] = actual_quat[env_ids]
    raw_env._debug_integrated_target_pos_env[env_ids] = raw_env._target_pos_env[env_ids]
    raw_env._capture_fixed_tool_to_book_transform(env_ids)
    raw_env._refresh_state_after_reset_acceptance(env_ids)
    raw_env.scene.write_data_to_sim()
    raw_env.sim.forward()
    raw_env.scene.update(dt=0.0)
    return _raw_observation(raw_env)[0].detach().cpu().numpy()


def _case_targets(count, perturbed, seed):
    if args_cli.takeover_states_json:
        document = json.loads(
            Path(args_cli.takeover_states_json).expanduser().read_text(encoding="utf-8")
        )
        states = document.get("states", [])
        if not states:
            raise ValueError("takeover-states JSON contains no states")
        return [
            (
                str(state.get("case_name", state.get("run", f"state_{index}"))),
                np.asarray(state["raw_observation"], dtype=np.float64),
                {key: value for key, value in state.items() if key != "raw_observation"},
            )
            for index, state in enumerate(states)
        ]
    if not perturbed:
        return [("corrected_real_reference", TARGET_RAW.copy(), {})]
    rng = np.random.default_rng(seed)
    targets = []
    for _ in range(count):
        value = TARGET_RAW.copy()
        value[1] += rng.uniform(-0.002, 0.002)
        value[3] += rng.uniform(-0.002, 0.002)
        value[4] += rng.uniform(-0.002, 0.002)
        value[5] += rng.uniform(-math.radians(1.0), math.radians(1.0))
        targets.append((f"perturbed_{len(targets)}", value, {}))
    return targets


def _info_scalar(infos, key, default=None):
    if isinstance(infos, dict):
        metrics = infos.get("episode_metrics", {})
        value = metrics.get(key, infos.get(f"episode_metric_{key}", default))
    else:
        value = infos[0].get("episode_metrics", {}).get(key, default)
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "__len__") and not isinstance(value, (str, bytes, dict)):
        value = value[0]
    return value.item() if hasattr(value, "item") else value


@hydra_task_config(TASK, "sb3_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = 1
    env_cfg.seed = int(args_cli.seed)
    env_cfg.enable_residual_reset_curriculum = False
    env_cfg.enable_residual_clearance_curriculum = False
    env_cfg.enable_residual_action_scale_curriculum = False
    env_cfg.enable_nominal_release_assist = False
    env_cfg.enable_policy_book_observation_bias = False
    env_cfg.enable_insert_gripper_observation_nuisance = False
    env_cfg.enable_insert_action_realization_dr = False
    env_cfg.reset_arm_joint_pos_noise = 0.0
    env_cfg.book_grasp_x_jitter = 0.0
    env_cfg.book_grasp_y_jitter = 0.0
    env_cfg.book_grasp_z_jitter = 0.0
    env_cfg.book_grasp_yaw_jitter = 0.0
    env_cfg.slot_lateral_clearance_min = 0.003
    env_cfg.slot_lateral_clearance_max = 0.003
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device
    env_cfg.log_dir = str(Path(args_cli.checkpoint).expanduser().resolve().parent)

    base_env = gym.make(TASK, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    raw_env = base_env.unwrapped
    if isinstance(raw_env, DirectMARLEnv):
        base_env = multi_agent_to_single_agent(base_env)
    if args_cli.video:
        video_dir = Path(args_cli.output).expanduser().resolve().parent / "video"
        base_env = gym.wrappers.RecordVideo(
            base_env,
            video_folder=str(video_dir),
            step_trigger=lambda step: step == 0,
            video_length=int(args_cli.video_length),
            disable_logger=True,
        )
    env = Sb3VecEnvWrapper(base_env, fast_variant=False)
    agent_cfg = process_sb3_cfg(agent_cfg, 1)

    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
    vec_path = checkpoint.with_name(
        checkpoint.name.replace("model", "model_vecnormalize", 1).removesuffix(".zip") + ".pkl"
    )
    env = VecNormalize.load(vec_path, env)
    env.training = False
    env.norm_reward = False
    agent = PPO.load(
        str(checkpoint.with_suffix("")), env=env,
        custom_objects=_custom_objects(env, agent_cfg),
    )

    targets = _case_targets(args_cli.cases, args_cli.perturbed, args_cli.seed)
    obs = env.reset()
    results = []
    for case_index, (case_name, target, case_metadata) in enumerate(targets):
        achieved = _set_takeover_state(raw_env, target)
        obs = env.normalize_obs(_policy_observation(raw_env))
        history = []
        reward_sum = 0.0
        release = None
        first_dx = None
        previous_rear = float(achieved[1])
        final_infos = None
        for step in range(int(raw_env.max_episode_length) + 5):
            before = _raw_observation(raw_env)[0].detach().cpu().numpy()
            action, _ = agent.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            reward_sum += float(rewards[0])
            after = _raw_observation(raw_env)[0].detach().cpu().numpy()
            rear_dx = float(after[1] - previous_rear)
            if first_dx is None:
                first_dx = rear_dx
            previous_rear = float(after[1])
            mode_before = int(round(float(before[0])))
            history.append({
                "step": step,
                "mode": mode_before,
                "rear_to_mouth": float(before[1]),
                "lateral": float(before[3]),
                "z": float(before[4]),
                "yaw": float(before[5]),
                "action": [float(v) for v in action[0]],
                "rear_progress": rear_dx,
            })
            if release is None and int(raw_env._release_step_buf[0].item()) >= 0:
                release = {
                    "step": step,
                    "rear_to_mouth": float(before[1]),
                    "front_to_back": float(before[2]),
                    "insertion_depth": float(raw_env.cfg.slot_x_back - raw_env._geom_mouth_x - before[2]),
                }
            if bool(dones[0]):
                final_infos = infos
                break
        success = bool(_info_scalar(final_infos, "success", reward_sum > 50.0))
        failure_code = _info_scalar(final_infos, "failure_code", None)
        insert_rows = [row for row in history if row["mode"] == 0]
        push_rows = [row for row in history if row["mode"] == 2]
        release_rear = release["rear_to_mouth"] if release else None
        result = {
            "case": case_index,
            "case_name": case_name,
            "case_metadata": case_metadata,
            "target_raw": target.tolist(),
            "achieved_raw": achieved.tolist(),
            "difference": (achieved - target).tolist(),
            "success": success,
            "failure_code": failure_code,
            "episode_reward": reward_sum,
            "steps": len(history),
            "first_rear_progress_m": first_dx,
            "progressed_forward": bool(insert_rows and max(r["rear_to_mouth"] for r in insert_rows) > achieved[1] + 0.01),
            "entered_slot": bool(insert_rows and max(r["rear_to_mouth"] for r in insert_rows) > -0.078),
            "released": release is not None,
            "never_released": release is None,
            "premature_release": bool(release_rear is not None and release_rear < -0.078),
            "correct_release": bool(release_rear is not None and release_rear >= -0.078),
            "release": release,
            "push_entered": bool(push_rows),
            "post_push_success": bool(success and push_rows),
            "maximum_initial_retreat_m": float(
                max(0.0, float(achieved[1]) - min(r["rear_to_mouth"] for r in insert_rows))
            ) if insert_rows else None,
            "insert_ranges": {
                key: [min(r[key] for r in insert_rows), max(r[key] for r in insert_rows), insert_rows[-1][key]]
                for key in ("rear_to_mouth", "lateral", "z", "yaw")
            } if insert_rows else {},
            "history": history,
        }
        results.append(result)
        print(
            f"[CASE {case_index}] success={success} first_dx_mm={1000*first_dx:+.3f} "
            f"forward={result['progressed_forward']} entered={result['entered_slot']} "
            f"released={result['released']} push={result['push_entered']} "
            f"initial_max_abs_error={np.max(np.abs(achieved-target)):.6g}",
            flush=True,
        )
    output = Path(args_cli.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"target": TARGET_RAW.tolist(), "results": results}, indent=2) + "\n")
    print(f"[RESULT] {output}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
