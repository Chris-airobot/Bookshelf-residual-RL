# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from Stable-Baselines3."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path

# Ensure the bookshelf package is importable regardless of install state.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_BOOKSHELF_SRC = _REPO_ROOT / "source" / "bookshelf"
if str(_BOOKSHELF_SRC) not in sys.path:
    sys.path.insert(0, str(_BOOKSHELF_SRC))

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from Stable-Baselines3.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="sb3_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--eval_slot_clearance",
    type=float,
    default=None,
    help="Evaluate at a fixed clearance with full randomization/residual authority and no release assistance.",
)
parser.add_argument(
    "--eval_old_reset_noise",
    action="store_true",
    default=False,
    help="Evaluate with the older larger reset/grasp noise: 3 deg arm, 8/6/3 mm grasp xyz, 8 deg yaw.",
)
parser.add_argument(
    "--eval_episodes",
    type=int,
    default=0,
    help="Stop after this many completed episodes; 0 runs until manually stopped.",
)
parser.add_argument(
    "--eval_output_dir",
    type=str,
    default=None,
    help="Scenario trace directory. Bookshelf residual/PPO evaluations create one automatically when omitted.",
)
parser.add_argument(
    "--eval_scenario_bank",
    type=str,
    default=None,
    help="Replay every scenario from a frozen evaluation bank exactly once.",
)
parser.add_argument(
    "--eval_nominal_only",
    action="store_true",
    default=False,
    help=(
        "Evaluate the geometric nominal controller with zero residual actions. "
        "This enables geometry-gated nominal release and does not load PPO."
    ),
)
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--keep_all_info",
    action="store_true",
    default=False,
    help="Use a slower SB3 wrapper but keep all the extra training info.",
)
parser.add_argument(
    "--print_residual_components",
    action="store_true",
    default=False,
    help="For residual envs, print nominal/residual/final action components during play.",
)
parser.add_argument(
    "--print_residual_interval",
    type=int,
    default=30,
    help="Print residual components every N sim steps when --print_residual_components is set.",
)
parser.add_argument(
    "--print_residual_env_index",
    type=int,
    default=0,
    help="Environment index to print when --print_residual_components is set.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import random
import time
import math
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict

from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_tasks.utils.parse_cfg import get_checkpoint_path

import bookshelf.tasks  # noqa: F401
from bookshelf.tasks.direct.bookshelf.frozen_scenario_bank import load_frozen_scenario_bank

from evaluation_scenarios import (
    EvaluationScenarioTrace,
    SCENARIO_FIELDS,
    SCENARIO_VECTOR_FIELDS,
    apply_evaluation_seed_after_agent_load,
    git_revision,
    sha256_file,
)


def _scalar(value, env_idx: int | None = None, default=None):
    if value is None:
        return default
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if env_idx is not None and hasattr(value, "__len__") and not isinstance(value, (str, bytes, dict)):
            value = value[env_idx]
        if hasattr(value, "item"):
            return value.item()
        return value
    except Exception:
        return default


def _episode_metric(infos, env_idx: int, key: str, default=None):
    if isinstance(infos, dict):
        metrics = infos.get("episode_metrics")
        if isinstance(metrics, dict) and key in metrics:
            return _scalar(metrics.get(key), env_idx, default)
        return _scalar(infos.get(f"episode_metric_{key}"), env_idx, default)
    if isinstance(infos, (list, tuple)) and env_idx < len(infos) and isinstance(infos[env_idx], dict):
        info = infos[env_idx]
        metrics = info.get("episode_metrics")
        if isinstance(metrics, dict) and key in metrics:
            return _scalar(metrics.get(key), None, default)
        return _scalar(info.get(f"episode_metric_{key}"), None, default)
    return default


def _episode_info_value(infos, env_idx: int, key: str, default=None):
    if not isinstance(infos, (list, tuple)) or env_idx >= len(infos):
        return default
    info = infos[env_idx]
    if not isinstance(info, dict):
        return default
    episode = info.get("episode")
    if not isinstance(episode, dict):
        return default
    return _scalar(episode.get(key), None, default)


def _episode_metric_vector(infos, env_idx: int, key: str) -> list | None:
    if not isinstance(infos, (list, tuple)) or env_idx >= len(infos):
        return None
    info = infos[env_idx]
    if not isinstance(info, dict):
        return None
    value = info.get(f"episode_metric_{key}")
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "tolist"):
            value = value.tolist()
        return list(value)
    except (TypeError, ValueError):
        return None


def _scenario_trace_row(infos, env_idx: int) -> dict:
    row = {
        "env_id": env_idx,
        "slot_center_y": _episode_metric(infos, env_idx, "slot_center_y"),
        "slot_clearance": _episode_metric(infos, env_idx, "slot_clearance"),
        "missing_book_index": _episode_metric(infos, env_idx, "missing_book_index"),
    }
    vector = _episode_metric_vector(infos, env_idx, "scenario_vector")
    if vector is not None and len(vector) == len(SCENARIO_VECTOR_FIELDS):
        row.update(zip(SCENARIO_VECTOR_FIELDS, vector))
    return row


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Play with stable-baselines agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")
    traced_tasks = {"Bookshelf-Residual-Direct-v0", "Bookshelf-PPO-Direct-v0"}
    frozen_bank = None
    if args_cli.eval_scenario_bank is not None:
        if task_name not in traced_tasks:
            raise ValueError(f"Frozen bookshelf scenarios are not supported for task {task_name!r}.")
        frozen_bank = load_frozen_scenario_bank(args_cli.eval_scenario_bank)
        bank_episode_count = int(frozen_bank["scenario_count"])
        if args_cli.eval_episodes not in (0, bank_episode_count):
            raise ValueError(
                "--eval_episodes must be omitted/zero or equal the frozen bank count: "
                f"{args_cli.eval_episodes} != {bank_episode_count}"
            )
        args_cli.eval_episodes = bank_episode_count
    trace_enabled = args_cli.eval_output_dir is not None or frozen_bank is not None or (
        args_cli.eval_episodes > 0 and task_name in traced_tasks
    )
    if trace_enabled and not args_cli.keep_all_info:
        args_cli.keep_all_info = True
        print("[INFO] Enabling full terminal metrics for scenario tracing.")
    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    if frozen_bank is not None:
        if env_cfg.scene.num_envs > int(frozen_bank["scenario_count"]):
            raise ValueError(
                "--num_envs cannot exceed the frozen bank scenario count: "
                f"{env_cfg.scene.num_envs} > {frozen_bank['scenario_count']}"
            )
        env_cfg.evaluation_scenario_bank = str(Path(args_cli.eval_scenario_bank).expanduser().resolve())
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["seed"]
    if hasattr(env_cfg, "enable_reset_acceptance_gate"):
        env_cfg.enable_reset_acceptance_gate = False
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if hasattr(env_cfg, "debug_print_residual_components"):
        env_cfg.debug_print_residual_components = bool(args_cli.print_residual_components)
        env_cfg.debug_print_residual_interval = int(args_cli.print_residual_interval)
        env_cfg.debug_print_residual_env_index = int(args_cli.print_residual_env_index)
    if args_cli.eval_slot_clearance is not None:
        if hasattr(env_cfg, "enable_residual_clearance_curriculum"):
            env_cfg.enable_residual_clearance_curriculum = False
        if hasattr(env_cfg, "enable_residual_reset_curriculum"):
            env_cfg.enable_residual_reset_curriculum = False
        if hasattr(env_cfg, "enable_residual_action_scale_curriculum"):
            env_cfg.enable_residual_action_scale_curriculum = False
        if hasattr(env_cfg, "enable_nominal_release_assist"):
            env_cfg.enable_nominal_release_assist = False
        env_cfg.slot_lateral_clearance_min = float(args_cli.eval_slot_clearance)
        env_cfg.slot_lateral_clearance_max = float(args_cli.eval_slot_clearance)
    if args_cli.eval_old_reset_noise:
        if hasattr(env_cfg, "enable_residual_reset_curriculum"):
            env_cfg.enable_residual_reset_curriculum = False
        env_cfg.reset_arm_joint_pos_noise = math.radians(3.0)
        env_cfg.book_grasp_x_jitter = 0.008
        env_cfg.book_grasp_y_jitter = 0.006
        env_cfg.book_grasp_z_jitter = 0.003
        env_cfg.book_grasp_yaw_jitter = math.radians(8.0)
    if args_cli.eval_nominal_only:
        if task_name != "Bookshelf-Residual-Direct-v0":
            raise ValueError("--eval_nominal_only requires Bookshelf-Residual-Direct-v0.")
        if not bool(getattr(env_cfg, "enable_nominal_controller", False)):
            raise ValueError("--eval_nominal_only requires the geometric nominal controller.")
        env_cfg.enable_nominal_release_assist = True
        env_cfg.nominal_release_assist_until_frac = 1.0

    # directory for logging into
    log_root_path = os.path.join("logs", "sb3", train_task_name)
    log_root_path = os.path.abspath(log_root_path)
    # checkpoint and log_dir stuff
    if args_cli.eval_nominal_only:
        if args_cli.checkpoint is not None:
            raise ValueError("Do not pass --checkpoint with --eval_nominal_only.")
        checkpoint_path = None
        log_dir = os.path.join(log_root_path, "nominal_only")
    elif args_cli.checkpoint is None:
        # FIXME: last checkpoint doesn't seem to really use the last one'
        if args_cli.use_last_checkpoint:
            checkpoint = "model_.*.zip"
        else:
            checkpoint = "model.zip"
        checkpoint_path = get_checkpoint_path(log_root_path, ".*", checkpoint, sort_alpha=False)
    else:
        checkpoint_path = args_cli.checkpoint
    if checkpoint_path is not None:
        log_dir = os.path.dirname(checkpoint_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg, env.unwrapped.num_envs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for stable baselines
    env = Sb3VecEnvWrapper(env, fast_variant=not args_cli.keep_all_info)

    vec_norm_path = None
    if checkpoint_path is not None:
        vec_norm_path = Path(
            checkpoint_path.replace("/model", "/model_vecnormalize").replace(".zip", ".pkl")
        )

    # normalize environment (if needed)
    if vec_norm_path is not None and vec_norm_path.exists():
        print(f"Loading saved normalization: {vec_norm_path}")
        env = VecNormalize.load(vec_norm_path, env)
        #  do not update them at test time
        env.training = False
        # reward normalization is not needed at test time
        env.norm_reward = False
    elif not args_cli.eval_nominal_only and "normalize_input" in agent_cfg:
        env = VecNormalize(
            env,
            training=True,
            norm_obs="normalize_input" in agent_cfg and agent_cfg.pop("normalize_input"),
            clip_obs="clip_obs" in agent_cfg and agent_cfg.pop("clip_obs"),
        )

    evaluation_seed = int(agent_cfg["seed"])
    agent = None
    checkpoint_agent_seed = None
    if args_cli.eval_nominal_only:
        print("[INFO] Evaluation policy: nominal controller only (zero residual actions).")
    else:
        print(f"Loading checkpoint from: {checkpoint_path}")
        agent = PPO.load(checkpoint_path, env, print_system_info=True)
        checkpoint_agent_seed = apply_evaluation_seed_after_agent_load(agent, evaluation_seed)
        print(
            "[INFO] Reapplied evaluation seed after PPO.load: "
            f"evaluation={evaluation_seed}, checkpoint={checkpoint_agent_seed}"
        )

    trace = None
    if trace_enabled:
        if args_cli.eval_output_dir is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            trace_output_dir = _REPO_ROOT / "logs" / "eval_scenarios" / (
                f"{timestamp}_{task_name}_seed{agent_cfg['seed']}"
            )
        else:
            trace_output_dir = Path(args_cli.eval_output_dir).expanduser().resolve()
        trace = EvaluationScenarioTrace(
            trace_output_dir,
            {
                "task": task_name,
                "evaluation_policy": "nominal_only" if args_cli.eval_nominal_only else "ppo",
                "seed": evaluation_seed,
                "checkpoint_agent_seed": checkpoint_agent_seed,
                "num_envs": int(env.num_envs),
                "requested_episodes": int(args_cli.eval_episodes),
                "checkpoint": str(Path(checkpoint_path).resolve()) if checkpoint_path is not None else None,
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "vecnormalize": (
                    str(vec_norm_path.resolve())
                    if vec_norm_path is not None and vec_norm_path.exists()
                    else None
                ),
                "vecnormalize_sha256": sha256_file(vec_norm_path),
                "git": git_revision(_REPO_ROOT),
                "evaluation": {
                    "fixed_slot_clearance": args_cli.eval_slot_clearance,
                    "old_reset_noise": bool(args_cli.eval_old_reset_noise),
                    "reset_arm_joint_pos_noise": float(
                        getattr(env_cfg, "reset_arm_joint_pos_noise", 0.0)
                    ),
                    "book_grasp_x_jitter": float(getattr(env_cfg, "book_grasp_x_jitter", 0.0)),
                    "book_grasp_y_jitter": float(getattr(env_cfg, "book_grasp_y_jitter", 0.0)),
                    "book_grasp_z_jitter": float(getattr(env_cfg, "book_grasp_z_jitter", 0.0)),
                    "book_grasp_yaw_jitter": float(getattr(env_cfg, "book_grasp_yaw_jitter", 0.0)),
                },
                "scenario_fields": SCENARIO_FIELDS,
                "scenario_vector_fields": SCENARIO_VECTOR_FIELDS,
                "frozen_scenario_bank": (
                    {
                        "path": frozen_bank["path"],
                        "scenario_count": frozen_bank["scenario_count"],
                        "scenario_sha256": frozen_bank["scenario_sha256"],
                    }
                    if frozen_bank is not None
                    else None
                ),
            },
        )
        print(f"[INFO] Evaluation scenario trace: {trace.output_dir}")

    dt = env.unwrapped.step_dt
    num_envs = env.num_envs

    # Episode tracking
    # success = terminated with reward > SUCCESS_THRESH (success_bonus=100, drop=-20)
    SUCCESS_REWARD_THRESH = 50.0
    ep_reward = [0.0] * num_envs
    n_success = 0
    n_timeout = 0
    n_drop = 0
    n_episodes = 0
    PRINT_EVERY = 10  # print rolling stats every N completed episodes

    # reset environment
    obs = env.reset()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            if args_cli.eval_nominal_only:
                actions = np.zeros((num_envs, *env.action_space.shape), dtype=np.float32)
            else:
                actions, _ = agent.predict(obs, deterministic=True)
            # env stepping
            obs, rewards, dones, infos = env.step(actions)

        for i in range(num_envs):
            ep_reward[i] += float(rewards[i])
            if dones[i]:
                trace_row = _scenario_trace_row(infos, i) if trace is not None else {}
                bank_index = trace_row.get("scenario_bank_index", -1)
                if frozen_bank is not None and (bank_index is None or int(bank_index) < 0):
                    ep_reward[i] = 0.0
                    continue
                if args_cli.eval_episodes > 0 and n_episodes >= args_cli.eval_episodes:
                    ep_reward[i] = 0.0
                    continue
                r = ep_reward[i]
                metric_success = _episode_metric(infos, i, "success", default=None)
                failure_code = _episode_metric(infos, i, "failure_code", default=None)
                outcome = "timeout"
                if metric_success is not None:
                    if bool(metric_success):
                        n_success += 1
                        outcome = "success"
                    elif int(failure_code or 0) == 3:
                        n_timeout += 1
                    else:
                        n_drop += 1
                        outcome = "drop"
                elif r > SUCCESS_REWARD_THRESH:
                    n_success += 1
                    outcome = "success"
                elif r < -10.0:
                    n_drop += 1
                    outcome = "drop"
                else:
                    n_timeout += 1
                n_episodes += 1
                if trace is not None:
                    trace_row.update(
                        {
                            "episode_index": n_episodes - 1,
                            "outcome": outcome,
                            "failure_code": failure_code,
                            "episode_reward": r,
                            "episode_length": _episode_info_value(infos, i, "l"),
                        }
                    )
                    trace.append(trace_row)
                ep_reward[i] = 0.0

                if n_episodes % PRINT_EVERY == 0:
                    total = n_success + n_timeout + n_drop
                    print(
                        f"  [{total:>4} eps]  "
                        f"success {n_success}/{total} ({100*n_success/max(total,1):.0f}%)  "
                        f"drop {n_drop}  timeout {n_timeout}"
                    )

        if args_cli.eval_episodes > 0 and n_episodes >= args_cli.eval_episodes:
            break

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # Final summary
    total = n_success + n_timeout + n_drop
    if total > 0:
        print("\n" + "=" * 55)
        print(f"  Episodes       : {total}")
        print(f"  Success        : {n_success} / {total}  ({100*n_success/total:.0f}%)")
        print(f"  Drop           : {n_drop} / {total}  ({100*n_drop/total:.0f}%)")
        print(f"  Timeout        : {n_timeout} / {total}  ({100*n_timeout/total:.0f}%)")
        print("=" * 55)

    if trace is not None:
        summary_path = trace.write()
        print(f"[INFO] Scenario trace summary: {summary_path}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
