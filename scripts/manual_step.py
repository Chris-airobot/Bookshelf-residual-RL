#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manually step an Isaac Lab environment and inspect obs/reward/dones.

Usage example:
  python scripts/manual_step.py --task=Template-Bookshelf-Direct-v0 --num_envs 1

Then type actions like (v4: dx dy dz dyaw release_trigger in [-1,1]):
  0.5 0.0 0.0 0.0 -1.0
  0.0 -0.2 0.05 0.0 -0.5
release_trigger <= 0 means "do not release"; release_trigger > 0.5 requests release.
Idle steps keep release_trigger at -1.0 unless you explicitly type a different value.
Press Enter to repeat the last action, or type 'q' to quit.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Manual stepping/debug script for Isaac Lab environments.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument(
    "--record_demo",
    type=str,
    default=None,
    help="Optional path to save a manual demonstration .pt file for BC data collection.",
)
parser.add_argument(
    "--record_idle",
    action="store_true",
    default=False,
    help="Record idle/wait steps too. By default only steps caused by an entered command are recorded.",
)
parser.add_argument(
    "--record_scripted",
    action="store_true",
    default=False,
    help="Record SCRIPTED mode samples too. By default they are skipped because policy actions are ignored there.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# Default runtime settings for this manual script.
DEFAULT_NUM_ENVS = 1
DEFAULT_DISABLE_FABRIC = False
DEFAULT_STEPS = 10_000_000
DEFAULT_EPISODE_LENGTH_S = 1_000_000.0
DEFAULT_PRINT_EVERY = 0
DEFAULT_PRINT_ON_COMMAND = True
DEFAULT_RESET_ON_TERMINATED = True
DEFAULT_RATE_HZ = 30.0
DEFAULT_ACTION_GAIN = 1.0
DEFAULT_CONTINUOUS_ACTION = False
DEFAULT_SYNC_USD_XFORM = False
DEFAULT_DEBUG_TERMINATE = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

# Ensure local extension package imports work even when launched outside repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_BOOKSHELF_SRC = _REPO_ROOT / "source" / "bookshelf"
if str(_BOOKSHELF_SRC) not in sys.path:
    sys.path.insert(0, str(_BOOKSHELF_SRC))

import bookshelf.tasks  # noqa: F401

from isaaclab.sim.utils.stage import get_current_stage
from pxr import Gf, UsdGeom


def _format_tensor(x: torch.Tensor, max_elems: int = 8, decimals: int = 6) -> str:
    x = x.detach().cpu().flatten()
    if x.numel() <= max_elems:
        fmt = "{:." + str(decimals) + "f}"
        return "[" + ", ".join(fmt.format(v) for v in x.tolist()) + "]"
    fmt = "{:." + str(decimals) + "f}"
    head = ", ".join(fmt.format(v) for v in x[:max_elems].tolist())
    return "[" + head + ", ...]"


def _policy_obs_tensor(obs) -> torch.Tensor | None:
    if isinstance(obs, dict):
        obs = obs.get("policy", None)
    if obs is None:
        return None
    return torch.as_tensor(obs)


def _resolve_demo_path(path_arg: str) -> Path:
    path = Path(path_arg).expanduser()
    if path.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    path.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path / f"manual_demo_{stamp}.pt"


def _increment_demo_path(path: Path) -> Path:
    """Increment the trailing number in a filename: demo_002.pt -> demo_003.pt."""
    import re
    stem, suffix, parent = path.stem, path.suffix, path.parent
    m = re.search(r"(\d+)$", stem)
    if m:
        new_stem = stem[: m.start(1)] + str(int(m.group(1)) + 1).zfill(len(m.group(1)))
    else:
        new_stem = stem + "_2"
    return parent / (new_stem + suffix)


def _print_success_status(env) -> None:
    """Print each success sub-condition for env 0 without stepping."""
    uw = env.unwrapped
    if not hasattr(uw, "_compute_task_metrics"):
        print("[status] env does not expose _compute_task_metrics; cannot print.")
        return
    with torch.no_grad():
        m = uw._compute_task_metrics()

    cfg = uw.cfg
    import math as _math

    mode = int(uw._mode[0].item())
    mode_name = {0: "INSERT", 1: "SCRIPTED", 2: "PUSH"}.get(mode, str(mode))
    in_push = mode == 2

    rear   = float(m["rear_to_mouth"][0].item())
    front  = float(m["front_to_back"][0].item())
    lat    = float(m["lat_extent"][0].item())
    z_err  = float(m["z_err"][0].item())
    yaw_e  = float(_math.degrees(abs(float(m["yaw_err"][0].item()))))

    rear_ok  = float(cfg.success_rear_to_mouth_min) <= rear <= float(cfg.success_rear_to_mouth_max)
    front_eps = float(cfg.success_front_clear_eps_m)
    front_ok = (float(cfg.success_front_clear_min) - front_eps) <= front <= (float(cfg.success_front_clear_max) + front_eps)

    curr_clearance = uw._current_slot_lateral_clearance()
    inner_half = 0.5 * (uw._neighbor_thick_y + curr_clearance)
    lat_limit = inner_half - float(cfg.success_lateral_margin)
    lat_eps   = float(cfg.success_lateral_extent_eps_m)
    lat_ok    = lat <= (lat_limit + lat_eps)

    z_ok   = abs(z_err) < float(cfg.success_z_thresh)
    yaw_ok = yaw_e < _math.degrees(float(cfg.success_yaw_thresh))
    upright = bool(uw._upright_ok()[0].item())
    ready  = bool((uw.episode_length_buf[0] > int(cfg.min_steps_before_success)).item())

    def tick(ok): return "OK" if ok else "FAIL"

    print(f"[status] mode={mode_name} (need PUSH)  {'OK' if in_push else 'FAIL'}")
    print(f"  rear_to_mouth  = {rear:+.4f}  range=[{cfg.success_rear_to_mouth_min:.4f}, {cfg.success_rear_to_mouth_max:.4f}]  {tick(rear_ok)}")
    print(f"  front_to_back  = {front:+.4f}  range=[{float(cfg.success_front_clear_min):.4f}, {float(cfg.success_front_clear_max):.4f}] (+/-eps)  {tick(front_ok)}")
    print(f"  lat_extent     = {lat:.4f}   limit={lat_limit + lat_eps:.4f}  {tick(lat_ok)}")
    print(f"  z_err          = {z_err:+.4f}  thresh={cfg.success_z_thresh:.4f}  {tick(z_ok)}")
    print(f"  yaw_err        = {yaw_e:.2f}deg   thresh={_math.degrees(float(cfg.success_yaw_thresh)):.1f}deg  {tick(yaw_ok)}")
    print(f"  upright        = {tick(upright)}")
    print(f"  ready          = {tick(ready)}")
    all_ok = in_push and rear_ok and front_ok and lat_ok and z_ok and yaw_ok and upright and ready
    print(f"  --> SUCCESS GATE: {'OPEN (counting)' if all_ok else 'CLOSED'}")


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=DEFAULT_NUM_ENVS,
        use_fabric=not DEFAULT_DISABLE_FABRIC,
    )
    # Important: Isaac Lab envs typically auto-reset internally when an episode is done (including timeouts).
    # For manual debugging, we often want to avoid timeouts entirely so the pose doesn't get reset unexpectedly.
    if DEFAULT_EPISODE_LENGTH_S is not None:
        env_cfg.episode_length_s = float(DEFAULT_EPISODE_LENGTH_S)
    if DEFAULT_DEBUG_TERMINATE and hasattr(env_cfg, "debug_print_terminate_reason"):
        env_cfg.debug_print_terminate_reason = True

    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO] observation_space: {env.observation_space}")
    print(f"[INFO] action_space: {env.action_space}")
    print("[INFO] Enter actions as space-separated floats (e.g. '0.1 -0.2').")
    print("[INFO] This script keeps stepping while waiting for input.")
    print("[INFO] Press Enter to repeat last action. Type 'q' to quit.")
    record_path = _resolve_demo_path(args_cli.record_demo) if args_cli.record_demo is not None else None
    episode_steps: list[dict[str, torch.Tensor | int | bool]] = []
    if record_path is not None:
        print(f"[INFO] Recording demos. First file: {record_path}")
        print("[INFO] Each successful episode is saved immediately and the filename auto-increments.")
        if not args_cli.record_idle:
            print("[INFO] Recording only command steps. Use --record_idle to include idle/wait steps.")
        if not args_cli.record_scripted:
            print("[INFO] Skipping SCRIPTED mode samples. Use --record_scripted to include them.")

    obs, info = env.reset()
    action_dim = int(env.action_space.shape[-1])
    num_envs = int(env.unwrapped.num_envs)
    zero_action = torch.zeros((num_envs, action_dim), device=env.unwrapped.device, dtype=torch.float32)
    last_nudge = zero_action.clone()
    # Bookshelf v4: action[4] is release_trigger; idle should default to "do not release".
    last_release_trigger = -1.0
    release_trigger_dim = action_dim >= 5
    if release_trigger_dim:
        last_nudge[..., -1] = last_release_trigger

    # Named action aliases (5-dim: dx dy dz dyaw release_trigger)
    ACTION_ALIASES: dict[str, list[float]] = {
        "u":      [ 0.0,  0.0,  1.0,  0.0, -1.0],
        "d":    [ 0.0,  0.0, -1.0,  0.0, -1.0],
        "b": [-1.0,  0.0,  0.0,  0.0, -1.0],
        "f":    [ 1.0,  0.0,  0.0,  0.0, -1.0],
        "r":   [ 0.0,  1.0,  0.0,  0.0, -1.0],
        "l":    [ 0.0, -1.0,  0.0,  0.0, -1.0],
        "o":    [ 0.0,  0.0,  0.0,  0.0,  1.0],
        "c":   [ 0.0,  0.0,  0.0,  0.0, -1.0],
        # "c":       [ 0.0,  0.0,  0.0,  0.0, -1.0],
    }
    print("[INFO] Named aliases: up, down, forward, back, right, left, open, close")
    print("[INFO] Type 's' or 'status' to print success sub-conditions.")

    # Optional USD Xform sync (so gizmo matches the moving book when Fabric is enabled).
    # This is purely for visualization/debug and does not change training.
    book_xformable = None
    translate_op = None
    orient_op = None
    if DEFAULT_SYNC_USD_XFORM:
        stage = get_current_stage()
        book_prim = stage.GetPrimAtPath("/World/envs/env_0/Book")
        if not book_prim.IsValid():
            print("[WARN] Could not find /World/envs/env_0/Book for USD sync.")
        else:
            book_xformable = UsdGeom.Xformable(book_prim)
            # Reuse existing ops if present to avoid precision/type conflicts (quatf vs quatd).
            # If missing, create translate (float) and orient (double) ops.
            ordered_ops = book_xformable.GetOrderedXformOps()
            for op in ordered_ops:
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate and translate_op is None:
                    translate_op = op
                if op.GetOpType() == UsdGeom.XformOp.TypeOrient and orient_op is None:
                    orient_op = op
            if translate_op is None:
                translate_op = book_xformable.AddTranslateOp()
            if orient_op is None:
                orient_op = book_xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)

    # background input reader (so simulation keeps stepping/rendering)
    cmd_queue: "queue.Queue[str]" = queue.Queue()
    stop_event = threading.Event()

    def _input_thread():
        while not stop_event.is_set():
            try:
                line = input("action> ")
            except EOFError:
                stop_event.set()
                break
            cmd_queue.put(line)

    t = threading.Thread(target=_input_thread, daemon=True)
    t.start()

    step_period_s = 1.0 / max(DEFAULT_RATE_HZ, 1.0)

    last_print_step = -1

    for step in range(DEFAULT_STEPS):
        # non-blocking read of latest command (if any)
        got_command = False
        nudge_this_step = False
        release_trigger_one_shot = False
        while True:
            try:
                line = cmd_queue.get_nowait()
            except queue.Empty:
                break

            line = line.strip()
            if line.lower() in {"q", "quit", "exit"}:
                stop_event.set()
                break
            if line == "":
                # repeat last nudge for a single step
                got_command = True
                nudge_this_step = True
                continue

            # Status query: print success sub-conditions without stepping.
            if line.lower() in {"s", "status"}:
                _print_success_status(env)
                continue

            # Expand named aliases before numeric parsing.
            alias_vals = ACTION_ALIASES.get(line.lower())
            if alias_vals is not None:
                if len(alias_vals) != action_dim:
                    print(f"[WARN] Alias '{line}' has {len(alias_vals)} dims but action_dim={action_dim}. Skipping.")
                    continue
                parts = [str(v) for v in alias_vals]
            else:
                parts = line.split()

            if len(parts) != action_dim:
                print(f"[WARN] Expected {action_dim} floats, got {len(parts)}. Try again.")
                continue
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                print("[WARN] Could not parse floats. Try again.")
                continue
            action = torch.tensor(vals, device=env.unwrapped.device, dtype=torch.float32).unsqueeze(0)
            action = action * float(DEFAULT_ACTION_GAIN)
            last_nudge = torch.clamp(action, -1.0, 1.0)
            if release_trigger_dim:
                last_release_trigger = float(last_nudge[0, -1].item())
                # Keep release as one-shot for cleaner logged demos.
                if last_release_trigger > 0.5:
                    release_trigger_one_shot = True
            got_command = True
            nudge_this_step = True

        if stop_event.is_set():
            break

        idle_action = zero_action.clone()
        if release_trigger_dim:
            idle_action[..., -1] = last_release_trigger

        # Default: single-step nudge then idle. With continuous mode, keep last_nudge until a new line is typed.
        if DEFAULT_CONTINUOUS_ACTION:
            if nudge_this_step:
                action = last_nudge
            else:
                action = last_nudge if last_nudge.abs().any() else idle_action
        else:
            if nudge_this_step:
                action = last_nudge
            else:
                action = idle_action

        obs_before = obs
        obs_before_policy = _policy_obs_tensor(obs_before)

        # Use no_grad instead of inference_mode.
        # inference_mode can mark internal Isaac Lab buffers as "inference tensors", and then env.reset()
        # may fail when it tries to update them outside inference_mode.
        with torch.no_grad():
            obs, reward, terminated, truncated, info = env.step(action)

        if record_path is not None and obs_before_policy is not None:
            mode0 = float(obs_before_policy[0, 0].detach().cpu().item())
            should_record_step = bool(args_cli.record_idle or got_command)
            if not args_cli.record_scripted and abs(mode0 - 0.5) < 1.0e-6:
                should_record_step = False

            if should_record_step:
                obs_after_policy = _policy_obs_tensor(obs)
                if obs_after_policy is not None:
                    episode_steps.append(
                        {
                            "step": int(step),
                            "commanded": bool(got_command),
                            "obs": obs_before_policy[0].detach().cpu().clone(),
                            "action": torch.as_tensor(action)[0].detach().cpu().clone(),
                            "next_obs": obs_after_policy[0].detach().cpu().clone(),
                            "reward": torch.as_tensor(reward)[0].detach().cpu().clone(),
                            "terminated": torch.as_tensor(terminated)[0].detach().cpu().clone(),
                            "truncated": torch.as_tensor(truncated)[0].detach().cpu().clone(),
                            "mode": torch.tensor(mode0, dtype=torch.float32),
                        }
                    )

        if release_trigger_one_shot and release_trigger_dim:
            last_release_trigger = -1.0
            last_nudge[..., -1] = last_release_trigger

        # keep USD Xform in sync with sim book pose for correct gizmo (Fabric-enabled debug)
        if book_xformable is not None:
            book = env.unwrapped.book
            pos_w = book.data.root_link_pos_w[0].detach().cpu()
            env_origin = env.unwrapped.scene.env_origins[0].detach().cpu()
            pos_local = pos_w - env_origin
            qwxyz = book.data.root_link_quat_w[0].detach().cpu()
            translate_op.Set(Gf.Vec3f(float(pos_local[0]), float(pos_local[1]), float(pos_local[2])))
            orient_op.Set(
                Gf.Quatd(
                    float(qwxyz[0].item()),
                    Gf.Vec3d(
                        float(qwxyz[1].item()),
                        float(qwxyz[2].item()),
                        float(qwxyz[3].item()),
                    ),
                )
            )

        should_print = False
        if DEFAULT_PRINT_ON_COMMAND and got_command:
            should_print = True
        if DEFAULT_PRINT_EVERY and DEFAULT_PRINT_EVERY > 0 and (step % DEFAULT_PRINT_EVERY == 0):
            should_print = True
        # avoid printing multiple times for the same step if queue had multiple lines
        if should_print and step == last_print_step:
            should_print = False

        if should_print:
            last_print_step = step
            # obs may be dict (policy) or plain tensor depending on wrappers
            if isinstance(obs, dict):
                obs_policy = obs.get("policy", None)
            else:
                obs_policy = obs

            print(f"step={step}")
            print(f"  action={_format_tensor(torch.as_tensor(action)[0], max_elems=8, decimals=3)}")
            if obs_policy is not None:
                print(f"  obs(policy)={_format_tensor(torch.as_tensor(obs_policy)[0], decimals=6)}")
            print(f"  reward={float(torch.as_tensor(reward)[0]):.6f}")
            print(f"  terminated={bool(torch.as_tensor(terminated)[0])} truncated={bool(torch.as_tensor(truncated)[0])}")
            # print current book pose (env_0) when available
            try:
                # Prefer the env's commanded buffer if present (stable across fabric/USD modes).
                if hasattr(env.unwrapped, "_book_pos_w"):
                    book_pos_w = env.unwrapped._book_pos_w[0].detach().cpu()
                    src = "cmd"
                else:
                    book_pos_w = env.unwrapped.book.data.root_pos_w[0].detach().cpu()
                    src = "sim"
                print(f"  book_pos_w[{src}]={_format_tensor(book_pos_w, decimals=6)}")
                env_origin = env.unwrapped.scene.env_origins[0].detach().cpu()
                print(f"  book_pos_env[{src}]={_format_tensor(book_pos_w - env_origin, decimals=6)}")
            except Exception:
                pass
            # episode info (if present)
            if isinstance(info, (list, tuple)) and len(info) > 0 and isinstance(info[0], dict):
                ep = info[0].get("episode", None)
                if isinstance(ep, dict):
                    keys = [k for k in ep.keys() if k not in {"r", "l"}]
                    extra = {k: ep[k] for k in keys[:8]}
                    print(f"  episode: r={ep.get('r', None)} l={ep.get('l', None)} extras={extra}")

        term0 = bool(torch.as_tensor(terminated)[0])
        trunc0 = bool(torch.as_tensor(truncated)[0])
        # Manual stepping behavior: ignore timeouts/truncations so you can keep inspecting.
        # Only treat terminated=True as an episode end (typically success/failure condition).
        if term0 and DEFAULT_RESET_ON_TERMINATED:
            # Save episode if it was a success (reward >> 0 due to success_bonus=100).
            if record_path is not None and episode_steps:
                rew0 = float(torch.as_tensor(reward)[0])
                if rew0 > 50.0:
                    record_path.parent.mkdir(parents=True, exist_ok=True)
                    payload = {
                        "task": args_cli.task,
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                        "obs_dim": int(env.observation_space.shape[-1]),
                        "action_dim": int(env.action_space.shape[-1]),
                        "record_idle": bool(args_cli.record_idle),
                        "record_scripted": bool(args_cli.record_scripted),
                        "num_steps": len(episode_steps),
                        "steps": episode_steps,
                    }
                    torch.save(payload, record_path)
                    print(f"[INFO] Saved {len(episode_steps)} steps -> {record_path}")
                    record_path = _increment_demo_path(record_path)
                    print(f"[INFO] Next episode will save to: {record_path}")
                else:
                    print(f"[INFO] Episode failed (reward={rew0:.1f}), discarding {len(episode_steps)} steps.")
            episode_steps = []

            with torch.no_grad():
                obs, info = env.reset()
            # After reset, hold zero action until user types a new command.
            last_nudge = zero_action.clone()
            last_release_trigger = -1.0
            if release_trigger_dim:
                last_nudge[..., -1] = last_release_trigger
            print(f"[INFO] reset() because terminated=True (truncated={trunc0} ignored); action reset to zero")

        if not simulation_app.is_running():
            break

        time.sleep(step_period_s)


    env.close()
    stop_event.set()


if __name__ == "__main__":
    main()
    simulation_app.close()
