#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manually step an Isaac Lab environment and inspect obs/reward/dones.

Usage example:
  python scripts/manual_step.py --task=Template-Bookshelf-Direct-v0 --num_envs 1

Then type actions like (v4: dx dy dz dyaw dgrip in [-1,1]; dgrip=-1 close, +1 open):
  0.5 0.0 0.0 0.0 -1.0
  0.0 -0.2 0.05 0.0 -0.5
Idle steps (no new line) repeat the last gripper command; they do not send g=0 (which would drift half-open).
After env reset, idle gripper defaults to closed (-1) until you type a new 5th value.
Press Enter to repeat the last action, or type 'q' to quit.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import queue
import threading
import time

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Manual stepping/debug script for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate (use 1 for manual).")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument(
    "--steps",
    type=int,
    default=10_000_000,
    help="Max steps to run (you can quit early by typing 'q').",
)
parser.add_argument(
    "--episode_length_s",
    type=float,
    default=1000000,
    help="Override env episode_length_s for this manual run (prevents timeouts/auto-resets).",
)
parser.add_argument(
    "--print_every",
    type=int,
    default=0,
    help="Print every N steps (0 disables periodic printing).",
)
parser.add_argument(
    "--print_on_command",
    action="store_true",
    default=True,
    help="Print a line immediately after you enter a command.",
)
parser.add_argument(
    "--reset_on_terminated",
    action="store_true",
    default=True,
    help="Reset automatically when terminated=True (timeouts/truncations are ignored in this script).",
)
parser.add_argument(
    "--rate_hz",
    type=float,
    default=30.0,
    help="Stepping rate while waiting for input (keeps viewport responsive).",
)
parser.add_argument(
    "--action_gain",
    type=float,
    default=1.0,
    help="Multiply typed actions by this gain (debug-only; still clamped to [-1, 1]).",
)
parser.add_argument(
    "--continuous_action",
    action="store_true",
    default=False,
    help="Keep applying the last typed action every step until you enter a new one (default: one-step nudge then zero).",
)
parser.add_argument(
    "--sync_usd_xform",
    action="store_true",
    default=False,
    help="Sync USD Xform of env_0 book to commanded pose each step (for correct gizmo with Fabric enabled).",
)
parser.add_argument(
    "--debug_terminate",
    action="store_true",
    default=True,
    help="Print which termination flag fired (Bookshelf v4: debug_print_terminate_reason on env cfg).",
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


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # Important: Isaac Lab envs typically auto-reset internally when an episode is done (including timeouts).
    # For manual debugging, we often want to avoid timeouts entirely so the pose doesn't get reset unexpectedly.
    if args_cli.episode_length_s is not None:
        env_cfg.episode_length_s = float(args_cli.episode_length_s)
    if args_cli.debug_terminate and hasattr(env_cfg, "debug_print_terminate_reason"):
        env_cfg.debug_print_terminate_reason = True

    # For manual "reset" testing we want deterministic starting poses:
    # - no arm joint noise
    # - no book grasp-frame jitter (position / yaw)
    for name in [
        "reset_arm_joint_pos_noise",
        "book_grasp_x_jitter",
        "book_grasp_y_jitter",
        "book_grasp_z_jitter",
        "book_grasp_yaw_jitter",
    ]:
        if hasattr(env_cfg, name):
            setattr(env_cfg, name, 0.0)
    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO] observation_space: {env.observation_space}")
    print(f"[INFO] action_space: {env.action_space}")
    print("[INFO] Enter actions as space-separated floats (e.g. '0.1 -0.2').")
    print("[INFO] This script keeps stepping while waiting for input.")
    print("[INFO] Press Enter to repeat last action. Type 'q' to quit.")

    obs, info = env.reset()
    zero_action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    last_action = zero_action.clone()
    last_nudge = zero_action.clone()
    # Bookshelf v4: idle steps must repeat the last gripper command to avoid accidental opens.
    last_gripper = -1.0
    if int(env.action_space.shape[-1]) in (5, 6):
        last_nudge[..., -1] = last_gripper

    # Optional USD Xform sync (so gizmo matches the moving book when Fabric is enabled).
    # This is purely for visualization/debug and does not change training.
    book_xformable = None
    translate_op = None
    orient_op = None
    if args_cli.sync_usd_xform:
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

    step_period_s = 1.0 / max(args_cli.rate_hz, 1.0)

    last_print_step = -1

    for step in range(args_cli.steps):
        # non-blocking read of latest command (if any)
        got_command = False
        nudge_this_step = False
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

            parts = line.split()
            if len(parts) != int(env.action_space.shape[-1]):
                print(f"[WARN] Expected {env.action_space.shape[-1]} floats, got {len(parts)}. Try again.")
                continue
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                print("[WARN] Could not parse floats. Try again.")
                continue
            action = torch.tensor(vals, device=env.unwrapped.device, dtype=torch.float32).unsqueeze(0)
            action = action * float(args_cli.action_gain)
            last_nudge = torch.clamp(action, -1.0, 1.0)
            if int(env.action_space.shape[-1]) in (5, 6):
                last_gripper = float(last_nudge[0, -1].item())
            got_command = True
            nudge_this_step = True

        if stop_event.is_set():
            break

        idle_action = zero_action.clone()
        if int(env.action_space.shape[-1]) in (5, 6):
            idle_action[..., -1] = last_gripper

        # Default: single-step nudge then idle. With --continuous_action, keep last_nudge until a new line is typed.
        if args_cli.continuous_action:
            if nudge_this_step:
                action = last_nudge
            else:
                action = last_nudge if last_nudge.abs().any() else idle_action
        else:
            if nudge_this_step:
                action = last_nudge
            else:
                action = idle_action

        # Use no_grad instead of inference_mode.
        # inference_mode can mark internal Isaac Lab buffers as "inference tensors", and then env.reset()
        # may fail when it tries to update them outside inference_mode.
        with torch.no_grad():
            obs, reward, terminated, truncated, info = env.step(action)

        if not args_cli.continuous_action and nudge_this_step:
            last_action = zero_action.clone()

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
        if args_cli.print_on_command and got_command:
            should_print = True
        if args_cli.print_every and args_cli.print_every > 0 and (step % args_cli.print_every == 0):
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
        if term0 and args_cli.reset_on_terminated:
            with torch.no_grad():
                obs, info = env.reset()
            # After reset, hold zero action until user types a new command.
            last_action = zero_action.clone()
            last_nudge = zero_action.clone()
            last_gripper = -1.0
            if int(env.action_space.shape[-1]) in (5, 6):
                last_nudge[..., -1] = last_gripper
            print(f"[INFO] reset() because terminated=True (truncated={trunc0} ignored); action reset to zero")

        if not simulation_app.is_running():
            break

        time.sleep(step_period_s)

    env.close()
    stop_event.set()


if __name__ == "__main__":
    main()
    simulation_app.close()

