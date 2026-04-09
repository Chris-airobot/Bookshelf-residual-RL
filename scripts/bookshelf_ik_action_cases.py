#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Run scripted action sequences on Bookshelf (or compatible) envs for IK sanity checks.

Cases (v4 action = [dx, dy, dz, dyaw, dgrip] in [-1, 1]):
  hold   — zeros (arm should stay in hold-IK mode if actions are exactly zero)
  xp, xm — +x / -x
  yp, ym — +y / -y
  zp, zm — +z / -z
  yawp, yawm — +dyaw / -dyaw
  gripc, gripo — close/open gripper
  all    — run hold, then each axis pair in order

Example:
  ./isaaclab.sh -p scripts/bookshelf_ik_action_cases.py \\
      --task Bookshelf-Direct-v4 --num_envs 1 --case all --hold_steps 60 --pulse_steps 120

Launch Isaac Sim via AppLauncher (same as manual_step.py).
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Scripted IK / action-axis tests for Bookshelf direct envs.")
parser.add_argument("--task", type=str, default="Bookshelf-Direct-v4", help="Gym task id.")
parser.add_argument("--num_envs", type=int, default=1, help="Use 1 to read env_0 metrics clearly.")
parser.add_argument(
    "--case",
    type=str,
    default="all",
    choices=("hold", "xp", "xm", "yp", "ym", "zp", "zm", "yawp", "yawm", "gripc", "gripo", "all"),
    help="Which scripted motion to run.",
)
parser.add_argument("--hold_steps", type=int, default=40, help="Steps for zero-action hold segment.")
parser.add_argument("--pulse_steps", type=int, default=80, help="Steps per non-zero action segment.")
parser.add_argument("--action_mag", type=float, default=0.5, help="Magnitude in [-1, 1] for pulse axes.")
parser.add_argument(
    "--episode_length_s",
    type=float,
    default=1.0e6,
    help="Avoid timeout resets during the sweep.",
)
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable Fabric (USD I/O).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import bookshelf.tasks  # noqa: F401


def _obs_tracking(env_unwrapped, eid: int = 0) -> tuple[float, float, float, float]:
    """Return raw (ex, ey, ez, eyaw) before obs scaling — v4 policy obs layout."""
    o = env_unwrapped._get_observations()["policy"][eid]
    scale = float(env_unwrapped.cfg.tracking_error_obs_scale)
    ex_s, ey_s, ez_s, eyaw_s = float(o[3].item()), float(o[4].item()), float(o[5].item()), float(o[6].item())
    return ex_s * scale, ey_s * scale, ez_s * scale, eyaw_s * scale


def _book_metrics(env_unwrapped, eid: int = 0) -> tuple[float, float, float]:
    """Return (ins, lat_err, book_x) for env frame.

    ins = book_x - slot_x_open (positive means deeper into the shelf).
    lat_err = |book_y - slot_center_y|.
    """
    book_pos_env = env_unwrapped.book.data.root_link_pos_w[eid] - env_unwrapped.scene.env_origins[eid]
    book_x = float(book_pos_env[0].item())
    ins = book_x - float(env_unwrapped.cfg.slot_x_open)
    lat_err = float(torch.abs(book_pos_env[1] - float(env_unwrapped.cfg.slot_center_y)).item())
    return float(ins), float(lat_err), book_x


def _ee_metrics(env_unwrapped, eid: int = 0) -> tuple[float, float, float, float]:
    """Return (ee_x, ee_y, target_x, target_y) in env frame."""
    ee_pos_w = env_unwrapped.robot.data.body_pos_w[eid, env_unwrapped._ee_body_idx]
    ee_pos_env = ee_pos_w - env_unwrapped.scene.env_origins[eid]
    target_pos_env = env_unwrapped._target_pos_env[eid]
    return (
        float(ee_pos_env[0].item()),
        float(ee_pos_env[1].item()),
        float(target_pos_env[0].item()),
        float(target_pos_env[1].item()),
    )


def _book_contact_norm(env_unwrapped, eid: int = 0) -> float:
    nf = env_unwrapped.book_contact.data.net_forces_w
    if nf is None:
        return 0.0
    # env_v4 helper uses index 0 of the ContactSensor's internal bodies list.
    return float(torch.linalg.norm(nf[eid, 0]).item())


def _segment(
    env,
    env_u,
    action: torch.Tensor,
    n: int,
    label: str,
) -> None:
    print(f"\n--- {label} ({n} steps) action={action[0].tolist()} ---", flush=True)
    ex0, ey0, ez0, eyaw0 = _obs_tracking(env_u, 0)
    ins0, lat0, book_x0 = _book_metrics(env_u, 0)
    ee_x0, ee_y0, tgt_x0, tgt_y0 = _ee_metrics(env_u, 0)
    contact0 = _book_contact_norm(env_u, 0)
    print(
        "  start:",
        f"ex,ey,ez,eyaw=({ex0:.5f},{ey0:.5f},{ez0:.5f},{eyaw0:.5f})",
        f"ins={ins0:.5f} lat_err={lat0:.5f} book_x={book_x0:.5f}",
        f"ee_xy=({ee_x0:.5f},{ee_y0:.5f}) tgt_xy=({tgt_x0:.5f},{tgt_y0:.5f})",
        f"book_contact_norm={contact0:.2f}",
        flush=True,
    )
    act = action.expand(env_u.num_envs, action.shape[-1])
    for _ in range(n):
        with torch.no_grad():
            env.step(act)
    ex1, ey1, ez1, eyaw1 = _obs_tracking(env_u, 0)
    ins1, lat1, book_x1 = _book_metrics(env_u, 0)
    ee_x1, ee_y1, tgt_x1, tgt_y1 = _ee_metrics(env_u, 0)
    contact1 = _book_contact_norm(env_u, 0)
    print(
        "  end:  ",
        f"ex,ey,ez,eyaw=({ex1:.5f},{ey1:.5f},{ez1:.5f},{eyaw1:.5f})",
        f"ins={ins1:.5f} lat_err={lat1:.5f} book_x={book_x1:.5f}",
        f"ee_xy=({ee_x1:.5f},{ee_y1:.5f}) tgt_xy=({tgt_x1:.5f},{tgt_y1:.5f})",
        f"book_contact_norm={contact1:.2f}",
        flush=True,
    )
    print(
        "  delta:",
        f"d_ex={ex1-ex0:.5f} d_ey={ey1-ey0:.5f} d_ez={ez1-ez0:.5f} d_eyaw={eyaw1-eyaw0:.5f}",
        f"d_ins={ins1-ins0:.5f} d_lat_err={lat1-lat0:.5f} d_book_x={book_x1-book_x0:.5f}",
        f"d_ee_xy=({ee_x1-ee_x0:.5f},{ee_y1-ee_y0:.5f})",
        f"d_contact_norm={contact1-contact0:.2f}",
        flush=True,
    )


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.episode_length_s = float(args_cli.episode_length_s)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env_u = env.unwrapped
    device = env_u.device
    adim = int(env.action_space.shape[-1])
    if adim != 6:
        print(f"[WARN] Expected action dim 6 for v4; got {adim}. Pulses pad with zeros beyond known indices.")

    mag = max(-1.0, min(1.0, float(args_cli.action_mag)))

    def _a6(a: list[float]) -> torch.Tensor:
        row = (a + [0.0] * adim)[:adim]  # pad to env action dimension
        return torch.tensor([row], device=device, dtype=torch.float32)

    z = torch.zeros((1, adim), device=device)
    cases: dict[str, torch.Tensor] = {
        "hold": z.clone(),
        # Action order is [dx, dy, dz, dpitch, dyaw, g_gripper]
        "xp": _a6([mag, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "xm": _a6([-mag, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "yp": _a6([0.0, mag, 0.0, 0.0, 0.0, 0.0]),
        "ym": _a6([0.0, -mag, 0.0, 0.0, 0.0, 0.0]),
        "zp": _a6([0.0, 0.0, mag, 0.0, 0.0, 0.0]),
        "zm": _a6([0.0, 0.0, -mag, 0.0, 0.0, 0.0]),
        "yawp": _a6([0.0, 0.0, 0.0, 0.0, mag, 0.0]),
        "yawm": _a6([0.0, 0.0, 0.0, 0.0, -mag, 0.0]),
        "gripc": _a6([0.0, 0.0, 0.0, 0.0, 0.0, -mag]),
        "gripo": _a6([0.0, 0.0, 0.0, 0.0, 0.0, mag]),
    }

    with torch.no_grad():
        env.reset()

    if args_cli.case == "all":
        order = ["hold", "xp", "xm", "yp", "ym", "zp", "zm", "yawp", "yawm", "gripc", "gripo"]
        for name in order:
            a = cases[name]
            n = args_cli.hold_steps if name == "hold" else args_cli.pulse_steps
            _segment(env, env_u, a, n, name)
    else:
        a = cases[args_cli.case]
        n = args_cli.hold_steps if args_cli.case == "hold" else args_cli.pulse_steps
        _segment(env, env_u, a, n, args_cli.case)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
