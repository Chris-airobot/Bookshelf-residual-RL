#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bookshelf-Direct-v3: planner-handoff residual pose control, impedance execution, contact-rich insertion."""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import math as math_utils
from isaaclab.utils.math import sample_uniform

from .bookshelf_env_cfg_v3 import BookshelfEnvCfg


def _geom_slot_mouth_x(cfg: BookshelfEnvCfg) -> float:
    """-X face of side walls (true channel entrance). slot_x_open is task ref, not this plane."""
    x0, x1 = cfg.slot_x_open, cfg.slot_x_back
    mid_x = 0.5 * (x0 + x1)
    depth_x = x1 - x0 + 0.06
    return mid_x - 0.5 * depth_x


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return (angle + torch.pi) % (2.0 * torch.pi) - torch.pi


def _yaw_from_quat_wxyz(q: torch.Tensor) -> torch.Tensor:
    """Extract world yaw (about +Z) from quaternion (w, x, y, z)."""
    qw, qx, qy, qz = q.unbind(dim=-1)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return torch.atan2(siny_cosp, cosy_cosp)


def _quat_mul_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product for quaternions (w, x, y, z)."""
    aw, ax, ay, az = a.unbind(dim=-1)
    bw, bx, by, bz = b.unbind(dim=-1)
    w = aw * bw - ax * bx - ay * by - az * bz
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((w, x, y, z), dim=-1)


class BookshelfEnv(DirectRLEnv):
    cfg: BookshelfEnvCfg

    def __init__(self, cfg: BookshelfEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self._env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self.book: RigidObject = self.scene.rigid_objects["book"]
        self.book_contact: ContactSensor = self.scene.sensors["book_contact"]

        self.success_steps_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.prev_insertion_depth = torch.zeros(self.num_envs, device=self.device)
        self._prev_lateral_err = torch.zeros(self.num_envs, device=self.device)
        self._prev_yaw_err = torch.zeros(self.num_envs, device=self.device)
        self._peak_ins = torch.zeros(self.num_envs, device=self.device)
        self._stall_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._jam_for_reward = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._forces_w = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self._torques_w = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self._last_actions = torch.zeros((self.num_envs, 3), device=self.device)

        # Residual control: target in env frame (slot frame aligned with env axes in this task).
        self._target_pos_env = torch.zeros((self.num_envs, 3), device=self.device)
        self._target_yaw = torch.zeros(self.num_envs, device=self.device)

        # Max COM x so OBB leading +X face stays left of geometric slot mouth (side wall -X face).
        h = torch.tensor(
            [self.cfg.book_size[i] * 0.5 for i in range(3)],
            device=self.device,
            dtype=torch.float32,
        )
        q = torch.tensor(self.cfg.book_standing_quat, device=self.device, dtype=torch.float32).unsqueeze(0)
        r0 = math_utils.matrix_from_quat(q)[0, 0, :].abs()
        half_x_plus = float((r0 * h).sum().item())
        mouth_x = _geom_slot_mouth_x(self.cfg)
        self._book_com_x_max = float(mouth_x - half_x_plus - self.cfg.initial_slot_mouth_clearance_x)

    def _setup_scene(self):
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        x0 = self.cfg.slot_x_open
        x1 = self.cfg.slot_x_back
        mid_x = 0.5 * (x0 + x1)
        depth_x = x1 - x0 + 0.06
        bthick = self.cfg.book_size[2]
        clearance = self.cfg.slot_lateral_clearance
        inner_half = 0.5 * (bthick + clearance)
        wt = self.cfg.wall_thickness
        zc = self.cfg.book_size[1] * 0.5 + 0.02
        wall_h = self.cfg.book_size[1] + 0.08

        kin = RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True)
        col = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
        mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.45, 0.35))

        def wall(name: str, size: tuple[float, float, float], pos: tuple[float, float, float]) -> None:
            cfg = sim_utils.MeshCuboidCfg(size=size, rigid_props=kin, collision_props=col, visual_material=mat)
            cfg.func(f"/World/envs/env_0/SlotWalls/{name}", cfg, translation=pos, orientation=(1.0, 0.0, 0.0, 0.0))

        wall(
            "LeftWall",
            (depth_x, wt, wall_h),
            (mid_x, inner_half + wt * 0.5, zc),
        )
        wall(
            "RightWall",
            (depth_x, wt, wall_h),
            (mid_x, -(inner_half + wt * 0.5), zc),
        )
        back_x = x1 + wt * 0.5
        span_y = 2.0 * inner_half + 2.5 * wt
        wall("BackWall", (wt, span_y, wall_h), (back_x, self.cfg.slot_center_y, zc))

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        # Update residual target in env frame (slot frame aligned with env axes).
        dx = self.actions[:, 0] * self.cfg.dx_action_scale
        dy = self.actions[:, 1] * self.cfg.dy_action_scale
        dyaw = self.actions[:, 2] * self.cfg.dyaw_action_scale

        self._target_pos_env[:, 0] += dx
        self._target_pos_env[:, 1] += dy
        self._target_yaw = _wrap_to_pi(self._target_yaw + dyaw)

        # Clip target to a local region around the slot mouth (planner-handoff assumption).
        x_clip, y_clip = self.cfg.target_pos_clip
        self._target_pos_env[:, 0] = torch.clamp(
            self._target_pos_env[:, 0],
            self.cfg.slot_x_open - x_clip,
            self.cfg.slot_x_open + x_clip,
        )
        self._target_pos_env[:, 1] = torch.clamp(
            self._target_pos_env[:, 1],
            self.cfg.slot_center_y - y_clip,
            self.cfg.slot_center_y + y_clip,
        )
        self._target_yaw = torch.clamp(self._target_yaw, -self.cfg.target_yaw_clip, self.cfg.target_yaw_clip)

        # Impedance to target: convert pose residual to external wrench.
        p = self._book_pos_env()
        v = self.book.data.root_link_vel_w
        yaw = _yaw_from_quat_wxyz(self.book.data.root_link_quat_w)

        ex = self._target_pos_env[:, 0] - p[:, 0]
        ey = self._target_pos_env[:, 1] - p[:, 1]
        eyaw = _wrap_to_pi(self._target_yaw - yaw)

        fx = self.cfg.k_pos_x * ex - self.cfg.d_vel_x * v[:, 0]
        fy = self.cfg.k_pos_y * ey - self.cfg.d_vel_y * v[:, 1]
        mz = self.cfg.k_yaw * eyaw - self.cfg.d_yaw_rate * v[:, 5]

        fx = torch.clamp(fx, -self.cfg.max_force, self.cfg.max_force)
        fy = torch.clamp(fy, -self.cfg.max_force, self.cfg.max_force)
        mz = torch.clamp(mz, -self.cfg.max_yaw_torque, self.cfg.max_yaw_torque)

        self._forces_w[:, 0, 0] = fx
        self._forces_w[:, 0, 1] = fy
        self._forces_w[:, 0, 2] = 0.0

        # Keep upright via roll/pitch damping; yaw is controller-driven above.
        av = self.book.data.root_link_vel_w[:, 3:6]
        kd_rp = 10.0
        self._torques_w[:, 0, 0] = -kd_rp * av[:, 0]
        self._torques_w[:, 0, 1] = -kd_rp * av[:, 1]
        self._torques_w[:, 0, 2] = mz

        self.book.set_external_force_and_torque(
            self._forces_w, self._torques_w, env_ids=self._env_ids, is_global=True
        )

    def _book_pos_env(self) -> torch.Tensor:
        return self.book.data.root_link_pos_w - self.scene.env_origins

    def _contact_wrench_world(self) -> tuple[torch.Tensor, torch.Tensor]:
        nf = self.book_contact.data.net_forces_w
        if nf is None:
            f = torch.zeros(self.num_envs, 3, device=self.device)
        else:
            f = nf[:, 0, :].clone()

        nt = getattr(self.book_contact.data, "net_torques_w", None)
        if nt is None:
            t = torch.zeros(self.num_envs, 3, device=self.device)
        else:
            t = nt[:, 0, :].clone()
        return f, t

    def _yaw_err(self) -> torch.Tensor:
        q = self.book.data.root_link_quat_w
        yaw = _yaw_from_quat_wxyz(q)
        return _wrap_to_pi(yaw)

    def _get_observations(self) -> dict:
        p = self._book_pos_env()
        v = self.book.data.root_link_vel_w
        lat = self.cfg.slot_center_y - p[:, 1]
        yaw_e = self._yaw_err()

        f, t = self._contact_wrench_world()
        fs = self.cfg.contact_obs_force_scale
        ts = self.cfg.contact_obs_torque_scale

        obs = torch.stack(
            (
                self.cfg.slot_x_back - p[:, 0],
                lat,
                yaw_e,
                v[:, 0],
                v[:, 1],
                v[:, 5],
                torch.clamp(f[:, 0] / fs, -1.0, 1.0),
                torch.clamp(f[:, 1] / fs, -1.0, 1.0),
                torch.clamp(t[:, 2] / ts, -1.0, 1.0),
                self._last_actions[:, 0],
                self._last_actions[:, 1],
                self._last_actions[:, 2],
            ),
            dim=-1,
        )
        obs = torch.clamp(obs, -self.cfg.obs_clip, self.cfg.obs_clip)
        return {"policy": obs}

    def _upright_ok(self) -> torch.Tensor:
        quat = self.book.data.root_link_quat_w
        spine_l = torch.zeros_like(quat[..., 1:4])
        spine_l[..., 1] = 1.0
        spine_w = math_utils.quat_apply(quat, spine_l)
        if spine_w.dim() == 1:
            return torch.abs(spine_w[2]) > self.cfg.upright_dot_thresh
        return torch.abs(spine_w[:, 2]) > self.cfg.upright_dot_thresh

    def _update_stall(self, ins: torch.Tensor) -> None:
        gained = ins > self._peak_ins + 0.003
        self._peak_ins = torch.maximum(self._peak_ins, ins)
        self._stall_buf = torch.where(gained, torch.zeros_like(self._stall_buf), self._stall_buf + 1)

    def _jammed(self, ins: torch.Tensor, f: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        high_side = torch.abs(f[:, 1]) > self.cfg.jam_lateral_force_thresh
        high_yaw = torch.abs(t[:, 2]) > self.cfg.jam_yaw_torque_thresh
        shallow = ins < self.cfg.success_min_insertion * 0.75
        return (self._stall_buf >= self.cfg.jam_consecutive_steps) & (high_side | high_yaw) & shallow

    def _get_rewards(self) -> torch.Tensor:
        p = self._book_pos_env()
        v = self.book.data.root_link_vel_w
        ins = p[:, 0] - self.cfg.slot_x_open
        lat_e = torch.abs(p[:, 1] - self.cfg.slot_center_y)
        yaw_e = torch.abs(self._yaw_err())

        d_ins = ins - self.prev_insertion_depth
        lat_gate = torch.exp(-((lat_e / self.cfg.progress_lateral_sigma) ** 2))
        yaw_gate = torch.exp(-((yaw_e / self.cfg.progress_yaw_sigma) ** 2))
        gate = lat_gate * yaw_gate
        progress_r = self.cfg.progress_scale * torch.clamp(d_ins, min=-0.02, max=0.02) * gate

        lateral_prog = self._prev_lateral_err - lat_e
        center_r = self.cfg.center_scale * torch.clamp(lateral_prog, min=-0.02, max=0.02)

        yaw_prog = self._prev_yaw_err - yaw_e
        yaw_r = self.cfg.yaw_scale * torch.clamp(yaw_prog, min=-0.08, max=0.08)

        f, t = self._contact_wrench_world()
        fn = torch.linalg.norm(f, dim=-1)
        contact_pen = self.cfg.contact_penalty_scale * torch.clamp((fn - 10.0) / 50.0, min=0.0, max=4.0)

        backward = torch.clamp(-d_ins, min=0.0)
        # Allow corrective backoff: penalize backoff mainly when already aligned.
        back_pen = self.cfg.backward_penalty_scale * backward * gate

        mis_lat = lat_e > self.cfg.misaligned_push_lateral_thresh
        mis_yaw = yaw_e > self.cfg.misaligned_push_yaw_thresh
        misaligned = (mis_lat | mis_yaw).float()
        misalign_push_pen = self.cfg.misaligned_push_scale * torch.clamp(d_ins, min=0.0) * misaligned

        dact = self.actions - self._last_actions
        dact_pen = self.cfg.action_delta_penalty_scale * torch.sum(dact**2, dim=-1)

        # Strong penalty when the book has slid outside the channel while still shallow.
        side_bad = (lat_e > self.cfg.side_bad_lateral_thresh) & (ins < self.cfg.side_bad_insertion_thresh)
        side_bad_pen = self.cfg.side_bad_penalty * side_bad.float()

        inserted = ins >= self.cfg.success_min_insertion
        aligned = lat_e < self.cfg.success_lateral_thresh
        yaw_ok = yaw_e < self.cfg.success_yaw_thresh
        upright = self._upright_ok()
        inside = inserted & aligned & yaw_ok & upright
        self.success_steps_buf = torch.where(
            inside, self.success_steps_buf + 1, torch.zeros_like(self.success_steps_buf)
        )
        success_mask = self.success_steps_buf >= self.cfg.success_steps

        success_r = self.cfg.success_bonus * success_mask.float()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        timeout_pen = self.cfg.timeout_penalty * (time_out & ~success_mask).float()

        jam_r = self.cfg.jam_penalty * (self._jam_for_reward & ~success_mask).float()

        oob = (torch.abs(p[:, 0]) > self.cfg.max_abs_xy) | (torch.abs(p[:, 1]) > self.cfg.max_abs_xy)
        fell = p[:, 2] < 0.04
        tipped = ~upright
        explode = fn > self.cfg.max_contact_force_norm
        fail_pen = (
            self.cfg.oob_penalty * (oob & ~success_mask).float()
            + self.cfg.fell_penalty * (fell & ~success_mask).float()
            + self.cfg.tipped_penalty * (tipped & ~success_mask).float()
            + self.cfg.explode_penalty * (explode & ~success_mask).float()
        )

        rew = (
            progress_r
            + center_r
            + yaw_r
            - dact_pen
            - contact_pen
            - back_pen
            - misalign_push_pen
            + success_r
            + timeout_pen
            + jam_r
            + fail_pen
            + side_bad_pen
        )

        self.prev_insertion_depth = ins.detach()
        self._prev_lateral_err = lat_e.detach()
        self._prev_yaw_err = yaw_e.detach()
        self._last_actions = self.actions.detach()

        self.extras.setdefault("log", {})
        self.extras["log"]["insertion_depth"] = ins.mean()
        self.extras["log"]["lateral_err"] = lat_e.mean()
        self.extras["log"]["yaw_err_abs"] = yaw_e.mean()
        self.extras["log"]["contact_norm"] = fn.mean()
        self.extras["log"]["success_rate"] = success_mask.float().mean()
        self.extras["log"]["jam_rate"] = self._jam_for_reward.float().mean()
        self.extras["log"]["side_bad_rate"] = side_bad.float().mean()

        return rew

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        p = self._book_pos_env()
        ins = p[:, 0] - self.cfg.slot_x_open
        f, t = self._contact_wrench_world()
        fn = torch.linalg.norm(f, dim=-1)
        explode = fn > self.cfg.max_contact_force_norm
        self._update_stall(ins)
        jammed = self._jammed(ins, f, t)
        self._jam_for_reward = jammed

        oob = (torch.abs(p[:, 0]) > self.cfg.max_abs_xy) | (torch.abs(p[:, 1]) > self.cfg.max_abs_xy)
        fell = p[:, 2] < 0.04
        tipped = ~self._upright_ok()

        success = self.success_steps_buf >= self.cfg.success_steps
        terminated = success | oob | explode | jammed | fell | tipped
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids_t = self._env_ids
        else:
            env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        super()._reset_idx(env_ids_t)

        book_default_state = self.book.data.default_root_state[env_ids_t].clone()
        book_pos = book_default_state[:, 0:3]
        x_hi = self._book_com_x_max
        x_lo = x_hi - self.cfg.initial_forward_back_span
        n = int(env_ids_t.numel())
        book_pos[:, 0] = sample_uniform(x_lo, x_hi, (n,), self.device)
        book_pos[:, 1] = self.cfg.slot_center_y + sample_uniform(
            self.cfg.initial_lateral_offset_range[0],
            self.cfg.initial_lateral_offset_range[1],
            (n,),
            self.device,
        )
        book_pos[:, 2] = self.cfg.book_size[1] / 2.0

        yaw = sample_uniform(
            self.cfg.initial_yaw_range[0],
            self.cfg.initial_yaw_range[1],
            (n,),
            self.device,
        )
        qyaw = torch.stack(
            (torch.cos(0.5 * yaw), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(0.5 * yaw)),
            dim=-1,
        )
        qstand = torch.tensor(self.cfg.book_standing_quat, device=self.device, dtype=book_default_state.dtype).expand(n, 4)
        quat = _quat_mul_wxyz(qyaw, qstand)

        book_default_state[:, 0:3] = book_pos + self.scene.env_origins[env_ids_t]
        book_default_state[:, 3:7] = quat
        book_default_state[:, 7:] = 0.0

        self.book.write_root_state_to_sim(book_default_state, env_ids=env_ids_t)

        # Initialize residual-control target to current state (planner-handoff "near slot").
        self._target_pos_env[env_ids_t] = book_pos
        self._target_yaw[env_ids_t] = yaw

        ins0 = book_pos[:, 0] - self.cfg.slot_x_open
        lat0 = torch.abs(book_pos[:, 1] - self.cfg.slot_center_y)
        self.prev_insertion_depth[env_ids_t] = ins0
        self._prev_lateral_err[env_ids_t] = lat0
        self._prev_yaw_err[env_ids_t] = torch.abs(yaw)
        self._last_actions[env_ids_t] = 0.0
        self.success_steps_buf[env_ids_t] = 0
        self._peak_ins[env_ids_t] = ins0
        self._stall_buf[env_ids_t] = 0
        self._jam_for_reward[env_ids_t] = False

