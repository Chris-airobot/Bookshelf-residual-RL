# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bookshelf-Direct-v1: planar force control, narrow slot with walls, contact-rich insertion."""

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import math as math_utils
from isaaclab.utils.math import sample_uniform

from .bookshelf_env_cfg_v1 import BookshelfEnvCfg


def _geom_slot_mouth_x(cfg: BookshelfEnvCfg) -> float:
    """-X face of side walls (true channel entrance). slot_x_open is task ref, not this plane."""
    x0, x1 = cfg.slot_x_open, cfg.slot_x_back
    mid_x = 0.5 * (x0 + x1)
    depth_x = x1 - x0 + 0.06
    return mid_x - 0.5 * depth_x


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
        self._peak_ins = torch.zeros(self.num_envs, device=self.device)
        self._stall_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._jam_for_reward = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._forces_w = torch.zeros((self.num_envs, 1, 3), device=self.device)
        self._torques_w = torch.zeros((self.num_envs, 1, 3), device=self.device)

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

        # Side walls (long along X, thin along Y)
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
        # Back wall (closes +X end of slot)
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
        fx = self.actions[:, 0] * self.cfg.force_scale
        fy = self.actions[:, 1] * self.cfg.force_scale
        self._forces_w[:, 0, 0] = fx
        self._forces_w[:, 0, 1] = fy
        self._forces_w[:, 0, 2] = 0.0

        av = self.book.data.root_link_vel_w[:, 3:6]
        kd = 10.0
        self._torques_w[:, 0, 0] = -kd * av[:, 0]
        self._torques_w[:, 0, 1] = -kd * av[:, 1]
        self._torques_w[:, 0, 2] = -3.0 * av[:, 2]

        self.book.set_external_force_and_torque(
            self._forces_w, self._torques_w, env_ids=self._env_ids, is_global=True
        )

    def _book_pos_env(self) -> torch.Tensor:
        return self.book.data.root_link_pos_w - self.scene.env_origins

    def _contact_f_world(self) -> torch.Tensor:
        nf = self.book_contact.data.net_forces_w
        if nf is None:
            return torch.zeros(self.num_envs, 3, device=self.device)
        return nf[:, 0, :].clone()

    def _get_observations(self) -> dict:
        p = self._book_pos_env()
        v = self.book.data.root_link_vel_w
        ins = p[:, 0] - self.cfg.slot_x_open
        lat = self.cfg.slot_center_y - p[:, 1]
        f = self._contact_f_world()
        fs = self.cfg.contact_obs_force_scale
        obs = torch.stack(
            (
                self.cfg.slot_x_back - p[:, 0],
                lat,
                v[:, 0],
                v[:, 1],
                torch.clamp(f[:, 0] / fs, -1.0, 1.0),
                torch.clamp(f[:, 1] / fs, -1.0, 1.0),
            ),
            dim=-1,
        )
        obs = torch.clamp(obs, -self.cfg.obs_clip, self.cfg.obs_clip)
        return {"policy": obs}

    def _upright_ok(self) -> torch.Tensor:
        quat = self.book.data.root_link_quat_w
        # IMPORTANT: quat_apply reshapes output back to vec.shape, so vec must match quat batch dims.
        # Build local "spine" (+Y) with same leading dims as quat.
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

    def _jammed(self, ins: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        high_side = torch.abs(f[:, 1]) > self.cfg.jam_lateral_force_thresh
        shallow = ins < self.cfg.success_min_insertion * 0.75
        return (self._stall_buf >= self.cfg.jam_consecutive_steps) & high_side & shallow

    def _get_rewards(self) -> torch.Tensor:
        p = self._book_pos_env()
        ins = p[:, 0] - self.cfg.slot_x_open
        lat_e = torch.abs(p[:, 1] - self.cfg.slot_center_y)

        d_ins = ins - self.prev_insertion_depth
        gate = torch.exp(-((lat_e / self.cfg.progress_lateral_sigma) ** 2))
        progress_r = self.cfg.progress_scale * torch.clamp(d_ins, min=-0.02, max=0.02) * gate
        prev_lat = torch.abs(self.prev_insertion_depth * 0.0 + (p[:, 1] - self.cfg.slot_center_y).abs())
        # lateral improvement vs previous step (use stored prev lateral)
        lateral_prog = self._prev_lateral_err - lat_e
        center_r = self.cfg.center_scale * torch.clamp(lateral_prog, min=-0.02, max=0.02)

        f = self._contact_f_world()
        fn = torch.linalg.norm(f, dim=-1)
        contact_pen = self.cfg.contact_penalty_scale * torch.clamp(fn / 50.0, max=4.0)

        backward = torch.clamp(-d_ins, min=0.0)
        back_pen = self.cfg.backward_penalty_scale * backward

        misaligned = lat_e > self.cfg.misaligned_push_thresh
        misalign_push_pen = self.cfg.misaligned_push_scale * torch.clamp(d_ins, min=0.0) * misaligned.float()

        act_pen = self.cfg.action_penalty_scale * torch.sum(self.actions**2, dim=-1)

        inserted = ins >= self.cfg.success_min_insertion
        aligned = lat_e < self.cfg.success_lateral_thresh
        upright = self._upright_ok()
        inside = inserted & aligned & upright
        self.success_steps_buf = torch.where(
            inside, self.success_steps_buf + 1, torch.zeros_like(self.success_steps_buf)
        )
        success_mask = self.success_steps_buf >= self.cfg.success_steps

        success_r = self.cfg.success_bonus * success_mask.float()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        timeout_pen = self.cfg.timeout_penalty * (time_out & ~success_mask).float()

        jam_r = self.cfg.jam_penalty * (self._jam_for_reward & ~success_mask).float()

        rew = (
            progress_r
            + center_r
            - act_pen
            - contact_pen
            - back_pen
            - misalign_push_pen
            + success_r
            + timeout_pen
            + jam_r
        )

        self.prev_insertion_depth = ins.detach()
        self._prev_lateral_err = lat_e.detach()

        self.extras.setdefault("log", {})
        self.extras["log"]["insertion_depth"] = ins.mean()
        self.extras["log"]["lateral_err"] = lat_e.mean()
        self.extras["log"]["contact_norm"] = fn.mean()
        self.extras["log"]["success_rate"] = success_mask.float().mean()
        self.extras["log"]["jam_rate"] = self._jam_for_reward.float().mean()

        return rew

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        p = self._book_pos_env()
        ins = p[:, 0] - self.cfg.slot_x_open
        f = self._contact_f_world()
        fn = torch.linalg.norm(f, dim=-1)
        explode = fn > self.cfg.max_contact_force_norm
        self._update_stall(ins)
        jammed = self._jammed(ins, f)
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
            env_ids = self._env_ids
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        super()._reset_idx(env_ids)

        book_default_state = self.book.data.default_root_state[env_ids].clone()
        book_pos = book_default_state[:, 0:3]
        x_hi = self._book_com_x_max
        x_lo = x_hi - self.cfg.initial_forward_back_span
        book_pos[:, 0] = sample_uniform(x_lo, x_hi, (len(env_ids),), self.device)
        book_pos[:, 1] = self.cfg.slot_center_y + sample_uniform(
            self.cfg.initial_lateral_offset_range[0],
            self.cfg.initial_lateral_offset_range[1],
            (len(env_ids),),
            self.device,
        )
        book_pos[:, 2] = self.cfg.book_size[1] / 2.0

        quat = torch.tensor(self.cfg.book_standing_quat, device=self.device, dtype=book_default_state.dtype).expand(
            len(env_ids), 4
        )
        book_default_state[:, 0:3] = book_pos + self.scene.env_origins[env_ids]
        book_default_state[:, 3:7] = quat
        book_default_state[:, 7:] = 0.0

        self.book.write_root_state_to_sim(book_default_state, env_ids=env_ids)

        ins0 = book_pos[:, 0] - self.cfg.slot_x_open
        lat0 = torch.abs(book_pos[:, 1] - self.cfg.slot_center_y)
        self.prev_insertion_depth[env_ids] = ins0
        self._prev_lateral_err[env_ids] = lat0
        self.success_steps_buf[env_ids] = 0
        self._peak_ins[env_ids] = ins0
        self._stall_buf[env_ids] = 0
        self._jam_for_reward[env_ids] = False
