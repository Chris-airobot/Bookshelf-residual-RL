# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .bookshelf_env_cfg_v0 import BookshelfEnvCfg


class BookshelfEnv(DirectRLEnv):
    cfg: BookshelfEnvCfg

    def __init__(self, cfg: BookshelfEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # control timestep (physics dt * action decimation)
        self.dt = self.cfg.sim.dt * self.cfg.decimation
        # buffers for simple 2D insertion task
        self._env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        # kinematic commanded state buffers (world frame)
        self._book_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._book_linvel_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_forward_error = torch.zeros(self.num_envs, device=self.device)
        self.prev_lateral_error = torch.zeros(self.num_envs, device=self.device)
        self.success_steps_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _setup_scene(self):
        # add rigid object that acts as the "book"
        self.book = RigidObject(self.cfg.book_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # spawn a simple visual slot marker at the target position in env_0 (same pose as standing book)
        # this is purely visual (no collisions) so the first env stays contact-free
        slot_z = self.cfg.book_size[1] / 2.0
        slot_frame_prim = sim_utils.create_prim(
            "/World/envs/env_0/Slot",
            "Xform",
            translation=(self.cfg.target_depth, self.cfg.target_lateral, slot_z),
            orientation=self.cfg.book_standing_quat,
        )
        _ = slot_frame_prim  # avoid unused-variable warnings
        # make the slot marker the same size as the book
        bx, by, bz = self.cfg.book_size
        # Use MeshCuboid so the marker's frame matches its visual center (no scaled-cube artifacts).
        slot_cfg = sim_utils.MeshCuboidCfg(
            size=(bx, by, bz),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.8, 0.2)),
        )
        slot_cfg.func("/World/envs/env_0/Slot/geometry", slot_cfg)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # register rigid object to scene
        self.scene.rigid_objects["book"] = self.book
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # actions are 2D: [forward, lateral] in [-1, 1]
        self.actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        # Env1 "clean" control: directly integrate planar motion and keep the book upright.
        #
        # Why: a thin upright cuboid is dynamically unstable on a ground plane. Even with purely planar commands,
        # contacts introduce small torques that tip the book over. For the first environment (no slot contact yet),
        # we want to learn planar correction, not balancing. So we enforce a fixed standing orientation and zero
        # angular velocity, and only allow translation in x/y.
        #
        # Control: actions command planar velocities (m/s). We integrate position with dt and write pose to sim.
        # We also write velocities so observations remain consistent.
        #
        # interpret actions as desired planar delta-pose residuals for the book (meters per control step)
        forward_delta = self.actions[:, 0] * self.cfg.forward_delta_scale
        lateral_delta = self.actions[:, 1] * self.cfg.lateral_delta_scale

        # commanded linear velocity in world frame (derived from delta / dt; no vertical motion)
        self._book_linvel_w.zero_()
        self._book_linvel_w[:, 0] = forward_delta / self.dt
        self._book_linvel_w[:, 1] = lateral_delta / self.dt

        # integrate commanded position in world frame
        self._book_pos_w[:, 0] += forward_delta
        self._book_pos_w[:, 1] += lateral_delta
        # keep a fixed height so the book stays on the ground
        self._book_pos_w[:, 2] = self.scene.env_origins[:, 2] + (self.cfg.book_size[1] / 2.0)

        # fixed standing orientation (w, x, y, z)
        quat_wxyz = torch.tensor(self.cfg.book_standing_quat, device=self.device, dtype=self._book_pos_w.dtype).repeat(
            (self.num_envs, 1)
        )

        # write full root state (pos, quat, linvel, angvel) for kinematic body
        root_state = torch.zeros((self.num_envs, 13), device=self.device, dtype=self._book_pos_w.dtype)
        root_state[:, 0:3] = self._book_pos_w
        root_state[:, 3:7] = quat_wxyz
        root_state[:, 7:10] = self._book_linvel_w
        # ang vel stays zero
        self.book.write_root_state_to_sim(root_state, env_ids=self._env_ids)

    def _get_observations(self) -> dict:
        # commanded book position/velocity in env frame (stable even with fabric/USD differences)
        book_pos = self._book_pos_w - self.scene.env_origins
        book_linvel = self._book_linvel_w

        forward_error = self.cfg.target_depth - book_pos[:, 0]
        lateral_error = self.cfg.target_lateral - book_pos[:, 1]

        obs = torch.stack(
            (
                forward_error,
                lateral_error,
                book_linvel[:, 0],
                book_linvel[:, 1],
            ),
            dim=-1,
        )
        obs = torch.clamp(obs, -self.cfg.obs_clip, self.cfg.obs_clip)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # compute current errors
        book_pos = self._book_pos_w - self.scene.env_origins

        forward_error = torch.abs(self.cfg.target_depth - book_pos[:, 0])
        lateral_error = torch.abs(self.cfg.target_lateral - book_pos[:, 1])

        # signed progress rewards: positive if closer, negative if farther
        forward_progress = self.prev_forward_error - forward_error
        lateral_progress = self.prev_lateral_error - lateral_error

        progress_reward = self.cfg.progress_scale * forward_progress
        centering_reward = self.cfg.center_scale * lateral_progress

        # smoothness penalty
        action_penalty = self.cfg.action_penalty_scale * torch.sum(self.actions**2, dim=-1)

        # success condition
        inside_forward = forward_error < self.cfg.success_forward_thresh
        inside_lateral = lateral_error < self.cfg.success_lateral_thresh
        inside_region = inside_forward & inside_lateral

        self.success_steps_buf = torch.where(
            inside_region, self.success_steps_buf + 1, torch.zeros_like(self.success_steps_buf)
        )
        success_mask = self.success_steps_buf >= self.cfg.success_steps

        success_reward = self.cfg.success_bonus * success_mask.to(self.device)

        # timeout penalty at episode end if not successful
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        timeout_penalty = self.cfg.timeout_penalty * (time_out & (~success_mask)).to(self.device)

        total_reward = progress_reward + centering_reward - action_penalty + success_reward + timeout_penalty

        # update previous errors
        self.prev_forward_error = forward_error.detach()
        self.prev_lateral_error = lateral_error.detach()

        # optional logging
        self.extras.setdefault("log", {})
        self.extras["log"]["forward_error"] = forward_error.mean()
        self.extras["log"]["lateral_error"] = lateral_error.mean()
        self.extras["log"]["forward_progress"] = forward_progress.mean()
        self.extras["log"]["lateral_progress"] = lateral_progress.mean()
        self.extras["log"]["success_rate"] = success_mask.float().mean()

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # terminate when book has been in the success region long enough
        success_terminated = self.success_steps_buf >= self.cfg.success_steps
        # terminate if the book drifts too far (prevents runaway states / NaNs)
        book_pos = self._book_pos_w - self.scene.env_origins
        out_of_bounds = (torch.abs(book_pos[:, 0]) > self.cfg.max_abs_xy) | (torch.abs(book_pos[:, 1]) > self.cfg.max_abs_xy)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return success_terminated | out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._env_ids
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        # base class handles episode bookkeeping
        super()._reset_idx(env_ids)

        # reset book root state near the target with small random offsets
        book_default_state = self.book.data.default_root_state[env_ids].clone()

        # position in env frame before adding env origins
        book_pos = book_default_state[:, 0:3]
        book_pos[:, 0] = self.cfg.target_depth
        book_pos[:, 1] = self.cfg.target_lateral
        # keep the book standing on the ground (center z = half of standing height)
        book_pos[:, 2] = self.cfg.book_size[1] / 2.0

        forward_offset = sample_uniform(
            self.cfg.initial_forward_offset_range[0],
            self.cfg.initial_forward_offset_range[1],
            (len(env_ids),),
            self.device,
        )
        lateral_offset = sample_uniform(
            self.cfg.initial_lateral_offset_range[0],
            self.cfg.initial_lateral_offset_range[1],
            (len(env_ids),),
            self.device,
        )
        book_pos[:, 0] += forward_offset
        book_pos[:, 1] += lateral_offset

        # move to world frame by adding env origins
        book_default_state[:, 0:3] = book_pos + self.scene.env_origins[env_ids]
        # zero velocities
        book_default_state[:, 7:] = 0.0

        self.book.write_root_pose_to_sim(book_default_state[:, :7], env_ids=env_ids)
        self.book.write_root_velocity_to_sim(book_default_state[:, 7:], env_ids=env_ids)

        # initialize commanded buffers (world frame)
        self._book_pos_w[env_ids] = book_default_state[:, 0:3]
        self._book_linvel_w[env_ids] = 0.0

        # reset reward-related buffers
        forward_error = torch.abs(self.cfg.target_depth - book_pos[:, 0])
        lateral_error = torch.abs(self.cfg.target_lateral - book_pos[:, 1])
        self.prev_forward_error[env_ids] = forward_error
        self.prev_lateral_error[env_ids] = lateral_error
        self.success_steps_buf[env_ids] = 0