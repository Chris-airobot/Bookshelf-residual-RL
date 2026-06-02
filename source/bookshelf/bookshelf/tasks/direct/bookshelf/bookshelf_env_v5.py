#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bookshelf-Direct-v5 environment.

V5 is v4 with randomized reset geometry. The side books are kinematic rigid
objects: they do not move during an episode, but their positions and exposed
heights are reset per environment. The row has ten logical book positions, one
sampled missing-book slot, and a mix of one-slot and two-slot side books.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import math as math_utils
from isaaclab.utils.math import sample_uniform

from .bookshelf_env_cfg_v5 import BookshelfEnvCfg
from .bookshelf_env_v4 import (
    _DONE_DEPTH,
    _DONE_DROP,
    _DONE_FELL,
    _DONE_LATERAL,
    _DONE_NONE,
    _DONE_NOT_PUSH,
    _DONE_OOB,
    _DONE_SUCCESS,
    _DONE_TIMEOUT,
    _DONE_UNSTABLE,
    _DONE_UPRIGHT,
    _DONE_YAW,
    _DONE_Z,
    _MODE_INSERT,
    _MODE_PUSH,
    _MODE_SCRIPTED,
    BookshelfEnv as BookshelfEnvV4,
    _neighbor_book_dims,
    _wrap_to_pi,
    _yaw_from_quat_wxyz,
)


class BookshelfEnv(BookshelfEnvV4):
    """V4 task with randomized row geometry and an extra upright-tilt action."""

    cfg: BookshelfEnvCfg

    def __init__(self, cfg: BookshelfEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._ensure_randomization_buffers()
        self._nominal_ee_roll_b = torch.tensor(0.0, device=self.device)
        self._nominal_ee_pitch_b = torch.tensor(0.0, device=self.device)
        self._nominal_ee_yaw_b = torch.tensor(0.0, device=self.device)
        self._capture_nominal_ee_orientation()

    def _slot_center_y(self) -> torch.Tensor:
        value = getattr(self, "_slot_center_y_env", None)
        if value is not None:
            return value
        return torch.full((self.num_envs,), float(self.cfg.slot_center_y), device=self.device, dtype=torch.float32)

    def _slot_lateral_clearance(self) -> torch.Tensor:
        value = getattr(self, "_slot_lateral_clearance_env", None)
        if value is not None:
            return value
        return torch.full(
            (self.num_envs,), float(self.cfg.slot_lateral_clearance), device=self.device, dtype=torch.float32
        )

    def _ensure_randomization_buffers(self) -> None:
        if not hasattr(self, "_slot_center_y_env"):
            self._slot_center_y_env = torch.full(
                (self.num_envs,), float(self.cfg.slot_center_y), device=self.device, dtype=torch.float32
            )
        if not hasattr(self, "_slot_lateral_clearance_env"):
            self._slot_lateral_clearance_env = torch.full(
                (self.num_envs,), float(self.cfg.slot_lateral_clearance), device=self.device, dtype=torch.float32
            )
        if not hasattr(self, "_missing_book_index_env"):
            self._missing_book_index_env = torch.full((self.num_envs,), 2, device=self.device, dtype=torch.long)
        if not hasattr(self, "_single_book_slot_env"):
            self._single_book_slot_env = torch.full(
                (self.num_envs, self._max_single_row_books()), -1, device=self.device, dtype=torch.long
            )
        if not hasattr(self, "_wide_book_start_slot_env"):
            self._wide_book_start_slot_env = torch.full(
                (self.num_envs, self._max_wide_row_books()), -1, device=self.device, dtype=torch.long
            )

    def _capture_nominal_ee_orientation(self) -> None:
        robot_default = self.robot.data.default_root_state.clone()
        robot_default[:, 0:3] += self.scene.env_origins
        joint_pos = self.robot.data.default_joint_pos.clone()
        joint_vel = self.robot.data.default_joint_vel.clone()

        current_root = self.robot.data.root_state_w.clone()
        current_joint_pos = self.robot.data.joint_pos.clone()
        current_joint_vel = self.robot.data.joint_vel.clone()

        self.robot.write_root_state_to_sim(robot_default)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)

        _, ee_quat_b = self._ee_pose_in_base()
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(ee_quat_b)
        self._nominal_ee_roll_b = roll.detach().mean()
        self._nominal_ee_pitch_b = pitch.detach().mean()
        self._nominal_ee_yaw_b = yaw.detach().mean()

        self.robot.write_root_state_to_sim(current_root)
        self.robot.write_joint_state_to_sim(current_joint_pos, current_joint_vel)
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)

    def _compute_ik_joint_targets_nominal_orientation(
        self, target_pos_env: torch.Tensor, env_ids_t: torch.Tensor
    ) -> torch.Tensor:
        self._target_pos_env[:] = target_pos_env
        self._target_yaw[:] = self._nominal_ee_yaw_b

        quat_des_b = math_utils.quat_from_euler_xyz(
            torch.full((self.num_envs,), float(self._nominal_ee_roll_b.item()), device=self.device),
            torch.full((self.num_envs,), float(self._nominal_ee_pitch_b.item()), device=self.device),
            torch.full((self.num_envs,), float(self._nominal_ee_yaw_b.item()), device=self.device),
        )

        offset_des_b = math_utils.quat_apply(quat_des_b, self._ik_body_offset_pos_b)
        body_pos_des_b = target_pos_env - offset_des_b

        self._ik_cmd[:, 0:3] = body_pos_des_b
        self._ik_cmd[:, 3:7] = quat_des_b
        self._ik.set_command(self._ik_cmd)

        ee_pos_b, ee_quat_b = self._ee_pose_in_base()
        jacobian = self.robot.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, self._jacobi_joint_ids]
        joint_pos = self.robot.data.joint_pos[:, self._arm_joint_ids]
        return self._ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

    def _compute_ik_joint_targets_from_tool(
        self, target_pos_env: torch.Tensor, target_yaw: torch.Tensor, target_pitch: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Absolute IK with yaw plus optional pitch control for upright tilt correction."""
        self._target_pos_env[:] = target_pos_env
        self._target_yaw[:] = target_yaw

        _, ee_quat_b = self._ee_pose_in_base()
        ee_roll_b, ee_pitch_b, _ = math_utils.euler_xyz_from_quat(ee_quat_b)
        pitch_des = ee_pitch_b if target_pitch is None else target_pitch
        quat_des_b = math_utils.quat_from_euler_xyz(ee_roll_b, pitch_des, target_yaw)

        offset_des_b = math_utils.quat_apply(quat_des_b, self._ik_body_offset_pos_b)
        body_pos_des_b = target_pos_env - offset_des_b

        self._ik_cmd[:, 0:3] = body_pos_des_b
        self._ik_cmd[:, 3:7] = quat_des_b
        self._ik.set_command(self._ik_cmd)

        ee_pos_b, ee_quat_b2 = self._ee_pose_in_base()
        jacobian = self.robot.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, self._jacobi_joint_ids]
        joint_pos = self.robot.data.joint_pos[:, self._arm_joint_ids]
        return self._ik.compute(ee_pos_b, ee_quat_b2, jacobian, joint_pos)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)
        self._mode_start = self._mode.clone()
        self._release_request = self.actions[:, -1] > float(self.cfg.release_trigger_threshold)

    def _apply_action(self) -> None:
        mode = self._mode

        dx = self.actions[:, 0] * self.cfg.dx_action_scale
        dy = self.actions[:, 1] * self.cfg.dy_action_scale
        dz = self.actions[:, 2] * self.cfg.dz_action_scale
        dyaw = self.actions[:, 3] * self.cfg.dyaw_action_scale
        dpitch = self.actions[:, 4] * self.cfg.dpitch_action_scale

        ee_tool_pos_env = self._ee_tool_pos_env()
        _, ee_quat_b = self._ee_pose_in_base()
        _, ee_pitch_b, ee_yaw_b = math_utils.euler_xyz_from_quat(ee_quat_b)

        target_pos_env = ee_tool_pos_env.clone()
        target_yaw = ee_yaw_b.clone()
        target_pitch = ee_pitch_b.clone()

        normal_mask = mode != _MODE_SCRIPTED
        if torch.any(normal_mask):
            delta = torch.stack((dx, dy, dz), dim=-1)
            target_pos_env[normal_mask] = target_pos_env[normal_mask] + delta[normal_mask]
            target_yaw[normal_mask] = _wrap_to_pi(ee_yaw_b[normal_mask] + dyaw[normal_mask])
            target_pitch[normal_mask] = _wrap_to_pi(ee_pitch_b[normal_mask] + dpitch[normal_mask])

        scripted_mask = mode == _MODE_SCRIPTED
        if torch.any(scripted_mask):
            retreat_mask = scripted_mask & (self._script_step_buf >= int(self.cfg.script_open_steps))
            if torch.any(retreat_mask):
                target_pos_env[retreat_mask, 0] += float(self.cfg.script_retreat_dx)
                target_pos_env[retreat_mask, 2] += float(self.cfg.script_retreat_dz)

        joint_pos_des = self._compute_ik_joint_targets_from_tool(target_pos_env, target_yaw, target_pitch)

        arm_act = self.actions[:, 0:5]
        act_small = arm_act.abs() < float(self.cfg.ik_hold_action_epsilon)
        hold_arm = normal_mask & act_small.all(dim=-1)
        move_arm = ~hold_arm

        move_exp = move_arm.unsqueeze(-1).expand_as(joint_pos_des)
        self._arm_hold_joint_pos = torch.where(move_exp, joint_pos_des, self._arm_hold_joint_pos)

        hold_exp = hold_arm.unsqueeze(-1).expand_as(joint_pos_des)
        joint_pos_des = torch.where(hold_exp, self._arm_hold_joint_pos, joint_pos_des)

        self.robot.set_joint_position_target(joint_pos_des, joint_ids=self._arm_joint_ids)

        if len(self._finger_joint_ids) > 0:
            c = float(self.cfg.gripper_closed_joint_pos)
            o = float(self.cfg.gripper_open_joint_pos)
            script_open_retreat = int(self.cfg.script_open_steps) + int(self.cfg.script_retreat_steps)
            gripper_should_open = (mode == _MODE_SCRIPTED) & (self._script_step_buf < script_open_retreat)
            finger_cmd = torch.where(
                gripper_should_open,
                torch.full((self.num_envs,), o, device=self.device, dtype=torch.float32),
                torch.full((self.num_envs,), c, device=self.device, dtype=torch.float32),
            )
            finger_des = finger_cmd.unsqueeze(-1).expand(self.num_envs, len(self._finger_joint_ids))
            self.robot.set_joint_position_target(finger_des, joint_ids=self._finger_joint_ids)

    def _book_upright_tilt_obs(self) -> tuple[torch.Tensor, torch.Tensor]:
        quat = self.book.data.root_link_quat_w
        spine_l = torch.zeros_like(quat[..., 1:4])
        spine_l[..., 1] = 1.0
        spine_w = math_utils.quat_apply(quat, spine_l)
        return torch.clamp(spine_w[:, 0], -1.0, 1.0), torch.clamp(spine_w[:, 1], -1.0, 1.0)

    def _get_observations(self) -> dict:
        obs = super()._get_observations()
        tilt_x, tilt_y = self._book_upright_tilt_obs()
        obs["policy"] = torch.cat((obs["policy"], tilt_x.unsqueeze(-1), tilt_y.unsqueeze(-1)), dim=-1)
        return obs

    def _single_book_names(self) -> list[str]:
        return [f"side_book_{i}" for i in range(self._max_single_row_books())]

    def _wide_book_names(self) -> list[str]:
        return [f"wide_side_book_{i}" for i in range(self._max_wide_row_books())]

    def _row_book_names(self) -> list[str]:
        return self._single_book_names() + self._wide_book_names()

    def _max_single_row_books(self) -> int:
        return max(0, int(getattr(self.cfg, "row_book_count", 10)) - 1)

    def _max_wide_row_books(self) -> int:
        return max(0, (int(getattr(self.cfg, "row_book_count", 10)) - 1) // 2)

    def _sample_row_layout(self, env_ids_t: torch.Tensor) -> None:
        self._ensure_randomization_buffers()
        p_merge = float(getattr(self.cfg, "side_book_merge_probability", 0.0))
        row_count = int(getattr(self.cfg, "row_book_count", 10))

        self._single_book_slot_env[env_ids_t] = -1
        self._wide_book_start_slot_env[env_ids_t] = -1

        for env_id in env_ids_t.tolist():
            missing = int(self._missing_book_index_env[env_id].item())
            visible_slots = [idx for idx in range(row_count) if idx != missing]
            single_order = torch.randperm(self._max_single_row_books(), device=self.device).tolist()
            wide_order = torch.randperm(self._max_wide_row_books(), device=self.device).tolist()
            single_cursor = 0
            wide_cursor = 0

            for side_slots in (
                [idx for idx in visible_slots if idx < missing],
                [idx for idx in visible_slots if idx > missing],
            ):
                cursor = 0
                while cursor < len(side_slots):
                    span = 1
                    can_merge = cursor + 1 < len(side_slots) and side_slots[cursor + 1] == side_slots[cursor] + 1
                    if can_merge and float(torch.rand((), device=self.device).item()) < p_merge:
                        span = 2
                    if span == 2 and wide_cursor < self._max_wide_row_books():
                        self._wide_book_start_slot_env[env_id, wide_order[wide_cursor]] = side_slots[cursor]
                        wide_cursor += 1
                    else:
                        for offset in range(span):
                            if single_cursor < self._max_single_row_books():
                                self._single_book_slot_env[env_id, single_order[single_cursor]] = (
                                    side_slots[cursor + offset]
                                )
                                single_cursor += 1
                    cursor += span

    def _current_slot_lateral_clearance(self) -> float:
        """Compatibility helper used by manual status printing."""
        if hasattr(self, "_slot_lateral_clearance_env"):
            return float(self._slot_lateral_clearance_env.mean().item())
        return super()._current_slot_lateral_clearance()

    def _row_book_root_states(self, env_ids_t: torch.Tensor) -> list[tuple[RigidObject, torch.Tensor]]:
        self._ensure_randomization_buffers()
        n = int(env_ids_t.numel())
        dtype = torch.float32
        x0 = float(self.cfg.slot_x_open)
        x1 = float(self.cfg.slot_x_back)
        mid_x = 0.5 * (x0 + x1)
        shelf_z = float(self.cfg.shelf_top_z + self.cfg.shelf_thickness)

        nb = _neighbor_book_dims(self.cfg)
        bheight = float(nb[1])
        bthick = float(nb[2])
        pitch_y = bthick

        clearance = self._slot_lateral_clearance_env[env_ids_t]
        missing_idx = self._missing_book_index_env[env_ids_t]
        row_center = 0.5 * float(int(getattr(self.cfg, "row_book_count", 10)) - 1)

        q = torch.tensor(self.cfg.book_standing_quat, device=self.device, dtype=dtype).unsqueeze(0).expand(n, 4)
        states: list[tuple[RigidObject, torch.Tensor]] = []

        def append_pool_states(names: list[str], slot_buffer: torch.Tensor, span_slots: float) -> None:
            for pool_idx, name in enumerate(names):
                obj = self.scene.rigid_objects[name]
                state = obj.data.default_root_state[env_ids_t].clone()
                slot_start = slot_buffer[env_ids_t, pool_idx]
                active = slot_start >= 0

                slot_mid = slot_start.to(dtype=dtype) + 0.5 * (span_slots - 1.0)
                side = torch.sign(slot_mid - missing_idx.to(dtype=dtype))
                y_active = (slot_mid - row_center) * pitch_y + side * 0.5 * clearance

                book_height = float(getattr(self.cfg, name).spawn.size[1])
                z_active = torch.full((n,), shelf_z + 0.5 * book_height, device=self.device, dtype=dtype)
                hidden_z = torch.full((n,), shelf_z - 5.0, device=self.device, dtype=dtype)

                state[:, 0:3] = torch.stack(
                    (
                        torch.full((n,), mid_x, device=self.device, dtype=dtype),
                        torch.where(active, y_active, torch.zeros_like(y_active)),
                        torch.where(active, z_active, hidden_z),
                    ),
                    dim=-1,
                )
                state[:, 0:3] += self.scene.env_origins[env_ids_t]
                state[:, 3:7] = q
                state[:, 7:] = 0.0
                states.append((obj, state))

        append_pool_states(self._single_book_names(), self._single_book_slot_env, 1.0)
        append_pool_states(self._wide_book_names(), self._wide_book_start_slot_env, 2.0)
        return states

    def _write_side_book_states(self, env_ids_t: torch.Tensor) -> None:
        for obj, state in self._row_book_root_states(env_ids_t):
            obj.write_root_state_to_sim(state, env_ids=env_ids_t)

    def _align_gripper_to_sampled_slot(self, env_ids_t: torch.Tensor) -> None:
        """Approximate the classical planner by starting in front of the sampled gap."""
        target_pos_env = self._ee_tool_pos_env().clone()
        target_pos_env[env_ids_t, 1] = self._slot_center_y_env[env_ids_t]

        joint_pos_des = self._compute_ik_joint_targets_nominal_orientation(target_pos_env, env_ids_t)

        joint_pos = self.robot.data.joint_pos[env_ids_t].clone()
        joint_vel = self.robot.data.joint_vel[env_ids_t].clone()
        joint_pos[:, self._arm_joint_ids] = joint_pos_des[env_ids_t]
        joint_vel[:, self._arm_joint_ids] = 0.0
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids_t)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids_t)

        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)

        snapped_book_state = self._snap_book_to_measured_grasp(env_ids_t)
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)

        self._arm_hold_joint_pos[env_ids_t] = self.robot.data.joint_pos[env_ids_t][:, self._arm_joint_ids].clone()
        self.robot.set_joint_position_target(
            self._arm_hold_joint_pos[env_ids_t], joint_ids=self._arm_joint_ids, env_ids=env_ids_t
        )

        warmup = int(getattr(self.cfg, "reset_warmup_steps", 0))
        if warmup > 0:
            for _ in range(warmup):
                self.book.write_root_state_to_sim(snapped_book_state, env_ids=env_ids_t)
                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                self.scene.update(dt=self.physics_dt)

        ee_body_pos_env = self.robot.data.body_pos_w[env_ids_t, self._ee_body_idx] - self.scene.env_origins[env_ids_t]
        ee_body_quat_w = self.robot.data.body_quat_w[env_ids_t, self._ee_body_idx]
        offset_w = math_utils.quat_apply(ee_body_quat_w, self._ik_body_offset_pos_b[env_ids_t])
        self._target_pos_env[env_ids_t] = ee_body_pos_env + offset_w

        _, ee_quat_b = self._ee_pose_in_base()
        _, _, ee_yaw_b = math_utils.euler_xyz_from_quat(ee_quat_b[env_ids_t])
        self._target_yaw[env_ids_t] = ee_yaw_b

        corners0 = self._book_corners_env()[env_ids_t]
        front_x0 = corners0[..., 0].max(dim=-1).values
        rear_x0 = corners0[..., 0].min(dim=-1).values
        mouth = float(self._geom_mouth_x)
        self._prev_rear_to_mouth[env_ids_t] = (rear_x0 - mouth).detach()
        self._prev_front_to_back[env_ids_t] = (float(self.cfg.slot_x_back) - front_x0).detach()

    def _compute_task_metrics(self) -> dict[str, torch.Tensor]:
        corners = self._book_corners_env()

        front_x = corners[..., 0].max(dim=-1).values
        rear_x = corners[..., 0].min(dim=-1).values
        mouth = float(self._geom_mouth_x)

        rear_to_mouth = rear_x - mouth
        front_to_back = float(self.cfg.slot_x_back) - front_x

        cy = self._slot_center_y()
        lat_err = cy - self._book_pos_env()[:, 1]
        lat_extent = torch.abs(corners[..., 1] - cy.view(-1, 1)).max(dim=-1).values

        p = self._book_pos_env()
        z_target = self.cfg.shelf_top_z + self.cfg.shelf_thickness + 0.5 * self.cfg.book_size[1]
        z_err = p[:, 2] - float(z_target)

        yaw = _yaw_from_quat_wxyz(self.book.data.root_link_quat_w)
        yaw_err = _wrap_to_pi(yaw)

        v = self.book.data.root_link_vel_w
        book_lin_speed = torch.linalg.norm(v[:, 0:3], dim=-1)
        book_ang_speed = torch.linalg.norm(v[:, 3:6], dim=-1)

        return {
            "rear_to_mouth": rear_to_mouth,
            "front_to_back": front_to_back,
            "lat_err": lat_err,
            "lat_extent": lat_extent,
            "z_err": z_err,
            "yaw_err": yaw_err,
            "book_lin_speed": book_lin_speed,
            "book_ang_speed": book_ang_speed,
            "gripper_open": self._gripper_open01(),
        }

    def _book_supported_on_shelf(self, p_env: torch.Tensor, lowest_z: torch.Tensor) -> torch.Tensor:
        x0 = float(self.cfg.slot_x_open)
        x1 = float(self.cfg.slot_x_back)
        mid_x = 0.5 * (x0 + x1)
        depth_x = x1 - x0 + 0.06
        hx = 0.5 * depth_x + float(self.cfg.shelf_footprint_x_pad_m)

        bthick = self._neighbor_thick_y
        clearance_max = float(getattr(self.cfg, "slot_lateral_clearance_max", self.cfg.slot_lateral_clearance))
        row_center = 0.5 * float(int(getattr(self.cfg, "row_book_count", 10)) - 1)
        hy = 0.5 * clearance_max + (row_center + 1.5) * bthick

        deck_z = float(self.cfg.shelf_top_z + self.cfg.shelf_thickness)
        slack = float(self.cfg.book_on_shelf_z_slack_m)

        in_xy = (
            (p_env[:, 0] >= mid_x - hx)
            & (p_env[:, 0] <= mid_x + hx)
            & (p_env[:, 1] >= -hy)
            & (p_env[:, 1] <= hy)
        )
        lowest_on_deck = lowest_z >= deck_z - slack
        return in_xy & lowest_on_deck

    def _setup_scene(self):
        x0 = self.cfg.slot_x_open
        x1 = self.cfg.slot_x_back
        mid_x = 0.5 * (x0 + x1)
        depth_x = x1 - x0 + 0.06
        nb = _neighbor_book_dims(self.cfg)
        bheight, bthick = float(nb[1]), float(nb[2])

        clearance_max = float(getattr(self.cfg, "slot_lateral_clearance_max", self.cfg.slot_lateral_clearance))
        row_center = 0.5 * float(int(getattr(self.cfg, "row_book_count", 10)) - 1)
        bookend_y = 0.5 * clearance_max + (row_center + 1.0) * bthick
        shelf_half_y = bookend_y + 0.5 * bthick

        wt = self.cfg.wall_thickness
        z_book = bheight * 0.5
        qw, qx, qy, qz = self.cfg.book_standing_quat
        standing_quat = (float(qw), float(qx), float(qy), float(qz))

        kin = RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True)
        col = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
        wood = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.45, 0.35))

        base = "/World/envs/env_0/Bookshelf"

        ground_cfg = sim_utils.MeshCuboidCfg(
            size=(3.0, 3.0, 0.02),
            rigid_props=kin,
            collision_props=col,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.25, 0.25)),
        )
        ground_cfg.func(
            "/World/envs/env_0/Ground",
            ground_cfg,
            translation=(0.0, 0.0, -0.01),
            orientation=(1.0, 0.0, 0.0, 0.0),
        )

        shelf_width = 2.0 * shelf_half_y
        shelf_thick = float(self.cfg.shelf_thickness)

        shelf_cfg = sim_utils.MeshCuboidCfg(
            size=(depth_x, shelf_width, shelf_thick),
            rigid_props=kin,
            collision_props=col,
            visual_material=wood,
        )
        shelf_cfg.func(
            f"{base}/Shelf",
            shelf_cfg,
            translation=(mid_x, self.cfg.slot_center_y, self.cfg.shelf_top_z + shelf_thick * 0.5),
            orientation=(1.0, 0.0, 0.0, 0.0),
        )

        top_clearance = float(getattr(self.cfg, "top_shelf_clearance", 0.005))
        top_center_z = self.cfg.shelf_top_z + shelf_thick + bheight + top_clearance + 0.5 * shelf_thick
        shelf_cfg.func(
            f"{base}/TopShelf",
            shelf_cfg,
            translation=(mid_x, self.cfg.slot_center_y, top_center_z),
            orientation=(1.0, 0.0, 0.0, 0.0),
        )

        back_x = x1 + wt * 0.5
        panel_cfg = sim_utils.MeshCuboidCfg(
            size=(wt, shelf_width, bheight + 0.1),
            rigid_props=kin,
            collision_props=col,
            visual_material=wood,
        )
        panel_cfg.func(
            f"{base}/BackPanel",
            panel_cfg,
            translation=(back_x, self.cfg.slot_center_y, self.cfg.shelf_top_z + shelf_thick + z_book),
            orientation=(1.0, 0.0, 0.0, 0.0),
        )

        bookend_cfg = sim_utils.MeshCuboidCfg(
            size=(float(nb[0]), bheight, bthick),
            rigid_props=kin,
            collision_props=col,
            visual_material=wood,
        )
        for name, y in (("LeftBookend", -bookend_y), ("RightBookend", bookend_y)):
            bookend_cfg.func(
                f"{base}/{name}",
                bookend_cfg,
                translation=(mid_x, y, self.cfg.shelf_top_z + shelf_thick + z_book),
                orientation=standing_quat,
            )

        robot = Articulation(self.cfg.robot)
        book = RigidObject(self.cfg.book)
        side_books = {name: RigidObject(getattr(self.cfg, name)) for name in self._row_book_names()}

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = robot
        self.scene.rigid_objects["book"] = book
        self.scene.rigid_objects.update(side_books)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        m = self._compute_task_metrics()
        self._step_metrics = m

        mode_before = self._mode.clone()

        accepted_release = (mode_before == _MODE_INSERT) & self._release_request
        self._release_step_buf = torch.where(
            accepted_release & (self._release_step_buf < 0),
            self.episode_length_buf.clone(),
            self._release_step_buf,
        )

        scripted_old = mode_before == _MODE_SCRIPTED
        self._script_step_buf = torch.where(scripted_old, self._script_step_buf + 1, self._script_step_buf)

        script_total = int(self.cfg.script_open_steps) + int(self.cfg.script_retreat_steps) + int(self.cfg.script_close_steps)
        script_done = scripted_old & (self._script_step_buf >= script_total)

        self._mode = torch.where(accepted_release, torch.full_like(self._mode, _MODE_SCRIPTED), self._mode)
        self._script_step_buf = torch.where(accepted_release, torch.zeros_like(self._script_step_buf), self._script_step_buf)

        self._mode = torch.where(script_done, torch.full_like(self._mode, _MODE_PUSH), self._mode)
        self._script_step_buf = torch.where(script_done, torch.zeros_like(self._script_step_buf), self._script_step_buf)
        self._push_start_step_buf = torch.where(
            script_done & (self._push_start_step_buf < 0),
            self.episode_length_buf.clone(),
            self._push_start_step_buf,
        )

        lat_extent = m["lat_extent"]
        z_err = m["z_err"]
        yaw_err = m["yaw_err"]
        front_to_back = m["front_to_back"]
        rear_to_mouth = m["rear_to_mouth"]
        yaw_e = torch.abs(yaw_err)
        upright = self._upright_ok()

        inner_half = 0.5 * (self._neighbor_thick_y + self._slot_lateral_clearance_env)
        lat_limit = inner_half - float(self.cfg.success_lateral_margin)
        lat_eps = float(self.cfg.success_lateral_extent_eps_m)
        front_eps = float(self.cfg.success_front_clear_eps_m)

        lat_ok = lat_extent <= (lat_limit + lat_eps)
        rear_ok = (rear_to_mouth >= float(self.cfg.success_rear_to_mouth_min)) & (
            rear_to_mouth <= float(self.cfg.success_rear_to_mouth_max)
        )
        front_ok = (front_to_back >= float(self.cfg.success_front_clear_min) - front_eps) & (
            front_to_back <= float(self.cfg.success_front_clear_max) + front_eps
        )
        z_ok = torch.abs(z_err) < float(self.cfg.success_z_thresh)
        yaw_ok = yaw_e < float(self.cfg.success_yaw_thresh)

        if float(self.cfg.success_max_lin_vel) > 0.0:
            stable_lin = m["book_lin_speed"] < float(self.cfg.success_max_lin_vel)
        else:
            stable_lin = torch.ones_like(lat_ok)

        if float(self.cfg.success_max_ang_vel) > 0.0:
            stable_ang = m["book_ang_speed"] < float(self.cfg.success_max_ang_vel)
        else:
            stable_ang = torch.ones_like(lat_ok)

        ready = self.episode_length_buf > int(self.cfg.min_steps_before_success)
        success_gate = (
            (mode_before == _MODE_PUSH)
            & rear_ok
            & front_ok
            & lat_ok
            & z_ok
            & yaw_ok
            & upright
            & stable_lin
            & stable_ang
            & ready
        )

        self._success_steps_buf = torch.where(
            success_gate, self._success_steps_buf + 1, torch.zeros_like(self._success_steps_buf)
        )
        success = self._success_steps_buf >= int(self.cfg.success_steps)

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        p = self._book_pos_env()
        corners_w = self._book_corners_env()
        lowest_z = corners_w[..., 2].min(dim=-1).values
        floor_z = float(self.cfg.book_floor_lowest_z_thresh)
        book_touches_ground = lowest_z < floor_z
        on_shelf = self._book_supported_on_shelf(p, lowest_z)
        book_dropped_to_ground = book_touches_ground & ~on_shelf

        if bool(self.cfg.enable_failure_terminations):
            oob = (torch.abs(p[:, 0]) > self.cfg.max_abs_xy) | (torch.abs(p[:, 1]) > self.cfg.max_abs_xy)
            fell = p[:, 2] < self.cfg.fell_height_thresh
            terminated = success | book_dropped_to_ground | oob | fell
        else:
            oob = torch.zeros_like(success)
            fell = torch.zeros_like(success)
            terminated = success | book_dropped_to_ground

        done = terminated | time_out
        depth_ok = rear_ok & front_ok
        stable_ok = stable_lin & stable_ang
        failure_code = torch.full((self.num_envs,), _DONE_NONE, dtype=torch.long, device=self.device)
        failure_code = torch.where(done & success, torch.full_like(failure_code, _DONE_SUCCESS), failure_code)
        failure_code = torch.where(
            done & ~success & book_dropped_to_ground, torch.full_like(failure_code, _DONE_DROP), failure_code
        )
        failure_code = torch.where(done & ~success & oob, torch.full_like(failure_code, _DONE_OOB), failure_code)
        failure_code = torch.where(done & ~success & fell, torch.full_like(failure_code, _DONE_FELL), failure_code)
        failure_code = torch.where(
            done & ~success & ~(mode_before == _MODE_PUSH), torch.full_like(failure_code, _DONE_NOT_PUSH), failure_code
        )
        failure_code = torch.where(
            done & ~success & (mode_before == _MODE_PUSH) & ~depth_ok,
            torch.full_like(failure_code, _DONE_DEPTH),
            failure_code,
        )
        failure_code = torch.where(
            done & ~success & (mode_before == _MODE_PUSH) & depth_ok & ~lat_ok,
            torch.full_like(failure_code, _DONE_LATERAL),
            failure_code,
        )
        failure_code = torch.where(
            done & ~success & (mode_before == _MODE_PUSH) & depth_ok & lat_ok & ~z_ok,
            torch.full_like(failure_code, _DONE_Z),
            failure_code,
        )
        failure_code = torch.where(
            done & ~success & (mode_before == _MODE_PUSH) & depth_ok & lat_ok & z_ok & ~yaw_ok,
            torch.full_like(failure_code, _DONE_YAW),
            failure_code,
        )
        failure_code = torch.where(
            done & ~success & (mode_before == _MODE_PUSH) & depth_ok & lat_ok & z_ok & yaw_ok & ~upright,
            torch.full_like(failure_code, _DONE_UPRIGHT),
            failure_code,
        )
        failure_code = torch.where(
            done & ~success & (mode_before == _MODE_PUSH) & depth_ok & lat_ok & z_ok & yaw_ok & upright & ~stable_ok,
            torch.full_like(failure_code, _DONE_UNSTABLE),
            failure_code,
        )
        failure_code = torch.where(
            done & ~success & time_out & (failure_code == _DONE_NONE),
            torch.full_like(failure_code, _DONE_TIMEOUT),
            failure_code,
        )

        push_steps = torch.where(
            self._push_start_step_buf >= 0,
            self.episode_length_buf - self._push_start_step_buf,
            torch.full_like(self.episode_length_buf, -1),
        )
        self.extras["episode_metric_done"] = done
        self.extras["episode_metric_slot_center_y"] = self._slot_center_y_env
        self.extras["episode_metric_slot_clearance"] = self._slot_lateral_clearance_env
        self.extras["episode_metric_missing_book_index"] = self._missing_book_index_env
        self.extras["episode_metric_success"] = success
        self.extras["episode_metric_failure_code"] = failure_code
        self.extras["episode_metric_final_lat_err"] = torch.abs(m["lat_err"])
        self.extras["episode_metric_final_z_err"] = torch.abs(z_err)
        self.extras["episode_metric_final_yaw_err_deg"] = torch.rad2deg(yaw_e)
        self.extras["episode_metric_final_rear_to_mouth"] = rear_to_mouth
        self.extras["episode_metric_final_front_to_back"] = front_to_back
        self.extras["episode_metric_release_step"] = self._release_step_buf
        self.extras["episode_metric_push_steps"] = push_steps
        self.extras["episode_metric_mode_at_done"] = mode_before

        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids_t = self._env_ids
        else:
            env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        n = int(env_ids_t.numel())
        cmin = float(getattr(self.cfg, "slot_lateral_clearance_min", self.cfg.slot_lateral_clearance))
        cmax = float(getattr(self.cfg, "slot_lateral_clearance_max", self.cfg.slot_lateral_clearance))

        self._ensure_randomization_buffers()

        self._slot_lateral_clearance_env[env_ids_t] = sample_uniform(cmin, cmax, (n,), self.device)
        row_count = int(getattr(self.cfg, "row_book_count", 10))
        forced_missing = int(getattr(self.cfg, "forced_missing_book_index", -1))
        if forced_missing >= 0:
            if forced_missing >= row_count:
                raise ValueError(f"forced_missing_book_index={forced_missing} must be in [0, {row_count - 1}]")
            self._missing_book_index_env[env_ids_t] = forced_missing
        else:
            self._missing_book_index_env[env_ids_t] = torch.randint(0, row_count, (n,), device=self.device)
        self._sample_row_layout(env_ids_t)

        nb = _neighbor_book_dims(self.cfg)
        pitch_y = float(nb[2])
        row_center = 0.5 * float(row_count - 1)
        self._slot_center_y_env[env_ids_t] = (
            self._missing_book_index_env[env_ids_t].to(dtype=torch.float32) - row_center
        ) * pitch_y

        self._write_side_book_states(env_ids_t)

        super()._reset_idx(env_ids_t)

        self._write_side_book_states(env_ids_t)
        self._align_gripper_to_sampled_slot(env_ids_t)
