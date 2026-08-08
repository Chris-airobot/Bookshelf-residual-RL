#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bookshelf residual-RL environment.

This task uses v5-style release mechanics and reset randomization, but composes
the Cartesian motion command as:

    delta_final = delta_nominal + delta_policy_residual

The PPO policy still owns the release trigger.  Once PPO requests release, the
environment executes the usual scripted release transition and enters push mode.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import os
import sys
import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import math as math_utils

from .bookshelf_residual_env_cfg import BookshelfEnvCfg
from .bookshelf_env_v4 import _MODE_INSERT, _MODE_PUSH, _MODE_SCRIPTED, _wrap_to_pi
from .bookshelf_env_v5 import BookshelfEnv as BookshelfEnvV5


class BookshelfEnv(BookshelfEnvV5):
    """Randomized bookshelf insertion with a nominal controller plus learned residual actions."""

    cfg: BookshelfEnvCfg

    def __init__(self, cfg: BookshelfEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._target_book_marker_path = "/World/envs/env_0/V6TargetBook50"
        self._target_ee_marker_path = "/World/Visuals/V6TargetEEFrame"
        self._current_ee_marker_path = "/World/Visuals/V6CurrentEEFrame"
        self._debug_rrt_plan = None
        self._debug_rrt_plan_step = 0
        self._debug_rrt_failed = False
        self._debug_rrt_warned = False
        self._debug_curobo_planner = None
        self._debug_curobo_plan = None
        self._debug_curobo_plan_step = 0
        self._debug_curobo_failed = False
        self._debug_curobo_warned = False
        self._debug_preinsert_hold_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._debug_position_only_ee_quat_b = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float32)
        self._debug_position_only_ee_quat_b[:, 0] = 1.0
        self._debug_tool_to_book_transform_frozen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._cleanup_legacy_debug_visuals()

    def _cleanup_legacy_debug_visuals(self) -> None:
        for prim_path in (
            "/Visuals/V6TargetEEFrame",
            "/Visuals/V6CurrentEEFrame",
            "/World/envs/env_0/V6TargetBook50",
            "/V6Target",
        ):
            if sim_utils.is_prim_path_valid(prim_path):
                sim_utils.delete_prim(prim_path)

    def _target_book_marker_pose(self, env_id: int = 0) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        frac = float(self.cfg.nominal_release_inside_fraction)
        book_depth_x = float(self.cfg.book_size[0])
        book_height_z = float(self.cfg.book_size[1])
        mouth_x = float(self._geom_mouth_x)
        center_y = float(self._slot_center_y()[env_id].item())
        pos_env = (
            mouth_x + (frac - 0.5) * book_depth_x,
            center_y,
            float(self.cfg.shelf_top_z + self.cfg.shelf_thickness) + 0.5 * book_height_z,
        )
        return pos_env, tuple(float(v) for v in self.cfg.book_standing_quat)

    def _target_book_pose_tensors(self, inside_fraction: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        frac = float(self.cfg.nominal_release_inside_fraction if inside_fraction is None else inside_fraction)
        book_depth_x = float(self.cfg.book_size[0])
        book_height_z = float(self.cfg.book_size[1])
        mouth_x = float(self._geom_mouth_x)

        pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        pos[:, 0] = mouth_x + (frac - 0.5) * book_depth_x
        pos[:, 1] = self._slot_center_y()
        pos[:, 2] = float(self.cfg.shelf_top_z + self.cfg.shelf_thickness) + 0.5 * book_height_z

        quat = torch.tensor(self.cfg.book_standing_quat, device=self.device, dtype=torch.float32)
        quat = quat.view(1, 4).expand(self.num_envs, 4)
        return pos, quat

    def _create_target_book_marker(self) -> None:
        if not bool(getattr(self.cfg, "show_target_book_marker", True)):
            return
        if not hasattr(self, "_target_book_marker_path"):
            self._target_book_marker_path = "/World/envs/env_0/V6TargetBook50"
        if sim_utils.is_prim_path_valid(self._target_book_marker_path):
            sim_utils.delete_prim(self._target_book_marker_path)
        pos, quat = self._target_book_marker_pose(0)
        sim_utils.create_prim(
            self._target_book_marker_path,
            "Xform",
            translation=pos,
            orientation=quat,
        )
        marker_cfg = sim_utils.MeshCuboidCfg(
            size=tuple(float(v) for v in self.cfg.book_size),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.2)),
        )
        marker_cfg.func(f"{self._target_book_marker_path}/geometry", marker_cfg)

    def _create_target_ee_marker(self) -> None:
        if not bool(getattr(self.cfg, "show_target_ee_marker", True)):
            return
        if not hasattr(self, "_target_ee_marker"):
            marker_cfg = FRAME_MARKER_CFG.copy()
            marker_cfg.prim_path = getattr(self, "_target_ee_marker_path", "/World/Visuals/V6TargetEEFrame")
            marker_scale = max(0.20, float(getattr(self.cfg, "target_ee_marker_axis_length", 0.20)))
            marker_cfg.markers["frame"].scale = (marker_scale, marker_scale, marker_scale)
            if sim_utils.is_prim_path_valid(marker_cfg.prim_path):
                sim_utils.delete_prim(marker_cfg.prim_path)
            self._target_ee_marker = VisualizationMarkers(marker_cfg)
            self._target_ee_marker.set_visibility(True)

        target_pos, target_quat = self._planned_tool_release_pose_quat()
        target_pos_w = target_pos[0:1] + self.scene.env_origins[0:1]
        self._target_ee_marker.visualize(target_pos_w, target_quat[0:1])

    def _create_current_ee_marker(self) -> None:
        if not bool(getattr(self.cfg, "show_current_ee_marker", True)):
            return
        if not hasattr(self, "_current_ee_marker"):
            marker_cfg = FRAME_MARKER_CFG.copy()
            marker_cfg.prim_path = getattr(self, "_current_ee_marker_path", "/World/Visuals/V6CurrentEEFrame")
            marker_scale = max(0.14, 0.7 * float(getattr(self.cfg, "target_ee_marker_axis_length", 0.20)))
            marker_cfg.markers["frame"].scale = (marker_scale, marker_scale, marker_scale)
            if sim_utils.is_prim_path_valid(marker_cfg.prim_path):
                sim_utils.delete_prim(marker_cfg.prim_path)
            self._current_ee_marker = VisualizationMarkers(marker_cfg)
            self._current_ee_marker.set_visibility(True)

        current_pos_w = self._ee_tool_pos_env()[0:1] + self.scene.env_origins[0:1]
        current_quat_w = self.robot.data.body_quat_w[0:1, self._ee_body_idx]
        self._current_ee_marker.visualize(current_pos_w, current_quat_w)

    def _refresh_debug_markers(self) -> None:
        if bool(getattr(self.cfg, "show_target_book_marker", True)) and not sim_utils.is_prim_path_valid(
            getattr(self, "_target_book_marker_path", "/World/envs/env_0/V6TargetBook50")
        ):
            self._create_target_book_marker()
        if bool(getattr(self.cfg, "show_target_ee_marker", True)):
            self._create_target_ee_marker()
        if bool(getattr(self.cfg, "show_current_ee_marker", True)):
            self._create_current_ee_marker()

    def _residual_curriculum_clearance_range(self) -> tuple[float, float]:
        if not bool(getattr(self.cfg, "enable_residual_clearance_curriculum", False)):
            return (
                float(getattr(self.cfg, "slot_lateral_clearance_min", self.cfg.slot_lateral_clearance)),
                float(getattr(self.cfg, "slot_lateral_clearance_max", self.cfg.slot_lateral_clearance)),
            )

        stage = self._residual_curriculum_stage()
        bounds = (
            getattr(self.cfg, "residual_curriculum_clearance_1", (0.010, 0.010)),
            getattr(self.cfg, "residual_curriculum_clearance_2", (0.006, 0.006)),
            getattr(self.cfg, "residual_curriculum_clearance_3", (0.004, 0.006)),
            getattr(self.cfg, "residual_curriculum_clearance_final", (0.002, 0.002)),
        )[stage]
        return float(bounds[0]), float(bounds[1])

    def _residual_curriculum_progress(self) -> float:
        total = max(1.0, float(getattr(self.cfg, "residual_curriculum_total_steps", 1)))
        return min(1.0, max(0.0, float(getattr(self, "common_step_counter", 0)) / total))

    def _residual_curriculum_stage(self) -> int:
        progress = self._residual_curriculum_progress()
        if progress < float(getattr(self.cfg, "residual_curriculum_1_frac", 0.10)):
            return 0
        if progress < float(getattr(self.cfg, "residual_curriculum_2_frac", 0.20)):
            return 1
        if progress < float(getattr(self.cfg, "residual_curriculum_3_frac", 0.30)):
            return 2
        return 3

    def _residual_action_scale(self) -> float:
        if not bool(getattr(self.cfg, "enable_residual_action_scale_curriculum", False)):
            return 1.0
        stage = self._residual_curriculum_stage()
        return float(
            (
                getattr(self.cfg, "residual_curriculum_action_scale_1", 0.30),
                getattr(self.cfg, "residual_curriculum_action_scale_2", 0.50),
                getattr(self.cfg, "residual_curriculum_action_scale_3", 0.75),
                getattr(self.cfg, "residual_curriculum_action_scale_final", 1.00),
            )[stage]
        )

    def _apply_residual_reset_curriculum(self) -> None:
        if not bool(getattr(self.cfg, "enable_residual_reset_curriculum", False)):
            return
        stage = self._residual_curriculum_stage()
        joint, x_jitter, y_jitter, z_jitter, yaw_jitter = (
            getattr(self.cfg, "residual_curriculum_reset_1"),
            getattr(self.cfg, "residual_curriculum_reset_2"),
            getattr(self.cfg, "residual_curriculum_reset_3"),
            getattr(self.cfg, "residual_curriculum_reset_final"),
        )[stage]
        self.cfg.reset_arm_joint_pos_noise = float(joint)
        self.cfg.book_grasp_x_jitter = float(x_jitter)
        self.cfg.book_grasp_y_jitter = float(y_jitter)
        self.cfg.book_grasp_z_jitter = float(z_jitter)
        self.cfg.book_grasp_yaw_jitter = float(yaw_jitter)

    def _nominal_release_assist_enabled(self) -> bool:
        if not bool(getattr(self.cfg, "enable_nominal_release_assist", False)):
            return False
        return self._residual_curriculum_progress() < float(getattr(self.cfg, "nominal_release_assist_until_frac", 0.30))

    def _reset_idx(self, env_ids: Sequence[int] | None):
        cmin, cmax = self._residual_curriculum_clearance_range()
        self.cfg.slot_lateral_clearance_min = cmin
        self.cfg.slot_lateral_clearance_max = cmax
        self._apply_residual_reset_curriculum()
        super()._reset_idx(env_ids)
        env_ids_t = self._env_ids if env_ids is None else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if bool(getattr(self.cfg, "debug_start_from_default_grasp_pose", False)):
            self._reset_to_default_grasp_start(env_ids_t)
        if bool(getattr(self.cfg, "debug_omit_bookshelf_obstacles", False)):
            self._omit_bookshelf_obstacles(env_ids_t)
        self._capture_fixed_tool_to_book_transform(env_ids_t)
        self._clear_debug_rrt_plan()
        self._clear_debug_curobo_plan()
        self._debug_preinsert_hold_buf[env_ids_t] = 0
        if bool(getattr(self.cfg, "debug_spawn_at_target_tool_pose", False)):
            self._spawn_at_planned_tool_pose(env_ids_t)
        _, ee_quat_b = self._ee_pose_in_base()
        self._debug_position_only_ee_quat_b[env_ids_t] = ee_quat_b[env_ids_t].detach().clone()
        reset_env0 = env_ids is None
        if env_ids is not None:
            reset_env0 = bool(torch.any(torch.as_tensor(env_ids, device=self.device, dtype=torch.long) == 0).item())
        if reset_env0:
            self._create_target_book_marker()
            self._create_target_ee_marker()
            self._create_current_ee_marker()
            if bool(getattr(self.cfg, "debug_print_sampled_grasp_joints", False)):
                self._print_env0_joint_values("[Bookshelf v6] sampled grasp joints")

    def _print_env0_joint_values(self, label: str) -> None:
        joint_pos = self.robot.data.joint_pos[0].detach().cpu()
        values = {
            name: float(joint_pos[idx].item())
            for idx, name in enumerate(self.robot.joint_names)
            if name.startswith("panda_joint") or name.startswith("panda_finger_joint")
        }
        print(f"{label}: {values}")

    def _reset_to_default_grasp_start(self, env_ids_t: torch.Tensor) -> None:
        robot_default = self.robot.data.default_root_state[env_ids_t].clone()
        robot_default[:, 0:3] += self.scene.env_origins[env_ids_t]
        self.robot.write_root_state_to_sim(robot_default, env_ids=env_ids_t)

        joint_pos = self.robot.data.default_joint_pos[env_ids_t].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids_t].clone()
        start_joint_pos = getattr(self.cfg, "debug_start_joint_pos", {})
        for joint_name, value in start_joint_pos.items():
            if joint_name not in self.robot.joint_names:
                continue
            joint_id = self.robot.joint_names.index(joint_name)
            joint_pos[:, joint_id] = float(value)
            joint_vel[:, joint_id] = 0.0
        start_joint_names = set(start_joint_pos.keys())
        if len(self._finger_joint_ids) > 0 and not any(
            self.robot.joint_names[joint_id] in start_joint_names for joint_id in self._finger_joint_ids
        ):
            joint_pos[:, self._finger_joint_ids] = float(self.cfg.gripper_closed_joint_pos)
            joint_vel[:, self._finger_joint_ids] = 0.0

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
        for _ in range(max(0, warmup)):
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

        self._success_steps_buf[env_ids_t] = 0
        self._mode[env_ids_t] = _MODE_INSERT
        self._mode_start[env_ids_t] = _MODE_INSERT
        self._script_step_buf[env_ids_t] = 0
        self._release_request[env_ids_t] = False
        self._release_step_buf[env_ids_t] = -1
        self._push_start_step_buf[env_ids_t] = -1

    def _omit_bookshelf_obstacles(self, env_ids_t: torch.Tensor) -> None:
        for env_id in env_ids_t.tolist():
            shelf_path = f"/World/envs/env_{env_id}/Bookshelf"
            if sim_utils.is_prim_path_valid(shelf_path):
                sim_utils.delete_prim(shelf_path)

        hidden_pos = torch.zeros((env_ids_t.numel(), 3), device=self.device, dtype=torch.float32)
        hidden_pos[:, 2] = -5.0
        hidden_pos = hidden_pos + self.scene.env_origins[env_ids_t]

        for name in self._row_book_names():
            if name not in self.scene.rigid_objects:
                continue
            obj = self.scene.rigid_objects[name]
            state = obj.data.default_root_state[env_ids_t].clone()
            state[:, 0:3] = hidden_pos
            state[:, 7:] = 0.0
            obj.write_root_state_to_sim(state, env_ids=env_ids_t)

        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)

    def _capture_fixed_tool_to_book_transform(self, env_ids_t: torch.Tensor) -> None:
        if not hasattr(self, "_book_offset_tool"):
            self._book_offset_tool = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
            self._book_rel_quat_tool = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float32)
            self._book_rel_quat_tool[:, 0] = 1.0

        if bool(getattr(self.cfg, "debug_freeze_tool_to_book_transform", False)):
            env_ids_t = env_ids_t[~self._debug_tool_to_book_transform_frozen[env_ids_t]]
            if env_ids_t.numel() == 0:
                return

        ee_body_pos_env = self.robot.data.body_pos_w[:, self._ee_body_idx] - self.scene.env_origins
        ee_body_quat_w = self.robot.data.body_quat_w[:, self._ee_body_idx]
        tool_pos_env = ee_body_pos_env + math_utils.quat_apply(ee_body_quat_w, self._ik_body_offset_pos_b)
        book_pos_env = self._book_pos_env()
        book_quat_w = self.book.data.root_link_quat_w

        self._book_offset_tool[env_ids_t] = math_utils.quat_apply_inverse(
            ee_body_quat_w[env_ids_t], book_pos_env[env_ids_t] - tool_pos_env[env_ids_t]
        )
        self._book_rel_quat_tool[env_ids_t] = math_utils.quat_mul(
            math_utils.quat_inv(ee_body_quat_w[env_ids_t]), book_quat_w[env_ids_t]
        )
        if bool(getattr(self.cfg, "debug_freeze_tool_to_book_transform", False)):
            self._debug_tool_to_book_transform_frozen[env_ids_t] = True

    def _write_book_from_fixed_tool_transform(self, env_ids_t: torch.Tensor) -> None:
        if not hasattr(self, "_book_offset_tool"):
            return

        ee_body_pos_env = self.robot.data.body_pos_w[:, self._ee_body_idx] - self.scene.env_origins
        ee_body_quat_w = self.robot.data.body_quat_w[:, self._ee_body_idx]
        tool_pos_env = ee_body_pos_env + math_utils.quat_apply(ee_body_quat_w, self._ik_body_offset_pos_b)

        book_pos_env = tool_pos_env + math_utils.quat_apply(ee_body_quat_w, self._book_offset_tool)
        book_quat_w = math_utils.quat_mul(ee_body_quat_w, self._book_rel_quat_tool)

        book_state = self.book.data.root_state_w[env_ids_t].clone()
        book_state[:, 0:3] = book_pos_env[env_ids_t] + self.scene.env_origins[env_ids_t]
        book_state[:, 3:7] = book_quat_w[env_ids_t]
        book_state[:, 7:] = 0.0
        self.book.write_root_state_to_sim(book_state, env_ids=env_ids_t)

    def _spawn_at_planned_tool_pose(self, env_ids_t: torch.Tensor) -> None:
        spawn_inside_fraction = float(getattr(self.cfg, "debug_spawn_inside_fraction", 0.0))
        target_tool_pos, target_tool_quat = self._planned_tool_release_pose_quat(spawn_inside_fraction)
        _, target_tool_quat_b = math_utils.subtract_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.robot.data.root_pos_w,
            target_tool_quat,
        )

        spawned_with_curobo = False
        if bool(getattr(self.cfg, "debug_spawn_with_curobo", False)) and self.num_envs == 1:
            was_curobo_enabled = bool(getattr(self.cfg, "debug_use_curobo_planner", False))
            self.cfg.debug_use_curobo_planner = True
            self._make_debug_curobo_plan(inside_fraction=spawn_inside_fraction)
            self.cfg.debug_use_curobo_planner = was_curobo_enabled
            if self._debug_curobo_plan is not None and self._debug_curobo_plan.shape[0] > 0:
                joint_pos = self.robot.data.joint_pos[env_ids_t].clone()
                joint_vel = self.robot.data.joint_vel[env_ids_t].clone()
                joint_pos[:, self._arm_joint_ids] = self._debug_curobo_plan[-1:].expand(env_ids_t.numel(), -1)
                joint_vel[:, self._arm_joint_ids] = 0.0
                if len(self._finger_joint_ids) > 0:
                    joint_pos[:, self._finger_joint_ids] = float(self.cfg.gripper_closed_joint_pos)
                    joint_vel[:, self._finger_joint_ids] = 0.0

                self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids_t)
                self.robot.set_joint_position_target(joint_pos, env_ids=env_ids_t)
                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                self.scene.update(dt=self.physics_dt)
                spawned_with_curobo = True
            self._clear_debug_curobo_plan()

        iters = 0 if spawned_with_curobo else max(1, int(getattr(self.cfg, "debug_spawn_ik_iters", 80)))
        for _ in range(iters):
            joint_pos_des = self._compute_ik_joint_targets_from_tool_quat(target_tool_pos, target_tool_quat_b)
            joint_pos = self.robot.data.joint_pos[env_ids_t].clone()
            joint_vel = self.robot.data.joint_vel[env_ids_t].clone()
            joint_pos[:, self._arm_joint_ids] = joint_pos_des[env_ids_t]
            joint_vel[:, self._arm_joint_ids] = 0.0

            if len(self._finger_joint_ids) > 0:
                joint_pos[:, self._finger_joint_ids] = float(self.cfg.gripper_closed_joint_pos)
                joint_vel[:, self._finger_joint_ids] = 0.0

            self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids_t)
            self.robot.set_joint_position_target(joint_pos, env_ids=env_ids_t)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

        # Keep the grasp physically consistent with wherever IK actually placed the tool.
        # The green marker remains the desired book pose; the controller should move
        # the grasped book there instead of starting from a teleported book pose.
        self._write_book_from_fixed_tool_transform(env_ids_t)
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)

        self._arm_hold_joint_pos[env_ids_t] = self.robot.data.joint_pos[env_ids_t][:, self._arm_joint_ids].clone()
        self.robot.set_joint_position_target(
            self._arm_hold_joint_pos[env_ids_t], joint_ids=self._arm_joint_ids, env_ids=env_ids_t
        )

    def _book_tilt_x(self) -> torch.Tensor:
        tilt_x, _ = self._book_upright_tilt_obs()
        return tilt_x

    def _nominal_orientation_aligned(self, metrics: dict[str, torch.Tensor]) -> torch.Tensor:
        tilt_x = self._book_tilt_x()
        return (
            (torch.abs(metrics["yaw_err"]) < float(self.cfg.nominal_align_yaw_thresh))
            & (torch.abs(tilt_x) < float(self.cfg.nominal_align_tilt_x_thresh))
        )

    def _nominal_position_aligned(self, metrics: dict[str, torch.Tensor]) -> torch.Tensor:
        return (
            (torch.abs(metrics["lat_err"]) < float(self.cfg.nominal_align_lat_thresh))
            & (torch.abs(metrics["z_err"]) < float(self.cfg.nominal_align_z_thresh))
        )

    def _nominal_alignment_mask(self, metrics: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._nominal_orientation_aligned(metrics) & self._nominal_position_aligned(metrics)

    def _planned_tool_release_pose_quat(self, inside_fraction: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        desired_book_pos, desired_book_quat = self._target_book_pose_tensors(inside_fraction)

        if not hasattr(self, "_book_offset_tool"):
            self._capture_fixed_tool_to_book_transform(self._env_ids)

        target_quat = math_utils.quat_mul(desired_book_quat, math_utils.quat_inv(self._book_rel_quat_tool))
        desired_tool_pos = desired_book_pos - math_utils.quat_apply(target_quat, self._book_offset_tool)
        return desired_tool_pos, target_quat

    def _planned_hand_release_pose_quat(self, inside_fraction: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        target_tool_pos, target_hand_quat = self._planned_tool_release_pose_quat(inside_fraction)
        target_hand_pos = target_tool_pos - math_utils.quat_apply(target_hand_quat, self._ik_body_offset_pos_b)
        return target_hand_pos, target_hand_quat

    def _planned_tool_release_pose(self, inside_fraction: float | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        desired_tool_pos, target_quat = self._planned_tool_release_pose_quat(inside_fraction)
        _, target_pitch, target_yaw = math_utils.euler_xyz_from_quat(target_quat)
        return desired_tool_pos, target_yaw, target_pitch

    def _compute_ik_joint_targets_from_tool_quat(
        self, target_pos_env: torch.Tensor, target_quat_b: torch.Tensor
    ) -> torch.Tensor:
        self._target_pos_env[:] = target_pos_env

        offset_des_b = math_utils.quat_apply(target_quat_b, self._ik_body_offset_pos_b)
        body_pos_des_b = target_pos_env - offset_des_b

        self._ik_cmd[:, 0:3] = body_pos_des_b
        self._ik_cmd[:, 3:7] = target_quat_b
        self._ik.set_command(self._ik_cmd)

        ee_pos_b, ee_quat_b = self._ee_pose_in_base()
        jacobian = self.robot.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, self._jacobi_joint_ids]
        joint_pos = self.robot.data.joint_pos[:, self._arm_joint_ids]
        return self._ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

    @staticmethod
    def _interpolate_cspace_path(path: np.ndarray, max_cspace_dist: float) -> np.ndarray:
        if path is None or path.shape[0] == 0:
            return np.empty((0, 0), dtype=np.float32)
        if path.shape[0] == 1:
            return path.astype(np.float32)

        max_cspace_dist = max(1.0e-4, float(max_cspace_dist))
        interpolated = []
        for i in range(path.shape[0] - 1):
            n_pts = int(np.ceil(np.amax(np.abs(path[i + 1] - path[i])) / max_cspace_dist))
            n_pts = max(1, n_pts)
            interpolated.append(np.linspace(path[i], path[i + 1], num=n_pts, endpoint=False))
        interpolated.append(path[np.newaxis, -1, :])
        return np.concatenate(interpolated, axis=0).astype(np.float32)

    def _clear_debug_rrt_plan(self) -> None:
        self._debug_rrt_plan = None
        self._debug_rrt_plan_step = 0
        self._debug_rrt_failed = False
        self._debug_rrt_warned = False

    def _clear_debug_curobo_plan(self) -> None:
        self._debug_curobo_plan = None
        self._debug_curobo_plan_step = 0
        self._debug_curobo_failed = False
        self._debug_curobo_warned = False

    @staticmethod
    def _ensure_curobo_importable() -> None:
        isaaclab_mimic_root = os.path.join("/home/chris/Chris/IsaacLab/source", "isaaclab_mimic")
        if os.path.isdir(isaaclab_mimic_root) and isaaclab_mimic_root not in sys.path:
            sys.path.insert(0, isaaclab_mimic_root)

    def _target_hand_pose_matrix_w(self, inside_fraction: float | None = None) -> torch.Tensor:
        target_pos_env, target_quat = self._planned_hand_release_pose_quat(inside_fraction)
        target_pos_w = target_pos_env[0] + self.scene.env_origins[0]
        target_rot_w = math_utils.matrix_from_quat(target_quat[0])
        target_pose = torch.eye(4, device=self.device, dtype=torch.float32)
        target_pose[:3, :3] = target_rot_w
        target_pose[:3, 3] = target_pos_w
        return target_pose

    def _make_debug_curobo_plan(self, inside_fraction: float | None = None) -> None:
        self._clear_debug_curobo_plan()
        if self.num_envs != 1:
            if not self._debug_curobo_warned:
                print("[Bookshelf v6] cuRobo debug planner only supports num_envs=1.")
                self._debug_curobo_warned = True
            self._debug_curobo_failed = True
            return

        try:
            self._ensure_curobo_importable()
            import torch as _torch
            from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
            from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg
        except Exception as exc:
            print(f"[Bookshelf v6] Could not import cuRobo planner: {exc}")
            self._debug_curobo_failed = True
            self._debug_curobo_warned = True
            return

        if not _torch.cuda.is_available():
            print("[Bookshelf v6] cuRobo is installed, but torch.cuda.is_available() is False in this process.")
            self._debug_curobo_failed = True
            self._debug_curobo_warned = True
            return

        try:
            # Isaac Lab policy/env stepping is commonly under no_grad/inference mode,
            # but cuRobo trajopt needs autograd internally, including during warmup.
            with torch.inference_mode(False), torch.enable_grad():
                if self._debug_curobo_planner is None:
                    planner_cfg = CuroboPlannerCfg.franka_config()
                    planner_cfg.robot_prim_path = "/World/envs/env_0/Robot"
                    planner_cfg.world_ignore_substrings = [
                        "/World/defaultGroundPlane",
                        "/curobo",
                        "/World/Visuals",
                        "/V6Target",
                        "V6Target",
                        "/Book",
                        "/SideBook",
                        "/WideSideBook",
                    ]
                    planner_cfg.motion_step_size = float(getattr(self.cfg, "debug_curobo_motion_step_size", 0.01))
                    planner_cfg.visualize_plan = False
                    planner_cfg.max_planning_attempts = 1
                    self._debug_curobo_planner = CuroboPlanner(env=self, robot=self.robot, config=planner_cfg)

                self._debug_curobo_planner.update_world()
                target_pose = self._target_hand_pose_matrix_w(inside_fraction).clone()
                success = self._debug_curobo_planner.plan_motion(
                    target_pose,
                    step_size=float(getattr(self.cfg, "debug_curobo_motion_step_size", 0.01)),
                    enable_retiming=True,
                )
        except Exception as exc:
            import traceback

            print(f"[Bookshelf v6] cuRobo planning failed: {exc}")
            traceback.print_exc()
            self._debug_curobo_failed = True
            self._debug_curobo_warned = True
            return

        plan = self._debug_curobo_planner.current_plan if success else None
        if plan is None or len(plan.position) <= 1:
            target_pose = self._target_hand_pose_matrix_w(inside_fraction)
            current_pos_w = self.robot.data.body_pos_w[0, self._ee_body_idx]
            target_pos_w = target_pose[:3, 3]
            pos_err = torch.linalg.norm(target_pos_w - current_pos_w).item()
            print(
                "[Bookshelf v6] cuRobo failed to generate a usable plan. "
                f"current_hand_w={current_pos_w.detach().cpu().tolist()}, "
                f"target_hand_w={target_pos_w.detach().cpu().tolist()}, "
                f"pos_err={pos_err:.4f} m"
            )
            self._debug_curobo_failed = True
            self._debug_curobo_warned = True
            return

        plan_joint_names = list(plan.joint_names)
        robot_joint_names = list(self.robot.joint_names)
        arm_joint_ids = list(self._arm_joint_ids)
        arm_plan = torch.zeros((len(plan.position), len(arm_joint_ids)), device=self.device, dtype=torch.float32)
        current_arm = self.robot.data.joint_pos[0, arm_joint_ids].detach().clone()
        arm_plan[:] = current_arm.unsqueeze(0)

        plan_pos = plan.position.detach().to(device=self.device, dtype=torch.float32)
        for plan_idx, joint_name in enumerate(plan_joint_names):
            if joint_name not in robot_joint_names:
                continue
            robot_joint_id = robot_joint_names.index(joint_name)
            if robot_joint_id not in arm_joint_ids:
                continue
            arm_idx = arm_joint_ids.index(robot_joint_id)
            arm_plan[:, arm_idx] = plan_pos[:, plan_idx]

        self._debug_curobo_plan = arm_plan
        self._debug_curobo_plan_step = 0
        print(f"[Bookshelf v6] cuRobo plan generated with {arm_plan.shape[0]} waypoints.")

    def _apply_debug_curobo_plan(self) -> bool:
        if not bool(getattr(self.cfg, "debug_use_curobo_planner", False)):
            return False
        if self.num_envs != 1:
            return False
        if self._debug_curobo_plan is None and not bool(getattr(self, "_debug_curobo_failed", False)):
            self._make_debug_curobo_plan()

        if self._debug_curobo_plan is None or self._debug_curobo_plan.shape[0] == 0:
            self.robot.set_joint_position_target(
                self._arm_hold_joint_pos[0:1], joint_ids=self._arm_joint_ids, env_ids=torch.tensor([0], device=self.device)
            )
            return True

        step = min(int(self._debug_curobo_plan_step), int(self._debug_curobo_plan.shape[0] - 1))
        joint_pos_des = self._debug_curobo_plan[step : step + 1]
        self.robot.set_joint_position_target(
            joint_pos_des, joint_ids=self._arm_joint_ids, env_ids=torch.tensor([0], device=self.device)
        )
        self._arm_hold_joint_pos[0:1] = joint_pos_des
        if self._debug_curobo_plan_step < int(self._debug_curobo_plan.shape[0] - 1):
            self._debug_curobo_plan_step += 1

        if len(self._finger_joint_ids) > 0:
            finger_des = torch.full(
                (1, len(self._finger_joint_ids)),
                float(self.cfg.gripper_closed_joint_pos),
                device=self.device,
                dtype=torch.float32,
            )
            self.robot.set_joint_position_target(
                finger_des, joint_ids=self._finger_joint_ids, env_ids=torch.tensor([0], device=self.device)
            )
        return True

    @staticmethod
    def _ensure_lula_rrt_importable() -> None:
        try:
            import isaacsim.robot_motion.motion_generation  # noqa: F401

            return
        except ModuleNotFoundError:
            pass

        try:
            import omni.kit.app

            ext_mgr = omni.kit.app.get_app().get_extension_manager()
            ext_mgr.set_extension_enabled_immediate("isaacsim.robot_motion.lula", True)
            ext_mgr.set_extension_enabled_immediate("isaacsim.robot_motion.motion_generation", True)
        except Exception:
            pass

        try:
            import isaacsim.robot_motion.motion_generation  # noqa: F401

            return
        except ModuleNotFoundError:
            pass

        isaacsim_root = os.environ.get("ISAACSIM_PATH", "/home/chris/isaacsim")
        ext_root = os.path.join(isaacsim_root, "exts")
        fallback_paths = (
            os.path.join(ext_root, "isaacsim.robot_motion.motion_generation"),
            os.path.join(ext_root, "isaacsim.robot_motion.lula"),
            os.path.join(ext_root, "isaacsim.robot_motion.lula", "pip_prebundle"),
        )
        for path in fallback_paths:
            if os.path.isdir(path) and path not in sys.path:
                sys.path.insert(0, path)

    @staticmethod
    def _fallback_franka_rrt_config() -> dict[str, str] | None:
        isaacsim_root = os.environ.get("ISAACSIM_PATH", "/home/chris/isaacsim")
        mg_root = os.path.join(isaacsim_root, "exts", "isaacsim.robot_motion.motion_generation")
        config = {
            "robot_description_path": os.path.join(
                mg_root, "motion_policy_configs", "franka", "rmpflow", "robot_descriptor.yaml"
            ),
            "urdf_path": os.path.join(mg_root, "motion_policy_configs", "franka", "lula_franka_gen.urdf"),
            "rrt_config_path": os.path.join(
                mg_root, "path_planner_configs", "franka", "rrt", "franka_planner_config.yaml"
            ),
            "end_effector_frame_name": "panda_hand",
        }
        if all(os.path.exists(path) for path in config.values() if path != "panda_hand"):
            return config
        return None

    def _make_debug_rrt_plan(self) -> None:
        self._clear_debug_rrt_plan()
        if self.num_envs != 1:
            if not self._debug_rrt_warned:
                print("[Bookshelf v6] Lula RRT debug planner only supports num_envs=1.")
                self._debug_rrt_warned = True
            self._debug_rrt_failed = True
            return

        try:
            self._ensure_lula_rrt_importable()
            import isaacsim.robot_motion.motion_generation.interface_config_loader as interface_config_loader
            from isaacsim.robot_motion.motion_generation.lula import RRT
        except Exception as exc:
            print(f"[Bookshelf v6] Could not import Isaac Sim Lula RRT planner: {exc}")
            self._debug_rrt_failed = True
            self._debug_rrt_warned = True
            return

        try:
            rrt_config = interface_config_loader.load_supported_path_planner_config("Franka", "RRT")
        except Exception:
            rrt_config = None
        if rrt_config is None:
            rrt_config = self._fallback_franka_rrt_config()
        if rrt_config is None:
            print("[Bookshelf v6] Could not load Franka RRT planner config.")
            self._debug_rrt_failed = True
            self._debug_rrt_warned = True
            return

        rrt_config["end_effector_frame_name"] = "panda_hand"
        planner = RRT(**rrt_config)
        planner.set_max_iterations(int(getattr(self.cfg, "debug_rrt_max_iterations", 50000)))

        root_pos = self.robot.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
        root_quat = self.robot.data.root_quat_w[0].detach().cpu().numpy().astype(np.float64)
        planner.set_robot_base_pose(root_pos, root_quat)

        target_pos_env, target_quat = self._planned_hand_release_pose_quat()
        target_pos_w = (target_pos_env[0] + self.scene.env_origins[0]).detach().cpu().numpy().astype(np.float64)
        target_quat_w = target_quat[0].detach().cpu().numpy().astype(np.float64)
        planner.set_end_effector_target(target_pos_w, target_quat_w)
        planner.update_world()

        active_joint_names = list(planner.get_active_joints())
        robot_joint_names = list(self.robot.joint_names)
        planner_joint_ids = [robot_joint_names.index(name) for name in active_joint_names]
        start_pos = self.robot.data.joint_pos[0, planner_joint_ids].detach().cpu().numpy().astype(np.float64)

        path = planner.compute_path(start_pos, np.array([], dtype=np.float64))
        if path is None or len(path) <= 1:
            print(f"[Bookshelf v6] Lula RRT failed to plan to target EE pose at {target_pos_w}.")
            self._debug_rrt_failed = True
            self._debug_rrt_warned = True
            return

        path = self._interpolate_cspace_path(path, float(getattr(self.cfg, "debug_rrt_interpolation_max_dist", 0.01)))

        arm_joint_ids = list(self._arm_joint_ids)
        arm_plan = torch.zeros((path.shape[0], len(arm_joint_ids)), device=self.device, dtype=torch.float32)
        current_arm = self.robot.data.joint_pos[0, arm_joint_ids].detach().clone()
        arm_plan[:] = current_arm.unsqueeze(0)

        for planner_path_idx, robot_joint_id in enumerate(planner_joint_ids):
            if robot_joint_id not in arm_joint_ids:
                continue
            arm_idx = arm_joint_ids.index(robot_joint_id)
            arm_plan[:, arm_idx] = torch.tensor(path[:, planner_path_idx], device=self.device, dtype=torch.float32)

        self._debug_rrt_plan = arm_plan
        self._debug_rrt_plan_step = 0
        print(f"[Bookshelf v6] Lula RRT plan generated with {path.shape[0]} interpolated waypoints.")

    def _apply_debug_rrt_plan(self) -> bool:
        if not bool(getattr(self.cfg, "debug_use_lula_rrt_planner", False)):
            return False
        if self.num_envs != 1:
            return False
        if self._debug_rrt_plan is None and not bool(getattr(self, "_debug_rrt_failed", False)):
            self._make_debug_rrt_plan()

        if self._debug_rrt_plan is None or self._debug_rrt_plan.shape[0] == 0:
            self.robot.set_joint_position_target(
                self._arm_hold_joint_pos[0:1], joint_ids=self._arm_joint_ids, env_ids=torch.tensor([0], device=self.device)
            )
            return True

        step = min(int(self._debug_rrt_plan_step), int(self._debug_rrt_plan.shape[0] - 1))
        joint_pos_des = self._debug_rrt_plan[step : step + 1]
        self.robot.set_joint_position_target(
            joint_pos_des, joint_ids=self._arm_joint_ids, env_ids=torch.tensor([0], device=self.device)
        )
        self._arm_hold_joint_pos[0:1] = joint_pos_des
        if self._debug_rrt_plan_step < int(self._debug_rrt_plan.shape[0] - 1):
            self._debug_rrt_plan_step += 1

        if len(self._finger_joint_ids) > 0:
            finger_des = torch.full(
                (1, len(self._finger_joint_ids)),
                float(self.cfg.gripper_closed_joint_pos),
                device=self.device,
                dtype=torch.float32,
            )
            self.robot.set_joint_position_target(
                finger_des, joint_ids=self._finger_joint_ids, env_ids=torch.tensor([0], device=self.device)
            )
        return True

    def _nominal_release_mask(self, metrics: dict[str, torch.Tensor]) -> torch.Tensor:
        mouth_x = float(self._geom_mouth_x)
        front_x = float(self.cfg.slot_x_back) - metrics["front_to_back"]
        rear_x = mouth_x + metrics["rear_to_mouth"]
        book_depth_x = torch.clamp(front_x - rear_x, min=1.0e-4)
        inside_fraction = torch.clamp((front_x - mouth_x) / book_depth_x, 0.0, 1.0)
        tilt_x = self._book_tilt_x()

        return (
            (self._mode == _MODE_INSERT)
            & (torch.abs(metrics["lat_err"]) < float(self.cfg.nominal_release_lat_thresh))
            & (torch.abs(metrics["z_err"]) < float(self.cfg.nominal_release_z_thresh))
            & (torch.abs(metrics["yaw_err"]) < float(self.cfg.nominal_release_yaw_thresh))
            & (torch.abs(tilt_x) < float(self.cfg.nominal_release_tilt_x_thresh))
            & (inside_fraction >= float(self.cfg.nominal_release_inside_fraction))
            & (metrics["front_to_back"] >= float(self.cfg.nominal_release_front_to_back_min))
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)
        self._mode_start = self._mode.clone()

        policy_release = self.actions[:, -1] > float(self.cfg.release_trigger_threshold)
        nominal_release = torch.zeros_like(policy_release)
        if (
            self._nominal_release_assist_enabled()
            and
            bool(getattr(self.cfg, "enable_nominal_controller", True))
            and not bool(getattr(self.cfg, "debug_freeze_nominal_controller", False))
        ):
            m = self._compute_task_metrics()
            nominal_release = self._nominal_release_mask(m)

        self._release_request = policy_release | nominal_release

    def _get_rewards(self) -> torch.Tensor:
        rew = super()._get_rewards()
        weight = float(getattr(self.cfg, "residual_action_l2_weight", 0.0))
        if weight <= 0.0:
            return rew

        residual_l2 = torch.mean(torch.square(self.actions[:, 0:5]), dim=-1)
        penalty = weight * residual_l2
        rew = rew - penalty
        self.extras.setdefault("log", {})
        self.extras["log"]["residual_action_l2_mean"] = residual_l2.mean()
        self.extras["log"]["residual_action_l2_penalty_mean"] = penalty.mean()
        return rew

    def _preinsert_reached_mask(self) -> torch.Tensor:
        target_pos_env, target_quat = self._planned_tool_release_pose_quat()
        current_pos_env = self._ee_tool_pos_env()
        current_quat = self.robot.data.body_quat_w[:, self._ee_body_idx]

        pos_err = torch.linalg.norm(target_pos_env - current_pos_env, dim=-1)
        if bool(getattr(self.cfg, "debug_position_only_target_ee", False)):
            return pos_err < float(self.cfg.debug_preinsert_pos_tol)

        rot_err = math_utils.quat_error_magnitude(target_quat, current_quat)
        return (pos_err < float(self.cfg.debug_preinsert_pos_tol)) & (
            rot_err < float(self.cfg.debug_preinsert_rot_tol)
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated, time_out = super()._get_dones()
        if bool(getattr(self.cfg, "debug_done_on_preinsert_reached", False)):
            preinsert_reached = self._preinsert_reached_mask()
            self._debug_preinsert_hold_buf = torch.where(
                preinsert_reached,
                self._debug_preinsert_hold_buf + 1,
                torch.zeros_like(self._debug_preinsert_hold_buf),
            )
            step_dt = float(getattr(self, "step_dt", getattr(self, "physics_dt", 1.0 / 60.0)))
            hold_steps = max(1, int(round(float(getattr(self.cfg, "debug_preinsert_hold_seconds", 1.0)) / step_dt)))
            hold_done = self._debug_preinsert_hold_buf >= hold_steps
            terminated = terminated | hold_done
            self.extras["episode_metric_preinsert_reached"] = preinsert_reached
            self.extras["episode_metric_preinsert_hold_steps"] = self._debug_preinsert_hold_buf
        return terminated, time_out

    def _nominal_cartesian_delta(self, mode: torch.Tensor) -> torch.Tensor:
        nominal = torch.zeros((self.num_envs, 5), device=self.device, dtype=torch.float32)
        if bool(getattr(self.cfg, "debug_freeze_nominal_controller", False)):
            return nominal
        if not bool(getattr(self.cfg, "enable_nominal_controller", True)):
            return nominal

        normal_mask = mode != _MODE_SCRIPTED
        if not torch.any(normal_mask):
            return nominal

        insert_mask = normal_mask & (mode != _MODE_PUSH)
        push_mask = normal_mask & (mode == _MODE_PUSH)

        metrics = self._compute_task_metrics()
        tilt_x = self._book_tilt_x()
        tool_pos = self._ee_tool_pos_env()

        aligned = (
            (torch.abs(metrics["lat_err"]) < float(self.cfg.nominal_align_lat_thresh))
            & (torch.abs(metrics["z_err"]) < float(self.cfg.nominal_align_z_thresh))
            & (torch.abs(metrics["yaw_err"]) < float(self.cfg.nominal_align_yaw_thresh))
            & (torch.abs(tilt_x) < float(self.cfg.nominal_align_tilt_x_thresh))
        )

        insert_dx = torch.full(
            (self.num_envs,), float(self.cfg.nominal_insert_dx), device=self.device, dtype=torch.float32
        )
        near_mouth = metrics["rear_to_mouth"] > float(self.cfg.nominal_slow_rear_to_mouth)
        insert_dx = torch.where(
            near_mouth,
            torch.full_like(insert_dx, float(self.cfg.nominal_insert_dx_near_mouth)),
            insert_dx,
        )
        insert_dx = torch.where(aligned, insert_dx, insert_dx * float(self.cfg.nominal_unaligned_dx_scale))
        insert_dy = torch.clamp(
            float(self.cfg.nominal_lateral_gain) * metrics["lat_err"],
            -float(self.cfg.nominal_dy_limit),
            float(self.cfg.nominal_dy_limit),
        )
        insert_z_err = metrics["z_err"] - float(getattr(self.cfg, "nominal_insert_z_offset", 0.0))
        insert_dz = torch.clamp(
            -float(self.cfg.nominal_height_gain) * insert_z_err,
            -float(self.cfg.nominal_dz_limit),
            float(self.cfg.nominal_dz_limit),
        )
        insert_dyaw = torch.clamp(
            -float(self.cfg.nominal_yaw_gain) * metrics["yaw_err"],
            -float(self.cfg.nominal_dyaw_limit),
            float(self.cfg.nominal_dyaw_limit),
        )
        insert_dpitch = torch.clamp(
            -float(self.cfg.nominal_pitch_gain) * tilt_x,
            -float(self.cfg.nominal_dpitch_limit),
            float(self.cfg.nominal_dpitch_limit),
        )

        if bool(getattr(self.cfg, "debug_position_only_target_ee", False)):
            insert_dyaw = torch.zeros_like(insert_dyaw)
            insert_dpitch = torch.zeros_like(insert_dpitch)

        nominal[:, 0] = torch.where(
            insert_mask, insert_dx, torch.full_like(insert_dx, float(self.cfg.nominal_push_dx))
        )
        nominal[:, 1] = torch.where(insert_mask, insert_dy, nominal[:, 1])
        nominal[:, 2] = torch.where(insert_mask, insert_dz, nominal[:, 2])
        nominal[:, 3] = torch.where(insert_mask, insert_dyaw, nominal[:, 3])
        nominal[:, 4] = torch.where(insert_mask, insert_dpitch, nominal[:, 4])

        if torch.any(push_mask):
            corners = self._book_corners_env()
            book_bottom_z = corners[..., 2].min(dim=-1).values
            book_top_z = corners[..., 2].max(dim=-1).values
            book_height = torch.clamp(book_top_z - book_bottom_z, min=1.0e-4)
            push_z = book_bottom_z + float(self.cfg.nominal_push_z_fraction_from_bottom) * book_height
            slot_y = self._slot_center_y()
            push_dy = torch.clamp(
                float(self.cfg.nominal_push_lateral_gain) * (slot_y - tool_pos[:, 1]),
                -float(self.cfg.nominal_push_dy_limit),
                float(self.cfg.nominal_push_dy_limit),
            )
            push_dz = torch.clamp(
                float(self.cfg.nominal_push_height_gain) * (push_z - tool_pos[:, 2]),
                -float(self.cfg.nominal_push_dz_limit),
                float(self.cfg.nominal_push_dz_limit),
            )
            push_dyaw = torch.clamp(
                -float(self.cfg.nominal_push_yaw_gain) * metrics["yaw_err"],
                -float(self.cfg.nominal_dyaw_limit),
                float(self.cfg.nominal_dyaw_limit),
            )
            push_dpitch = torch.clamp(
                -float(self.cfg.nominal_push_pitch_gain) * tilt_x,
                -float(self.cfg.nominal_dpitch_limit),
                float(self.cfg.nominal_dpitch_limit),
            )
            if bool(getattr(self.cfg, "debug_position_only_target_ee", False)):
                push_dyaw = torch.zeros_like(push_dyaw)
                push_dpitch = torch.zeros_like(push_dpitch)
            nominal[:, 1] = torch.where(push_mask, push_dy, nominal[:, 1])
            nominal[:, 2] = torch.where(push_mask, push_dz, nominal[:, 2])
            nominal[:, 3] = torch.where(push_mask, push_dyaw, nominal[:, 3])
            nominal[:, 4] = torch.where(push_mask, push_dpitch, nominal[:, 4])

        nominal = torch.where(normal_mask.unsqueeze(-1), nominal, torch.zeros_like(nominal))
        return nominal

    def _apply_action(self) -> None:
        mode = self._mode
        self._refresh_debug_markers()

        if bool(getattr(self.cfg, "debug_hold_book_fixed_to_tool", False)):
            # Ideal grasp is only a pre-release debug aid. Stop overwriting the
            # book immediately when release is requested so opening/retreat and
            # the subsequent push operate on the physical book.
            hold_ids = self._env_ids[(mode == _MODE_INSERT) & ~self._release_request]
            if hold_ids.numel() > 0:
                self._write_book_from_fixed_tool_transform(hold_ids)

        if self._apply_debug_curobo_plan():
            return

        if self._apply_debug_rrt_plan():
            return

        residual = torch.stack(
            (
                self.actions[:, 0] * self.cfg.dx_action_scale,
                self.actions[:, 1] * self.cfg.dy_action_scale,
                self.actions[:, 2] * self.cfg.dz_action_scale,
                self.actions[:, 3] * self.cfg.dyaw_action_scale,
                self.actions[:, 4] * self.cfg.dpitch_action_scale,
            ),
            dim=-1,
        )
        residual = residual * self._residual_action_scale()
        nominal = self._nominal_cartesian_delta(mode)
        delta = nominal + residual
        delta[:, 0] = torch.clamp(delta[:, 0], -float(self.cfg.final_dx_limit), float(self.cfg.final_dx_limit))
        delta[:, 1] = torch.clamp(delta[:, 1], -float(self.cfg.final_dy_limit), float(self.cfg.final_dy_limit))
        delta[:, 2] = torch.clamp(delta[:, 2], -float(self.cfg.final_dz_limit), float(self.cfg.final_dz_limit))
        delta[:, 3] = torch.clamp(delta[:, 3], -float(self.cfg.final_dyaw_limit), float(self.cfg.final_dyaw_limit))
        delta[:, 4] = torch.clamp(
            delta[:, 4], -float(self.cfg.final_dpitch_limit), float(self.cfg.final_dpitch_limit)
        )
        if bool(getattr(self.cfg, "debug_print_residual_components", False)):
            interval = max(1, int(getattr(self.cfg, "debug_print_residual_interval", 30)))
            env_i = int(getattr(self.cfg, "debug_print_residual_env_index", 0))
            if env_i < self.num_envs and int(self.common_step_counter) % interval == 0:
                n = nominal[env_i]
                r = residual[env_i]
                d = delta[env_i]
                mode_i = int(mode[env_i].item())
                raw_release = float(self.actions[env_i, -1].item())
                release_req = bool(self._release_request[env_i].item())
                print(
                    "[residual debug] "
                    f"step={int(self.common_step_counter)} env={env_i} mode={mode_i} "
                    f"release_action={raw_release:+.3f} release_req={release_req} "
                    f"nom=[dx {float(n[0].item()):+.4f}, dy {float(n[1].item()):+.4f}, "
                    f"dz {float(n[2].item()):+.4f}, dyaw {float(torch.rad2deg(n[3]).item()):+.3f}deg, "
                    f"dpitch {float(torch.rad2deg(n[4]).item()):+.3f}deg] "
                    f"res=[dx {float(r[0].item()):+.4f}, dy {float(r[1].item()):+.4f}, "
                    f"dz {float(r[2].item()):+.4f}, dyaw {float(torch.rad2deg(r[3]).item()):+.3f}deg, "
                    f"dpitch {float(torch.rad2deg(r[4]).item()):+.3f}deg] "
                    f"final=[dx {float(d[0].item()):+.4f}, dy {float(d[1].item()):+.4f}, "
                    f"dz {float(d[2].item()):+.4f}, dyaw {float(torch.rad2deg(d[3]).item()):+.3f}deg, "
                    f"dpitch {float(torch.rad2deg(d[4]).item()):+.3f}deg]"
                )

        ee_tool_pos_env = self._ee_tool_pos_env()
        _, ee_quat_b = self._ee_pose_in_base()
        _, ee_pitch_b, ee_yaw_b = math_utils.euler_xyz_from_quat(ee_quat_b)

        target_pos_env = ee_tool_pos_env.clone()
        target_yaw = ee_yaw_b.clone()
        target_pitch = ee_pitch_b.clone()

        normal_mask = mode != _MODE_SCRIPTED
        push_mask = normal_mask & (mode == _MODE_PUSH)
        if torch.any(normal_mask):
            target_pos_env[normal_mask] = target_pos_env[normal_mask] + delta[normal_mask, 0:3]
            target_yaw[normal_mask] = _wrap_to_pi(ee_yaw_b[normal_mask] + delta[normal_mask, 3])
            target_pitch[normal_mask] = _wrap_to_pi(ee_pitch_b[normal_mask] + delta[normal_mask, 4])

        scripted_mask = mode == _MODE_SCRIPTED
        if torch.any(scripted_mask):
            retreat_mask = scripted_mask & (self._script_step_buf >= int(self.cfg.script_open_steps))
            if torch.any(retreat_mask):
                target_pos_env[retreat_mask, 0] += float(self.cfg.script_retreat_dx)
                target_pos_env[retreat_mask, 2] += float(self.cfg.script_retreat_dz)

        if bool(getattr(self.cfg, "debug_position_only_target_ee", False)):
            joint_pos_des = self._compute_ik_joint_targets_from_tool_quat(
                target_pos_env, self._debug_position_only_ee_quat_b
            )
        elif bool(getattr(self.cfg, "debug_use_full_target_ee_quat", False)):
            _, full_target_quat = self._planned_tool_release_pose_quat()
            _, push_target_quat = self._planned_tool_release_pose_quat(inside_fraction=0.9)
            full_target_quat = torch.where(push_mask.unsqueeze(-1), push_target_quat, full_target_quat)
            _, full_target_quat_b = math_utils.subtract_frame_transforms(
                self.robot.data.root_pos_w,
                self.robot.data.root_quat_w,
                self.robot.data.root_pos_w,
                full_target_quat,
            )

            target_quat = ee_quat_b.clone()
            if torch.any(normal_mask):
                target_quat[normal_mask] = full_target_quat_b[normal_mask]
            joint_pos_des = self._compute_ik_joint_targets_from_tool_quat(target_pos_env, target_quat)
        else:
            joint_pos_des = self._compute_ik_joint_targets_from_tool(target_pos_env, target_yaw, target_pitch)

        act_small = delta.abs() < float(self.cfg.ik_hold_action_epsilon)
        hold_arm = normal_mask & act_small.all(dim=-1)
        move_arm = ~hold_arm

        move_exp = move_arm.unsqueeze(-1).expand_as(joint_pos_des)
        self._arm_hold_joint_pos = torch.where(move_exp, joint_pos_des, self._arm_hold_joint_pos)

        hold_exp = hold_arm.unsqueeze(-1).expand_as(joint_pos_des)
        joint_pos_des = torch.where(hold_exp, self._arm_hold_joint_pos, joint_pos_des)

        self.robot.set_joint_position_target(joint_pos_des, joint_ids=self._arm_joint_ids)

        if len(self._finger_joint_ids) > 0:
            hold_width = float(self.cfg.gripper_closed_joint_pos)
            push_closed = float(getattr(self.cfg, "gripper_push_closed_joint_pos", 0.0))
            o = float(self.cfg.gripper_open_joint_pos)
            script_open_retreat = int(self.cfg.script_open_steps) + int(self.cfg.script_retreat_steps)
            gripper_should_open = (mode == _MODE_SCRIPTED) & (self._script_step_buf < script_open_retreat)
            gripper_should_close = (mode == _MODE_PUSH) | (
                (mode == _MODE_SCRIPTED) & (self._script_step_buf >= script_open_retreat)
            )
            finger_cmd = torch.where(
                gripper_should_open,
                torch.full((self.num_envs,), o, device=self.device, dtype=torch.float32),
                torch.where(
                    gripper_should_close,
                    torch.full((self.num_envs,), push_closed, device=self.device, dtype=torch.float32),
                    torch.full((self.num_envs,), hold_width, device=self.device, dtype=torch.float32),
                ),
            )
            finger_des = finger_cmd.unsqueeze(-1).expand(self.num_envs, len(self._finger_joint_ids))
            self.robot.set_joint_position_target(finger_des, joint_ids=self._finger_joint_ids)
