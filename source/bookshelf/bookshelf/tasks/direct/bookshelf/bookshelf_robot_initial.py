#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bookshelf-Direct-v4 (spawn-only environment).

This environment intentionally contains **no task logic** (no meaningful actions, observations, rewards,
or success/failure terminations). It only spawns the scene: robot, book, and bookshelf geometry.

Placeholders are kept so the Isaac Lab `DirectRLEnv` loop can run without special-casing.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import math as math_utils
from isaaclab.utils.math import sample_uniform

from .bookshelf_env_cfg_v4 import BookshelfEnvCfg


class BookshelfEnv(DirectRLEnv):
    """Minimal env that only sets up the assets."""

    cfg: BookshelfEnvCfg

    def __init__(self, cfg: BookshelfEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self.robot: Articulation = self.scene.articulations["robot"]
        self.book: RigidObject = self.scene.rigid_objects["book"]

        # Finger bodies for grasp midpoint (used to spawn book in the gripper).
        lf_ids, _ = self.robot.find_bodies("panda_leftfinger")
        rf_ids, _ = self.robot.find_bodies("panda_rightfinger")
        if len(lf_ids) != 1 or len(rf_ids) != 1:
            raise RuntimeError("Expected panda_leftfinger and panda_rightfinger bodies for grasp frame.")
        self._left_finger_body_idx = lf_ids[0]
        self._right_finger_body_idx = rf_ids[0]
        hand_ids, hand_names = self.robot.find_bodies("panda_hand")
        if len(hand_ids) != 1:
            raise RuntimeError(f"Expected one panda_hand body. Got {len(hand_ids)}: {hand_names}")
        self._hand_body_idx = hand_ids[0]

    def _grasp_frame_pose_w(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lf_pos = self.robot.data.body_pos_w[env_ids, self._left_finger_body_idx]
        rf_pos = self.robot.data.body_pos_w[env_ids, self._right_finger_body_idx]
        grasp_pos_w = 0.5 * (lf_pos + rf_pos)
        grasp_quat_w = self.robot.data.body_quat_w[env_ids, self._hand_body_idx]
        return grasp_pos_w, grasp_quat_w

    def _q_body_to_hand_grasp(self, n: int, dtype: torch.dtype) -> torch.Tensor:
        mode = self.cfg.book_grasp_orientation_in_hand
        if mode == "franka_axes":
            fw, fx, fy, fz = self.cfg.book_to_hand_quat_franka_axes_wxyz
            return torch.tensor([fw, fx, fy, fz], device=self.device, dtype=dtype).unsqueeze(0).expand(n, 4).clone()
        if mode == "manual_quat":
            rw, rx, ry, rz = self.cfg.book_grasp_rel_quat_wxyz
            return torch.tensor([rw, rx, ry, rz], device=self.device, dtype=dtype).unsqueeze(0).expand(n, 4).clone()
        raise ValueError(f"Unknown book_grasp_orientation_in_hand: {mode}")

    @staticmethod
    def _quat_world_yaw_half(yaw: torch.Tensor) -> torch.Tensor:
        n = yaw.shape[0]
        device, dtype = yaw.device, yaw.dtype
        return torch.stack(
            (
                torch.cos(0.5 * yaw),
                torch.zeros(n, device=device, dtype=dtype),
                torch.zeros(n, device=device, dtype=dtype),
                torch.sin(0.5 * yaw),
            ),
            dim=-1,
        )

    def _snap_book_to_measured_grasp(self, env_ids_t: torch.Tensor) -> None:
        """Place book from measured finger-midpoint so it starts inside the gripper."""
        n = int(env_ids_t.numel())
        dtype = torch.float32

        # Book COM offset in grasp frame (origin = finger midpoint, axes = panda_hand body axes).
        hx, hy, hz = self.cfg.book_grasp_offset_hand
        off = (
            torch.tensor([hx, hy, hz], device=self.device, dtype=dtype).unsqueeze(0).expand(n, 3).clone()
        )
        if float(self.cfg.book_grasp_x_jitter) != 0.0:
            off[:, 0] += sample_uniform(
                -float(self.cfg.book_grasp_x_jitter), float(self.cfg.book_grasp_x_jitter), (n,), self.device
            )
        if float(self.cfg.book_grasp_y_jitter) != 0.0:
            off[:, 1] += sample_uniform(
                -float(self.cfg.book_grasp_y_jitter), float(self.cfg.book_grasp_y_jitter), (n,), self.device
            )
        yaw_delta = (
            sample_uniform(
                -float(self.cfg.book_grasp_yaw_jitter),
                float(self.cfg.book_grasp_yaw_jitter),
                (n,),
                self.device,
            )
            if float(self.cfg.book_grasp_yaw_jitter) != 0.0
            else torch.zeros(n, device=self.device, dtype=dtype)
        )

        grasp_pos_w, grasp_quat_w = self._grasp_frame_pose_w(env_ids_t)
        book_pos_w = grasp_pos_w + math_utils.quat_apply(grasp_quat_w, off)
        book_pos_env = book_pos_w - self.scene.env_origins[env_ids_t]

        q_b2h = self._q_body_to_hand_grasp(n, grasp_quat_w.dtype)
        qyaw = self._quat_world_yaw_half(yaw_delta)
        book_quat_w = math_utils.quat_mul(grasp_quat_w, math_utils.quat_mul(q_b2h, qyaw))

        book_state = self.book.data.default_root_state[env_ids_t].clone()
        book_state[:, 0:3] = book_pos_env + self.scene.env_origins[env_ids_t]
        book_state[:, 3:7] = book_quat_w
        book_state[:, 7:] = 0.0
        self.book.write_root_state_to_sim(book_state, env_ids=env_ids_t)

    def _setup_scene(self):
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        x0 = self.cfg.slot_x_open
        x1 = self.cfg.slot_x_back
        mid_x = 0.5 * (x0 + x1)
        depth_x = x1 - x0 + 0.06
        bthick = self.cfg.book_size[2]
        bheight = self.cfg.book_size[1]
        blen = self.cfg.book_size[0]
        clearance = self.cfg.slot_lateral_clearance
        inner_half = 0.5 * (bthick + clearance)
        half_lateral_y = 0.5 * bthick
        n_extra = max(0, int(self.cfg.shelf_extra_books_per_side))
        pitch_y = bthick + float(self.cfg.neighbor_book_pitch_gap)
        wt = self.cfg.wall_thickness
        z_book = bheight * 0.5
        qw, qx, qy, qz = self.cfg.book_standing_quat
        standing_quat = (float(qw), float(qx), float(qy), float(qz))

        kin = RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True)
        col = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
        wood = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.45, 0.35))
        book_vis = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.25, 0.15))

        base = "/World/envs/env_0/Bookshelf"

        # Shelf surface.
        shelf_depth = depth_x
        shelf_half_y = inner_half + bthick + n_extra * pitch_y
        shelf_width = 2.0 * shelf_half_y + 0.1
        shelf_thick = float(self.cfg.shelf_thickness)
        shelf_cfg = sim_utils.MeshCuboidCfg(
            size=(shelf_depth, shelf_width, shelf_thick),
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

        # Slot-defining neighbor books (standing).
        left_center_y = self.cfg.slot_center_y - inner_half - half_lateral_y
        right_center_y = self.cfg.slot_center_y + inner_half + half_lateral_y
        neighbor_cfg = sim_utils.MeshCuboidCfg(
            size=(blen, bheight, bthick),
            rigid_props=kin,
            collision_props=col,
            visual_material=book_vis,
        )
        neighbor_cfg.func(
            f"{base}/LeftNeighborBook",
            neighbor_cfg,
            translation=(mid_x, left_center_y, self.cfg.shelf_top_z + shelf_thick + z_book),
            orientation=standing_quat,
        )
        neighbor_cfg.func(
            f"{base}/RightNeighborBook",
            neighbor_cfg,
            translation=(mid_x, right_center_y, self.cfg.shelf_top_z + shelf_thick + z_book),
            orientation=standing_quat,
        )
        for i in range(1, n_extra + 1):
            y_left = left_center_y - i * pitch_y
            y_right = right_center_y + i * pitch_y
            neighbor_cfg.func(
                f"{base}/LeftNeighborExtra{i}",
                neighbor_cfg,
                translation=(mid_x, y_left, self.cfg.shelf_top_z + shelf_thick + z_book),
                orientation=standing_quat,
            )
            neighbor_cfg.func(
                f"{base}/RightNeighborExtra{i}",
                neighbor_cfg,
                translation=(mid_x, y_right, self.cfg.shelf_top_z + shelf_thick + z_book),
                orientation=standing_quat,
            )

        # Back panel.
        back_x = x1 + wt * 0.5
        panel_cfg = sim_utils.MeshCuboidCfg(
            size=(wt, shelf_width + 0.1, bheight + 0.1),
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

        # Spawn under env_0 only, clone once, then register (Factory-style pattern).
        robot = Articulation(self.cfg.robot)
        book = RigidObject(self.cfg.book)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = robot
        self.scene.rigid_objects["book"] = book

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ---- placeholders so DirectRLEnv can run ----

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        pass

    def _apply_action(self) -> None:
        pass

    def _get_observations(self) -> dict:
        obs = torch.zeros((self.num_envs, self.cfg.observation_space), device=self.device, dtype=torch.float32)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids_t = self._env_ids
        else:
            env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        super()._reset_idx(env_ids_t)

        robot_default = self.robot.data.default_root_state[env_ids_t].clone()
        robot_default[:, 0:3] += self.scene.env_origins[env_ids_t]
        self.robot.write_root_state_to_sim(robot_default, env_ids=env_ids_t)
        joint_pos = self.robot.data.default_joint_pos[env_ids_t].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids_t].clone()
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids_t)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids_t)

        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)

        # Ensure book starts in the gripper for the (possibly updated) default robot joint pose.
        self._snap_book_to_measured_grasp(env_ids_t)
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)
