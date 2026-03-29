#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bookshelf-Direct-v4 environment.

- Actions: [dx, dy, dz, dyaw, g_gripper] residuals; gripper maps to finger position targets
- Success: geometry at slot mouth (hold for a few steps)
- Failure: book COM on the floor (exempt if still over shelf deck footprint)
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import math as math_utils
from isaaclab.utils.math import sample_uniform

from .bookshelf_env_cfg_v4 import BookshelfEnvCfg


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return (angle + torch.pi) % (2.0 * torch.pi) - torch.pi


def _yaw_from_quat_wxyz(q: torch.Tensor) -> torch.Tensor:
    """Extract world yaw (about +Z) from quaternion (w, x, y, z)."""
    qw, qx, qy, qz = q.unbind(dim=-1)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return torch.atan2(siny_cosp, cosy_cosp)


def _cuboid_corners_local(half_extents: torch.Tensor) -> torch.Tensor:
    """Return 8 cuboid corners in local frame given half-extents (hx, hy, hz)."""
    hx, hy, hz = half_extents.unbind(dim=-1)
    return torch.stack(
        (
            torch.stack((+hx, +hy, +hz), dim=-1),
            torch.stack((+hx, +hy, -hz), dim=-1),
            torch.stack((+hx, -hy, +hz), dim=-1),
            torch.stack((+hx, -hy, -hz), dim=-1),
            torch.stack((-hx, +hy, +hz), dim=-1),
            torch.stack((-hx, +hy, -hz), dim=-1),
            torch.stack((-hx, -hy, +hz), dim=-1),
            torch.stack((-hx, -hy, -hz), dim=-1),
        ),
        dim=-2,
    )


class BookshelfEnv(DirectRLEnv):
    """Bookshelf insertion env with Cartesian + yaw + gripper actions."""

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

        # Arm/finger joints and jacobian indices for IK.
        self._arm_joint_ids, _ = self.robot.find_joints("panda_joint.*")
        self._finger_joint_ids, _ = self.robot.find_joints("panda_finger_joint.*")
        self._ee_body_idx = self._hand_body_idx
        if self.robot.is_fixed_base:
            self._jacobi_body_idx = self._ee_body_idx - 1
            self._jacobi_joint_ids = self._arm_joint_ids
        else:
            self._jacobi_body_idx = self._ee_body_idx
            self._jacobi_joint_ids = [i + 6 for i in self._arm_joint_ids]

        self._ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            num_envs=self.num_envs,
            device=str(self.device),
        )
        self._ik_body_offset_pos_b = torch.tensor(self.cfg.ik_body_offset_pos, device=self.device, dtype=torch.float32)
        self._ik_body_offset_pos_b = self._ik_body_offset_pos_b.view(1, 3).expand(self.num_envs, 3)
        self._ik_cmd = torch.zeros((self.num_envs, 7), device=self.device)

        # Buffers.
        self._last_actions = torch.zeros((self.num_envs, 5), device=self.device)
        self._success_steps_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._prev_front_to_mouth = torch.zeros(self.num_envs, device=self.device)
        self._prev_abs_lat_err = torch.zeros(self.num_envs, device=self.device)
        self._prev_abs_yaw_err = torch.zeros(self.num_envs, device=self.device)

        # Integrated Cartesian target (env frame + base yaw).
        self._target_pos_env = torch.zeros((self.num_envs, 3), device=self.device)
        self._target_yaw = torch.zeros(self.num_envs, device=self.device)

        # Precompute book corners for geometry checks.
        half = 0.5 * torch.tensor(self.cfg.book_size, device=self.device, dtype=torch.float32)
        self._book_corners_local = _cuboid_corners_local(half).to(device=self.device, dtype=torch.float32)

        # Hold arm target when action≈0 (avoid sag under load).
        self._arm_hold_joint_pos = self.robot.data.default_joint_pos[:, self._arm_joint_ids].clone()
        self._slot_reach_entry_credited = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

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
        # Optional: small in-hand Z jitter (if present in cfg).
        z_j = float(getattr(self.cfg, "book_grasp_z_jitter", 0.0))
        if z_j != 0.0:
            off[:, 2] += sample_uniform(-z_j, z_j, (n,), self.device)
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

    def _ee_pose_in_base(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos_w = self.robot.data.body_pos_w[:, self._ee_body_idx]
        ee_quat_w = self.robot.data.body_quat_w[:, self._ee_body_idx]
        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        return ee_pos_b, ee_quat_b

    def _compute_ik_joint_targets_from_tool(self, target_pos_env: torch.Tensor, target_yaw: torch.Tensor) -> torch.Tensor:
        """Absolute IK: reach tool target in env frame with yaw in base frame; keep current roll/pitch."""
        self._target_pos_env[:] = target_pos_env
        self._target_yaw[:] = target_yaw
        _, ee_quat_b = self._ee_pose_in_base()
        ee_roll_b, ee_pitch_b, _ = math_utils.euler_xyz_from_quat(ee_quat_b)
        quat_des_b = math_utils.quat_from_euler_xyz(ee_roll_b, ee_pitch_b, target_yaw)
        offset_des_b = math_utils.quat_apply(quat_des_b, self._ik_body_offset_pos_b)
        body_pos_des_b = target_pos_env - offset_des_b
        self._ik_cmd[:, 0:3] = body_pos_des_b
        self._ik_cmd[:, 3:7] = quat_des_b
        self._ik.set_command(self._ik_cmd)
        ee_pos_b, ee_quat_b2 = self._ee_pose_in_base()
        jacobian = self.robot.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, self._jacobi_joint_ids]
        joint_pos = self.robot.data.joint_pos[:, self._arm_joint_ids]
        return self._ik.compute(ee_pos_b, ee_quat_b2, jacobian, joint_pos)

    def _book_pos_env(self) -> torch.Tensor:
        return self.book.data.root_link_pos_w - self.scene.env_origins

    def _book_corners_env(self) -> torch.Tensor:
        """Book corners in env frame: (num_envs, 8, 3)."""
        pos_w = self.book.data.root_link_pos_w
        quat_w = self.book.data.root_link_quat_w
        corners_l = self._book_corners_local.view(1, 8, 3).expand(self.num_envs, 8, 3)
        quat_rep = quat_w.view(self.num_envs, 1, 4).expand(self.num_envs, 8, 4)
        corners_w = math_utils.quat_apply(quat_rep.reshape(-1, 4), corners_l.reshape(-1, 3)).view(
            self.num_envs, 8, 3
        ) + pos_w.view(self.num_envs, 1, 3)
        return corners_w - self.scene.env_origins.view(self.num_envs, 1, 3)

    def _upright_ok(self) -> torch.Tensor:
        quat = self.book.data.root_link_quat_w
        spine_l = torch.zeros_like(quat[..., 1:4])
        spine_l[..., 1] = 1.0
        spine_w = math_utils.quat_apply(quat, spine_l)
        return torch.abs(spine_w[:, 2]) > self.cfg.upright_dot_thresh

    def _stage_metrics(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (front_to_true_mouth, lat_err, lat_extent, z_err, yaw_err_signed, front_to_back)."""
        corners = self._book_corners_env()
        front_x = corners[..., 0].max(dim=-1).values
        front_to_true_mouth = front_x - float(self.cfg.slot_x_open)
        front_to_back = float(self.cfg.slot_x_back) - front_x

        lat_err = self.cfg.slot_center_y - self._book_pos_env()[:, 1]
        lat_extent = torch.abs(corners[..., 1] - self.cfg.slot_center_y).max(dim=-1).values

        p = self._book_pos_env()
        z_target = self.cfg.shelf_top_z + self.cfg.shelf_thickness + 0.5 * self.cfg.book_size[1]
        z_err = p[:, 2] - float(z_target)

        yaw = _yaw_from_quat_wxyz(self.book.data.root_link_quat_w)
        yaw_err = _wrap_to_pi(yaw)
        return front_to_true_mouth, lat_err, lat_extent, z_err, yaw_err, front_to_back

    def _book_supported_on_shelf(self, p_env: torch.Tensor, lowest_z: torch.Tensor) -> torch.Tensor:
        """COM over shelf footprint and **lowest corner** on/near the deck — not merely COM height (avoids floor false negatives)."""
        x0 = float(self.cfg.slot_x_open)
        x1 = float(self.cfg.slot_x_back)
        mid_x = 0.5 * (x0 + x1)
        depth_x = x1 - x0 + 0.06
        hx = 0.5 * depth_x + float(self.cfg.shelf_footprint_x_pad_m)
        bthick = self.cfg.book_size[2]
        clearance = self.cfg.slot_lateral_clearance
        inner_half = 0.5 * (bthick + clearance)
        half_lateral_y = 0.5 * bthick
        n_extra = max(0, int(self.cfg.shelf_extra_books_per_side))
        pitch_y = bthick + float(self.cfg.neighbor_book_pitch_gap)
        shelf_half_y = inner_half + bthick + n_extra * pitch_y
        hy = shelf_half_y + float(self.cfg.shelf_footprint_y_pad_m)
        cy = float(self.cfg.slot_center_y)
        deck_z = float(self.cfg.shelf_top_z + self.cfg.shelf_thickness)
        slack = float(self.cfg.book_on_shelf_z_slack_m)
        in_xy = (p_env[:, 0] >= mid_x - hx) & (p_env[:, 0] <= mid_x + hx) & (p_env[:, 1] >= cy - hy) & (p_env[:, 1] <= cy + hy)
        lowest_on_deck = lowest_z >= deck_z - slack
        return in_xy & lowest_on_deck

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

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        dx = self.actions[:, 0] * self.cfg.dx_action_scale
        dy = self.actions[:, 1] * self.cfg.dy_action_scale
        dz = self.actions[:, 2] * self.cfg.dz_action_scale
        dyaw = self.actions[:, 3] * self.cfg.dyaw_action_scale

        # Residual-on-current style: target is computed from current measured EE(tool) pose each step.
        ee_body_pos_env = self.robot.data.body_pos_w[:, self._ee_body_idx] - self.scene.env_origins
        ee_body_quat_w = self.robot.data.body_quat_w[:, self._ee_body_idx]
        ee_tool_offset_w = math_utils.quat_apply(ee_body_quat_w, self._ik_body_offset_pos_b)
        ee_tool_pos_env = ee_body_pos_env + ee_tool_offset_w

        _, ee_quat_b = self._ee_pose_in_base()
        _, _, ee_yaw_b = math_utils.euler_xyz_from_quat(ee_quat_b)

        target_pos_env = ee_tool_pos_env + torch.stack((dx, dy, dz), dim=-1)
        target_yaw = _wrap_to_pi(ee_yaw_b + dyaw)

        joint_pos_des = self._compute_ik_joint_targets_from_tool(target_pos_env, target_yaw)

        arm_act = self.actions[:, 0:4]
        act_small = arm_act.abs() < float(self.cfg.ik_hold_action_epsilon)
        hold_arm = act_small.all(dim=-1)
        move_arm = ~hold_arm
        move_exp = move_arm.unsqueeze(-1).expand_as(joint_pos_des)
        self._arm_hold_joint_pos = torch.where(move_exp, joint_pos_des, self._arm_hold_joint_pos)
        hold_exp = hold_arm.unsqueeze(-1).expand_as(joint_pos_des)
        joint_pos_des = torch.where(hold_exp, self._arm_hold_joint_pos, joint_pos_des)
        self.robot.set_joint_position_target(joint_pos_des, joint_ids=self._arm_joint_ids)

        # Gripper: g=-1 → closed, g=+1 → open (same target for both finger joints).
        if len(self._finger_joint_ids) > 0:
            g = self.actions[:, 4].clamp(-1.0, 1.0)
            c = float(self.cfg.gripper_closed_joint_pos)
            o = float(self.cfg.gripper_open_joint_pos)
            finger_cmd = 0.5 * (1.0 - g) * c + 0.5 * (1.0 + g) * o
            n_f = len(self._finger_joint_ids)
            finger_des = finger_cmd.unsqueeze(-1).expand(self.num_envs, n_f)
            self.robot.set_joint_position_target(finger_des, joint_ids=self._finger_joint_ids)

    def _get_observations(self) -> dict:
        front_to_mouth, lat_err, _, z_err, yaw_err, _ = self._stage_metrics()
        v = self.book.data.root_link_vel_w

        f2m_s = torch.clamp(front_to_mouth / float(self.cfg.front_to_mouth_obs_scale), -1.0, 1.0)
        lat_s = torch.clamp(lat_err / float(self.cfg.lat_err_obs_scale), -1.0, 1.0)
        yaw_s = torch.clamp(yaw_err / float(self.cfg.yaw_err_obs_scale), -1.0, 1.0)
        z_s = torch.clamp(z_err / float(self.cfg.z_err_obs_scale), -1.0, 1.0)
        vx_s = torch.clamp(v[:, 0] / float(self.cfg.lin_vel_obs_scale), -1.0, 1.0)
        vy_s = torch.clamp(v[:, 1] / float(self.cfg.lin_vel_obs_scale), -1.0, 1.0)
        wz_s = torch.clamp(v[:, 5] / float(self.cfg.ang_vel_obs_scale), -1.0, 1.0)

        ee_body_pos_env = self.robot.data.body_pos_w[:, self._ee_body_idx] - self.scene.env_origins
        ee_body_quat_w = self.robot.data.body_quat_w[:, self._ee_body_idx]
        ee_tool_offset_w = math_utils.quat_apply(ee_body_quat_w, self._ik_body_offset_pos_b)
        ee_tool_pos_env = ee_body_pos_env + ee_tool_offset_w

        _, ee_quat_b_obs = self._ee_pose_in_base()
        _, _, ee_yaw_b = math_utils.euler_xyz_from_quat(ee_quat_b_obs)

        ex = self._target_pos_env[:, 0] - ee_tool_pos_env[:, 0]
        ey = self._target_pos_env[:, 1] - ee_tool_pos_env[:, 1]
        ez = self._target_pos_env[:, 2] - ee_tool_pos_env[:, 2]
        eyaw = _wrap_to_pi(self._target_yaw - ee_yaw_b)

        te = float(self.cfg.tracking_error_obs_scale)
        ex_s = torch.clamp(ex / te, -1.0, 1.0)
        ey_s = torch.clamp(ey / te, -1.0, 1.0)
        ez_s = torch.clamp(ez / te, -1.0, 1.0)
        eyaw_s = torch.clamp(eyaw / te, -1.0, 1.0)

        c = float(self.cfg.gripper_closed_joint_pos)
        o = float(self.cfg.gripper_open_joint_pos)
        if len(self._finger_joint_ids) > 0:
            fp = self.robot.data.joint_pos[:, self._finger_joint_ids]
            fmean = fp.mean(dim=-1)
            grip_open01 = (fmean - c) / (o - c + 1e-9)
            grip_s = torch.clamp(2.0 * grip_open01 - 1.0, -1.0, 1.0)
        else:
            grip_s = torch.zeros(self.num_envs, device=self.device)

        obs = torch.stack(
            (
                f2m_s,
                lat_s,
                yaw_s,
                z_s,
                ex_s,
                ey_s,
                ez_s,
                eyaw_s,
                vx_s,
                vy_s,
                wz_s,
                grip_s,
                self._last_actions[:, 0],
                self._last_actions[:, 1],
                self._last_actions[:, 2],
                self._last_actions[:, 3],
                self._last_actions[:, 4],
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        front_to_mouth, lat_err, lat_extent, z_err, yaw_err, _ = self._stage_metrics()
        abs_lat = torch.abs(lat_err)
        abs_yaw = torch.abs(yaw_err)
        yaw_e = abs_yaw

        d_mouth = front_to_mouth - self._prev_front_to_mouth
        progress = self.cfg.progress_scale * torch.clamp(d_mouth, min=-0.02, max=0.02)

        enter_m = float(self.cfg.success_enter_margin)
        fm_pos = torch.clamp(front_to_mouth, min=0.0, max=enter_m)
        depth = self.cfg.depth_reward_scale * fm_pos
        norm_depth = torch.clamp(front_to_mouth / enter_m, 0.0, 1.0)
        depth_sq = float(self.cfg.depth_corridor_progress_scale) * (norm_depth**2)

        # Alignment progress (constructive shaping): reward reductions in |lat_err| and |yaw_err|.
        center_prog = self.cfg.center_progress_scale * torch.clamp(
            self._prev_abs_lat_err - abs_lat, min=-0.01, max=0.01
        )
        yaw_prog = self.cfg.yaw_progress_scale * torch.clamp(
            self._prev_abs_yaw_err - abs_yaw, min=-0.05, max=0.05
        )

        inner_half = 0.5 * (self.cfg.book_size[2] + self.cfg.slot_lateral_clearance)
        clearance_limit = inner_half - float(self.cfg.success_lateral_margin)
        clearance_violation = torch.clamp(lat_extent - clearance_limit, min=0.0)
        base_align = (
            self.cfg.lateral_penalty_scale * abs_lat
            + self.cfg.yaw_penalty_scale * yaw_e
            + self.cfg.z_penalty_scale * torch.abs(z_err)
        )
        clr_scale = float(self.cfg.clearance_violation_scale)
        ramp_d = float(getattr(self.cfg, "clearance_ramp_depth_m", 0.03))
        ramp = torch.clamp(front_to_mouth / ramp_d, 0.0, 1.0)
        # Past mouth: ramp clearance cost from 50% -> 100% over ramp_d (ease cliff at insertion).
        clr_ramp = torch.where(front_to_mouth > 0.0, 0.5 + 0.5 * ramp, torch.ones_like(front_to_mouth))
        clearance_pen = clr_scale * clr_ramp * clearance_violation
        align_pen = base_align + clearance_pen

        # Extra forward incentive when roughly aligned: encourages "align then insert" instead of dithering.
        aligned_for_bonus = (abs_lat < float(self.cfg.aligned_bonus_lat_thresh)) & (
            abs_yaw < float(self.cfg.aligned_bonus_yaw_thresh)
        )
        aligned_forward = self.cfg.aligned_forward_bonus_scale * torch.clamp(d_mouth, min=0.0, max=0.02)
        aligned_forward = torch.where(aligned_for_bonus, aligned_forward, torch.zeros_like(aligned_forward))

        # Slot funnel: one-time entry reward + small per-step (avoids multi-thousand return from camping).
        sr_lat = float(getattr(self.cfg, "slot_reach_lat_thresh", 0.012))
        sr_yaw = float(getattr(self.cfg, "slot_reach_yaw_thresh", 0.21))
        sr_z = float(getattr(self.cfg, "slot_reach_z_thresh", 0.018))
        sr_win = float(getattr(self.cfg, "slot_reach_mouth_window_m", 0.10))
        in_slot_funnel = (
            (abs_lat < sr_lat)
            & (abs_yaw < sr_yaw)
            & (torch.abs(z_err) < sr_z)
            & (front_to_mouth > -sr_win)
            & (front_to_mouth < enter_m)
        )
        first_funnel_enter = in_slot_funnel & ~self._slot_reach_entry_credited
        entry_bonus = float(getattr(self.cfg, "slot_reach_entry_bonus", 10.0))
        slot_reach_once = entry_bonus * first_funnel_enter.float()
        self._slot_reach_entry_credited = self._slot_reach_entry_credited | first_funnel_enter
        per_step = float(getattr(self.cfg, "slot_reach_per_step_scale", 0.12))
        slot_reach = slot_reach_once + per_step * in_slot_funnel.float()

        corners_r = self._book_corners_env()
        lowest_z = corners_r[..., 2].min(dim=-1).values
        p_env = self._book_pos_env()
        on_shelf = self._book_supported_on_shelf(p_env, lowest_z)
        post_push = float(getattr(self.cfg, "post_insert_push_scale", 0.0)) * torch.clamp(d_mouth, 0.0, 0.02)
        post_push = post_push * on_shelf.float() * (front_to_mouth > 0.0).float() * (front_to_mouth < enter_m).float()
        stall_thr = float(getattr(self.cfg, "shelf_stall_d_mouth_thresh", 0.0008))
        stall_pen_s = float(getattr(self.cfg, "shelf_stagnation_penalty_scale", 0.0))
        # Only past mouth, pre-success: penalize idle random actions instead of pushing deeper.
        shelf_stall = (
            on_shelf & (front_to_mouth > 0.0) & (front_to_mouth < enter_m) & (d_mouth < stall_thr)
        )
        shelf_stagnation_pen = stall_pen_s * shelf_stall.float()

        dact = self.actions - self._last_actions
        dact_pen = self.cfg.action_delta_penalty_scale * torch.sum(dact**2, dim=-1)

        # Archive-inspired anti-dither: penalize x-action sign flips near mouth plane.
        near_mouth = torch.abs(front_to_mouth) < float(self.cfg.mouth_dither_window)
        dx_now = self.actions[:, 0]
        dx_prev = self._last_actions[:, 0]
        flip = (dx_now * dx_prev) < 0.0
        active = (dx_now.abs() > float(self.cfg.dx_signflip_active_thresh)) & (
            dx_prev.abs() > float(self.cfg.dx_signflip_active_thresh)
        )
        dx_signflip_pen = self.cfg.dx_signflip_penalty_scale * (near_mouth & flip & active).float()

        # Before reaching mouth, soften harsh alignment penalties to avoid local probe/retreat traps.
        pre_mouth = front_to_mouth < 0.0
        pen_scale = torch.where(
            pre_mouth,
            torch.full_like(front_to_mouth, float(self.cfg.pre_mouth_penalty_scale)),
            torch.ones_like(front_to_mouth),
        )
        align_pen = align_pen * pen_scale

        success = self._success_steps_buf >= self.cfg.success_steps
        success_r = self.cfg.success_bonus * success.float()

        rew = (
            progress
            + depth
            + depth_sq
            + center_prog
            + yaw_prog
            + aligned_forward
            + slot_reach
            + post_push
            - shelf_stagnation_pen
            - align_pen
            - dact_pen
            - dx_signflip_pen
            + self.cfg.step_penalty
            + success_r
        )

        self._prev_front_to_mouth = front_to_mouth.detach()
        self._prev_abs_lat_err = abs_lat.detach()
        self._prev_abs_yaw_err = abs_yaw.detach()
        self._last_actions = self.actions.detach()
        return rew

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        front_to_mouth, _, lat_extent, z_err, yaw_err, front_to_back = self._stage_metrics()
        yaw_e = torch.abs(yaw_err)
        upright = self._upright_ok()

        inner_half = 0.5 * (self.cfg.book_size[2] + self.cfg.slot_lateral_clearance)
        lat_limit = inner_half - float(self.cfg.success_lateral_margin)
        lat_ok = lat_extent <= (inner_half - float(self.cfg.success_lateral_margin) + 1e-4)
        depth_ok = front_to_mouth > float(self.cfg.success_enter_margin)
        if float(self.cfg.success_back_margin) > 0.0:
            not_too_deep = front_to_back > float(self.cfg.success_back_margin)
        else:
            not_too_deep = torch.ones_like(depth_ok)
        z_ok = torch.abs(z_err) < float(self.cfg.success_z_thresh)
        yaw_ok = yaw_e < float(self.cfg.success_yaw_thresh)

        # Optional stability gate.
        v = self.book.data.root_link_vel_w
        lin_speed = torch.linalg.norm(v[:, 0:3], dim=-1)
        ang_speed = torch.linalg.norm(v[:, 3:6], dim=-1)
        if float(self.cfg.success_max_lin_vel) > 0.0:
            stable_lin = lin_speed < float(self.cfg.success_max_lin_vel)
        else:
            stable_lin = torch.ones_like(depth_ok)
        if float(self.cfg.success_max_ang_vel) > 0.0:
            stable_ang = ang_speed < float(self.cfg.success_max_ang_vel)
        else:
            stable_ang = torch.ones_like(depth_ok)

        # Avoid "success on reset": require a few env steps before success can start counting.
        ready = self.episode_length_buf > int(self.cfg.min_steps_before_success)
        stage_insert = depth_ok & not_too_deep & lat_ok & z_ok & yaw_ok & upright & stable_lin & stable_ang & ready

        if bool(getattr(self.cfg, "debug_print_success", False)):
            every = max(1, int(getattr(self.cfg, "debug_print_success_every", 1)))
            step_ctr = self.common_step_counter.item() if hasattr(self.common_step_counter, "item") else self.common_step_counter
            if int(step_ctr) % every == 0:
                i = 0
                print(
                    "[SUCCESS_GATES] env=0 "
                    f"ep_len={int(self.episode_length_buf[i].item())} "
                    f"front_to_mouth={float(front_to_mouth[i].item()):+.4f}>(enter {float(self.cfg.success_enter_margin):.4f})={bool(depth_ok[i].item())} "
                    f"front_to_back={float(front_to_back[i].item()):+.4f}>(back {float(self.cfg.success_back_margin):.4f})={bool(not_too_deep[i].item())} "
                    f"lat_extent={float(lat_extent[i].item()):.4f}<(lim {float(lat_limit):.4f})={bool(lat_ok[i].item())} "
                    f"z_err={float(z_err[i].item()):+.4f}<(thr {float(self.cfg.success_z_thresh):.4f})={bool(z_ok[i].item())} "
                    f"yaw_e={float(yaw_e[i].item()):.4f}<(thr {float(self.cfg.success_yaw_thresh):.4f})={bool(yaw_ok[i].item())} "
                    f"upright={bool(upright[i].item())} "
                    f"lin={float(lin_speed[i].item()):.3f}<(thr {float(self.cfg.success_max_lin_vel):.3f})={bool(stable_lin[i].item())} "
                    f"ang={float(ang_speed[i].item()):.3f}<(thr {float(self.cfg.success_max_ang_vel):.3f})={bool(stable_ang[i].item())} "
                    f"hold_steps={int(self._success_steps_buf[i].item())}/{int(self.cfg.success_steps)} "
                    f"stage_insert={bool(stage_insert[i].item())}"
                )

        self._success_steps_buf = torch.where(
            stage_insert, self._success_steps_buf + 1, torch.zeros_like(self._success_steps_buf)
        )
        success = self._success_steps_buf >= int(self.cfg.success_steps)

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        p = self._book_pos_env()
        corners_w = self._book_corners_env()
        lowest_z = corners_w[..., 2].min(dim=-1).values
        floor_z = float(getattr(self.cfg, "book_floor_lowest_z_thresh", 0.042))
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

        # Debug: print which termination triggered (first terminated env only).
        if bool(torch.any(terminated)):
            i = int(torch.nonzero(terminated, as_tuple=False)[0, 0].item())
            # print(
            #     "[DONE] env="
            #     f"{i} success={bool(success[i].item())} book_drop={bool(book_dropped_to_ground[i].item())} "
            #     f"oob={bool(oob[i].item())} fell={bool(fell[i].item())} "
            #     f"steps={int(self._success_steps_buf[i].item())} "
            #     f"front_to_mouth={float(front_to_mouth[i].item()):+.4f} "
            #     f"front_to_back={float(front_to_back[i].item()):+.4f} "
            #     f"lat_extent={float(lat_extent[i].item()):.4f} "
            #     f"z_err={float(z_err[i].item()):+.4f} "
            #     f"yaw_e={float(yaw_e[i].item()):.4f} "
            #     f"upright={bool(upright[i].item())} "
            #     f"lin={float(lin_speed[i].item()):.3f} ang={float(ang_speed[i].item()):.3f}"
            # )
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids_t = self._env_ids
        else:
            env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        super()._reset_idx(env_ids_t)

        # Clear any previously latched action for these envs.
        # (Manual stepping can continue calling step() even when no new input was parsed.)
        if hasattr(self, "actions") and isinstance(self.actions, torch.Tensor):
            self.actions[env_ids_t] = 0.0

        robot_default = self.robot.data.default_root_state[env_ids_t].clone()
        robot_default[:, 0:3] += self.scene.env_origins[env_ids_t]
        self.robot.write_root_state_to_sim(robot_default, env_ids=env_ids_t)
        joint_pos = self.robot.data.default_joint_pos[env_ids_t].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids_t].clone()

        # Reset randomization on joints (book pose changes indirectly via grasp frame).
        noise = float(getattr(self.cfg, "reset_arm_joint_pos_noise", 0.0))
        if noise > 0.0 and len(self._arm_joint_ids) > 0:
            n = int(env_ids_t.numel())
            j = len(self._arm_joint_ids)
            dq = sample_uniform(-noise, noise, (n, j), self.device)
            joint_pos[:, self._arm_joint_ids] = joint_pos[:, self._arm_joint_ids] + dq
            # Clamp to soft limits to avoid invalid reset poses.
            lo = self.robot.data.soft_joint_pos_limits[env_ids_t][:, self._arm_joint_ids, 0]
            hi = self.robot.data.soft_joint_pos_limits[env_ids_t][:, self._arm_joint_ids, 1]
            joint_pos[:, self._arm_joint_ids] = torch.max(torch.min(joint_pos[:, self._arm_joint_ids], hi), lo)

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

        # IMPORTANT: re-seed the arm hold target from the current measured joint pose after reset/settle.
        # Otherwise, action==0 can still "continue" towards an old hold setpoint from the previous episode.
        self._arm_hold_joint_pos[env_ids_t] = self.robot.data.joint_pos[env_ids_t][:, self._arm_joint_ids].clone()
        self.robot.set_joint_position_target(
            self._arm_hold_joint_pos[env_ids_t], joint_ids=self._arm_joint_ids, env_ids=env_ids_t
        )
        if len(self._finger_joint_ids) > 0:
            finger_des = self.robot.data.default_joint_pos[env_ids_t][:, self._finger_joint_ids]
            self.robot.set_joint_position_target(finger_des, joint_ids=self._finger_joint_ids, env_ids=env_ids_t)

        # Initialize integrated target to current EE tool pose and yaw.
        ee_body_pos_env = self.robot.data.body_pos_w[env_ids_t, self._ee_body_idx] - self.scene.env_origins[env_ids_t]
        ee_body_quat_w = self.robot.data.body_quat_w[env_ids_t, self._ee_body_idx]
        offset_w = math_utils.quat_apply(ee_body_quat_w, self._ik_body_offset_pos_b[env_ids_t])
        ee_tool_pos_env = ee_body_pos_env + offset_w
        self._target_pos_env[env_ids_t] = ee_tool_pos_env

        _, ee_quat_b = self._ee_pose_in_base()
        _, _, ee_yaw_b = math_utils.euler_xyz_from_quat(ee_quat_b[env_ids_t])
        self._target_yaw[env_ids_t] = ee_yaw_b

        corners0 = self._book_corners_env()[env_ids_t]
        front_x0 = corners0[..., 0].max(dim=-1).values
        self._prev_front_to_mouth[env_ids_t] = (front_x0 - float(self.cfg.slot_x_open)).detach()
        self._last_actions[env_ids_t] = 0.0
        self._success_steps_buf[env_ids_t] = 0
        # Reset progress buffers.
        _, lat_err0, _, _, yaw_err0, _ = self._stage_metrics()
        self._prev_abs_lat_err[env_ids_t] = torch.abs(lat_err0)[env_ids_t].detach()
        self._prev_abs_yaw_err[env_ids_t] = torch.abs(yaw_err0)[env_ids_t].detach()
        self._slot_reach_entry_credited[env_ids_t] = False
