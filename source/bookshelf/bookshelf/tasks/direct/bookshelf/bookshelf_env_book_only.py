#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bookshelf-Direct-v4 (insert-only environment).

Insertion-only RL task:
- Robot keeps grasping the book (no release, no gripper action)
- Policy uses residual-on-current EE(tool) pose + differential IK; zero actions hold last arm setpoint (reduces sag)
- Success is geometry-only at the slot mouth (hold for a few steps)
- Full reset: desired book pose in front of the slot → inverse grasp → IK on ``panda_hand`` pose, then snap book root
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
    """Insertion-only env with minimal action/obs/reward."""

    cfg: BookshelfEnvCfg

    def __init__(self, cfg: BookshelfEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self.robot: Articulation = self.scene.articulations["robot"]
        self.book: RigidObject = self.scene.rigid_objects["book"]

        # Buffers.
        self._last_actions = torch.zeros((self.num_envs, 4), device=self.device)
        self._success_steps_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._prev_front_to_mouth = torch.zeros(self.num_envs, device=self.device)

        # Differential IK control on end-effector.
        self._arm_joint_ids, _ = self.robot.find_joints("panda_joint.*")
        self._finger_joint_ids, _ = self.robot.find_joints("panda_finger_joint.*")
        body_ids, body_names = self.robot.find_bodies("panda_hand")
        if len(body_ids) != 1:
            raise RuntimeError(f"Expected one panda_hand body. Got {len(body_ids)}: {body_names}")
        self._ee_body_idx = body_ids[0]
        lf_ids, _ = self.robot.find_bodies("panda_leftfinger")
        rf_ids, _ = self.robot.find_bodies("panda_rightfinger")
        if len(lf_ids) != 1 or len(rf_ids) != 1:
            raise RuntimeError("Expected panda_leftfinger and panda_rightfinger bodies for grasp frame.")
        self._left_finger_body_idx = lf_ids[0]
        self._right_finger_body_idx = rf_ids[0]

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

        # Persistent Cartesian target in env frame + yaw in base frame.
        self._target_pos_env = torch.zeros((self.num_envs, 3), device=self.device)
        self._target_yaw = torch.zeros(self.num_envs, device=self.device)

        # Precompute book corners for lateral extent and front-to-mouth.
        half = 0.5 * torch.tensor(self.cfg.book_size, device=self.device, dtype=torch.float32)
        self._book_corners_local = _cuboid_corners_local(half).to(device=self.device, dtype=torch.float32)

        # Arm joint hold target while action≈0 (do not chase measured q under book load).
        self._arm_hold_joint_pos = self.robot.data.default_joint_pos[:, self._arm_joint_ids].clone()

        # Pre-insert: max env +X of book cuboid corners when COM at origin with ``book_standing_quat``.
        self._book_front_offset_x_standing = self._compute_book_front_offset_x_standing()
        # Finger-midpoint offset from ``panda_hand`` body origin in hand frame (filled on first reset).
        self._grasp_mid_from_hand_body_hand: torch.Tensor | None = None

    def _compute_book_front_offset_x_standing(self) -> float:
        half = 0.5 * torch.tensor(self.cfg.book_size, device=self.device, dtype=torch.float32)
        corners_l = _cuboid_corners_local(half).to(self.device)
        sw, sx, sy, sz = self.cfg.book_standing_quat
        q = torch.tensor([[sw, sx, sy, sz]], device=self.device, dtype=torch.float32).expand(8, 4)
        c_rot = math_utils.quat_apply(q.reshape(-1, 4), corners_l.reshape(-1, 3)).view(8, 3)
        return float(c_rot[:, 0].max())

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

    def _ee_pose_in_base(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos_w = self.robot.data.body_pos_w[:, self._ee_body_idx]
        ee_quat_w = self.robot.data.body_quat_w[:, self._ee_body_idx]
        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        return ee_pos_b, ee_quat_b

    def _compute_ik_joint_targets_from_tool(
        self, target_pos_env: torch.Tensor, target_yaw: torch.Tensor
    ) -> torch.Tensor:
        """Absolute IK: reach ``target_pos_env`` (tool tip) with yaw ``target_yaw``; keep current roll/pitch."""
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

    def _compute_ik_joint_targets_from_hand_body(
        self, hand_pos_env: torch.Tensor, hand_quat_wxyz: torch.Tensor
    ) -> torch.Tensor:
        """IK ``panda_hand`` body: ``hand_pos_env`` is COM-style env offset (``pos_w - env_origins``); quat is world."""
        hand_pos_w = hand_pos_env + self.scene.env_origins
        hand_pos_b, hand_quat_b = math_utils.subtract_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            hand_pos_w,
            hand_quat_wxyz,
        )
        self._ik_cmd[:, 0:3] = hand_pos_b
        self._ik_cmd[:, 3:7] = hand_quat_b
        self._ik.set_command(self._ik_cmd)
        ee_pos_b, ee_quat_b = self._ee_pose_in_base()
        jacobian = self.robot.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, self._jacobi_joint_ids]
        joint_pos = self.robot.data.joint_pos[:, self._arm_joint_ids]
        return self._ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

    def _grasp_frame_pose_w(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lf_pos = self.robot.data.body_pos_w[env_ids, self._left_finger_body_idx]
        rf_pos = self.robot.data.body_pos_w[env_ids, self._right_finger_body_idx]
        grasp_pos_w = 0.5 * (lf_pos + rf_pos)
        grasp_quat_w = self.robot.data.body_quat_w[env_ids, self._ee_body_idx]
        return grasp_pos_w, grasp_quat_w

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

    def _stage_metrics(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (front_to_mouth, lat_err, lat_extent, z_err, yaw_err_abs)."""
        corners = self._book_corners_env()
        front_x = corners[..., 0].max(dim=-1).values
        # True slot mouth plane is at cfg.slot_x_open.
        front_to_mouth = front_x - float(self.cfg.slot_x_open)

        # Lateral metrics.
        lat_err = self.cfg.slot_center_y - self._book_pos_env()[:, 1]
        lat_extent = torch.abs(corners[..., 1] - self.cfg.slot_center_y).max(dim=-1).values

        # Height error relative to shelf standing target.
        p = self._book_pos_env()
        z_target = self.cfg.shelf_top_z + self.cfg.shelf_thickness + 0.5 * self.cfg.book_size[1]
        z_err = p[:, 2] - z_target

        yaw = _yaw_from_quat_wxyz(self.book.data.root_link_quat_w)
        yaw_err_abs = torch.abs(_wrap_to_pi(yaw))
        return front_to_mouth, lat_err, lat_extent, z_err, yaw_err_abs

    def _cache_grasp_mid_offset_hand_if_needed(self) -> None:
        """Fill ``panda_hand`` origin → finger-midpoint vector in hand frame (from current sim)."""
        if self._grasp_mid_from_hand_body_hand is not None:
            return
        hand_pos_w = self.robot.data.body_pos_w[:, self._ee_body_idx]
        grasp_pos_w, _ = self._grasp_frame_pose_w(self._env_ids)
        hand_quat_w = self.robot.data.body_quat_w[:, self._ee_body_idx]
        delta_w = grasp_pos_w - hand_pos_w
        delta_h = math_utils.quat_apply(math_utils.quat_conjugate(hand_quat_w), delta_w)
        self._grasp_mid_from_hand_body_hand = delta_h[0].detach().clone()

    def _q_body_to_hand_grasp(self, n: int, dtype: torch.dtype) -> torch.Tensor:
        mode = self.cfg.book_grasp_orientation_in_hand
        if mode == "franka_axes":
            fw, fx, fy, fz = self.cfg.book_to_hand_quat_franka_axes_wxyz
            return (
                torch.tensor([fw, fx, fy, fz], device=self.device, dtype=dtype).unsqueeze(0).expand(n, 4).clone()
            )
        if mode == "manual_quat":
            rw, rx, ry, rz = self.cfg.book_grasp_rel_quat_wxyz
            return (
                torch.tensor([rw, rx, ry, rz], device=self.device, dtype=dtype).unsqueeze(0).expand(n, 4).clone()
            )
        raise ValueError(f"Unknown book_grasp_orientation_in_hand: {mode}")

    def _pre_insert_book_desired_for(self, env_ids_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Desired book COM (env frame) and world quat for each resetting env: standing in front of slot."""
        n = int(env_ids_t.numel())
        device = self.device
        dtype = torch.float32
        target_front_x = float(self.cfg.slot_x_open + self.cfg.pre_insert_book_front_offset_from_mouth)
        com_x = target_front_x - self._book_front_offset_x_standing
        pos = torch.zeros(n, 3, device=device, dtype=dtype)
        pos[:, 0] = com_x
        pos[:, 1] = float(self.cfg.slot_center_y)
        pos[:, 2] = float(self.cfg.shelf_top_z + self.cfg.shelf_thickness + 0.5 * self.cfg.book_size[1])

        yaw_w = sample_uniform(
            -float(self.cfg.book_grasp_yaw_jitter),
            float(self.cfg.book_grasp_yaw_jitter),
            (n,),
            device,
        )
        q_wyaw = torch.stack(
            (
                torch.cos(0.5 * yaw_w),
                torch.zeros(n, device=device, dtype=dtype),
                torch.zeros(n, device=device, dtype=dtype),
                torch.sin(0.5 * yaw_w),
            ),
            dim=-1,
        )
        sw, sx, sy, sz = self.cfg.book_standing_quat
        standing = torch.tensor([sw, sx, sy, sz], device=device, dtype=dtype).unsqueeze(0).expand(n, 4).clone()
        q_book = math_utils.quat_mul(q_wyaw, standing)
        return pos, q_book

    def _hand_pose_for_preinsert_book(
        self,
        book_pos_env: torch.Tensor,
        book_quat_w: torch.Tensor,
        hand_offset_hand: torch.Tensor,
        grasp_yaw_delta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inverse of spawn: ``q_book = q_hand * q_b2h * q_grasp_yaw``, COM from finger mid + offset."""
        n = book_pos_env.shape[0]
        q_b2h = self._q_body_to_hand_grasp(n, book_quat_w.dtype)
        qyaw = self._quat_world_yaw_half(grasp_yaw_delta)
        q_comp = math_utils.quat_mul(q_b2h, qyaw)
        q_hand = math_utils.quat_mul(book_quat_w, math_utils.quat_inv(q_comp))

        p_grasp_env = book_pos_env - math_utils.quat_apply(q_hand, hand_offset_hand)
        v_mid = self._grasp_mid_from_hand_body_hand
        if v_mid is None:
            raise RuntimeError("Grasp mid offset not cached; call _cache_grasp_mid_offset_hand_if_needed first.")
        v_mid_b = v_mid.view(1, 3).expand(n, 3)
        p_hand_env = p_grasp_env - math_utils.quat_apply(q_hand, v_mid_b)
        return p_hand_env, q_hand

    def _write_book_root_state(
        self, env_ids_t: torch.Tensor, book_pos_env: torch.Tensor, book_quat_w: torch.Tensor
    ) -> None:
        book_state = self.book.data.default_root_state[env_ids_t].clone()
        book_state[:, 0:3] = book_pos_env + self.scene.env_origins[env_ids_t]
        book_state[:, 3:7] = book_quat_w
        book_state[:, 7:] = 0.0
        self.book.write_root_state_to_sim(book_state, env_ids=env_ids_t)

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

    def _snap_book_to_measured_grasp(
        self,
        env_ids_t: torch.Tensor,
        hand_offset_hand: torch.Tensor,
        grasp_yaw_delta: torch.Tensor,
    ) -> None:
        """Place book from **measured** finger mid + ``panda_hand`` quat so it stays in the gripper."""
        n = int(env_ids_t.numel())
        grasp_pos_w, grasp_quat_w = self._grasp_frame_pose_w(env_ids_t)
        offset_w = math_utils.quat_apply(grasp_quat_w, hand_offset_hand)
        book_pos_w = grasp_pos_w + offset_w
        book_pos_env = book_pos_w - self.scene.env_origins[env_ids_t]

        q_b2h = self._q_body_to_hand_grasp(n, grasp_quat_w.dtype)
        qyaw = self._quat_world_yaw_half(grasp_yaw_delta)
        book_quat_w = math_utils.quat_mul(grasp_quat_w, math_utils.quat_mul(q_b2h, qyaw))
        self._write_book_root_state(env_ids_t, book_pos_env, book_quat_w)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        dx = self.actions[:, 0] * self.cfg.dx_action_scale
        dy = self.actions[:, 1] * self.cfg.dy_action_scale
        dz = self.actions[:, 2] * self.cfg.dz_action_scale
        dyaw = self.actions[:, 3] * self.cfg.dyaw_action_scale

        # Integrated target: actions are residual deltas applied to a persistent (pos,yaw) command.
        # This is easier to learn than "residual on current pose" for insertion since it reduces setpoint jitter.
        target_pos_env = self._target_pos_env + torch.stack((dx, dy, dz), dim=-1)
        target_yaw = _wrap_to_pi(self._target_yaw + dyaw)

        joint_pos_des = self._compute_ik_joint_targets_from_tool(target_pos_env, target_yaw)

        act_small = self.actions.abs() < float(self.cfg.ik_hold_action_epsilon)
        hold_arm = act_small.all(dim=-1)
        move_arm = ~hold_arm
        move_exp = move_arm.unsqueeze(-1).expand_as(joint_pos_des)
        self._arm_hold_joint_pos = torch.where(move_exp, joint_pos_des, self._arm_hold_joint_pos)
        hold_exp = hold_arm.unsqueeze(-1).expand_as(joint_pos_des)
        joint_pos_des = torch.where(hold_exp, self._arm_hold_joint_pos, joint_pos_des)
        self.robot.set_joint_position_target(joint_pos_des, joint_ids=self._arm_joint_ids)

        # Keep gripper fixed at default grasp (no learning).
        if len(self._finger_joint_ids) > 0:
            finger_des = self.robot.data.default_joint_pos[:, self._finger_joint_ids]
            self.robot.set_joint_position_target(finger_des, joint_ids=self._finger_joint_ids)

    def _get_observations(self) -> dict:
        front_to_mouth, lat_err, _, z_err, yaw_e = self._stage_metrics()
        v = self.book.data.root_link_vel_w

        # Light scaling/clipping for raw geometry + velocity channels.
        f2m_s = torch.clamp(front_to_mouth / float(self.cfg.front_to_mouth_obs_scale), -1.0, 1.0)
        lat_s = torch.clamp(lat_err / float(self.cfg.lat_err_obs_scale), -1.0, 1.0)
        yaw_s = torch.clamp(yaw_e / float(self.cfg.yaw_err_obs_scale), -1.0, 1.0)
        z_s = torch.clamp(z_err / float(self.cfg.z_err_obs_scale), -1.0, 1.0)
        vx_s = torch.clamp(v[:, 0] / float(self.cfg.lin_vel_obs_scale), -1.0, 1.0)
        vy_s = torch.clamp(v[:, 1] / float(self.cfg.lin_vel_obs_scale), -1.0, 1.0)
        wz_s = torch.clamp(v[:, 5] / float(self.cfg.ang_vel_obs_scale), -1.0, 1.0)

        # Tracking error (target vs current EE tool position in env frame, plus base yaw).
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
                self._last_actions[:, 0],
                self._last_actions[:, 1],
                self._last_actions[:, 2],
                self._last_actions[:, 3],
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        front_to_mouth, lat_err, lat_extent, z_err, yaw_e = self._stage_metrics()

        # Progress toward crossing the mouth plane.
        d_mouth = front_to_mouth - self._prev_front_to_mouth
        progress = self.cfg.progress_scale * torch.clamp(d_mouth, min=-0.02, max=0.02)

        # Small depth incentive after crossing.
        depth = self.cfg.depth_reward_scale * torch.clamp(front_to_mouth, min=0.0, max=0.08)

        # Alignment penalties.
        # Do not penalize raw lat_extent (it includes book size). Use centerline error and/or clearance violation.
        inner_half = 0.5 * (self.cfg.book_size[2] + self.cfg.slot_lateral_clearance)
        clearance_limit = inner_half - float(self.cfg.success_lateral_margin)
        clearance_violation = torch.clamp(lat_extent - clearance_limit, min=0.0)
        align_pen = (
            self.cfg.lateral_penalty_scale * torch.abs(lat_err)
            + self.cfg.yaw_penalty_scale * yaw_e
            + self.cfg.z_penalty_scale * torch.abs(z_err)
            + 10.0 * clearance_violation
        )

        dact = self.actions - self._last_actions
        dact_pen = self.cfg.action_delta_penalty_scale * torch.sum(dact**2, dim=-1)

        # Success bonus (sparse) when success is achieved (and stays achieved).
        success = self._success_steps_buf >= self.cfg.success_steps
        success_r = self.cfg.success_bonus * success.float()

        rew = progress + depth - align_pen - dact_pen + self.cfg.step_penalty + success_r

        self._prev_front_to_mouth = front_to_mouth.detach()
        self._last_actions = self.actions.detach()
        return rew

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        front_to_mouth, _, lat_extent, z_err, yaw_e = self._stage_metrics()
        upright = self._upright_ok()

        inner_half = 0.5 * (self.cfg.book_size[2] + self.cfg.slot_lateral_clearance)
        lat_ok = lat_extent < (inner_half - float(self.cfg.success_lateral_margin))
        mouth_ok = front_to_mouth > 0.0
        z_ok = torch.abs(z_err) < float(self.cfg.success_z_thresh)
        yaw_ok = yaw_e < float(self.cfg.success_yaw_thresh)

        stage_insert = mouth_ok & lat_ok & z_ok & yaw_ok & upright
        self._success_steps_buf = torch.where(
            stage_insert, self._success_steps_buf + 1, torch.zeros_like(self._success_steps_buf)
        )
        success = self._success_steps_buf >= int(self.cfg.success_steps)

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if bool(self.cfg.enable_failure_terminations):
            p = self._book_pos_env()
            oob = (torch.abs(p[:, 0]) > self.cfg.max_abs_xy) | (torch.abs(p[:, 1]) > self.cfg.max_abs_xy)
            fell = p[:, 2] < self.cfg.fell_height_thresh
            terminated = success | oob | fell
        else:
            terminated = success
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids_t = self._env_ids
        else:
            env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        super()._reset_idx(env_ids_t)

        # Reset robot to default pose (pre-grasp).
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

        self._cache_grasp_mid_offset_hand_if_needed()

        nidx = int(env_ids_t.numel())
        book_pos_des, book_quat_des = self._pre_insert_book_desired_for(env_ids_t)
        hx, hy, hz = self.cfg.book_grasp_offset_hand
        hand_off = (
            torch.tensor([hx, hy, hz], device=self.device, dtype=book_pos_des.dtype)
            .unsqueeze(0)
            .expand(nidx, 3)
            .clone()
        )
        hand_off[:, 0] += sample_uniform(
            -self.cfg.book_grasp_x_jitter, self.cfg.book_grasp_x_jitter, (nidx,), self.device
        )
        hand_off[:, 1] += sample_uniform(
            -self.cfg.book_grasp_y_jitter, self.cfg.book_grasp_y_jitter, (nidx,), self.device
        )
        grasp_yaw_delta = sample_uniform(
            -float(self.cfg.book_grasp_yaw_jitter),
            float(self.cfg.book_grasp_yaw_jitter),
            (nidx,),
            self.device,
        )

        full_reset = nidx == int(self.num_envs)
        hand_pos_env, hand_quat = self._hand_pose_for_preinsert_book(
            book_pos_des, book_quat_des, hand_off, grasp_yaw_delta
        )

        ids_snap = self._env_ids if full_reset else env_ids_t
        self._snap_book_to_measured_grasp(ids_snap, hand_off, grasp_yaw_delta)

        if full_reset and int(self.cfg.reset_preinsert_ik_steps) > 0:
            for _ in range(int(self.cfg.reset_preinsert_ik_steps)):
                joint_pos_des = self._compute_ik_joint_targets_from_hand_body(hand_pos_env, hand_quat)
                self._arm_hold_joint_pos[:] = joint_pos_des.clone()
                self.robot.set_joint_position_target(joint_pos_des, joint_ids=self._arm_joint_ids)
                if len(self._finger_joint_ids) > 0:
                    finger_des = self.robot.data.default_joint_pos[:, self._finger_joint_ids]
                    self.robot.set_joint_position_target(finger_des, joint_ids=self._finger_joint_ids)
                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                self.scene.update(dt=self.physics_dt)
                self._snap_book_to_measured_grasp(self._env_ids, hand_off, grasp_yaw_delta)

        for _ in range(int(self.cfg.reset_book_contact_settle_steps)):
            if full_reset and int(self.cfg.reset_preinsert_ik_steps) > 0:
                joint_pos_des = self._compute_ik_joint_targets_from_hand_body(hand_pos_env, hand_quat)
                self._arm_hold_joint_pos[:] = joint_pos_des.clone()
                self.robot.set_joint_position_target(joint_pos_des, joint_ids=self._arm_joint_ids)
                if len(self._finger_joint_ids) > 0:
                    finger_des = self.robot.data.default_joint_pos[:, self._finger_joint_ids]
                    self.robot.set_joint_position_target(finger_des, joint_ids=self._finger_joint_ids)
            self._snap_book_to_measured_grasp(ids_snap, hand_off, grasp_yaw_delta)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

        # Refresh hold pose after settle so action=0 does not sag with the book.
        self._arm_hold_joint_pos[env_ids_t] = self.robot.data.joint_pos[env_ids_t][:, self._arm_joint_ids].clone()

        # Initialize target to current EE tool pose (env frame) and current yaw (base frame).
        ee_body_pos_env = self.robot.data.body_pos_w[env_ids_t, self._ee_body_idx] - self.scene.env_origins[env_ids_t]
        ee_body_quat_w = self.robot.data.body_quat_w[env_ids_t, self._ee_body_idx]
        offset_w = math_utils.quat_apply(ee_body_quat_w, self._ik_body_offset_pos_b[env_ids_t])
        ee_tool_pos_env = ee_body_pos_env + offset_w
        self._target_pos_env[env_ids_t] = ee_tool_pos_env

        _, ee_quat_b = self._ee_pose_in_base()
        _, _, ee_yaw_b = math_utils.euler_xyz_from_quat(ee_quat_b[env_ids_t])
        self._target_yaw[env_ids_t] = ee_yaw_b

        # Reset buffers.
        corners0 = self._book_corners_env()[env_ids_t]
        front_x0 = corners0[..., 0].max(dim=-1).values
        self._prev_front_to_mouth[env_ids_t] = (front_x0 - float(self.cfg.slot_x_open)).detach()
        self._last_actions[env_ids_t] = 0.0
        self._success_steps_buf[env_ids_t] = 0
