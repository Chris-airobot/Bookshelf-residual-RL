#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bookshelf-Direct-v4 environment.

Hybrid control:
- learned INSERT mode
- scripted RELEASE+RETREAT mode
- learned PUSH mode

One shared RL policy is used for INSERT and PUSH.
In INSERT, action[4] > cfg.release_trigger_threshold requests scripted release.
In SCRIPTED and PUSH, gripper is forced open.
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
# from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import math as math_utils
from isaaclab.utils.math import sample_uniform

from .bookshelf_env_cfg_v4 import BookshelfEnvCfg

_MODE_INSERT = 0
_MODE_SCRIPTED = 1
_MODE_PUSH = 2


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


def _neighbor_book_dims(cfg: BookshelfEnvCfg) -> tuple[float, float, float]:
    """Cuboid (L, H, T) for slot-defining neighbor meshes; defaults to book_size."""
    nbs = getattr(cfg, "neighbor_book_size", None)
    if nbs is not None:
        return (float(nbs[0]), float(nbs[1]), float(nbs[2]))
    b = cfg.book_size
    return (float(b[0]), float(b[1]), float(b[2]))


def _geom_slot_mouth_x_from_neighbor(cfg: BookshelfEnvCfg) -> float:
    """Env-X of the robot-facing side of the slot-defining neighbor books."""
    x0 = float(cfg.slot_x_open)
    x1 = float(cfg.slot_x_back)
    mid_x = 0.5 * (x0 + x1)
    nb = _neighbor_book_dims(cfg)
    half = 0.5 * torch.tensor(list(nb), dtype=torch.float32)
    corners_l = _cuboid_corners_local(half)
    qw, qx, qy, qz = cfg.book_standing_quat
    quat = torch.tensor([qw, qx, qy, qz], dtype=torch.float32).unsqueeze(0).expand(8, 4)
    corners_rot = math_utils.quat_apply(quat, corners_l)
    return float(mid_x + corners_rot[:, 0].min().item())


class BookshelfEnv(DirectRLEnv):
    """Bookshelf placement env with one shared policy and minimal hybrid mode logic."""

    cfg: BookshelfEnvCfg

    def __init__(self, cfg: BookshelfEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self.robot: Articulation = self.scene.articulations["robot"]
        self.book: RigidObject = self.scene.rigid_objects["book"]

        # Finger bodies for grasp midpoint
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

        # Hybrid mode buffers
        self._mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._mode_start = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._script_step_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self._release_request = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._success_steps_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._prev_rear_to_mouth = torch.zeros(self.num_envs, device=self.device)
        self._prev_front_to_back = torch.zeros(self.num_envs, device=self.device)
        self._step_metrics: dict[str, torch.Tensor] = {}

        # Integrated Cartesian target
        self._target_pos_env = torch.zeros((self.num_envs, 3), device=self.device)
        self._target_yaw = torch.zeros(self.num_envs, device=self.device)

        # Precompute book corners for geometry checks.
        half = 0.5 * torch.tensor(self.cfg.book_size, device=self.device, dtype=torch.float32)
        self._book_corners_local = _cuboid_corners_local(half).to(device=self.device, dtype=torch.float32)

        # Mouth plane derived from neighbor geometry
        self._geom_mouth_x = _geom_slot_mouth_x_from_neighbor(self.cfg)
        self._neighbor_thick_y = float(_neighbor_book_dims(self.cfg)[2])

        # Hold arm target when action≈0
        self._arm_hold_joint_pos = self.robot.data.default_joint_pos[:, self._arm_joint_ids].clone()

    def _grasp_frame_pose_w(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lf_pos = self.robot.data.body_pos_w[env_ids, self._left_finger_body_idx]
        rf_pos = self.robot.data.body_pos_w[env_ids, self._right_finger_body_idx]
        grasp_pos_w = 0.5 * (lf_pos + rf_pos)
        grasp_quat_w = self.robot.data.body_quat_w[env_ids, self._hand_body_idx]
        return grasp_pos_w, grasp_quat_w

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

    def _snap_book_to_measured_grasp(self, env_ids_t: torch.Tensor) -> torch.Tensor:
        """Place book from measured finger-midpoint so it starts inside the gripper.

        Returns the written book world state (N, 13) so the caller can hold the book
        at this exact pose during warmup without re-sampling jitter.
        """
        n = int(env_ids_t.numel())
        dtype = torch.float32

        hx, hy, hz = self.cfg.book_grasp_offset_hand
        off = torch.tensor([hx, hy, hz], device=self.device, dtype=dtype).unsqueeze(0).expand(n, 3).clone()

        if float(self.cfg.book_grasp_x_jitter) != 0.0:
            off[:, 0] += sample_uniform(
                -float(self.cfg.book_grasp_x_jitter), float(self.cfg.book_grasp_x_jitter), (n,), self.device
            )
        if float(self.cfg.book_grasp_y_jitter) != 0.0:
            off[:, 1] += sample_uniform(
                -float(self.cfg.book_grasp_y_jitter), float(self.cfg.book_grasp_y_jitter), (n,), self.device
            )

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

        q_stand = (
            torch.tensor(self.cfg.book_standing_quat, device=self.device, dtype=dtype).unsqueeze(0).expand(n, 4).clone()
        )
        qyaw = self._quat_world_yaw_half(yaw_delta)
        book_quat_w = math_utils.quat_mul(q_stand, qyaw)

        book_state = self.book.data.default_root_state[env_ids_t].clone()
        book_state[:, 0:3] = book_pos_env + self.scene.env_origins[env_ids_t]
        book_state[:, 3:7] = book_quat_w
        book_state[:, 7:] = 0.0
        self.book.write_root_state_to_sim(book_state, env_ids=env_ids_t)
        return book_state

    def _ee_pose_in_base(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos_w = self.robot.data.body_pos_w[:, self._ee_body_idx]
        ee_quat_w = self.robot.data.body_quat_w[:, self._ee_body_idx]
        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        return ee_pos_b, ee_quat_b

    def _compute_ik_joint_targets_from_tool(self, target_pos_env: torch.Tensor, target_yaw: torch.Tensor) -> torch.Tensor:
        """Absolute IK: reach tool target in env frame with target yaw in base frame.

        Roll and pitch are kept from the current measured end-effector pose.
        """
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

    def _current_slot_lateral_clearance(self) -> float:
        """Return curriculum-adjusted slot clearance in meters."""
        if not bool(getattr(self.cfg, "enable_slot_clearance_curriculum", False)):
            return float(self.cfg.slot_lateral_clearance)

        c0 = float(getattr(self.cfg, "slot_lateral_clearance_start", self.cfg.slot_lateral_clearance))
        c1 = float(getattr(self.cfg, "slot_lateral_clearance_end", self.cfg.slot_lateral_clearance))
        steps = max(1, int(getattr(self.cfg, "slot_clearance_curriculum_steps", 1)))
        step_ctr = self.common_step_counter.item() if hasattr(self.common_step_counter, "item") else self.common_step_counter
        alpha = float(min(1.0, max(0.0, float(step_ctr) / float(steps))))
        return (1.0 - alpha) * c0 + alpha * c1

    def _gripper_open01(self) -> torch.Tensor:
        c = float(self.cfg.gripper_closed_joint_pos)
        o = float(self.cfg.gripper_open_joint_pos)
        if len(self._finger_joint_ids) > 0:
            fp = self.robot.data.joint_pos[:, self._finger_joint_ids]
            fmean = fp.mean(dim=-1)
            return torch.clamp((fmean - c) / (o - c + 1e-9), 0.0, 1.0)
        return torch.zeros(self.num_envs, device=self.device)

    def _ee_tool_pos_env(self) -> torch.Tensor:
        ee_body_pos_env = self.robot.data.body_pos_w[:, self._ee_body_idx] - self.scene.env_origins
        ee_body_quat_w = self.robot.data.body_quat_w[:, self._ee_body_idx]
        return ee_body_pos_env + math_utils.quat_apply(ee_body_quat_w, self._ik_body_offset_pos_b)

    def _compute_task_metrics(self) -> dict[str, torch.Tensor]:
        """Compact task metrics for hybrid insert/scripted/push control."""
        corners = self._book_corners_env()

        front_x = corners[..., 0].max(dim=-1).values
        rear_x = corners[..., 0].min(dim=-1).values
        mouth = float(self._geom_mouth_x)

        rear_to_mouth = rear_x - mouth
        front_to_back = float(self.cfg.slot_x_back) - front_x

        lat_err = self.cfg.slot_center_y - self._book_pos_env()[:, 1]
        lat_extent = torch.abs(corners[..., 1] - self.cfg.slot_center_y).max(dim=-1).values

        p = self._book_pos_env()
        z_target = self.cfg.shelf_top_z + self.cfg.shelf_thickness + 0.5 * self.cfg.book_size[1]
        z_err = p[:, 2] - float(z_target)

        yaw = _yaw_from_quat_wxyz(self.book.data.root_link_quat_w)
        yaw_err = _wrap_to_pi(yaw)

        v = self.book.data.root_link_vel_w
        book_lin_speed = torch.linalg.norm(v[:, 0:3], dim=-1)
        book_ang_speed = torch.linalg.norm(v[:, 3:6], dim=-1)

        gripper_open = self._gripper_open01()

        return {
            "rear_to_mouth": rear_to_mouth,
            "front_to_back": front_to_back,
            "lat_err": lat_err,
            "lat_extent": lat_extent,
            "z_err": z_err,
            "yaw_err": yaw_err,
            "book_lin_speed": book_lin_speed,
            "book_ang_speed": book_ang_speed,
            "gripper_open": gripper_open,
        }

    def _book_supported_on_shelf(self, p_env: torch.Tensor, lowest_z: torch.Tensor) -> torch.Tensor:
        """COM over shelf footprint and lowest corner on/near the deck."""
        x0 = float(self.cfg.slot_x_open)
        x1 = float(self.cfg.slot_x_back)
        mid_x = 0.5 * (x0 + x1)
        depth_x = x1 - x0 + 0.06
        hx = 0.5 * depth_x + float(self.cfg.shelf_footprint_x_pad_m)

        bthick = self._neighbor_thick_y
        clearance = self._current_slot_lateral_clearance()
        inner_half = 0.5 * (bthick + clearance)
        n_extra = max(0, int(self.cfg.shelf_extra_books_per_side))
        pitch_y = bthick + float(self.cfg.neighbor_book_pitch_gap)
        shelf_half_y = inner_half + bthick + n_extra * pitch_y
        hy = shelf_half_y + float(self.cfg.shelf_footprint_y_pad_m)

        cy = float(self.cfg.slot_center_y)
        deck_z = float(self.cfg.shelf_top_z + self.cfg.shelf_thickness)
        slack = float(self.cfg.book_on_shelf_z_slack_m)

        in_xy = (
            (p_env[:, 0] >= mid_x - hx)
            & (p_env[:, 0] <= mid_x + hx)
            & (p_env[:, 1] >= cy - hy)
            & (p_env[:, 1] <= cy + hy)
        )
        lowest_on_deck = lowest_z >= deck_z - slack
        return in_xy & lowest_on_deck

    def _setup_scene(self):
        # spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        x0 = self.cfg.slot_x_open
        x1 = self.cfg.slot_x_back
        mid_x = 0.5 * (x0 + x1)
        depth_x = x1 - x0 + 0.06
        nb = _neighbor_book_dims(self.cfg)
        blen, bheight, bthick = float(nb[0]), float(nb[1]), float(nb[2])

        if bool(getattr(self.cfg, "enable_slot_clearance_curriculum", False)):
            clearance = float(getattr(self.cfg, "slot_lateral_clearance_start", self.cfg.slot_lateral_clearance))
        else:
            clearance = float(self.cfg.slot_lateral_clearance)

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

        # Slot-defining neighbor books.
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

        # Spawn under env_0 only, clone once, then register.
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
        self._mode_start = self._mode.clone()
        self._release_request = self.actions[:, 4] > float(self.cfg.release_trigger_threshold)

    def _apply_action(self) -> None:
        mode = self._mode

        dx = self.actions[:, 0] * self.cfg.dx_action_scale
        dy = self.actions[:, 1] * self.cfg.dy_action_scale
        dz = self.actions[:, 2] * self.cfg.dz_action_scale
        dyaw = self.actions[:, 3] * self.cfg.dyaw_action_scale

        ee_tool_pos_env = self._ee_tool_pos_env()
        _, ee_quat_b = self._ee_pose_in_base()
        _, _, ee_yaw_b = math_utils.euler_xyz_from_quat(ee_quat_b)

        target_pos_env = ee_tool_pos_env.clone()
        target_yaw = ee_yaw_b.clone()

        normal_mask = mode != _MODE_SCRIPTED
        if torch.any(normal_mask):
            delta = torch.stack((dx, dy, dz), dim=-1)
            target_pos_env[normal_mask] = target_pos_env[normal_mask] + delta[normal_mask]
            target_yaw[normal_mask] = _wrap_to_pi(ee_yaw_b[normal_mask] + dyaw[normal_mask])

        scripted_mask = mode == _MODE_SCRIPTED
        if torch.any(scripted_mask):
            retreat_mask = scripted_mask & (self._script_step_buf >= int(self.cfg.script_open_steps))
            if torch.any(retreat_mask):
                target_pos_env[retreat_mask, 0] += float(self.cfg.script_retreat_dx)
                target_pos_env[retreat_mask, 2] += float(self.cfg.script_retreat_dz)

        joint_pos_des = self._compute_ik_joint_targets_from_tool(target_pos_env, target_yaw)

        arm_act = self.actions[:, 0:4]
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
            # Gripper is open only during the open+retreat sub-phases of SCRIPTED.
            # INSERT, PUSH, and the final close sub-phase of SCRIPTED all use closed.
            script_open_retreat = int(self.cfg.script_open_steps) + int(self.cfg.script_retreat_steps)
            gripper_should_open = (mode == _MODE_SCRIPTED) & (self._script_step_buf < script_open_retreat)
            finger_cmd = torch.where(
                gripper_should_open,
                torch.full((self.num_envs,), o, device=self.device, dtype=torch.float32),
                torch.full((self.num_envs,), c, device=self.device, dtype=torch.float32),
            )
            finger_des = finger_cmd.unsqueeze(-1).expand(self.num_envs, len(self._finger_joint_ids))
            self.robot.set_joint_position_target(finger_des, joint_ids=self._finger_joint_ids)

    def _get_observations(self) -> dict:
        m = self._compute_task_metrics()

        mode_obs = torch.where(
            self._mode == _MODE_INSERT,
            torch.full((self.num_envs,), float(self.cfg.mode_obs_insert), device=self.device),
            torch.where(
                self._mode == _MODE_SCRIPTED,
                torch.full((self.num_envs,), float(self.cfg.mode_obs_scripted), device=self.device),
                torch.full((self.num_envs,), float(self.cfg.mode_obs_push), device=self.device),
            ),
        )

        rear_s = torch.clamp(m["rear_to_mouth"] / float(self.cfg.rear_to_mouth_obs_scale), -1.0, 1.0)
        back_s = torch.clamp(m["front_to_back"] / float(self.cfg.front_to_back_obs_scale), -1.0, 1.0)
        lat_s = torch.clamp(m["lat_err"] / float(self.cfg.lat_err_obs_scale), -1.0, 1.0)
        z_s = torch.clamp(m["z_err"] / float(self.cfg.z_err_obs_scale), -1.0, 1.0)
        yaw_s = torch.clamp(m["yaw_err"] / float(self.cfg.yaw_err_obs_scale), -1.0, 1.0)

        tool_pos = self._ee_tool_pos_env()
        book_pos = self._book_pos_env()
        tool_to_book = tool_pos - book_pos
        ttb = float(self.cfg.tool_to_book_pos_obs_scale)
        hx_s = torch.clamp(tool_to_book[:, 0] / ttb, -1.0, 1.0)
        hy_s = torch.clamp(tool_to_book[:, 1] / ttb, -1.0, 1.0)
        hz_s = torch.clamp(tool_to_book[:, 2] / ttb, -1.0, 1.0)

        g_s = m["gripper_open"]

        obs = torch.stack(
            (
                mode_obs,
                rear_s,
                back_s,
                lat_s,
                z_s,
                yaw_s,
                hx_s,
                hy_s,
                hz_s,
                g_s,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        m = self._step_metrics
        if not m:
            m = self._compute_task_metrics()
            self._step_metrics = m

        mode_start = self._mode_start

        d_rear = m["rear_to_mouth"] - self._prev_rear_to_mouth
        d_back = self._prev_front_to_back - m["front_to_back"]  # positive when final seating improves

        insert_rew = (
            float(self.cfg.insert_progress_scale) * torch.clamp(d_rear, min=-0.02, max=0.02)
            - float(self.cfg.insert_lat_penalty_scale) * torch.abs(m["lat_err"])
            - float(self.cfg.insert_z_penalty_scale) * torch.abs(m["z_err"])
            - float(self.cfg.insert_yaw_penalty_scale) * torch.abs(m["yaw_err"])
        )

        push_rew = (
            float(self.cfg.push_progress_scale) * torch.clamp(d_back, min=-0.02, max=0.02)
            - float(self.cfg.push_lat_penalty_scale) * torch.abs(m["lat_err"])
            - float(self.cfg.push_z_penalty_scale) * torch.abs(m["z_err"])
            - float(self.cfg.push_yaw_penalty_scale) * torch.abs(m["yaw_err"])
        )

        scripted_rew = torch.zeros_like(insert_rew)

        rew_mode = torch.where(
            mode_start == _MODE_INSERT,
            insert_rew,
            torch.where(mode_start == _MODE_PUSH, push_rew, scripted_rew),
        )

        success = self._success_steps_buf >= int(self.cfg.success_steps)

        corners_w = self._book_corners_env()
        lowest_z = corners_w[..., 2].min(dim=-1).values
        floor_z = float(self.cfg.book_floor_lowest_z_thresh)
        book_touches_ground = lowest_z < floor_z
        p_env = self._book_pos_env()
        on_shelf = self._book_supported_on_shelf(p_env, lowest_z)
        book_dropped_to_ground = book_touches_ground & ~on_shelf

        rew = (
            rew_mode
            + float(self.cfg.step_penalty)
            + float(self.cfg.success_bonus) * success.float()
            + float(self.cfg.drop_penalty) * book_dropped_to_ground.float()
        )

        self._prev_rear_to_mouth = m["rear_to_mouth"].detach()
        self._prev_front_to_back = m["front_to_back"].detach()

        self.extras.setdefault("log", {})
        self.extras["log"]["reward_mean"] = rew.mean()
        self.extras["log"]["insert_mode_frac"] = (self._mode == _MODE_INSERT).float().mean()
        self.extras["log"]["scripted_mode_frac"] = (self._mode == _MODE_SCRIPTED).float().mean()
        self.extras["log"]["push_mode_frac"] = (self._mode == _MODE_PUSH).float().mean()

        return rew

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        m = self._compute_task_metrics()
        self._step_metrics = m

        mode_before = self._mode.clone()

        accepted_release = (mode_before == _MODE_INSERT) & self._release_request

        # Advance already-scripted envs
        scripted_old = mode_before == _MODE_SCRIPTED
        self._script_step_buf = torch.where(scripted_old, self._script_step_buf + 1, self._script_step_buf)

        script_total = int(self.cfg.script_open_steps) + int(self.cfg.script_retreat_steps) + int(self.cfg.script_close_steps)
        script_done = scripted_old & (self._script_step_buf >= script_total)

        # Enter scripted block
        self._mode = torch.where(accepted_release, torch.full_like(self._mode, _MODE_SCRIPTED), self._mode)
        self._script_step_buf = torch.where(accepted_release, torch.zeros_like(self._script_step_buf), self._script_step_buf)

        # Exit scripted block into push mode
        self._mode = torch.where(script_done, torch.full_like(self._mode, _MODE_PUSH), self._mode)
        self._script_step_buf = torch.where(script_done, torch.zeros_like(self._script_step_buf), self._script_step_buf)

        # Success only in PUSH mode
        lat_extent = m["lat_extent"]
        z_err = m["z_err"]
        yaw_err = m["yaw_err"]
        front_to_back = m["front_to_back"]
        rear_to_mouth = m["rear_to_mouth"]
        yaw_e = torch.abs(yaw_err)
        upright = self._upright_ok()

        curr_clearance = self._current_slot_lateral_clearance()
        inner_half = 0.5 * (self._neighbor_thick_y + curr_clearance)
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
            terminated = success | book_dropped_to_ground

        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids_t = self._env_ids
        else:
            env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        super()._reset_idx(env_ids_t)

        if hasattr(self, "actions") and isinstance(self.actions, torch.Tensor):
            self.actions[env_ids_t] = 0.0

        robot_default = self.robot.data.default_root_state[env_ids_t].clone()
        robot_default[:, 0:3] += self.scene.env_origins[env_ids_t]
        self.robot.write_root_state_to_sim(robot_default, env_ids=env_ids_t)

        joint_pos = self.robot.data.default_joint_pos[env_ids_t].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids_t].clone()

        noise = float(getattr(self.cfg, "reset_arm_joint_pos_noise", 0.0))
        if noise > 0.0 and len(self._arm_joint_ids) > 0:
            n = int(env_ids_t.numel())
            j = len(self._arm_joint_ids)
            dq = sample_uniform(-noise, noise, (n, j), self.device)
            joint_pos[:, self._arm_joint_ids] = joint_pos[:, self._arm_joint_ids] + dq

            lo = self.robot.data.soft_joint_pos_limits[env_ids_t][:, self._arm_joint_ids, 0]
            hi = self.robot.data.soft_joint_pos_limits[env_ids_t][:, self._arm_joint_ids, 1]
            joint_pos[:, self._arm_joint_ids] = torch.max(torch.min(joint_pos[:, self._arm_joint_ids], hi), lo)

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids_t)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids_t)

        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)

        # Ensure book starts in the gripper for the (possibly updated) default robot joint pose.
        # Capture the exact written state before any physics runs (avoids residual shelf contacts
        # corrupting the position on subsequent resets).
        snapped_book_state = self._snap_book_to_measured_grasp(env_ids_t)
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)

        # Re-seed hold target
        self._arm_hold_joint_pos[env_ids_t] = self.robot.data.joint_pos[env_ids_t][:, self._arm_joint_ids].clone()
        self.robot.set_joint_position_target(
            self._arm_hold_joint_pos[env_ids_t], joint_ids=self._arm_joint_ids, env_ids=env_ids_t
        )

        if len(self._finger_joint_ids) > 0:
            finger_des = self.robot.data.default_joint_pos[env_ids_t][:, self._finger_joint_ids]
            self.robot.set_joint_position_target(finger_des, joint_ids=self._finger_joint_ids, env_ids=env_ids_t)

        # Hold the book at the exact snapped pose while gripper fingers converge.
        # Use the state returned by _snap_book_to_measured_grasp (pre-physics) so that
        # residual contact forces from the shelf on prior episodes cannot corrupt the target.
        warmup = int(getattr(self.cfg, "reset_warmup_steps", 0))
        if warmup > 0:
            for _ in range(warmup):
                self.book.write_root_state_to_sim(snapped_book_state, env_ids=env_ids_t)
                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                self.scene.update(dt=self.physics_dt)

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
        rear_x0 = corners0[..., 0].min(dim=-1).values
        mouth = float(self._geom_mouth_x)

        self._prev_rear_to_mouth[env_ids_t] = (rear_x0 - mouth).detach()
        self._prev_front_to_back[env_ids_t] = (float(self.cfg.slot_x_back) - front_x0).detach()

        self._success_steps_buf[env_ids_t] = 0
        self._mode[env_ids_t] = _MODE_INSERT
        self._mode_start[env_ids_t] = _MODE_INSERT
        self._script_step_buf[env_ids_t] = 0
        self._release_request[env_ids_t] = False