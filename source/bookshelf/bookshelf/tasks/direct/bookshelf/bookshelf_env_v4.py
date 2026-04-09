#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bookshelf-Direct-v4 environment.

Internal phases (single policy): insert (grasped) → release → retreat → push; terminal success only
in the push phase when full-placement geometry + stability gates hold.
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

_PHASE_INSERT = 0
_PHASE_RELEASE = 1
_PHASE_RETREAT = 2
_PHASE_PUSH = 3


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
    """Cuboid (L, H, T) for slot-defining neighbor meshes; defaults to ``book_size``."""
    nbs = getattr(cfg, "neighbor_book_size", None)
    if nbs is not None:
        return (float(nbs[0]), float(nbs[1]), float(nbs[2]))
    b = cfg.book_size
    return (float(b[0]), float(b[1]), float(b[2]))


def _geom_slot_mouth_x_from_neighbor(cfg: BookshelfEnvCfg) -> float:
    """Env-X of the robot-facing side of the slot-defining neighbor books.

    Neighbors are spawned as ``MeshCuboidCfg(size=neighbor_book_size or book_size)`` at ``mid_x`` with
    ``book_standing_quat``. The mouth plane for metrics is the minimum world-X over their 8 corners.
    """
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
    """Bookshelf placement env with Cartesian + pitch + yaw + gripper and an internal phase FSM."""

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
        self._success_steps_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._prev_front_to_mouth = torch.zeros(self.num_envs, device=self.device)
        self._prev_rear_to_mouth = torch.zeros(self.num_envs, device=self.device)
        self._phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._phase_start = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._release_ready_hold_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._release_open_stable_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._retreat_clear_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._prev_hand_clearance = torch.zeros(self.num_envs, device=self.device)
        self._prev_gripper_open = torch.zeros(self.num_envs, device=self.device)
        self._step_metrics: dict[str, torch.Tensor] = {}
        self._crossed_mouth_bonus_given = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._pending_insert_release_bonus = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Integrated Cartesian target (env frame + base yaw).
        self._target_pos_env = torch.zeros((self.num_envs, 3), device=self.device)
        self._target_yaw = torch.zeros(self.num_envs, device=self.device)

        # Precompute book corners for geometry checks.
        half = 0.5 * torch.tensor(self.cfg.book_size, device=self.device, dtype=torch.float32)
        self._book_corners_local = _cuboid_corners_local(half).to(device=self.device, dtype=torch.float32)

        # Slot mouth plane for metrics: derived from neighbor cuboid geometry (same as _setup_scene).
        self._geom_mouth_x = _geom_slot_mouth_x_from_neighbor(self.cfg)
        self._neighbor_thick_y = float(_neighbor_book_dims(self.cfg)[2])

        # Hold arm target when action≈0 (avoid sag under load).
        self._arm_hold_joint_pos = self.robot.data.default_joint_pos[:, self._arm_joint_ids].clone()

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
        # Keep book orientation fixed at the standing quaternion during reset.
        # We still allow position jitter (via `off` and the measured grasp pose), but avoid
        # random quaternion differences that can make early learning harder.
        book_quat_w = (
            torch.tensor(self.cfg.book_standing_quat, device=self.device, dtype=dtype)
            .unsqueeze(0)
            .expand(n, 4)
            .clone()
        )

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

    def _compute_ik_joint_targets_from_tool(
        self, target_pos_env: torch.Tensor, target_yaw: torch.Tensor, target_pitch_delta: torch.Tensor
    ) -> torch.Tensor:
        """Absolute IK: reach tool target in env frame with yaw+pitch in base frame.

        Roll is kept from the current measured tool pose. Pitch can be adjusted by `target_pitch_delta`.
        """
        self._target_pos_env[:] = target_pos_env
        self._target_yaw[:] = target_yaw
        _, ee_quat_b = self._ee_pose_in_base()
        ee_roll_b, ee_pitch_b, _ = math_utils.euler_xyz_from_quat(ee_quat_b)
        quat_des_b = math_utils.quat_from_euler_xyz(ee_roll_b, ee_pitch_b + target_pitch_delta, target_yaw)
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
        """Geometry, support, release readiness, speeds, and hand–book clearance (env frame).

        Book X naming matches the rest of this env: ``front_x = max(corner X)`` is the deepest edge into the
        slot (+X); ``rear_x = min(corner X)`` is the robot-side edge. ``mouth`` is the slot mouth plane.
        """
        corners = self._book_corners_env()
        front_x = corners[..., 0].max(dim=-1).values
        rear_x = corners[..., 0].min(dim=-1).values
        mouth = float(self._geom_mouth_x)
        front_to_mouth = front_x - mouth
        rear_to_mouth = rear_x - mouth
        front_to_back = float(self.cfg.slot_x_back) - front_x

        lat_err = self.cfg.slot_center_y - self._book_pos_env()[:, 1]
        lat_extent = torch.abs(corners[..., 1] - self.cfg.slot_center_y).max(dim=-1).values

        p = self._book_pos_env()
        z_target = self.cfg.shelf_top_z + self.cfg.shelf_thickness + 0.5 * self.cfg.book_size[1]
        z_err = p[:, 2] - float(z_target)

        yaw = _yaw_from_quat_wxyz(self.book.data.root_link_quat_w)
        yaw_err = _wrap_to_pi(yaw)

        lowest_z = corners[..., 2].min(dim=-1).values
        supported = self._book_supported_on_shelf(p, lowest_z)

        # Leading edge past mouth ⇒ front_to_mouth > threshold (not <= 0; that would be still outside / at plane).
        thr_fm = float(self.cfg.phase_release_ready_min_front_to_mouth)
        leading_edge_past_mouth = front_to_mouth > thr_fm
        if bool(self.cfg.phase_release_ready_requires_past_mouth):
            pm = leading_edge_past_mouth
        else:
            pm = torch.ones_like(leading_edge_past_mouth)
        partial_rear = rear_to_mouth >= float(self.cfg.phase_release_ready_rear_to_mouth_min)
        lat_ok = torch.abs(lat_err) <= float(self.cfg.phase_release_ready_max_abs_lat_err)
        yaw_ok = torch.abs(yaw_err) <= float(self.cfg.phase_release_ready_max_abs_yaw_err)
        z_ok = torch.abs(z_err) <= float(self.cfg.phase_release_ready_max_abs_z_err)
        sup_req = bool(self.cfg.phase_release_ready_requires_supported)
        sup_ok = supported if sup_req else torch.ones_like(supported)
        ready_time = self.episode_length_buf > int(self.cfg.min_steps_before_success)
        release_ready = pm & partial_rear & lat_ok & yaw_ok & z_ok & sup_ok & ready_time

        v = self.book.data.root_link_vel_w
        book_lin_speed = torch.linalg.norm(v[:, 0:3], dim=-1)
        book_ang_speed = torch.linalg.norm(v[:, 3:6], dim=-1)

        tool_pos = self._ee_tool_pos_env()
        # Proxy clearance: IK tool point vs book vertices (not finger collision geometry).
        dists = torch.linalg.norm(corners - tool_pos.unsqueeze(1), dim=-1)
        hand_clearance = dists.min(dim=-1).values

        gripper_open = self._gripper_open01()

        return {
            "front_to_mouth": front_to_mouth,
            "lat_err": lat_err,
            "lat_extent": lat_extent,
            "z_err": z_err,
            "yaw_err": yaw_err,
            "front_to_back": front_to_back,
            "rear_to_mouth": rear_to_mouth,
            "supported_on_shelf": supported,
            "release_ready": release_ready,
            "book_lin_speed": book_lin_speed,
            "book_ang_speed": book_ang_speed,
            "hand_clearance": hand_clearance,
            "gripper_open": gripper_open,
        }

    def _book_supported_on_shelf(self, p_env: torch.Tensor, lowest_z: torch.Tensor) -> torch.Tensor:
        """COM over shelf footprint and **lowest corner** on/near the deck — not merely COM height (avoids floor false negatives)."""
        x0 = float(self.cfg.slot_x_open)
        x1 = float(self.cfg.slot_x_back)
        mid_x = 0.5 * (x0 + x1)
        depth_x = x1 - x0 + 0.06
        hx = 0.5 * depth_x + float(self.cfg.shelf_footprint_x_pad_m)
        bthick = self._neighbor_thick_y
        clearance = self._current_slot_lateral_clearance()
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
        nb = _neighbor_book_dims(self.cfg)
        blen, bheight, bthick = float(nb[0]), float(nb[1]), float(nb[2])
        # Shelf is spawned once; if curriculum is enabled, start with the wider setting.
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
        self._phase_start = self._phase.clone()

    def _apply_action(self) -> None:
        ph = self._phase
        s_ins = float(self.cfg.phase_insert_arm_scale)
        s_rel = float(self.cfg.phase_release_arm_scale)
        s_ret = float(self.cfg.phase_retreat_arm_scale)
        s_push = float(self.cfg.phase_push_arm_scale)
        arm_scale = torch.where(
            ph == _PHASE_INSERT,
            torch.full((self.num_envs,), s_ins, device=self.device, dtype=torch.float32),
            torch.where(
                ph == _PHASE_RELEASE,
                torch.full((self.num_envs,), s_rel, device=self.device, dtype=torch.float32),
                torch.where(
                    ph == _PHASE_RETREAT,
                    torch.full((self.num_envs,), s_ret, device=self.device, dtype=torch.float32),
                    torch.full((self.num_envs,), s_push, device=self.device, dtype=torch.float32),
                ),
            ),
        )
        dx_ins_m = float(getattr(self.cfg, "phase_insert_dx_multiplier", 1.0))
        dx_mul = torch.where(
            ph == _PHASE_INSERT,
            torch.full((self.num_envs,), dx_ins_m, device=self.device, dtype=torch.float32),
            torch.ones((self.num_envs,), device=self.device, dtype=torch.float32),
        )

        dx = self.actions[:, 0] * self.cfg.dx_action_scale * arm_scale * dx_mul
        dy = self.actions[:, 1] * self.cfg.dy_action_scale * arm_scale
        dz = self.actions[:, 2] * self.cfg.dz_action_scale * arm_scale
        dpitch = self.actions[:, 3] * self.cfg.dpitch_action_scale * arm_scale
        dyaw = self.actions[:, 4] * self.cfg.dyaw_action_scale * arm_scale

        ee_body_pos_env = self.robot.data.body_pos_w[:, self._ee_body_idx] - self.scene.env_origins
        ee_body_quat_w = self.robot.data.body_quat_w[:, self._ee_body_idx]
        ee_tool_offset_w = math_utils.quat_apply(ee_body_quat_w, self._ik_body_offset_pos_b)
        ee_tool_pos_env = ee_body_pos_env + ee_tool_offset_w

        _, ee_quat_b = self._ee_pose_in_base()
        _, _, ee_yaw_b = math_utils.euler_xyz_from_quat(ee_quat_b)

        target_pos_env = ee_tool_pos_env + torch.stack((dx, dy, dz), dim=-1)
        target_yaw = _wrap_to_pi(ee_yaw_b + dyaw)

        joint_pos_des = self._compute_ik_joint_targets_from_tool(target_pos_env, target_yaw, dpitch)

        arm_act = self.actions[:, 0:5]
        act_small = arm_act.abs() < float(self.cfg.ik_hold_action_epsilon)
        hold_arm = act_small.all(dim=-1)
        if bool(getattr(self.cfg, "ik_hold_disable_in_insert", False)):
            hold_arm = hold_arm & (ph != _PHASE_INSERT)
        move_arm = ~hold_arm
        move_exp = move_arm.unsqueeze(-1).expand_as(joint_pos_des)
        self._arm_hold_joint_pos = torch.where(move_exp, joint_pos_des, self._arm_hold_joint_pos)
        hold_exp = hold_arm.unsqueeze(-1).expand_as(joint_pos_des)
        joint_pos_des = torch.where(hold_exp, self._arm_hold_joint_pos, joint_pos_des)
        self.robot.set_joint_position_target(joint_pos_des, joint_ids=self._arm_joint_ids)

        if len(self._finger_joint_ids) > 0:
            g = self.actions[:, 5].clamp(-1.0, 1.0)
            c = float(self.cfg.gripper_closed_joint_pos)
            o = float(self.cfg.gripper_open_joint_pos)
            thr = float(self.cfg.gripper_open_action_threshold)
            # INSERT / RETREAT / PUSH: keep gripper closed or open deterministically.
            finger_cmd = torch.full_like(g, c)
            in_release = ph == _PHASE_RELEASE
            finger_cmd = torch.where(in_release & (g >= thr), torch.full_like(g, o), finger_cmd)
            finger_cmd = torch.where(
                (ph == _PHASE_RETREAT) | (ph == _PHASE_PUSH), torch.full_like(g, o), finger_cmd
            )
            n_f = len(self._finger_joint_ids)
            finger_des = finger_cmd.unsqueeze(-1).expand(self.num_envs, n_f)
            self.robot.set_joint_position_target(finger_des, joint_ids=self._finger_joint_ids)

    def _get_observations(self) -> dict:
        m = self._compute_task_metrics()
        phase_s = torch.clamp(self._phase.to(dtype=torch.float32) / 3.0, 0.0, 1.0)

        f2m_s = torch.clamp(m["front_to_mouth"] / float(self.cfg.front_to_mouth_obs_scale), -1.0, 1.0)
        r2m_s = torch.clamp(m["rear_to_mouth"] / float(self.cfg.rear_to_mouth_obs_scale), -1.0, 1.0)
        f2b_s = torch.clamp(m["front_to_back"] / float(self.cfg.front_to_back_obs_scale), -1.0, 1.0)
        lat_s = torch.clamp(m["lat_err"] / float(self.cfg.lat_err_obs_scale), -1.0, 1.0)
        yaw_s = torch.clamp(m["yaw_err"] / float(self.cfg.yaw_err_obs_scale), -1.0, 1.0)
        z_s = torch.clamp(m["z_err"] / float(self.cfg.z_err_obs_scale), -1.0, 1.0)

        gripper_open = m["gripper_open"]
        sup_f = m["supported_on_shelf"].to(dtype=torch.float32)
        rr_f = m["release_ready"].to(dtype=torch.float32)
        hc_s = torch.clamp(m["hand_clearance"] / float(self.cfg.hand_clearance_obs_scale), -1.0, 1.0)

        ee_body_quat_w = self.robot.data.body_quat_w[:, self._ee_body_idx]
        tool_pos_env = self._ee_tool_pos_env()
        book_pos_env = self._book_pos_env()
        tool_to_book = tool_pos_env - book_pos_env
        ttb = float(self.cfg.tool_to_book_pos_obs_scale)
        tool_to_book_x_s = torch.clamp(tool_to_book[:, 0] / ttb, -1.0, 1.0)
        tool_to_book_y_s = torch.clamp(tool_to_book[:, 1] / ttb, -1.0, 1.0)
        tool_to_book_z_s = torch.clamp(tool_to_book[:, 2] / ttb, -1.0, 1.0)

        tool_yaw_w = _yaw_from_quat_wxyz(ee_body_quat_w)
        book_yaw_w = _yaw_from_quat_wxyz(self.book.data.root_link_quat_w)
        tool_yaw_minus_book_yaw = _wrap_to_pi(tool_yaw_w - book_yaw_w)
        tyaw_scale = float(self.cfg.tool_yaw_minus_book_yaw_obs_scale)
        tool_yaw_minus_book_yaw_s = torch.clamp(tool_yaw_minus_book_yaw / tyaw_scale, -1.0, 1.0)

        _, tool_p_w, _ = math_utils.euler_xyz_from_quat(ee_body_quat_w)
        _, book_p_w, _ = math_utils.euler_xyz_from_quat(self.book.data.root_link_quat_w)
        tool_pitch_minus_book_pitch = _wrap_to_pi(tool_p_w - book_p_w)
        tp_scale = float(self.cfg.tool_pitch_minus_book_pitch_obs_scale)
        tool_pitch_minus_book_pitch_s = torch.clamp(tool_pitch_minus_book_pitch / tp_scale, -1.0, 1.0)

        obs = torch.stack(
            (
                phase_s,
                f2m_s,
                lat_s,
                yaw_s,
                z_s,
                f2b_s,
                r2m_s,
                gripper_open,
                sup_f,
                rr_f,
                hc_s,
                tool_to_book_x_s,
                tool_to_book_y_s,
                tool_to_book_z_s,
                tool_yaw_minus_book_yaw_s,
                tool_pitch_minus_book_pitch_s,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # DirectRLEnv calls `_get_dones()` before `_get_rewards()`; recompute if cache is empty (defensive).
        m = self._step_metrics
        if not m:
            m = self._compute_task_metrics()
            self._step_metrics = m
        ps = self._phase_start

        front_to_mouth = m["front_to_mouth"]
        lat_err = m["lat_err"]
        z_err = m["z_err"]
        yaw_err = m["yaw_err"]
        rear_to_mouth = m["rear_to_mouth"]
        d_rear = rear_to_mouth - self._prev_rear_to_mouth
        fm_prev = self._prev_front_to_mouth
        d_front = front_to_mouth - fm_prev

        # --- INSERT ---
        rear_p = float(self.cfg.rew_insert_rear_progress_scale) * torch.clamp(d_rear, min=-0.02, max=0.02)
        # <= 0: leading edge at/before mouth; > 0: truly inside slot (matches release_ready mouth sign).
        pre_entry = front_to_mouth <= 0.0
        post_entry = front_to_mouth > 0.0
        pre_entry_forward = (
            float(self.cfg.rew_insert_pre_entry_forward_scale)
            * pre_entry.float()
            * torch.clamp(d_front, min=0.0, max=0.02)
        )
        post_push = (
            float(self.cfg.rew_insert_post_entry_push_scale)
            * post_entry.float()
            * torch.clamp(d_rear, min=0.0, max=0.02)
        )
        stall_mask = post_entry & (torch.abs(d_rear) < float(self.cfg.rew_insert_stall_thresh_m))
        post_stall_pen = float(self.cfg.rew_insert_stall_penalty_scale) * stall_mask.float()
        bottom_bonus = (
            float(self.cfg.rew_insert_bottom_reward_scale)
            * (m["supported_on_shelf"] & post_entry).float()
        )
        rr_bonus = float(self.cfg.rew_insert_release_ready_bonus_scale) * m["release_ready"].float()

        lo_h = float(self.cfg.rew_insert_hover_mouth_low_m)
        hi_h = float(self.cfg.rew_insert_hover_mouth_high_m)
        in_mouth_band = (front_to_mouth >= lo_h) & (front_to_mouth <= hi_h)
        tiny_rear = torch.abs(d_rear) < float(self.cfg.rew_insert_hover_d_rear_thresh_m)
        hover_mask = in_mouth_band & tiny_rear & (~m["release_ready"])
        hover_pen = float(self.cfg.rew_insert_hover_penalty_scale) * hover_mask.float()

        r_lo = float(self.cfg.rew_insert_abs_rear_to_mouth_clip_lo)
        r_hi = float(self.cfg.rew_insert_abs_rear_to_mouth_clip_hi)
        abs_rear = float(self.cfg.rew_insert_abs_rear_to_mouth_scale) * torch.clamp(rear_to_mouth, min=r_lo, max=r_hi)

        align_full = (
            float(self.cfg.rew_insert_lat_penalty_scale) * torch.abs(lat_err)
            + float(self.cfg.rew_insert_z_penalty_scale) * torch.abs(z_err)
            + float(self.cfg.rew_insert_yaw_penalty_scale) * torch.abs(yaw_err)
        )
        pre_mul = float(getattr(self.cfg, "rew_insert_align_penalty_pre_mul", 1.0))
        align_scale = pre_entry.float() * pre_mul + post_entry.float()
        align_ins = align_scale * align_full

        just_crossed = (fm_prev <= 0.0) & (front_to_mouth > 0.0) & (~self._crossed_mouth_bonus_given)
        mouth_cross_b = float(self.cfg.rew_insert_cross_mouth_bonus) * just_crossed.float()
        self._crossed_mouth_bonus_given = self._crossed_mouth_bonus_given | just_crossed

        trans_b = float(self.cfg.rew_insert_to_release_transition_bonus) * self._pending_insert_release_bonus.float()
        self._pending_insert_release_bonus.fill_(False)

        rew_ins = (
            rear_p
            + pre_entry_forward
            + post_push
            - post_stall_pen
            + bottom_bonus
            + rr_bonus
            + abs_rear
            - hover_pen
            + mouth_cross_b
            + trans_b
            - align_ins
        )

        # --- RELEASE ---
        dg = m["gripper_open"] - self._prev_gripper_open
        rew_open = float(self.cfg.rew_release_open_progress_scale) * torch.clamp(dg, min=0.0, max=1.0)
        sp = m["book_lin_speed"] + 0.12 * m["book_ang_speed"]
        rel_speed_pen = float(self.cfg.rew_release_speed_penalty_scale) * sp
        align_rel = (
            float(self.cfg.rew_release_lat_penalty_scale) * torch.abs(lat_err)
            + float(self.cfg.rew_release_z_penalty_scale) * torch.abs(z_err)
            + float(self.cfg.rew_release_yaw_penalty_scale) * torch.abs(yaw_err)
        )
        rew_rel = rew_open - rel_speed_pen - align_rel

        # --- RETREAT ---
        d_clear = m["hand_clearance"] - self._prev_hand_clearance
        rew_clear = float(self.cfg.rew_retreat_clearance_delta_scale) * torch.clamp(d_clear, min=0.0, max=0.05)
        ret_sp = m["book_lin_speed"] + 0.15 * m["book_ang_speed"]
        ret_speed_pen = float(self.cfg.rew_retreat_book_speed_penalty_scale) * ret_sp
        rear_pull_pen = float(self.cfg.rew_retreat_rear_pull_penalty_scale) * torch.clamp(-d_rear, min=0.0, max=0.02)
        rew_ret = rew_clear - ret_speed_pen - rear_pull_pen

        # --- PUSH --- (same strict inside-slot definition as INSERT)
        inside_slot = front_to_mouth > 0.0
        rear_pp = float(self.cfg.rew_push_rear_progress_scale) * torch.clamp(d_rear, min=-0.02, max=0.02)
        post_push_p = float(self.cfg.rew_push_post_push_scale) * inside_slot.float() * torch.clamp(d_rear, min=0.0, max=0.02)
        stall_p = inside_slot & (torch.abs(d_rear) < float(self.cfg.rew_push_stall_thresh_m))
        post_stall_p = float(self.cfg.rew_push_stall_penalty_scale) * stall_p.float()
        bottom_p = float(self.cfg.rew_push_bottom_reward_scale) * (m["supported_on_shelf"] & inside_slot).float()
        align_pu = (
            float(self.cfg.rew_push_lat_penalty_scale) * torch.abs(lat_err)
            + float(self.cfg.rew_push_z_penalty_scale) * torch.abs(z_err)
            + float(self.cfg.rew_push_yaw_penalty_scale) * torch.abs(yaw_err)
        )
        rew_push = rear_pp + post_push_p - post_stall_p + bottom_p - align_pu

        ins_m = (ps == _PHASE_INSERT).to(dtype=torch.float32)
        rel_m = (ps == _PHASE_RELEASE).to(dtype=torch.float32)
        ret_m = (ps == _PHASE_RETREAT).to(dtype=torch.float32)
        psh_m = (ps == _PHASE_PUSH).to(dtype=torch.float32)
        rew_phase = ins_m * rew_ins + rel_m * rew_rel + ret_m * rew_ret + psh_m * rew_push

        step_pen = float(self.cfg.step_penalty)
        success = self._success_steps_buf >= int(self.cfg.success_steps)
        success_r = float(self.cfg.success_bonus) * success.float()

        corners_w = self._book_corners_env()
        lowest_z = corners_w[..., 2].min(dim=-1).values
        floor_z = float(self.cfg.book_floor_lowest_z_thresh)
        book_touches_ground = lowest_z < floor_z
        p_env = self._book_pos_env()
        on_shelf = self._book_supported_on_shelf(p_env, lowest_z)
        book_dropped_to_ground = book_touches_ground & ~on_shelf
        drop_pen = float(self.cfg.drop_penalty) * book_dropped_to_ground.float()

        rew = rew_phase + step_pen + success_r + drop_pen

        self.extras.setdefault("log", {})
        self.extras["log"]["reward_mean"] = rew.mean()
        self.extras["log"]["rear_to_mouth_mean"] = rear_to_mouth.mean()
        self.extras["log"]["front_to_mouth_mean"] = front_to_mouth.mean()
        self.extras["log"]["success_signal_mean"] = success.float().mean()
        self.extras["log"]["drop_signal_mean"] = book_dropped_to_ground.float().mean()
        self.extras["log"]["rew_insert_mean"] = (ins_m * rew_ins).mean()
        self.extras["log"]["rew_release_mean"] = (rel_m * rew_rel).mean()
        self.extras["log"]["rew_retreat_mean"] = (ret_m * rew_ret).mean()
        self.extras["log"]["rew_push_mean"] = (psh_m * rew_push).mean()
        # INSERT debug: fractions are over all envs (insert envs contribute; others zero).
        self.extras["log"]["insert_mouth_band_frac"] = (ins_m * in_mouth_band.to(dtype=torch.float32)).mean()
        self.extras["log"]["insert_post_entry_frac"] = (ins_m * post_entry.float()).mean()
        self.extras["log"]["insert_tiny_d_rear_frac"] = (ins_m * tiny_rear.to(dtype=torch.float32)).mean()
        self.extras["log"]["insert_hover_pen_mean"] = (ins_m * hover_pen).mean()
        self.extras["log"]["insert_mouth_cross_event_mean"] = just_crossed.float().mean()
        self.extras["log"]["insert_abs_rear_mean"] = (ins_m * abs_rear).mean()

        self._prev_front_to_mouth = front_to_mouth.detach()
        self._prev_rear_to_mouth = rear_to_mouth.detach()
        self._prev_hand_clearance = m["hand_clearance"].detach()
        self._prev_gripper_open = m["gripper_open"].detach()
        return rew

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        m = self._compute_task_metrics()
        self._step_metrics = m

        phase_b = self._phase
        rr = m["release_ready"]
        self._release_ready_hold_buf = torch.where(
            phase_b == _PHASE_INSERT,
            torch.where(rr, self._release_ready_hold_buf + 1, torch.zeros_like(self._release_ready_hold_buf)),
            torch.zeros_like(self._release_ready_hold_buf),
        )
        open_ok = m["gripper_open"] >= float(self.cfg.phase_release_gripper_open_frac)
        stable = (m["book_lin_speed"] < float(self.cfg.phase_release_max_lin_vel)) & (
            m["book_ang_speed"] < float(self.cfg.phase_release_max_ang_vel)
        )
        rel_ok = open_ok & stable
        self._release_open_stable_buf = torch.where(
            phase_b == _PHASE_RELEASE,
            torch.where(rel_ok, self._release_open_stable_buf + 1, torch.zeros_like(self._release_open_stable_buf)),
            torch.zeros_like(self._release_open_stable_buf),
        )
        clear_ok = m["hand_clearance"] >= float(self.cfg.phase_retreat_min_hand_clearance_m)
        self._retreat_clear_buf = torch.where(
            phase_b == _PHASE_RETREAT,
            torch.where(clear_ok, self._retreat_clear_buf + 1, torch.zeros_like(self._retreat_clear_buf)),
            torch.zeros_like(self._retreat_clear_buf),
        )

        go_rel = (phase_b == _PHASE_INSERT) & (
            self._release_ready_hold_buf >= int(self.cfg.phase_insert_min_release_ready_hold_steps)
        )
        self._pending_insert_release_bonus.copy_(go_rel)
        go_ret = (phase_b == _PHASE_RELEASE) & (
            self._release_open_stable_buf >= int(self.cfg.phase_release_min_open_stable_steps)
        )
        go_push = (phase_b == _PHASE_RETREAT) & (
            self._retreat_clear_buf >= int(self.cfg.phase_retreat_min_clear_hold_steps)
        )

        phase_a = phase_b.clone()
        phase_a = torch.where(go_rel, torch.full_like(phase_a, _PHASE_RELEASE), phase_a)
        phase_a = torch.where(go_ret, torch.full_like(phase_a, _PHASE_RETREAT), phase_a)
        phase_a = torch.where(go_push, torch.full_like(phase_a, _PHASE_PUSH), phase_a)
        self._phase = phase_a

        tr01 = ((phase_b == _PHASE_INSERT) & (phase_a == _PHASE_RELEASE)).float().mean()
        tr12 = ((phase_b == _PHASE_RELEASE) & (phase_a == _PHASE_RETREAT)).float().mean()
        tr23 = ((phase_b == _PHASE_RETREAT) & (phase_a == _PHASE_PUSH)).float().mean()

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
        lat_limit_eff = lat_limit + lat_eps
        front_min_eff = float(self.cfg.success_front_clear_min) - front_eps
        front_max_eff = float(self.cfg.success_front_clear_max) + front_eps
        lat_ok = lat_extent <= lat_limit_eff
        rear_ok = (rear_to_mouth >= float(self.cfg.success_rear_to_mouth_min)) & (
            rear_to_mouth <= float(self.cfg.success_rear_to_mouth_max)
        )
        front_band_ok = (front_to_back >= front_min_eff) & (front_to_back <= front_max_eff)
        z_ok = torch.abs(z_err) < float(self.cfg.success_z_thresh)
        yaw_ok = yaw_e < float(self.cfg.success_yaw_thresh)

        lin_speed = m["book_lin_speed"]
        ang_speed = m["book_ang_speed"]
        if float(self.cfg.success_max_lin_vel) > 0.0:
            stable_lin = lin_speed < float(self.cfg.success_max_lin_vel)
        else:
            stable_lin = torch.ones_like(lat_ok)
        if float(self.cfg.success_max_ang_vel) > 0.0:
            stable_ang = ang_speed < float(self.cfg.success_max_ang_vel)
        else:
            stable_ang = torch.ones_like(lat_ok)

        ready = self.episode_length_buf > int(self.cfg.min_steps_before_success)
        placement_ok = (
            rear_ok & front_band_ok & lat_ok & z_ok & yaw_ok & upright & stable_lin & stable_ang & ready
        )
        # Success is evaluated on `phase_b` before this step's transitions. If we RETREAT→PUSH and are
        # already placement_ok, the success counter starts on the *next* env step (first full step in PUSH).
        push_phase = phase_b == _PHASE_PUSH
        stage_push = push_phase & placement_ok

        if bool(self.cfg.debug_print_success):
            every = max(1, int(self.cfg.debug_print_success_every))
            step_ctr = self.common_step_counter.item() if hasattr(self.common_step_counter, "item") else self.common_step_counter
            if int(step_ctr) % every == 0:
                i = 0
                print(
                    "[SUCCESS_GATES] env=0 "
                    f"phase={int(phase_b[i].item())} ep_len={int(self.episode_length_buf[i].item())} "
                    f"rear_to_mouth={float(rear_to_mouth[i].item()):+.4f} "
                    f"front_to_back={float(front_to_back[i].item()):+.4f} "
                    f"placement_ok={bool(placement_ok[i].item())} push_phase={bool(push_phase[i].item())} "
                    f"hold_steps={int(self._success_steps_buf[i].item())}/{int(self.cfg.success_steps)}"
                )

        self._success_steps_buf = torch.where(
            stage_push, self._success_steps_buf + 1, torch.zeros_like(self._success_steps_buf)
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

        self.extras.setdefault("log", {})
        self.extras["log"]["phase_insert_frac"] = (self._phase == _PHASE_INSERT).float().mean()
        self.extras["log"]["phase_release_frac"] = (self._phase == _PHASE_RELEASE).float().mean()
        self.extras["log"]["phase_retreat_frac"] = (self._phase == _PHASE_RETREAT).float().mean()
        self.extras["log"]["phase_push_frac"] = (self._phase == _PHASE_PUSH).float().mean()
        self.extras["log"]["release_ready_mean"] = m["release_ready"].float().mean()
        self.extras["log"]["supported_on_shelf_mean"] = m["supported_on_shelf"].float().mean()
        self.extras["log"]["trans_insert_to_release_mean"] = tr01
        self.extras["log"]["trans_release_to_retreat_mean"] = tr12
        self.extras["log"]["trans_retreat_to_push_mean"] = tr23

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
        rear_x0 = corners0[..., 0].min(dim=-1).values
        mouth = float(self._geom_mouth_x)
        self._prev_front_to_mouth[env_ids_t] = (front_x0 - mouth).detach()
        self._prev_rear_to_mouth[env_ids_t] = (rear_x0 - mouth).detach()
        self._success_steps_buf[env_ids_t] = 0
        self._phase[env_ids_t] = 0
        self._phase_start[env_ids_t] = 0
        self._release_ready_hold_buf[env_ids_t] = 0
        self._release_open_stable_buf[env_ids_t] = 0
        self._retreat_clear_buf[env_ids_t] = 0
        self._crossed_mouth_bonus_given[env_ids_t] = False
        self._pending_insert_release_bonus[env_ids_t] = False

        m0 = self._compute_task_metrics()
        self._prev_hand_clearance[env_ids_t] = m0["hand_clearance"][env_ids_t].detach()
        self._prev_gripper_open[env_ids_t] = m0["gripper_open"][env_ids_t].detach()
