#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bookshelf-Direct-v4: robot-held insertion, action [dx,dy,dz,dyaw,dgrip] integrates a Cartesian target; differential IK
uses absolute pose mode (Isaac Lab reach ``ik_abs_env_cfg`` style) so roll/pitch stay tied to the reset posture while
yaw follows ``_target_yaw``."""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import math as math_utils
from isaaclab.utils.math import sample_uniform

from .bookshelf_env_cfg_v4 import BookshelfEnvCfg


def _geom_slot_mouth_x(cfg: BookshelfEnvCfg) -> float:
    """-X face of side walls (true channel entrance)."""
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
        self.robot: Articulation = self.scene.articulations["robot"]
        self.book: RigidObject = self.scene.rigid_objects["book"]
        self.book_contact: ContactSensor = self.scene.sensors["book_contact"]

        self.success_steps_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.prev_insertion_depth = torch.zeros(self.num_envs, device=self.device)
        self._prev_lateral_err = torch.zeros(self.num_envs, device=self.device)
        self._prev_yaw_err = torch.zeros(self.num_envs, device=self.device)
        self._peak_ins = torch.zeros(self.num_envs, device=self.device)
        self._stall_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._jam_for_reward = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._last_actions = torch.zeros((self.num_envs, 5), device=self.device)
        self._release_steps_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._prev_front_to_mouth = torch.zeros(self.num_envs, device=self.device)

        self._target_pos_env = torch.zeros((self.num_envs, 3), device=self.device)
        self._target_yaw = torch.zeros(self.num_envs, device=self.device)

        # Differential IK control on the end-effector (panda_hand) in robot base frame.
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
        self._debug_grasp_prints = 0

        self._ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            num_envs=self.num_envs,
            device=str(self.device),
        )
        # Isaac Lab reach (ik_abs) uses an end-effector frame offset from `panda_hand`.
        # Mirror that so our [dx,dy,dz,dyaw] targets move the same tool frame.
        self._ik_body_offset_pos_b = torch.tensor(self.cfg.ik_body_offset_pos, device=self.device, dtype=torch.float32)
        self._ik_body_offset_pos_b = self._ik_body_offset_pos_b.view(1, 3).expand(self.num_envs, 3)
        # Absolute pose command (x,y,z,qw,qx,qy,qz) in robot base frame; same convention as reach ik_abs_env_cfg.
        self._ik_cmd = torch.zeros((self.num_envs, 7), device=self.device)
        # Freeze roll/pitch (base frame, XYZ Euler) at reset; policy varies yaw via _target_yaw only.
        self._ik_fixed_roll = torch.zeros(self.num_envs, device=self.device)
        self._ik_fixed_pitch = torch.zeros(self.num_envs, device=self.device)
        # Match default_joint_pos (e.g. 0.01 m finger closure); zeros would command open and weaken grasp.
        self._gripper_pos_des = self.robot.data.default_joint_pos[:, self._finger_joint_ids].clone()
        if len(self._finger_joint_ids) > 0:
            finger_limits = self.robot.data.soft_joint_pos_limits[0, self._finger_joint_ids]
            self._finger_lower_limits = finger_limits[:, 0].unsqueeze(0).expand(self.num_envs, -1).clone()
            self._finger_upper_limits = finger_limits[:, 1].unsqueeze(0).expand(self.num_envs, -1).clone()
        else:
            self._finger_lower_limits = torch.empty((self.num_envs, 0), device=self.device)
            self._finger_upper_limits = torch.empty((self.num_envs, 0), device=self.device)
        # Fixed joint targets while action≈0 (do not track measured q — that follows sag under book weight).
        self._arm_hold_joint_pos = self.robot.data.default_joint_pos[:, self._arm_joint_ids].clone()

        # h = torch.tensor(
        #     [self.cfg.book_size[i] * 0.5 for i in range(3)],
        #     device=self.device,
        #     dtype=torch.float32,
        # )
        # q = torch.tensor(self.cfg.book_standing_quat, device=self.device, dtype=torch.float32).unsqueeze(0)
        # r0 = math_utils.matrix_from_quat(q)[0, 0, :].abs()
        # half_x_plus = float((r0 * h).sum().item())
        # mouth_x = _geom_slot_mouth_x(self.cfg)
        # self._book_com_x_max = float(mouth_x - half_x_plus - self.cfg.initial_slot_mouth_clearance_x)

        half = 0.5 * torch.tensor(self.cfg.book_size, device=self.device, dtype=torch.float32)
        self._book_corners_local = _cuboid_corners_local(half).to(device=self.device, dtype=torch.float32)

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
        # Standing book: book_standing_quat maps local +Z (thickness) → world ±Y, so lateral half-span is bthick/2 — not bheight/2.
        half_lateral_y = 0.5 * bthick
        n_extra = max(0, int(self.cfg.shelf_extra_books_per_side))
        pitch_y = bthick + float(self.cfg.neighbor_book_pitch_gap)
        wt = self.cfg.wall_thickness
        z_book = bheight * 0.5
        # standing_quat = (1.0, 0.0, 0.0, 0.0)
        qw, qx, qy, qz = self.cfg.book_standing_quat
        standing_quat = (float(qw), float(qx), float(qy), float(qz))

        kin = RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True)
        col = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
        wood = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.45, 0.35))
        book_vis = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.25, 0.15))

        base = "/World/envs/env_0/Bookshelf"

        # Shelf surface: horizontal plank books sit on.
        shelf_depth = depth_x
        shelf_half_y = inner_half + bthick + n_extra * pitch_y
        shelf_width = 2.0 * shelf_half_y + 0.1
        shelf_thick = 0.02
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

        # Slot-defining neighbor books (same cuboid as inserted book, standing).
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

        # Back panel of the bookshelf.
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

        # Factory-style: spawn under env_0 only, clone once, then register (same order as factory_env._setup_scene).
        robot = Articulation(self.cfg.robot)
        book = RigidObject(self.cfg.book)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # Contact sensor after clone so /World/envs/env_.*/Book matches every env (only env_0 exists before clone).
        book_contact = ContactSensor(self.cfg.book_contact)

        self.scene.articulations["robot"] = robot
        self.scene.rigid_objects["book"] = book
        self.scene.sensors["book_contact"] = book_contact

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _book_corners_env(self) -> torch.Tensor:
        """Book cuboid corners in env frame: shape (num_envs, 8, 3)."""
        pos_w = self.book.data.root_link_pos_w
        quat_w = self.book.data.root_link_quat_w
        # IsaacLab quat_apply doesn't broadcast over the corner dimension; flatten env×corner.
        corners_l = self._book_corners_local.view(1, 8, 3).expand(self.num_envs, 8, 3)
        quat_rep = quat_w.view(self.num_envs, 1, 4).expand(self.num_envs, 8, 4)
        corners_w = math_utils.quat_apply(
            quat_rep.reshape(-1, 4),
            corners_l.reshape(-1, 3),
        ).view(self.num_envs, 8, 3) + pos_w.view(self.num_envs, 1, 3)
        return corners_w - self.scene.env_origins.view(self.num_envs, 1, 3)

    def _stage_metrics(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (front_to_mouth, front_to_back, lat_extent, z_err, yaw_err_abs)."""
        corners = self._book_corners_env()
        front_x = corners[..., 0].max(dim=-1).values
        mouth_x = torch.tensor(_geom_slot_mouth_x(self.cfg), device=self.device, dtype=front_x.dtype)
        front_to_mouth = front_x - mouth_x
        front_to_back = self.cfg.slot_x_back - front_x

        lat_extent = torch.abs(corners[..., 1] - self.cfg.slot_center_y).max(dim=-1).values

        p = self._book_pos_env()
        z_target = self.cfg.shelf_top_z + self.cfg.shelf_thickness + 0.5 * self.cfg.book_size[1]
        z_err = p[:, 2] - z_target

        yaw_err_abs = torch.abs(self._yaw_err())
        return front_to_mouth, front_to_back, lat_extent, z_err, yaw_err_abs

    def _gripper_open_ratio(self) -> torch.Tensor:
        """Commanded gripper opening ratio in [0, 1] based on finger joint soft limits."""
        if len(self._finger_joint_ids) == 0:
            return torch.zeros(self.num_envs, device=self.device)
        finger_cmd = self._gripper_pos_des.mean(dim=-1)
        finger_low = self._finger_lower_limits.mean(dim=-1)
        finger_high = self._finger_upper_limits.mean(dim=-1)
        denom = torch.clamp(finger_high - finger_low, min=1.0e-6)
        return torch.clamp((finger_cmd - finger_low) / denom, 0.0, 1.0)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        # Policy action: [dx, dy, dz, dyaw, dgrip] as residual command on CURRENT EE/tool pose (Factory-style).
        # We do not integrate deltas into a long-lived target memory for motion execution.
        dx = self.actions[:, 0] * self.cfg.dx_action_scale
        dy = self.actions[:, 1] * self.cfg.dy_action_scale
        dz = self.actions[:, 2] * self.cfg.dz_action_scale
        dyaw = self.actions[:, 3] * self.cfg.dyaw_action_scale
        dgrip = self.actions[:, 4] * self.cfg.dgrip_action_scale

        # Current EE/tool position in env frame.
        ee_body_pos_env = self.robot.data.body_pos_w[:, self._ee_body_idx] - self.scene.env_origins
        ee_body_quat_w = self.robot.data.body_quat_w[:, self._ee_body_idx]
        ee_tool_offset_w = math_utils.quat_apply(ee_body_quat_w, self._ik_body_offset_pos_b)
        ee_tool_pos_env = ee_body_pos_env + ee_tool_offset_w

        # Current EE orientation in base frame (for yaw extraction and roll/pitch preservation).
        _, ee_quat_b = self._ee_pose_in_base()
        ee_roll_b, ee_pitch_b, ee_yaw_b = math_utils.euler_xyz_from_quat(ee_quat_b)

        # Residual action is applied on current measured EE/tool pose.
        target_pos_env = ee_tool_pos_env + torch.stack((dx, dy, dz), dim=-1)
        target_yaw = _wrap_to_pi(ee_yaw_b + dyaw)

        if self.cfg.enable_target_clamp:
            x_clip, y_clip = self.cfg.target_pos_clip
            target_pos_env[:, 0] = torch.clamp(
                target_pos_env[:, 0],
                self.cfg.slot_x_open - x_clip,
                self.cfg.slot_x_open + x_clip,
            )
            target_pos_env[:, 1] = torch.clamp(
                target_pos_env[:, 1],
                self.cfg.slot_center_y - y_clip,
                self.cfg.slot_center_y + y_clip,
            )
            target_yaw = torch.clamp(target_yaw, -self.cfg.target_yaw_clip, self.cfg.target_yaw_clip)
            target_pos_env[:, 2] = torch.clamp(
                target_pos_env[:, 2],
                self.cfg.target_pos_env_z_min,
                self.cfg.target_pos_env_z_max,
            )

        # Keep command buffers synced for observations/debugging (ex/ey/ez/eyaw are target-current).
        self._target_pos_env[:] = target_pos_env
        self._target_yaw[:] = target_yaw

        # Absolute IK command in base frame. Preserve current roll/pitch and apply yaw residual.
        quat_des_b = math_utils.quat_from_euler_xyz(ee_roll_b, ee_pitch_b, target_yaw)
        # Our action target corresponds to the tool-tip frame offset from `panda_hand`.
        # DifferentialIKController controls the `panda_hand` body, so we subtract the rotated offset.
        offset_des_b = math_utils.quat_apply(quat_des_b, self._ik_body_offset_pos_b)
        body_pos_des_b = target_pos_env - offset_des_b
        self._ik_cmd[:, 0:3] = body_pos_des_b
        self._ik_cmd[:, 3:7] = quat_des_b
        self._ik.set_command(self._ik_cmd)

        ee_pos_b, ee_quat_b = self._ee_pose_in_base()
        jacobian = self.robot.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, self._jacobi_joint_ids]
        joint_pos = self.robot.data.joint_pos[:, self._arm_joint_ids]
        act_small = self.actions.abs() < self.cfg.ik_hold_action_epsilon
        hold_ik = act_small.all(dim=-1)
        joint_pos_des = self._ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        # Idle: command fixed setpoint (refreshed on last non-idle IK). Using measured joint_pos as target drifts down with load.
        move_arm = ~hold_ik
        move_exp = move_arm.unsqueeze(-1).expand_as(joint_pos_des)
        self._arm_hold_joint_pos = torch.where(move_exp, joint_pos_des, self._arm_hold_joint_pos)
        hold_exp = hold_ik.unsqueeze(-1).expand_as(joint_pos_des)
        joint_pos_des = torch.where(hold_exp, self._arm_hold_joint_pos, joint_pos_des)
        self.robot.set_joint_position_target(joint_pos_des, joint_ids=self._arm_joint_ids)

        if len(self._finger_joint_ids) > 0:
            # Residual scalar controls both finger joints symmetrically, then clamp to finger soft limits.
            self._gripper_pos_des = torch.clamp(
                self._gripper_pos_des + dgrip.unsqueeze(-1),
                min=self._finger_lower_limits,
                max=self._finger_upper_limits,
            )
            self.robot.set_joint_position_target(self._gripper_pos_des, joint_ids=self._finger_joint_ids)

    def _ee_pose_in_base(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos_w = self.robot.data.body_pos_w[:, self._ee_body_idx]
        ee_quat_w = self.robot.data.body_quat_w[:, self._ee_body_idx]
        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        return ee_pos_b, ee_quat_b

    def _grasp_frame_pose_w(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Use midpoint between fingers as grasp-frame origin; orientation from panda_hand."""
        lf_pos = self.robot.data.body_pos_w[env_ids, self._left_finger_body_idx]
        rf_pos = self.robot.data.body_pos_w[env_ids, self._right_finger_body_idx]
        grasp_pos_w = 0.5 * (lf_pos + rf_pos)
        grasp_quat_w = self.robot.data.body_quat_w[env_ids, self._ee_body_idx]
        return grasp_pos_w, grasp_quat_w

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

        # v4: controller state — target vs end-effector in env frame; yaw error uses same XYZ-Euler yaw as IK.
        body_pos_env = self.robot.data.body_pos_w[:, self._ee_body_idx] - self.scene.env_origins
        ee_quat_w = self.robot.data.body_quat_w[:, self._ee_body_idx]
        offset_w = math_utils.quat_apply(ee_quat_w, self._ik_body_offset_pos_b)
        ee_pos_env = body_pos_env + offset_w
        _, ee_quat_b_obs = self._ee_pose_in_base()
        _, _, ee_yaw_b = math_utils.euler_xyz_from_quat(ee_quat_b_obs)
        ex = self._target_pos_env[:, 0] - ee_pos_env[:, 0]
        ey = self._target_pos_env[:, 1] - ee_pos_env[:, 1]
        ez = self._target_pos_env[:, 2] - ee_pos_env[:, 2]
        eyaw = _wrap_to_pi(self._target_yaw - ee_yaw_b)
        te_scale = self.cfg.tracking_error_obs_scale
        ex_s = torch.clamp(ex / te_scale, -1.0, 1.0)
        ey_s = torch.clamp(ey / te_scale, -1.0, 1.0)
        ez_s = torch.clamp(ez / te_scale, -1.0, 1.0)
        eyaw_s = torch.clamp(eyaw / te_scale, -1.0, 1.0)

        f, t = self._contact_wrench_world()
        fs = self.cfg.contact_obs_force_scale
        ts = self.cfg.contact_obs_torque_scale

        front_to_mouth, front_to_back, _, z_err, _ = self._stage_metrics()
        grip_open = self._gripper_open_ratio()
        post_release = (grip_open > self.cfg.post_release_push_open_thresh).float()

        obs = torch.stack(
            (
                self.cfg.slot_x_back - p[:, 0],
                lat,
                yaw_e,
                ex_s,
                ey_s,
                ez_s,
                eyaw_s,
                v[:, 0],
                v[:, 1],
                v[:, 5],
                torch.clamp(f[:, 0] / fs, -1.0, 1.0),
                torch.clamp(f[:, 1] / fs, -1.0, 1.0),
                torch.clamp(t[:, 2] / ts, -1.0, 1.0),
                self._last_actions[:, 0],
                self._last_actions[:, 1],
                self._last_actions[:, 2],
                self._last_actions[:, 3],
                self._last_actions[:, 4],
                grip_open,
                z_err,
                front_to_mouth,
                front_to_back,
                post_release,
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
        # v = self.book.data.root_link_vel_w
        ins = p[:, 0] - self.cfg.slot_x_open
        lat_e = torch.abs(p[:, 1] - self.cfg.slot_center_y)
        front_to_mouth, front_to_back, lat_extent, z_err, yaw_e = self._stage_metrics()

        d_ins = ins - self.prev_insertion_depth
        lat_gate = torch.exp(-((lat_e / self.cfg.progress_lateral_sigma) ** 2))
        yaw_gate = torch.exp(-((yaw_e / self.cfg.progress_yaw_sigma) ** 2))
        gate = lat_gate * yaw_gate
        gated_progress = self.cfg.progress_gate_floor + (1.0 - self.cfg.progress_gate_floor) * gate
        progress_r = self.cfg.progress_scale * torch.clamp(d_ins, min=-0.02, max=0.02) * gated_progress
        insertion_pos_r = self.cfg.insertion_pos_reward_scale * torch.clamp(ins, min=-0.20, max=0.08)

        # Dense "approach mouth plane" reward (pre-insertion): encourage front edge to move toward crossing the mouth.
        d_mouth = front_to_mouth - self._prev_front_to_mouth
        mouth_lat_gate = torch.exp(-((lat_extent / self.cfg.mouth_progress_lateral_sigma) ** 2))
        mouth_yaw_gate = torch.exp(-((yaw_e / self.cfg.mouth_progress_yaw_sigma) ** 2))
        mouth_gate = mouth_lat_gate * mouth_yaw_gate
        mouth_progress_r = (
            self.cfg.mouth_approach_progress_scale * torch.clamp(d_mouth, min=-0.02, max=0.02) * mouth_gate
        )

        lateral_prog = self._prev_lateral_err - lat_e
        center_r = self.cfg.center_scale * torch.clamp(lateral_prog, min=-0.02, max=0.02)

        yaw_prog = self._prev_yaw_err - yaw_e
        yaw_r = self.cfg.yaw_scale * torch.clamp(yaw_prog, min=-0.08, max=0.08)

        f, t = self._contact_wrench_world()
        fn = torch.linalg.norm(f, dim=-1)
        contact_pen = self.cfg.contact_penalty_scale * torch.clamp((fn - 10.0) / 50.0, min=0.0, max=4.0)

        backward = torch.clamp(-d_ins, min=0.0)
        # v4: backoff allowance — when misaligned, use lower backward penalty so policy can retreat then reinsert.
        misaligned_for_backoff = ((lat_e > self.cfg.backoff_allowance_lateral_thresh) | (yaw_e > self.cfg.backoff_allowance_yaw_thresh)) & (
            fn > self.cfg.backoff_contact_norm_thresh
        )
        back_scale = torch.where(
            misaligned_for_backoff,
            torch.full_like(backward, self.cfg.backoff_backward_penalty_scale),
            torch.full_like(backward, self.cfg.backward_penalty_scale),
        )
        # Do not multiply by the alignment gate: otherwise, when misaligned (gate -> 0),
        # retreat becomes "cheap" and the policy learns to back off until it is better aligned.
        back_pen = back_scale * backward

        mis_lat = lat_e > self.cfg.misaligned_push_lateral_thresh
        mis_yaw = yaw_e > self.cfg.misaligned_push_yaw_thresh
        misaligned = (mis_lat | mis_yaw).float()
        misalign_push_pen = self.cfg.misaligned_push_scale * torch.clamp(d_ins, min=0.0) * misaligned

        # Before reaching the mouth plane, reduce harsh penalties so exploration can approach the slot.
        pre_mouth = front_to_mouth < 0.0
        pre_scale = torch.full_like(fn, self.cfg.pre_mouth_penalty_scale)
        one = torch.ones_like(fn)
        pen_scale = torch.where(pre_mouth, pre_scale, one)
        contact_pen = contact_pen * pen_scale
        misalign_push_pen = misalign_push_pen * pen_scale

        dact = self.actions - self._last_actions
        dact_pen = self.cfg.action_delta_penalty_scale * torch.sum(dact**2, dim=-1)
        dgrip_pen = self.cfg.grip_action_delta_penalty_scale * (dact[:, 4] ** 2)

        # Anti-dither: discourage forward/backward chattering near the mouth plane.
        near_mouth = torch.abs(front_to_mouth) < self.cfg.mouth_dither_window
        dx_now = self.actions[:, 0]
        dx_prev = self._last_actions[:, 0]
        flip = (dx_now * dx_prev) < 0.0
        active = (dx_now.abs() > 0.2) & (dx_prev.abs() > 0.2)
        dx_signflip_pen = self.cfg.dx_signflip_penalty_scale * (near_mouth & flip & active).float()

        # Release shaping: reward opening only when reasonably aligned/inserted.
        open_cmd = torch.clamp(self.actions[:, 4] - self.cfg.release_open_action_deadband, min=0.0)
        inner_half = 0.5 * (self.cfg.book_size[2] + self.cfg.slot_lateral_clearance)
        mouth_ok = front_to_mouth > 0.0
        z_ok = torch.abs(z_err) < self.cfg.success_z_thresh
        release_ok = (
            mouth_ok
            & (lat_extent < inner_half)
            & z_ok
            & (yaw_e < self.cfg.release_align_yaw_thresh)
            & (ins > self.cfg.release_min_insertion)
        )
        release_bonus = self.cfg.release_bonus_scale * open_cmd * release_ok.float()
        premature_open_pen = self.cfg.premature_open_penalty_scale * open_cmd * (~release_ok).float()

        gripper_open_ratio = self._gripper_open_ratio()

        # State-based early-open penalty: if the gripper is open before a clean insertion, penalize each step.
        upright = self._upright_ok()
        stage1_insert = (
            (front_to_mouth > 0.0)
            & (lat_extent < inner_half)
            & (yaw_e < self.cfg.success_yaw_thresh)
            & (torch.abs(z_err) < self.cfg.success_z_thresh)
            & upright
        )
        early_open = (gripper_open_ratio > self.cfg.early_open_ratio_thresh) & (~stage1_insert)
        early_open_state_pen = self.cfg.early_open_state_penalty_scale * gripper_open_ratio * early_open.float()

        drop_after_open = (
            (gripper_open_ratio > self.cfg.gripper_open_norm_thresh)
            & (ins < self.cfg.release_min_insertion)
            & (p[:, 2] < self.cfg.drop_height_thresh)
        ).float()
        drop_after_open_pen = self.cfg.drop_after_open_penalty_scale * drop_after_open

        # Post-release push shaping: encourage continuing to push inward after opening.
        post_release = gripper_open_ratio > self.cfg.post_release_push_open_thresh
        push_progress_r = (
            self.cfg.post_release_push_progress_scale * torch.clamp(d_ins, min=0.0, max=0.02) * post_release.float()
        )
        push_depth_r = self.cfg.post_release_push_depth_scale * torch.clamp(ins, min=0.0, max=0.08) * post_release.float()
        insertion_stage_r = post_release.float() * (
            self.cfg.insertion_stage_1_bonus * (ins > self.cfg.insertion_stage_1_thresh).float()
            + self.cfg.insertion_stage_2_bonus * (ins > self.cfg.insertion_stage_2_thresh).float()
            + self.cfg.insertion_stage_3_bonus * (ins > self.cfg.insertion_stage_3_thresh).float()
        )

        side_bad = (lat_e > self.cfg.side_bad_lateral_thresh) & (ins < self.cfg.side_bad_insertion_thresh)
        side_bad_pen = self.cfg.side_bad_penalty * side_bad.float()

        # # v4: gripper collision penalty — no sensor yet, so zero. Add sensor + check when robot/gripper exist.
        # gripper_pen = torch.zeros(self.num_envs, device=self.device)

        tipped_for_fail = (~upright) & (ins >= self.cfg.tipped_check_min_ins)
        released = gripper_open_ratio > self.cfg.release_open_ratio_thresh

        v_lin = self.book.data.root_link_vel_w[:, 0:3]
        v_ang = self.book.data.root_link_vel_w[:, 3:6]
        stable = (torch.linalg.norm(v_lin, dim=-1) < self.cfg.release_stable_linvel_thresh) & (
            torch.linalg.norm(v_ang, dim=-1) < self.cfg.release_stable_angvel_thresh
        )
        stage2_release_stable = stage1_insert & released & stable
        self._release_steps_buf = torch.where(
            stage2_release_stable, self._release_steps_buf + 1, torch.zeros_like(self._release_steps_buf)
        )
        release_dwell_ok = self._release_steps_buf >= self.cfg.release_stable_steps

        stage3_push = stage1_insert & release_dwell_ok & (front_to_back < self.cfg.success_final_depth_margin) & stable

        self.success_steps_buf = torch.where(
            stage3_push, self.success_steps_buf + 1, torch.zeros_like(self.success_steps_buf)
        )
        success_mask = self.success_steps_buf >= self.cfg.success_steps

        success_r = self.cfg.success_bonus * success_mask.float()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        timeout_pen = self.cfg.timeout_penalty * (time_out & ~success_mask).float()

        jam_r = self.cfg.jam_penalty * (self._jam_for_reward & ~success_mask).float()

        oob = (torch.abs(p[:, 0]) > self.cfg.max_abs_xy) | (torch.abs(p[:, 1]) > self.cfg.max_abs_xy)
        fell = p[:, 2] < self.cfg.fell_height_thresh
        explode = fn > self.cfg.max_contact_force_norm
        fail_pen = (
            self.cfg.oob_penalty * (oob & ~success_mask).float()
            + self.cfg.fell_penalty * (fell & ~success_mask).float()
            + self.cfg.tipped_penalty * (tipped_for_fail & ~success_mask).float()
            + self.cfg.explode_penalty * (explode & ~success_mask).float()
        )

        rew = (
            progress_r
            + insertion_pos_r
            + mouth_progress_r
            + center_r
            + yaw_r
            + self.cfg.step_penalty
            - dact_pen
            - dgrip_pen
            - dx_signflip_pen
            - contact_pen
            - back_pen
            - misalign_push_pen
            + release_bonus
            + push_progress_r
            + push_depth_r
            + insertion_stage_r
            - premature_open_pen
            - early_open_state_pen
            - drop_after_open_pen
            # + gripper_pen
            + success_r
            + timeout_pen
            + jam_r
            + fail_pen
            + side_bad_pen
        )

        self.prev_insertion_depth = ins.detach()
        self._prev_lateral_err = lat_e.detach()
        self._prev_yaw_err = yaw_e.detach()
        self._prev_front_to_mouth = front_to_mouth.detach()
        self._last_actions = self.actions.detach()

        self.extras.setdefault("log", {})
        self.extras["log"]["insertion_depth"] = ins.mean()
        self.extras["log"]["insertion_pos_r"] = insertion_pos_r.mean()
        self.extras["log"]["mouth_progress_r"] = mouth_progress_r.mean()
        self.extras["log"]["dx_signflip_pen"] = dx_signflip_pen.mean()
        self.extras["log"]["lateral_err"] = lat_e.mean()
        self.extras["log"]["yaw_err_abs"] = yaw_e.mean()
        self.extras["log"]["contact_norm"] = fn.mean()
        self.extras["log"]["success_rate"] = success_mask.float().mean()
        self.extras["log"]["jam_rate"] = self._jam_for_reward.float().mean()
        self.extras["log"]["side_bad_rate"] = side_bad.float().mean()
        self.extras["log"]["release_bonus"] = release_bonus.mean()
        self.extras["log"]["push_progress_r"] = push_progress_r.mean()
        self.extras["log"]["push_depth_r"] = push_depth_r.mean()
        self.extras["log"]["insertion_stage_r"] = insertion_stage_r.mean()
        self.extras["log"]["premature_open_pen"] = premature_open_pen.mean()
        self.extras["log"]["early_open_state_pen"] = early_open_state_pen.mean()
        self.extras["log"]["drop_after_open_pen"] = drop_after_open_pen.mean()
        self.extras["log"]["gripper_open_ratio"] = gripper_open_ratio.mean()
        self.extras["log"]["early_open_rate"] = early_open.float().mean()
        self.extras["log"]["stage1_insert_rate"] = stage1_insert.float().mean()
        self.extras["log"]["release_dwell_ok_rate"] = release_dwell_ok.float().mean()

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
        fell = p[:, 2] < self.cfg.fell_height_thresh
        tipped = (~self._upright_ok()) & (ins >= self.cfg.tipped_check_min_ins)

        # v4: bad gripper collision — no sensor yet, so always False. Add when robot/gripper exist.
        bad_gripper_collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        success = self.success_steps_buf >= self.cfg.success_steps
        if self.cfg.enable_failure_terminations:
            terminated = success | oob | explode | jammed | fell | tipped | bad_gripper_collision
        else:
            terminated = success
            if bool(self.cfg.terminate_on_fell):
                terminated = terminated | fell
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        self.extras.setdefault("log", {})
        self.extras["log"]["fell_rate"] = fell.float().mean()

        if self.cfg.debug_print_terminate_reason and bool(terminated.any().item()):
            for e in terminated.nonzero(as_tuple=False).reshape(-1).tolist():
                e = int(e)
                print(
                    "[Bookshelf v4 terminate]",
                    f"env={e}",
                    f"success={bool(success[e].item())}(buf={int(self.success_steps_buf[e].item())})",
                    f"oob={bool(oob[e].item())}",
                    f"explode={bool(explode[e].item())}(fn={float(fn[e].item()):.2f}/{self.cfg.max_contact_force_norm})",
                    f"jammed={bool(jammed[e].item())}",
                    f"fell={bool(fell[e].item())}(z={float(p[e, 2].item()):.4f})",
                    f"tipped={bool(tipped[e].item())}",
                    f"p_env_xy={float(p[e, 0].item()):.4f},{float(p[e, 1].item()):.4f}",
                    f"ins={float(ins[e].item()):.4f}",
                )

        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids_t = self._env_ids
        else:
            env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        super()._reset_idx(env_ids_t)

        # Reset robot to pre-insertion pose. default_root_state positions are env-local (from cfg.init_state.pos);
        # write_root_state_to_sim expects world frame — add env_origins like factory_env._set_assets_to_default_pose.
        robot_default = self.robot.data.default_root_state[env_ids_t].clone()
        robot_default[:, 0:3] += self.scene.env_origins[env_ids_t]
        self.robot.write_root_state_to_sim(robot_default, env_ids=env_ids_t)
        joint_pos = self.robot.data.default_joint_pos[env_ids_t].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids_t].clone()
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids_t)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids_t)
        if len(self._finger_joint_ids) > 0:
            finger_des = self.robot.data.default_joint_pos[env_ids_t][:, self._finger_joint_ids].clone()
            self.robot.set_joint_position_target(finger_des, joint_ids=self._finger_joint_ids, env_ids=env_ids_t)
            self._gripper_pos_des[env_ids_t] = finger_des

        # One sim step + scene update so articulation body poses reflect the state we just wrote.
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)

        # Default hold pose (will be refreshed again after the book is spawned and contact settles).
        self._arm_hold_joint_pos[env_ids_t] = self.robot.data.joint_pos[env_ids_t][:, self._arm_joint_ids].clone()

        n = int(env_ids_t.numel())
        grasp_pos_w, grasp_quat_w = self._grasp_frame_pose_w(env_ids_t)
        hx, hy, hz = self.cfg.book_grasp_offset_hand
        hand_offset = (
            torch.tensor([hx, hy, hz], device=self.device, dtype=grasp_pos_w.dtype).unsqueeze(0).expand(n, 3).clone()
        )
        hand_offset[:, 0] += sample_uniform(-self.cfg.book_grasp_x_jitter, self.cfg.book_grasp_x_jitter, (n,), self.device)
        hand_offset[:, 1] += sample_uniform(-self.cfg.book_grasp_y_jitter, self.cfg.book_grasp_y_jitter, (n,), self.device)
        offset_w = math_utils.quat_apply(grasp_quat_w, hand_offset)
        book_pos_w = grasp_pos_w + offset_w
        book_pos_env = book_pos_w - self.scene.env_origins[env_ids_t]

        # Optional jitter quaternion (same component used in both orient modes); composed per book_reset_orient_mode.
        yaw_delta = sample_uniform(
            -self.cfg.book_grasp_yaw_jitter,
            self.cfg.book_grasp_yaw_jitter,
            (n,),
            self.device,
        )
        qyaw_delta = torch.stack(
            (
                torch.cos(0.5 * yaw_delta),
                torch.zeros_like(yaw_delta),
                torch.zeros_like(yaw_delta),
                torch.sin(0.5 * yaw_delta),
            ),
            dim=-1,
        )
        sw, sx, sy, sz = self.cfg.book_standing_quat
        standing = (
            torch.tensor([sw, sx, sy, sz], device=self.device, dtype=grasp_quat_w.dtype)
            .unsqueeze(0)
            .expand(n, 4)
            .clone()
        )
        if self.cfg.book_reset_orient_mode == "standing_world":
            # Shelf-style standing in world: yaw about +Z, then same base quat as neighbor books.
            quat = _quat_mul_wxyz(qyaw_delta, standing)
        elif self.cfg.book_reset_orient_mode == "grasp_relative":
            # q_world = q_panda_hand * q_book_in_hand * q_yaw_jitter (offsets use same hand axes).
            mode = self.cfg.book_grasp_orientation_in_hand
            if mode == "franka_axes":
                # panda_hand: +Z approach (between fingers), +Y close, +X third axis.
                # Cuboid: +X_b || +Z_h, +Y_b || +X_h, +Z_b || +Y_h  =>  q maps book basis in hand frame.
                fw, fx, fy, fz = self.cfg.book_to_hand_quat_franka_axes_wxyz
                q_body_to_grasp = (
                    torch.tensor([fw, fx, fy, fz], device=self.device, dtype=grasp_quat_w.dtype)
                    .unsqueeze(0)
                    .expand(n, 4)
                    .clone()
                )
            elif mode == "manual_quat":
                rw, rx, ry, rz = self.cfg.book_grasp_rel_quat_wxyz
                q_body_to_grasp = (
                    torch.tensor([rw, rx, ry, rz], device=self.device, dtype=grasp_quat_w.dtype)
                    .unsqueeze(0)
                    .expand(n, 4)
                    .clone()
                )
            else:
                raise ValueError(f"Unknown book_grasp_orientation_in_hand: {mode}")
            quat = _quat_mul_wxyz(grasp_quat_w, _quat_mul_wxyz(q_body_to_grasp, qyaw_delta))
        else:
            raise ValueError(f"Unknown book_reset_orient_mode: {self.cfg.book_reset_orient_mode}")

        book_default_state = self.book.data.default_root_state[env_ids_t].clone()
        book_default_state[:, 0:3] = book_pos_env + self.scene.env_origins[env_ids_t]
        book_default_state[:, 3:7] = quat
        book_default_state[:, 7:] = 0.0

        self.book.write_root_state_to_sim(book_default_state, env_ids=env_ids_t)

        # Settle book–robot contact, then freeze IK roll/pitch (base XYZ Euler) for absolute pose control.
        self.scene.write_data_to_sim()
        for _ in range(self.cfg.reset_book_contact_settle_steps):
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

        # Refresh the hold pose after contact settling so action=[0,0,0,0] keeps the current grasp.
        self._arm_hold_joint_pos[env_ids_t] = self.robot.data.joint_pos[env_ids_t][:, self._arm_joint_ids].clone()

        _, ee_quat_b = self._ee_pose_in_base()
        roll, pitch, yaw_b = math_utils.euler_xyz_from_quat(ee_quat_b[env_ids_t])
        self._ik_fixed_roll[env_ids_t] = roll
        self._ik_fixed_pitch[env_ids_t] = pitch
        # Match IK / observation yaw convention (base XYZ Euler), so eyaw ≈ 0 after reset.
        self._target_yaw[env_ids_t] = yaw_b

        # EE Cartesian target in env frame (coincides with base for fixed-base Franka at env origin).
        ee_pos_w = self.robot.data.body_pos_w[env_ids_t, self._ee_body_idx]
        ee_quat_w = self.robot.data.body_quat_w[env_ids_t, self._ee_body_idx]
        # Convert the panda_hand offset into world/env frame.
        offset_w = math_utils.quat_apply(ee_quat_w, self._ik_body_offset_pos_b[env_ids_t])
        ee_eff_pos_w = ee_pos_w + offset_w
        ee_eff_pos_env = ee_eff_pos_w - self.scene.env_origins[env_ids_t]
        self._target_pos_env[env_ids_t, 0] = ee_eff_pos_env[:, 0]
        self._target_pos_env[env_ids_t, 1] = ee_eff_pos_env[:, 1]
        self._target_pos_env[env_ids_t, 2] = ee_eff_pos_env[:, 2]

        # IMPORTANT: keep reset target consistent with _apply_action() clamping.
        # Otherwise the first non-hold step will snap _target_pos_env to the clamp boundary,
        # biasing motion direction even for random actions.
        if self.cfg.enable_target_clamp:
            x_clip, y_clip = self.cfg.target_pos_clip
            self._target_pos_env[env_ids_t, 0] = torch.clamp(
                self._target_pos_env[env_ids_t, 0],
                self.cfg.slot_x_open - x_clip,
                self.cfg.slot_x_open + x_clip,
            )
            self._target_pos_env[env_ids_t, 1] = torch.clamp(
                self._target_pos_env[env_ids_t, 1],
                self.cfg.slot_center_y - y_clip,
                self.cfg.slot_center_y + y_clip,
            )
            self._target_yaw[env_ids_t] = torch.clamp(
                self._target_yaw[env_ids_t],
                -self.cfg.target_yaw_clip,
                self.cfg.target_yaw_clip,
            )
            self._target_pos_env[env_ids_t, 2] = torch.clamp(
                self._target_pos_env[env_ids_t, 2],
                self.cfg.target_pos_env_z_min,
                self.cfg.target_pos_env_z_max,
            )

        if self.cfg.debug_print_grasp_frame and self._debug_grasp_prints < 10 and int(env_ids_t[0].item()) == 0:
            lf_pos_w = self.robot.data.body_pos_w[0, self._left_finger_body_idx].detach().cpu()
            rf_pos_w = self.robot.data.body_pos_w[0, self._right_finger_body_idx].detach().cpu()
            finger_mid_w = 0.5 * (lf_pos_w + rf_pos_w)
            book_pos_w_dbg = self.book.data.root_link_pos_w[0].detach().cpu()
            dist_mid_to_book = torch.linalg.norm(finger_mid_w - book_pos_w_dbg).item()

            contact_f, _ = self._contact_wrench_world()
            book_contact_force_norm = torch.linalg.norm(contact_f[0]).item()

            finger_joint_des = self._gripper_pos_des[0].detach().cpu()
            finger_joint_meas = self.robot.data.joint_pos[0][self._finger_joint_ids].detach().cpu()

            xw = math_utils.quat_apply(
                grasp_quat_w[0:1], torch.tensor([[1.0, 0.0, 0.0]], device=self.device, dtype=grasp_pos_w.dtype)
            )[0]
            yw = math_utils.quat_apply(
                grasp_quat_w[0:1], torch.tensor([[0.0, 1.0, 0.0]], device=self.device, dtype=grasp_pos_w.dtype)
            )[0]
            zw = math_utils.quat_apply(
                grasp_quat_w[0:1], torch.tensor([[0.0, 0.0, 1.0]], device=self.device, dtype=grasp_pos_w.dtype)
            )[0]
            print(
                "[v4 reset debug] grasp_pos_w=", grasp_pos_w[0].detach().cpu().tolist(),
                "grasp_quat_w=", grasp_quat_w[0].detach().cpu().tolist(),
                "offset_local=", hand_offset[0].detach().cpu().tolist(),
                "offset_world=", offset_w[0].detach().cpu().tolist(),
                "axes_world(x,y,z)=", xw.detach().cpu().tolist(), yw.detach().cpu().tolist(), zw.detach().cpu().tolist(),
                "finger_mid_w=", finger_mid_w.tolist(),
                "book_pos_w=", book_pos_w_dbg.tolist(),
                f"dist_mid_to_book={dist_mid_to_book:.4f}",
                f"book_contact_force_norm={book_contact_force_norm:.2f}",
                "finger_joint_des=", finger_joint_des.tolist(),
                "finger_joint_meas=", finger_joint_meas.tolist(),
            )
            self._debug_grasp_prints += 1

        ins0 = book_pos_env[:, 0] - self.cfg.slot_x_open
        lat0 = torch.abs(book_pos_env[:, 1] - self.cfg.slot_center_y)
        self.prev_insertion_depth[env_ids_t] = ins0
        self._prev_lateral_err[env_ids_t] = lat0
        book_yaw_wrapped = _wrap_to_pi(_yaw_from_quat_wxyz(quat))
        self._prev_yaw_err[env_ids_t] = torch.abs(book_yaw_wrapped)
        self._last_actions[env_ids_t] = 0.0
        self.success_steps_buf[env_ids_t] = 0
        self._release_steps_buf[env_ids_t] = 0
        # Initialize mouth metric buffer at reset.
        corners0 = self._book_corners_env()[env_ids_t]
        front_x0 = corners0[..., 0].max(dim=-1).values
        mouth_x = _geom_slot_mouth_x(self.cfg)
        self._prev_front_to_mouth[env_ids_t] = (front_x0 - mouth_x).detach()
        self._peak_ins[env_ids_t] = ins0
        self._stall_buf[env_ids_t] = 0
        self._jam_for_reward[env_ids_t] = False
