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
from isaaclab.utils import math as math_utils
from isaaclab.utils.math import sample_uniform

from .bookshelf_env_cfg_v4 import BookshelfEnvCfg
from .drop_logic import book_dropped_mask
from .frozen_scenario_bank import FrozenScenarioAllocator, load_frozen_scenario_bank

_MODE_INSERT = 0
_MODE_SCRIPTED = 1
_MODE_PUSH = 2

_DONE_NONE = 0
_DONE_SUCCESS = 1
_DONE_DROP = 2
_DONE_TIMEOUT = 3
_DONE_NOT_PUSH = 4
_DONE_DEPTH = 5
_DONE_LATERAL = 6
_DONE_Z = 7
_DONE_YAW = 8
_DONE_UPRIGHT = 9
_DONE_UNSTABLE = 10
_DONE_OOB = 11
_DONE_FELL = 12


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

        # Robot-specific names live in the config so the task logic can be
        # shared by Panda and xArm7 embodiments.
        lf_name = str(self.cfg.robot_left_finger_body_name)
        rf_name = str(self.cfg.robot_right_finger_body_name)
        lf_ids, _ = self.robot.find_bodies(lf_name)
        rf_ids, _ = self.robot.find_bodies(rf_name)
        if len(lf_ids) != 1 or len(rf_ids) != 1:
            raise RuntimeError(
                f"Expected one {lf_name!r} and one {rf_name!r} body for the grasp frame."
            )
        self._left_finger_body_idx = lf_ids[0]
        self._right_finger_body_idx = rf_ids[0]

        hand_name = str(self.cfg.robot_hand_body_name)
        hand_ids, hand_names = self.robot.find_bodies(hand_name)
        if len(hand_ids) != 1:
            raise RuntimeError(f"Expected one {hand_name!r} body. Got {len(hand_ids)}: {hand_names}")
        self._hand_body_idx = hand_ids[0]

        ee_name = str(self.cfg.robot_ee_body_name)
        ee_ids, ee_names = self.robot.find_bodies(ee_name)
        if len(ee_ids) != 1:
            raise RuntimeError(f"Expected one {ee_name!r} EE body. Got {len(ee_ids)}: {ee_names}")
        self._ee_body_idx = ee_ids[0]

        grasp_name = str(getattr(self.cfg, "robot_grasp_frame_body_name", "") or "")
        self._grasp_frame_body_idx = None
        if grasp_name:
            grasp_ids, grasp_names = self.robot.find_bodies(grasp_name)
            if len(grasp_ids) != 1:
                raise RuntimeError(
                    f"Expected one {grasp_name!r} grasp-frame body. Got {len(grasp_ids)}: {grasp_names}"
                )
            self._grasp_frame_body_idx = grasp_ids[0]

        # Arm/finger joints and jacobian indices for IK.
        self._arm_joint_ids, _ = self.robot.find_joints(
            str(self.cfg.robot_arm_joint_names_expr), preserve_order=True
        )
        self._finger_joint_ids, _ = self.robot.find_joints(
            str(self.cfg.robot_finger_joint_names_expr), preserve_order=True
        )
        self._gripper_command_joint_ids, gripper_command_joint_names = self.robot.find_joints(
            str(self.cfg.robot_gripper_command_joint_names_expr), preserve_order=True
        )
        if len(self._arm_joint_ids) != 7:
            raise RuntimeError(
                "The bookshelf task requires seven ordered arm joints; "
                f"{self.cfg.robot_arm_joint_names_expr!r} matched {len(self._arm_joint_ids)}."
            )
        if len(self._gripper_command_joint_ids) < 1:
            raise RuntimeError(
                "The bookshelf task requires at least one gripper command joint; "
                f"{self.cfg.robot_gripper_command_joint_names_expr!r} matched "
                f"{len(self._gripper_command_joint_ids)}: {gripper_command_joint_names}."
            )

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
        self._release_step_buf = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._push_start_step_buf = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)

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
        self._ensure_scenario_trace_buffers()

    def _ensure_scenario_trace_buffers(self) -> None:
        """Create reset-condition buffers lazily so derived envs can reuse them."""
        if hasattr(self, "_scenario_reset_count_env"):
            return
        self._scenario_reset_count_env = torch.full(
            (self.num_envs,), -1, device=self.device, dtype=torch.long
        )
        self._scenario_bank_index_env = torch.full(
            (self.num_envs,), -1, device=self.device, dtype=torch.long
        )
        self._scenario_joint_noise_env = torch.zeros(
            (self.num_envs, len(self._arm_joint_ids)), device=self.device, dtype=torch.float32
        )
        self._scenario_applied_joint_noise_env = torch.zeros_like(
            self._scenario_joint_noise_env
        )
        self._scenario_grasp_jitter_env = torch.zeros(
            (self.num_envs, 4), device=self.device, dtype=torch.float32
        )
        self._scenario_initial_book_pose_env = torch.zeros(
            (self.num_envs, 7), device=self.device, dtype=torch.float32
        )
        self._scenario_initial_tool_pose_env = torch.zeros(
            (self.num_envs, 7), device=self.device, dtype=torch.float32
        )
        self._scenario_row_wide_mask_env = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.long
        )
        self._scenario_single_book_slot_env = torch.full(
            (self.num_envs, 9), -1, device=self.device, dtype=torch.long
        )
        self._scenario_wide_book_start_slot_env = torch.full(
            (self.num_envs, 4), -1, device=self.device, dtype=torch.long
        )
        self._frozen_joint_noise_env = torch.zeros_like(self._scenario_joint_noise_env)
        self._frozen_grasp_jitter_env = torch.zeros_like(self._scenario_grasp_jitter_env)
        self._frozen_slot_center_y_env = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.float32
        )
        self._frozen_slot_clearance_env = torch.zeros_like(self._frozen_slot_center_y_env)
        self._frozen_missing_book_index_env = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.long
        )
        self._frozen_row_wide_mask_env = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.long
        )
        self._frozen_single_book_slot_env = torch.full_like(
            self._scenario_single_book_slot_env, -1
        )
        self._frozen_wide_book_start_slot_env = torch.full_like(
            self._scenario_wide_book_start_slot_env, -1
        )

    def _ensure_frozen_scenario_bank(self) -> None:
        if hasattr(self, "_frozen_scenario_allocator"):
            return
        bank_path = str(getattr(self.cfg, "evaluation_scenario_bank", "") or "")
        self._frozen_scenario_bank = None
        self._frozen_scenario_allocator = None
        if not bank_path:
            return
        bank = load_frozen_scenario_bank(bank_path)
        if int(bank["scenario_count"]) < self.num_envs:
            raise ValueError(
                "Frozen scenario bank must contain at least as many scenarios as active environments: "
                f"{bank['scenario_count']} < {self.num_envs}"
            )
        self._frozen_scenario_bank = bank
        self._frozen_scenario_allocator = FrozenScenarioAllocator(bank["scenarios"], self.num_envs)
        print(
            "[INFO] Frozen evaluation scenario bank loaded: "
            f"{bank['scenario_count']} scenarios, sha256={bank['scenario_sha256']}"
        )

    def _assign_frozen_scenarios(self, env_ids_t: torch.Tensor) -> torch.Tensor:
        """Assign fresh bank rows to reset envs and return the active mask."""
        self._ensure_scenario_trace_buffers()
        self._ensure_frozen_scenario_bank()
        self._scenario_bank_index_env[env_ids_t] = -1
        if self._frozen_scenario_allocator is None:
            return torch.zeros(len(env_ids_t), device=self.device, dtype=torch.bool)

        assignments = self._frozen_scenario_allocator.allocate(env_ids_t.tolist())
        active = torch.zeros(len(env_ids_t), device=self.device, dtype=torch.bool)
        local_index = {int(env_id): index for index, env_id in enumerate(env_ids_t.tolist())}
        for env_id, scenario in assignments.items():
            if scenario is None:
                continue
            index = local_index[env_id]
            active[index] = True
            self._scenario_bank_index_env[env_id] = int(scenario["scenario_id"])
            self._frozen_slot_center_y_env[env_id] = float(scenario["slot_center_y"])
            self._frozen_slot_clearance_env[env_id] = float(scenario["slot_clearance"])
            self._frozen_missing_book_index_env[env_id] = int(scenario["missing_book_index"])
            self._frozen_row_wide_mask_env[env_id] = int(scenario["row_wide_mask"])
            self._frozen_joint_noise_env[env_id] = torch.tensor(
                [scenario[f"joint_noise_{joint}"] for joint in range(1, 8)],
                device=self.device,
                dtype=torch.float32,
            )
            self._frozen_grasp_jitter_env[env_id] = torch.tensor(
                [
                    scenario["grasp_jitter_x"],
                    scenario["grasp_jitter_y"],
                    scenario["grasp_jitter_z"],
                    scenario["grasp_jitter_yaw"],
                ],
                device=self.device,
                dtype=torch.float32,
            )
            self._frozen_single_book_slot_env[env_id] = torch.tensor(
                [scenario[f"single_book_slot_{slot}"] for slot in range(9)],
                device=self.device,
                dtype=torch.long,
            )
            self._frozen_wide_book_start_slot_env[env_id] = torch.tensor(
                [scenario[f"wide_book_start_slot_{slot}"] for slot in range(4)],
                device=self.device,
                dtype=torch.long,
            )
        return active

    def _capture_scenario_initial_pose(self, env_ids_t: torch.Tensor) -> None:
        self._ensure_scenario_trace_buffers()
        book_pos_env = self.book.data.root_link_pos_w[env_ids_t] - self.scene.env_origins[env_ids_t]
        book_quat_w = self.book.data.root_link_quat_w[env_ids_t]
        tool_pos_env = self._ee_tool_pos_env()[env_ids_t]
        tool_quat_w = self.robot.data.body_quat_w[env_ids_t, self._ee_body_idx]
        self._scenario_initial_book_pose_env[env_ids_t] = torch.cat((book_pos_env, book_quat_w), dim=-1)
        self._scenario_initial_tool_pose_env[env_ids_t] = torch.cat((tool_pos_env, tool_quat_w), dim=-1)

    def _write_scenario_episode_metrics(self) -> None:
        """Snapshot reset conditions before DirectRLEnv resets completed envs."""
        # Fixed order is mirrored by SCENARIO_VECTOR_FIELDS in
        # scripts/sb3/evaluation_scenarios.py.
        self.extras["episode_metric_scenario_vector"] = torch.cat(
            (
                self._scenario_reset_count_env.to(dtype=torch.float32).unsqueeze(-1),
                self._scenario_bank_index_env.to(dtype=torch.float32).unsqueeze(-1),
                self._scenario_row_wide_mask_env.to(dtype=torch.float32).unsqueeze(-1),
                self._scenario_joint_noise_env,
                self._scenario_grasp_jitter_env,
                self._scenario_single_book_slot_env.to(dtype=torch.float32),
                self._scenario_wide_book_start_slot_env.to(dtype=torch.float32),
                self._scenario_initial_book_pose_env,
                self._scenario_initial_tool_pose_env,
            ),
            dim=-1,
        ).clone()

    def _grasp_frame_pose_w(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._grasp_frame_body_idx is not None:
            return (
                self.robot.data.body_pos_w[env_ids, self._grasp_frame_body_idx],
                self.robot.data.body_quat_w[env_ids, self._grasp_frame_body_idx],
            )
        lf_pos = self.robot.data.body_pos_w[env_ids, self._left_finger_body_idx]
        rf_pos = self.robot.data.body_pos_w[env_ids, self._right_finger_body_idx]
        grasp_pos_w = 0.5 * (lf_pos + rf_pos)
        grasp_quat_w = self.robot.data.body_quat_w[env_ids, self._hand_body_idx]
        return grasp_pos_w, grasp_quat_w

    def debug_grasp_snapshot(self, env_index: int = 0) -> dict:
        """Return one JSON-serializable grasp snapshot for visual debugging."""
        if env_index < 0 or env_index >= self.num_envs:
            raise IndexError(f"env_index {env_index} is outside [0, {self.num_envs})")

        env_id = torch.tensor([env_index], device=self.device, dtype=torch.long)
        origin = self.scene.env_origins[env_index]
        grasp_pos_w, grasp_quat_w = self._grasp_frame_pose_w(env_id)
        book_pos_w = self.book.data.root_link_pos_w[env_id]
        book_quat_w = self.book.data.root_link_quat_w[env_id]
        book_pos_g, book_quat_g = math_utils.subtract_frame_transforms(
            grasp_pos_w, grasp_quat_w, book_pos_w, book_quat_w
        )
        ee_body_pos_w = self.robot.data.body_pos_w[env_id, self._ee_body_idx]
        ee_body_quat_w = self.robot.data.body_quat_w[env_id, self._ee_body_idx]
        tool_pos_w = ee_body_pos_w + math_utils.quat_apply(
            ee_body_quat_w,
            self._ik_body_offset_pos_b[env_id],
        )

        left_pos_w = self.robot.data.body_pos_w[env_index, self._left_finger_body_idx]
        right_pos_w = self.robot.data.body_pos_w[env_index, self._right_finger_body_idx]
        finger_delta_h = math_utils.quat_apply_inverse(
            self.robot.data.body_quat_w[env_index, self._hand_body_idx].unsqueeze(0),
            (left_pos_w - right_pos_w).unsqueeze(0),
        )[0]
        book_center_w = book_pos_w[0]

        joint_targets = getattr(self.robot.data, "joint_pos_target", None)
        applied_torque = getattr(self.robot.data, "applied_torque", None)
        arm_joints = {}
        arm_target_errors = []
        for joint_id in self._arm_joint_ids:
            joint_id = int(joint_id)
            position = float(self.robot.data.joint_pos[env_index, joint_id].item())
            target = (
                None
                if joint_targets is None
                else float(joint_targets[env_index, joint_id].item())
            )
            if target is not None:
                arm_target_errors.append(abs(position - target))
            arm_joints[self.robot.joint_names[joint_id]] = {
                "position_rad": position,
                "target_rad": target,
            }
        finger_joints = {}
        for joint_id in self._finger_joint_ids:
            joint_id = int(joint_id)
            finger_joints[self.robot.joint_names[joint_id]] = {
                "position_rad": float(self.robot.data.joint_pos[env_index, joint_id].item()),
                "target_rad": (
                    None
                    if joint_targets is None
                    else float(joint_targets[env_index, joint_id].item())
                ),
                "applied_torque_nm": (
                    None
                    if applied_torque is None
                    else float(applied_torque[env_index, joint_id].item())
                ),
            }

        book_corners = self._book_corners_env()[env_index]
        book_pos_env = book_center_w - origin
        corner_offsets_w = book_corners - book_pos_env
        hand_quat_w = self.robot.data.body_quat_w[env_index, self._hand_body_idx]
        corner_offsets_h = math_utils.quat_apply_inverse(
            hand_quat_w.unsqueeze(0).expand(corner_offsets_w.shape[0], 4),
            corner_offsets_w,
        )
        book_half_extent_across_gripper = 0.5 * (
            corner_offsets_h[:, 1].max() - corner_offsets_h[:, 1].min()
        )
        return {
            "env_index": env_index,
            "episode_step": int(self.episode_length_buf[env_index].item()),
            "scenario_reset_count": int(self._scenario_reset_count_env[env_index].item()),
            "mode": int(self._mode[env_index].item()),
            "robot_is_fixed_base": bool(self.robot.is_fixed_base),
            "robot_root_position_env_m": [
                float(value)
                for value in (self.robot.data.root_pos_w[env_index] - origin).detach().cpu().tolist()
            ],
            "arm_joints": arm_joints,
            "arm_max_target_error_rad": (
                None if not arm_target_errors else max(arm_target_errors)
            ),
            "book_position_env_m": [
                float(value) for value in (book_center_w - origin).detach().cpu().tolist()
            ],
            "book_quaternion_wxyz": [
                float(value) for value in book_quat_w[0].detach().cpu().tolist()
            ],
            "book_linear_velocity_world_mps": [
                float(value)
                for value in self.book.data.root_link_lin_vel_w[env_index].detach().cpu().tolist()
            ],
            "book_angular_velocity_world_radps": [
                float(value)
                for value in self.book.data.root_link_ang_vel_w[env_index].detach().cpu().tolist()
            ],
            "book_lowest_z_env_m": float(book_corners[:, 2].min().item()),
            "grasp_position_env_m": [
                float(value) for value in (grasp_pos_w[0] - origin).detach().cpu().tolist()
            ],
            "tool_position_env_m": [
                float(value) for value in (tool_pos_w[0] - origin).detach().cpu().tolist()
            ],
            "tool_quaternion_wxyz": [
                float(value) for value in ee_body_quat_w[0].detach().cpu().tolist()
            ],
            "book_position_in_grasp_frame_m": [
                float(value) for value in book_pos_g[0].detach().cpu().tolist()
            ],
            "book_quaternion_in_grasp_frame_wxyz": [
                float(value) for value in book_quat_g[0].detach().cpu().tolist()
            ],
            "finger_origin_delta_in_hand_frame_m": [
                float(value) for value in finger_delta_h.detach().cpu().tolist()
            ],
            "finger_origin_distance_m": float(torch.linalg.norm(finger_delta_h).item()),
            "book_half_extent_across_gripper_m": float(book_half_extent_across_gripper.item()),
            "book_center_to_left_finger_origin_m": float(
                torch.linalg.norm(book_center_w - left_pos_w).item()
            ),
            "book_center_to_right_finger_origin_m": float(
                torch.linalg.norm(book_center_w - right_pos_w).item()
            ),
            "finger_joints": finger_joints,
        }

    def debug_grasp_batch_snapshot(self) -> dict[str, torch.Tensor]:
        """Return vectorized reset/grasp diagnostics for offline preflight tools."""

        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        grasp_pos_w, grasp_quat_w = self._grasp_frame_pose_w(env_ids)
        book_pos_w = self.book.data.root_link_pos_w
        book_quat_w = self.book.data.root_link_quat_w
        book_pos_g, book_quat_g = math_utils.subtract_frame_transforms(
            grasp_pos_w,
            grasp_quat_w,
            book_pos_w,
            book_quat_w,
        )
        self._ensure_scenario_trace_buffers()
        expected_book_pos_w, expected_book_quat_w = self._book_reset_pose_w(
            env_ids,
            grasp_pos_w,
            grasp_quat_w,
            self._scenario_grasp_jitter_env,
        )
        expected_book_pos_g, expected_book_quat_g = math_utils.subtract_frame_transforms(
            grasp_pos_w,
            grasp_quat_w,
            expected_book_pos_w,
            expected_book_quat_w,
        )

        book_pos_env = book_pos_w - self.scene.env_origins
        book_corners = self._book_corners_env()
        corner_offsets_w = book_corners - book_pos_env.unsqueeze(1)
        hand_quat_w = self.robot.data.body_quat_w[:, self._hand_body_idx]
        corner_offsets_h = math_utils.quat_apply_inverse(
            hand_quat_w.unsqueeze(1).expand(-1, corner_offsets_w.shape[1], -1).reshape(-1, 4),
            corner_offsets_w.reshape(-1, 3),
        ).reshape(self.num_envs, corner_offsets_w.shape[1], 3)
        book_half_extent_across_gripper = 0.5 * (
            corner_offsets_h[:, :, 1].amax(dim=1)
            - corner_offsets_h[:, :, 1].amin(dim=1)
        )

        left_pos_w = self.robot.data.body_pos_w[:, self._left_finger_body_idx]
        right_pos_w = self.robot.data.body_pos_w[:, self._right_finger_body_idx]
        finger_delta_h = math_utils.quat_apply_inverse(
            hand_quat_w,
            left_pos_w - right_pos_w,
        )

        joint_targets = getattr(self.robot.data, "joint_pos_target", None)
        if joint_targets is None:
            arm_max_target_error = torch.full(
                (self.num_envs,), float("nan"), device=self.device
            )
        else:
            arm_error = torch.abs(
                _wrap_to_pi(
                    joint_targets[:, self._arm_joint_ids]
                    - self.robot.data.joint_pos[:, self._arm_joint_ids]
                )
            )
            arm_max_target_error = arm_error.amax(dim=-1)

        slot_center_y = getattr(self, "_slot_center_y", None)
        if callable(slot_center_y):
            slot_center_y_env = slot_center_y().detach().clone()
        else:
            slot_center_y_env = torch.full(
                (self.num_envs,),
                float(self.cfg.slot_center_y),
                device=self.device,
                dtype=torch.float32,
            )
        return {
            "scenario_bank_index": self._scenario_bank_index_env.detach().clone(),
            "scenario_reset_count": self._scenario_reset_count_env.detach().clone(),
            "missing_book_index": self._missing_book_index_env.detach().clone(),
            "slot_center_y_m": slot_center_y_env,
            "slot_clearance_m": self._slot_lateral_clearance_env.detach().clone(),
            "row_wide_mask": self._scenario_row_wide_mask_env.detach().clone(),
            "joint_noise_rad": self._scenario_joint_noise_env.detach().clone(),
            "applied_joint_noise_rad": (
                self._scenario_applied_joint_noise_env.detach().clone()
            ),
            "grasp_jitter": self._scenario_grasp_jitter_env.detach().clone(),
            "expected_book_position_in_grasp_frame_m": (
                expected_book_pos_g.detach().clone()
            ),
            "expected_book_quaternion_in_grasp_frame_wxyz": (
                expected_book_quat_g.detach().clone()
            ),
            "book_position_in_grasp_frame_m": book_pos_g.detach().clone(),
            "book_quaternion_in_grasp_frame_wxyz": book_quat_g.detach().clone(),
            "book_position_env_m": book_pos_env.detach().clone(),
            "book_lowest_z_env_m": book_corners[:, :, 2].amin(dim=1).detach().clone(),
            "book_linear_speed_mps": torch.linalg.norm(
                self.book.data.root_link_lin_vel_w, dim=-1
            ).detach().clone(),
            "book_angular_speed_radps": torch.linalg.norm(
                self.book.data.root_link_ang_vel_w, dim=-1
            ).detach().clone(),
            "arm_max_target_error_rad": arm_max_target_error.detach().clone(),
            "finger_origin_distance_m": torch.linalg.norm(
                finger_delta_h, dim=-1
            ).detach().clone(),
            "book_half_extent_across_gripper_m": (
                book_half_extent_across_gripper.detach().clone()
            ),
        }

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

    def _book_grasp_relative_quat(self, n: int, dtype: torch.dtype) -> torch.Tensor:
        mode = str(self.cfg.book_grasp_orientation_in_hand)
        if mode == "franka_axes":
            values = self.cfg.book_to_hand_quat_franka_axes_wxyz
        elif mode == "manual_quat":
            values = self.cfg.book_grasp_rel_quat_wxyz
        else:
            raise ValueError(f"Unknown book_grasp_orientation_in_hand: {mode}")
        return torch.tensor(values, device=self.device, dtype=dtype).unsqueeze(0).expand(n, 4).clone()

    def _book_grasp_world_quat(
        self,
        grasp_quat_w: torch.Tensor,
        yaw_delta: torch.Tensor,
    ) -> torch.Tensor:
        """Resolve the configured reset orientation without changing Panda defaults."""
        n = int(grasp_quat_w.shape[0])
        qyaw = self._quat_world_yaw_half(yaw_delta)
        source = str(getattr(self.cfg, "book_grasp_orientation_source", "world_standing"))
        if source == "world_standing":
            q_stand = (
                torch.tensor(self.cfg.book_standing_quat, device=self.device, dtype=grasp_quat_w.dtype)
                .unsqueeze(0)
                .expand(n, 4)
                .clone()
            )
            return math_utils.quat_mul(q_stand, qyaw)
        if source == "grasp_relative":
            q_book_in_grasp = self._book_grasp_relative_quat(n, grasp_quat_w.dtype)
            return math_utils.quat_mul(
                grasp_quat_w,
                math_utils.quat_mul(q_book_in_grasp, qyaw),
            )
        raise ValueError(f"Unknown book_grasp_orientation_source: {source}")

    def _book_reset_pose_w(
        self,
        env_ids_t: torch.Tensor,
        grasp_pos_w: torch.Tensor,
        grasp_quat_w: torch.Tensor,
        jitter: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve the held-book reset pose from one explicit frame source."""

        n = int(env_ids_t.numel())
        dtype = grasp_pos_w.dtype
        source = str(getattr(self.cfg, "book_grasp_pose_source", "finger_midpoint"))
        if source == "finger_midpoint":
            base_offset = torch.tensor(
                self.cfg.book_grasp_offset_hand,
                device=self.device,
                dtype=dtype,
            ).unsqueeze(0).expand(n, 3)
            book_pos_w = grasp_pos_w + math_utils.quat_apply(
                grasp_quat_w,
                base_offset + jitter[:, :3],
            )
            book_quat_w = self._book_grasp_world_quat(
                grasp_quat_w,
                jitter[:, 3],
            )
            return book_pos_w, book_quat_w

        if source in ("eef_calibrated", "eef_calibrated_position"):
            eef_pos_w = self.robot.data.body_pos_w[env_ids_t, self._ee_body_idx]
            eef_quat_w = self.robot.data.body_quat_w[env_ids_t, self._ee_body_idx]
            eef_book_translation = torch.tensor(
                self.cfg.eef_book_translation_xyz,
                device=self.device,
                dtype=dtype,
            ).unsqueeze(0).expand(n, 3)
            book_pos_w = eef_pos_w + math_utils.quat_apply(
                eef_quat_w,
                eef_book_translation,
            )
            # Keep the existing grasp-randomization axes while centering every
            # sample on the approved physical xArm calibration.
            book_pos_w += math_utils.quat_apply(grasp_quat_w, jitter[:, :3])
            if source == "eef_calibrated_position":
                book_quat_w = self._book_grasp_world_quat(
                    grasp_quat_w,
                    jitter[:, 3],
                )
            else:
                eef_book_quaternion = torch.tensor(
                    self.cfg.eef_book_quaternion_wxyz,
                    device=self.device,
                    dtype=dtype,
                ).unsqueeze(0).expand(n, 4)
                book_quat_w = math_utils.quat_mul(
                    eef_quat_w,
                    eef_book_quaternion,
                )
                book_quat_w = math_utils.quat_mul(
                    book_quat_w,
                    self._quat_world_yaw_half(jitter[:, 3]),
                )
            return book_pos_w, book_quat_w

        raise ValueError(
            "book_grasp_pose_source must be 'finger_midpoint', "
            "'eef_calibrated', or 'eef_calibrated_position', "
            f"got {source!r}"
        )

    def _snap_book_to_measured_grasp(self, env_ids_t: torch.Tensor) -> torch.Tensor:
        """Place the book from the configured grasp-frame calibration.

        Returns the written book world state (N, 13) so the caller can hold the book
        at this exact pose during warmup without re-sampling jitter.
        """
        n = int(env_ids_t.numel())
        dtype = torch.float32

        self._ensure_scenario_trace_buffers()
        jitter = torch.zeros((n, 4), device=self.device, dtype=dtype)

        translation_min = getattr(self.cfg, "book_grasp_translation_jitter_min", None)
        translation_max = getattr(self.cfg, "book_grasp_translation_jitter_max", None)
        if (translation_min is None) != (translation_max is None):
            raise ValueError(
                "book_grasp_translation_jitter_min and max must either both be set or both be None"
            )
        if translation_min is not None:
            lower = torch.tensor(translation_min, device=self.device, dtype=dtype)
            upper = torch.tensor(translation_max, device=self.device, dtype=dtype)
            if lower.shape != (3,) or upper.shape != (3,) or torch.any(lower > upper):
                raise ValueError("book grasp translation jitter bounds must be ordered xyz triples")
            jitter[:, :3] = lower + (upper - lower) * torch.rand((n, 3), device=self.device, dtype=dtype)
        else:
            symmetric = (
                float(self.cfg.book_grasp_x_jitter),
                float(self.cfg.book_grasp_y_jitter),
                float(getattr(self.cfg, "book_grasp_z_jitter", 0.0)),
            )
            for axis, amount in enumerate(symmetric):
                if amount != 0.0:
                    jitter[:, axis] = sample_uniform(-amount, amount, (n,), self.device)

        if float(self.cfg.book_grasp_yaw_jitter) != 0.0:
            jitter[:, 3] = sample_uniform(
                -float(self.cfg.book_grasp_yaw_jitter),
                float(self.cfg.book_grasp_yaw_jitter),
                (n,),
                self.device,
            )
        bank_active = self._scenario_bank_index_env[env_ids_t] >= 0
        if torch.any(bank_active):
            jitter[bank_active] = self._frozen_grasp_jitter_env[env_ids_t][bank_active]
        self._scenario_grasp_jitter_env[env_ids_t] = jitter

        grasp_pos_w, grasp_quat_w = self._grasp_frame_pose_w(env_ids_t)
        book_pos_w, book_quat_w = self._book_reset_pose_w(
            env_ids_t,
            grasp_pos_w,
            grasp_quat_w,
            jitter,
        )
        book_pos_env = book_pos_w - self.scene.env_origins[env_ids_t]

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

    def _position_env_to_base(self, position_env: torch.Tensor) -> torch.Tensor:
        """Convert an env-relative point into the robot-base frame."""
        root_pos_env = self.robot.data.root_pos_w - self.scene.env_origins
        return math_utils.quat_apply_inverse(
            self.robot.data.root_quat_w,
            position_env - root_pos_env,
        )

    def _compute_ik_joint_targets_from_tool(self, target_pos_env: torch.Tensor, target_yaw: torch.Tensor) -> torch.Tensor:
        """Absolute IK: reach tool target in env frame with target yaw in base frame.

        Roll and pitch are kept from the current measured end-effector pose.
        """
        self._target_pos_env[:] = target_pos_env
        self._target_yaw[:] = target_yaw

        _, ee_quat_b = self._ee_pose_in_base()
        ee_roll_b, ee_pitch_b, _ = math_utils.euler_xyz_from_quat(ee_quat_b)
        quat_des_b = math_utils.quat_from_euler_xyz(ee_roll_b, ee_pitch_b, target_yaw)

        target_pos_b = self._position_env_to_base(target_pos_env)
        offset_des_b = math_utils.quat_apply(quat_des_b, self._ik_body_offset_pos_b)
        body_pos_des_b = target_pos_b - offset_des_b

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
        if len(self._gripper_command_joint_ids) > 0:
            fp = self.robot.data.joint_pos[:, self._gripper_command_joint_ids]
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

    def _book_dropped_for_mode(self, mode: torch.Tensor) -> torch.Tensor:
        """Return drops using grasp-aware INSERT and released-book thresholds."""
        p_env = self._book_pos_env()
        lowest_z = self._book_corners_env()[..., 2].min(dim=-1).values
        on_shelf = self._book_supported_on_shelf(p_env, lowest_z)
        return book_dropped_mask(
            lowest_z=lowest_z,
            on_shelf=on_shelf,
            mode=mode,
            insert_mode=_MODE_INSERT,
            true_ground_z=float(self.cfg.book_true_ground_lowest_z_thresh),
            shelf_drop_z=float(self.cfg.book_floor_lowest_z_thresh),
        )

    def _setup_scene(self):
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

        if len(self._gripper_command_joint_ids) > 0:
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
            finger_des = finger_cmd.unsqueeze(-1).expand(self.num_envs, len(self._gripper_command_joint_ids))
            self.robot.set_joint_position_target(finger_des, joint_ids=self._gripper_command_joint_ids)

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

        book_dropped_to_ground = self._book_dropped_for_mode(mode_start)

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
        self._release_step_buf = torch.where(
            accepted_release & (self._release_step_buf < 0),
            self.episode_length_buf.clone(),
            self._release_step_buf,
        )

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
        self._push_start_step_buf = torch.where(
            script_done & (self._push_start_step_buf < 0),
            self.episode_length_buf.clone(),
            self._push_start_step_buf,
        )

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
        # Insertion depth is monotonic. Once both depth thresholds have been
        # crossed, moving farther into the slot must not turn success into a
        # failure.
        rear_ok = rear_to_mouth >= float(self.cfg.success_rear_to_mouth_min)
        front_ok = front_to_back <= float(self.cfg.success_front_clear_max) + front_eps
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
        book_dropped_to_ground = self._book_dropped_for_mode(mode_before)

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
        failure_code = torch.where(
            done & ~success & (failure_code == _DONE_NONE) & oob,
            torch.full_like(failure_code, _DONE_OOB),
            failure_code,
        )
        failure_code = torch.where(
            done & ~success & (failure_code == _DONE_NONE) & fell,
            torch.full_like(failure_code, _DONE_FELL),
            failure_code,
        )
        failure_code = torch.where(
            done & ~success & (failure_code == _DONE_NONE) & ~(mode_before == _MODE_PUSH),
            torch.full_like(failure_code, _DONE_NOT_PUSH),
            failure_code,
        )
        failure_code = torch.where(
            done & ~success & (failure_code == _DONE_NONE) & (mode_before == _MODE_PUSH) & ~depth_ok,
            torch.full_like(failure_code, _DONE_DEPTH),
            failure_code,
        )
        failure_code = torch.where(
            done & ~success & (failure_code == _DONE_NONE) & (mode_before == _MODE_PUSH) & depth_ok & ~lat_ok,
            torch.full_like(failure_code, _DONE_LATERAL),
            failure_code,
        )
        failure_code = torch.where(
            done & ~success & (failure_code == _DONE_NONE) & (mode_before == _MODE_PUSH) & depth_ok & lat_ok & ~z_ok,
            torch.full_like(failure_code, _DONE_Z),
            failure_code,
        )
        failure_code = torch.where(
            done & ~success & (failure_code == _DONE_NONE) & (mode_before == _MODE_PUSH) & depth_ok & lat_ok & z_ok & ~yaw_ok,
            torch.full_like(failure_code, _DONE_YAW),
            failure_code,
        )
        failure_code = torch.where(
            done & ~success & (failure_code == _DONE_NONE) & (mode_before == _MODE_PUSH) & depth_ok & lat_ok & z_ok & yaw_ok & ~upright,
            torch.full_like(failure_code, _DONE_UPRIGHT),
            failure_code,
        )
        failure_code = torch.where(
            done & ~success & (failure_code == _DONE_NONE) & (mode_before == _MODE_PUSH) & depth_ok & lat_ok & z_ok & yaw_ok & upright & ~stable_ok,
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
        self.extras["episode_metric_done"] = done.clone()
        self.extras["episode_metric_slot_clearance"] = torch.full(
            (self.num_envs,), float(curr_clearance), device=self.device
        )
        self.extras["episode_metric_success"] = success.clone()
        self.extras["episode_metric_failure_code"] = failure_code.clone()
        self.extras["episode_metric_final_lat_err"] = torch.abs(m["lat_err"]).clone()
        self.extras["episode_metric_final_z_err"] = torch.abs(z_err).clone()
        self.extras["episode_metric_final_yaw_err_deg"] = torch.rad2deg(yaw_e).clone()
        self.extras["episode_metric_final_rear_to_mouth"] = rear_to_mouth.clone()
        self.extras["episode_metric_final_front_to_back"] = front_to_back.clone()
        self.extras["episode_metric_release_step"] = self._release_step_buf.clone()
        self.extras["episode_metric_push_steps"] = push_steps.clone()
        self.extras["episode_metric_mode_at_done"] = mode_before.clone()
        self._write_scenario_episode_metrics()

        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids_t = self._env_ids
        else:
            env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        self._ensure_scenario_trace_buffers()
        self._scenario_reset_count_env[env_ids_t] += 1
        self._scenario_joint_noise_env[env_ids_t] = 0.0
        self._scenario_applied_joint_noise_env[env_ids_t] = 0.0
        self._scenario_grasp_jitter_env[env_ids_t] = 0.0
        super()._reset_idx(env_ids_t)

        if hasattr(self, "actions") and isinstance(self.actions, torch.Tensor):
            self.actions[env_ids_t] = 0.0

        robot_default = self.robot.data.default_root_state[env_ids_t].clone()
        robot_default[:, 0:3] += self.scene.env_origins[env_ids_t]
        self.robot.write_root_state_to_sim(robot_default, env_ids=env_ids_t)

        joint_pos = self.robot.data.default_joint_pos[env_ids_t].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids_t].clone()

        noise = float(getattr(self.cfg, "reset_arm_joint_pos_noise", 0.0))
        bank_active = self._scenario_bank_index_env[env_ids_t] >= 0
        if (noise > 0.0 or torch.any(bank_active)) and len(self._arm_joint_ids) > 0:
            n = int(env_ids_t.numel())
            j = len(self._arm_joint_ids)
            dq = torch.zeros((n, j), device=self.device, dtype=torch.float32)
            if noise > 0.0:
                dq = sample_uniform(-noise, noise, (n, j), self.device)
            if torch.any(bank_active):
                dq[bank_active] = self._frozen_joint_noise_env[env_ids_t][bank_active]
            self._scenario_joint_noise_env[env_ids_t] = dq
            joint_pos[:, self._arm_joint_ids] = joint_pos[:, self._arm_joint_ids] + dq

            lo = self.robot.data.soft_joint_pos_limits[env_ids_t][:, self._arm_joint_ids, 0]
            hi = self.robot.data.soft_joint_pos_limits[env_ids_t][:, self._arm_joint_ids, 1]
            joint_pos[:, self._arm_joint_ids] = torch.max(torch.min(joint_pos[:, self._arm_joint_ids], hi), lo)

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids_t)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids_t)

        robot_target_pose_only = bool(
            getattr(self.cfg, "debug_robot_target_pose_only", False)
        )
        if robot_target_pose_only:
            # This branch deliberately performs no physics step here. The
            # derived debug environment removes every obstacle before physics
            # is allowed to advance, so the configured arm pose is tested in
            # isolation instead of being contaminated by book/shelf contact.
            snapped_book_state = self.book.data.default_root_state[env_ids_t].clone()
            snapped_book_state[:, 0:3] = self.scene.env_origins[env_ids_t]
            snapped_book_state[:, 2] -= 5.0
            snapped_book_state[:, 7:] = 0.0
            self.book.write_root_state_to_sim(snapped_book_state, env_ids=env_ids_t)
            self.scene.write_data_to_sim()
            self._arm_hold_joint_pos[env_ids_t] = joint_pos[:, self._arm_joint_ids]
        else:
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

            # Re-seed hold target from the state reached by the normal reset.
            self._arm_hold_joint_pos[env_ids_t] = self.robot.data.joint_pos[env_ids_t][
                :, self._arm_joint_ids
            ].clone()
        self.robot.set_joint_position_target(
            self._arm_hold_joint_pos[env_ids_t], joint_ids=self._arm_joint_ids, env_ids=env_ids_t
        )

        if len(self._gripper_command_joint_ids) > 0:
            finger_des = self.robot.data.default_joint_pos[env_ids_t][:, self._gripper_command_joint_ids]
            self.robot.set_joint_position_target(
                finger_des, joint_ids=self._gripper_command_joint_ids, env_ids=env_ids_t
            )

        # Hold the book at the exact snapped pose while gripper fingers converge.
        # Use the state returned by _snap_book_to_measured_grasp (pre-physics) so that
        # residual contact forces from the shelf on prior episodes cannot corrupt the target.
        warmup = 0 if robot_target_pose_only else int(getattr(self.cfg, "reset_warmup_steps", 0))
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
        self._release_step_buf[env_ids_t] = -1
        self._push_start_step_buf[env_ids_t] = -1
        self._capture_scenario_initial_pose(env_ids_t)
