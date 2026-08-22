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

import math
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
        self._robot_base_reference_marker_path = "/World/envs/env_0/RobotBaseReference"
        self._robot_base_reference_frame_path = "/World/Visuals/RobotBaseReferenceFrame"
        self._reachable_grasp_target_frame_path = "/World/Visuals/XArmReachableGraspTargetFrame"
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
        self._debug_integrated_target_pos_env = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float32
        )
        self._debug_scripted_retreat_start_pos_env = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float32
        )
        self._debug_scripted_retreat_initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._debug_nominal_push_tracking_paused = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._debug_nominal_push_alignment_complete = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._debug_nominal_push_line_y_env = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._debug_nominal_push_target_z_env = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._debug_nominal_push_lowering_complete = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._debug_nominal_push_line_initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._debug_missing_index_sequence_cursor = 0
        pose_ik_rotation_weight = getattr(
            self.cfg, "debug_pose_ik_rotation_weight", None
        )
        if pose_ik_rotation_weight is not None:
            pose_ik_rotation_weight = float(pose_ik_rotation_weight)
            if (
                not math.isfinite(pose_ik_rotation_weight)
                or not 0.0 < pose_ik_rotation_weight <= 1.0
            ):
                raise ValueError(
                    "debug_pose_ik_rotation_weight must be in the interval (0, 1]"
                )
        push_target_lead = getattr(
            self.cfg, "debug_nominal_push_max_target_lead_m", None
        )
        if push_target_lead is not None:
            push_target_lead = float(push_target_lead)
            if not math.isfinite(push_target_lead) or push_target_lead <= 0.0:
                raise ValueError(
                    "debug_nominal_push_max_target_lead_m must be positive and finite"
                )
        vertical_target_lead = getattr(
            self.cfg, "debug_nominal_push_max_vertical_target_lead_m", None
        )
        if vertical_target_lead is not None:
            vertical_target_lead = float(vertical_target_lead)
            if not math.isfinite(vertical_target_lead) or vertical_target_lead <= 0.0:
                raise ValueError(
                    "debug_nominal_push_max_vertical_target_lead_m must be positive and finite"
                )
        if bool(getattr(self.cfg, "debug_nominal_push_tracking_pause_enabled", False)):
            pause_error = float(
                getattr(self.cfg, "debug_nominal_push_tracking_pause_joint_error_rad", 0.12)
            )
            resume_error = float(
                getattr(self.cfg, "debug_nominal_push_tracking_resume_joint_error_rad", 0.08)
            )
            if not 0.0 <= resume_error < pause_error:
                raise ValueError(
                    "nominal PUSH tracking thresholds must satisfy "
                    "0 <= resume_joint_error < pause_joint_error"
                )
        if bool(getattr(self.cfg, "debug_nominal_push_spine_tracking_enabled", False)):
            pause_error = float(
                getattr(self.cfg, "debug_nominal_push_spine_pause_lateral_error_m", 0.004)
            )
            resume_error = float(
                getattr(self.cfg, "debug_nominal_push_spine_resume_lateral_error_m", 0.002)
            )
            if not 0.0 <= resume_error < pause_error:
                raise ValueError(
                    "nominal PUSH spine thresholds must satisfy "
                    "0 <= resume_lateral_error < pause_lateral_error"
                )
        self._debug_tool_to_book_transform_frozen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._reachable_grasp_sequence_step = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._reachable_grasp_sequence_complete = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._robot_target_gripper_ramp_step = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._robot_target_gripper_ramp_start = torch.zeros(
            (self.num_envs, len(self._gripper_command_joint_ids)),
            dtype=torch.float32,
            device=self.device,
        )
        self._robot_target_gripper_held_book_state = self.book.data.root_state_w.clone()
        self._robot_forward_backward_wait_step = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._robot_forward_backward_motion_step = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._robot_forward_backward_initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._robot_forward_backward_origin_pos_env = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device
        )
        self._robot_forward_backward_origin_quat_b = torch.zeros(
            (self.num_envs, 4), dtype=torch.float32, device=self.device
        )
        self._robot_forward_backward_origin_quat_b[:, 0] = 1.0
        self._robot_nominal_handoff_wait_step = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._robot_nominal_handoff_complete = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._reset_acceptance_attempt_env = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._reset_acceptance_sampled_total = 0
        self._reset_acceptance_accepted_total = 0
        self._reset_acceptance_rejected_total = 0
        self._reset_acceptance_reason_totals = {
            "non_finite": 0,
            "translation_drift": 0,
            "rotation_drift": 0,
            "book_dropped": 0,
            "arm_tracking": 0,
        }
        self._policy_release_guard_mode = str(
            getattr(self.cfg, "policy_release_guard_mode", "none")
        ).strip().lower()
        if self._policy_release_guard_mode not in ("none", "observable_geometry"):
            raise ValueError(
                "policy_release_guard_mode must be 'none' or "
                "'observable_geometry'"
            )
        premature_release_penalty = float(
            getattr(self.cfg, "premature_release_penalty", 0.0)
        )
        if not math.isfinite(premature_release_penalty) or premature_release_penalty < 0.0:
            raise ValueError("premature_release_penalty must be finite and non-negative")
        self._raw_policy_release_request = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._blocked_policy_release_request = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._validate_reset_acceptance_configuration()
        self._cleanup_legacy_debug_visuals()

    def _validate_reset_acceptance_configuration(self) -> None:
        if not bool(getattr(self.cfg, "enable_reset_acceptance_gate", False)):
            return

        validation_steps = int(self.cfg.reset_acceptance_validation_steps)
        max_attempts = int(self.cfg.reset_acceptance_max_attempts)
        if validation_steps < 1:
            raise ValueError("reset_acceptance_validation_steps must be positive")
        if max_attempts < 1:
            raise ValueError("reset_acceptance_max_attempts must be positive")

        positive_limits = (
            "reset_acceptance_translation_limit_m",
            "reset_acceptance_rotation_limit_rad",
            "reset_acceptance_arm_error_limit_rad",
        )
        for name in positive_limits:
            value = float(getattr(self.cfg, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        ground_height = float(self.cfg.reset_acceptance_ground_height_m)
        if not math.isfinite(ground_height) or ground_height < 0.0:
            raise ValueError(
                "reset_acceptance_ground_height_m must be finite and nonnegative"
            )

        print(
            "[RESET_ACCEPTANCE] enabled "
            f"validation_steps={validation_steps} "
            f"max_attempts={max_attempts} "
            "translation_limit_mm="
            f"{1000.0 * float(self.cfg.reset_acceptance_translation_limit_m):.3f} "
            "rotation_limit_deg="
            f"{math.degrees(float(self.cfg.reset_acceptance_rotation_limit_rad)):.3f} "
            "arm_error_limit_deg="
            f"{math.degrees(float(self.cfg.reset_acceptance_arm_error_limit_rad)):.3f}",
            flush=True,
        )

    def _reset_acceptance_snapshot(
        self, env_ids_t: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        grasp_pos_w, grasp_quat_w = self._grasp_frame_pose_w(env_ids_t)
        book_pos_w = self.book.data.root_link_pos_w[env_ids_t]
        book_quat_w = self.book.data.root_link_quat_w[env_ids_t]
        book_pos_g, book_quat_g = math_utils.subtract_frame_transforms(
            grasp_pos_w,
            grasp_quat_w,
            book_pos_w,
            book_quat_w,
        )

        joint_targets = getattr(self.robot.data, "joint_pos_target", None)
        if joint_targets is None:
            arm_error = torch.full(
                (env_ids_t.numel(),), float("nan"), device=self.device
            )
        else:
            arm_error = torch.abs(
                _wrap_to_pi(
                    joint_targets[env_ids_t][:, self._arm_joint_ids]
                    - self.robot.data.joint_pos[env_ids_t][:, self._arm_joint_ids]
                )
            ).amax(dim=-1)

        return {
            "book_position_grasp": book_pos_g,
            "book_quaternion_grasp": book_quat_g,
            "book_lowest_z": self._book_corners_env()[env_ids_t, :, 2].amin(dim=1),
            "arm_error": arm_error,
        }

    def _validate_randomized_reset(
        self, env_ids_t: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        initial = self._reset_acceptance_snapshot(env_ids_t)
        initial_pos = initial["book_position_grasp"].clone()
        initial_quat = initial["book_quaternion_grasp"].clone()
        max_translation = torch.zeros(
            env_ids_t.numel(), device=self.device, dtype=torch.float32
        )
        max_rotation = torch.zeros_like(max_translation)
        max_arm_error = initial["arm_error"].clone()
        min_lowest_z = initial["book_lowest_z"].clone()

        for _ in range(int(self.cfg.reset_acceptance_validation_steps)):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
            current = self._reset_acceptance_snapshot(env_ids_t)
            translation = torch.linalg.norm(
                current["book_position_grasp"] - initial_pos, dim=-1
            )
            rotation = math_utils.quat_error_magnitude(
                initial_quat, current["book_quaternion_grasp"]
            )
            max_translation = torch.maximum(max_translation, translation)
            max_rotation = torch.maximum(max_rotation, rotation)
            max_arm_error = torch.maximum(max_arm_error, current["arm_error"])
            min_lowest_z = torch.minimum(min_lowest_z, current["book_lowest_z"])

        finite = torch.stack(
            (max_translation, max_rotation, max_arm_error, min_lowest_z), dim=-1
        ).isfinite().all(dim=-1)
        reason_masks = {
            "non_finite": ~finite,
            "translation_drift": finite
            & (
                max_translation
                > float(self.cfg.reset_acceptance_translation_limit_m)
            ),
            "rotation_drift": finite
            & (max_rotation > float(self.cfg.reset_acceptance_rotation_limit_rad)),
            "book_dropped": finite
            & (min_lowest_z <= float(self.cfg.reset_acceptance_ground_height_m)),
            "arm_tracking": finite
            & (max_arm_error > float(self.cfg.reset_acceptance_arm_error_limit_rad)),
        }
        invalid = torch.zeros_like(finite)
        for mask in reason_masks.values():
            invalid |= mask
        return invalid, reason_masks

    def _write_reset_acceptance_metrics(self) -> None:
        sampled = max(1, self._reset_acceptance_sampled_total)
        log = self.extras.setdefault("log", {})
        log["reset_gate_acceptance_rate"] = torch.tensor(
            self._reset_acceptance_accepted_total / sampled,
            device=self.device,
            dtype=torch.float32,
        )
        log["reset_gate_rejection_rate"] = torch.tensor(
            self._reset_acceptance_rejected_total / sampled,
            device=self.device,
            dtype=torch.float32,
        )
        log["reset_gate_sampled_total"] = torch.tensor(
            float(self._reset_acceptance_sampled_total), device=self.device
        )
        for reason, count in self._reset_acceptance_reason_totals.items():
            log[f"reset_gate_{reason}_total"] = torch.tensor(
                float(count), device=self.device
            )

    def _refresh_state_after_reset_acceptance(
        self, env_ids_t: torch.Tensor
    ) -> None:
        tool_pos_env = self._ee_tool_pos_env()[env_ids_t]
        _, tool_quat_b = self._ee_pose_in_base()
        tool_quat_b = tool_quat_b[env_ids_t]
        self._target_pos_env[env_ids_t] = tool_pos_env
        _, _, tool_yaw = math_utils.euler_xyz_from_quat(tool_quat_b)
        self._target_yaw[env_ids_t] = tool_yaw
        self._debug_position_only_ee_quat_b[env_ids_t] = tool_quat_b.detach().clone()
        self._debug_integrated_target_pos_env[env_ids_t] = tool_pos_env.detach().clone()
        self._debug_scripted_retreat_start_pos_env[env_ids_t] = (
            tool_pos_env.detach().clone()
        )

        corners = self._book_corners_env()[env_ids_t]
        front_x = corners[..., 0].max(dim=-1).values
        rear_x = corners[..., 0].min(dim=-1).values
        self._prev_rear_to_mouth[env_ids_t] = (
            rear_x - float(self._geom_mouth_x)
        ).detach()
        self._prev_front_to_back[env_ids_t] = (
            float(self.cfg.slot_x_back) - front_x
        ).detach()
        self._capture_fixed_tool_to_book_transform(env_ids_t)
        self._capture_scenario_initial_pose(env_ids_t)

    def _apply_reset_acceptance_gate(self, env_ids_t: torch.Tensor) -> None:
        if not bool(getattr(self.cfg, "enable_reset_acceptance_gate", False)):
            return

        invalid, reason_masks = self._validate_randomized_reset(env_ids_t)
        valid = ~invalid
        accepted_count = int(valid.sum().item())
        rejected_count = int(invalid.sum().item())
        self._reset_acceptance_sampled_total += int(env_ids_t.numel())
        self._reset_acceptance_accepted_total += accepted_count
        self._reset_acceptance_rejected_total += rejected_count
        self._reset_acceptance_attempt_env[env_ids_t[valid]] = 0
        for reason, mask in reason_masks.items():
            self._reset_acceptance_reason_totals[reason] += int(mask.sum().item())

        if rejected_count:
            rejected_ids = env_ids_t[invalid]
            self._reset_acceptance_attempt_env[rejected_ids] += 1
            reason_counts = {
                reason: int(mask.sum().item())
                for reason, mask in reason_masks.items()
                if bool(mask.any().item())
            }
            print(
                "[RESET_ACCEPTANCE] rejected "
                f"envs={rejected_ids.detach().cpu().tolist()} "
                f"reasons={reason_counts} "
                "attempts="
                f"{self._reset_acceptance_attempt_env[rejected_ids].detach().cpu().tolist()}",
                flush=True,
            )
            exhausted = (
                self._reset_acceptance_attempt_env[rejected_ids]
                >= int(self.cfg.reset_acceptance_max_attempts)
            )
            if bool(exhausted.any().item()):
                exhausted_ids = rejected_ids[exhausted].detach().cpu().tolist()
                raise RuntimeError(
                    "randomized grasp reset failed the acceptance gate after "
                    f"{int(self.cfg.reset_acceptance_max_attempts)} attempts for "
                    f"environments {exhausted_ids}; reason totals="
                    f"{self._reset_acceptance_reason_totals}"
                )
            self._reset_idx(rejected_ids)

        self._refresh_state_after_reset_acceptance(env_ids_t)
        self._write_reset_acceptance_metrics()

    def _cleanup_legacy_debug_visuals(self) -> None:
        for prim_path in (
            "/Visuals/V6TargetEEFrame",
            "/Visuals/V6CurrentEEFrame",
            "/World/envs/env_0/V6TargetBook50",
            "/World/envs/env_0/RobotBaseReference",
            "/World/Visuals/RobotBaseReferenceFrame",
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

    def _create_robot_base_reference_marker(self) -> None:
        if not bool(getattr(self.cfg, "show_robot_base_reference_marker", False)):
            return

        marker_path = getattr(
            self,
            "_robot_base_reference_marker_path",
            "/World/envs/env_0/RobotBaseReference",
        )
        base_pos_env = torch.tensor(
            getattr(self.cfg, "robot_base_reference_pos", (0.0, 0.0, 0.0)),
            device=self.device,
            dtype=torch.float32,
        ).view(1, 3)
        base_pos_w = base_pos_env + self.scene.env_origins[0:1]

        if not sim_utils.is_prim_path_valid(marker_path):
            footprint_cfg = sim_utils.MeshCylinderCfg(
                radius=0.09,
                height=0.004,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.35, 0.05)),
            )
            footprint_pos = base_pos_w[0].detach().cpu().tolist()
            footprint_pos[2] += 0.002
            footprint_cfg.func(
                marker_path,
                footprint_cfg,
                translation=tuple(float(value) for value in footprint_pos),
                orientation=(1.0, 0.0, 0.0, 0.0),
            )

        if not hasattr(self, "_robot_base_reference_frame"):
            frame_cfg = FRAME_MARKER_CFG.copy()
            frame_cfg.prim_path = getattr(
                self,
                "_robot_base_reference_frame_path",
                "/World/Visuals/RobotBaseReferenceFrame",
            )
            frame_cfg.markers["frame"].scale = (0.18, 0.18, 0.18)
            if sim_utils.is_prim_path_valid(frame_cfg.prim_path):
                sim_utils.delete_prim(frame_cfg.prim_path)
            self._robot_base_reference_frame = VisualizationMarkers(frame_cfg)
            self._robot_base_reference_frame.set_visibility(True)

        base_quat_w = torch.zeros((1, 4), device=self.device, dtype=torch.float32)
        base_quat_w[:, 0] = 1.0
        self._robot_base_reference_frame.visualize(base_pos_w, base_quat_w)

    def _create_reachable_grasp_target_frame(self) -> None:
        if not bool(getattr(self.cfg, "show_reachable_grasp_target_frame", False)):
            return

        if not hasattr(self, "_reachable_grasp_target_frame"):
            marker_cfg = FRAME_MARKER_CFG.copy()
            marker_cfg.prim_path = getattr(
                self,
                "_reachable_grasp_target_frame_path",
                "/World/Visuals/XArmReachableGraspTargetFrame",
            )
            marker_cfg.markers["frame"].scale = (0.14, 0.14, 0.14)
            if sim_utils.is_prim_path_valid(marker_cfg.prim_path):
                sim_utils.delete_prim(marker_cfg.prim_path)
            self._reachable_grasp_target_frame = VisualizationMarkers(marker_cfg)
            self._reachable_grasp_target_frame.set_visibility(True)

        source = str(
            getattr(self.cfg, "reachable_grasp_target_frame_source", "slot_relative")
        )
        if source == "sequence_target" and hasattr(
            self, "_reachable_grasp_sequence_target_pos_w"
        ):
            target_pos_w = self._reachable_grasp_sequence_target_pos_w[0:1]
            target_quat_w = self._reachable_grasp_sequence_target_quat_w[0:1]
        elif source == "current_tool":
            ee_body_pos_w = self.robot.data.body_pos_w[0:1, self._ee_body_idx]
            target_quat_w = self.robot.data.body_quat_w[0:1, self._ee_body_idx]
            target_pos_w = ee_body_pos_w + math_utils.quat_apply(
                target_quat_w,
                self._ik_body_offset_pos_b[0:1],
            )
        elif source == "slot_relative":
            offset = torch.tensor(
                self.cfg.reset_tool_offset_slot_xyz,
                device=self.device,
                dtype=torch.float32,
            )
            target_pos_env = torch.tensor(
                (
                    float(self.cfg.slot_x_open) + float(offset[0].item()),
                    float(self.cfg.slot_center_y) + float(offset[1].item()),
                    float(self.cfg.shelf_top_z)
                    + float(self.cfg.shelf_thickness)
                    + 0.5 * float(self.cfg.book_size[1])
                    + float(offset[2].item()),
                ),
                device=self.device,
                dtype=torch.float32,
            ).view(1, 3)
            target_pos_w = target_pos_env + self.scene.env_origins[0:1]

            target_quat_b = torch.tensor(
                self.cfg.reset_tool_quaternion_slot_wxyz,
                device=self.device,
                dtype=torch.float32,
            ).view(1, 4)
            target_quat_b = target_quat_b / torch.linalg.norm(
                target_quat_b, dim=-1, keepdim=True
            )
            target_quat_w = math_utils.quat_mul(
                self.robot.data.root_quat_w[0:1],
                target_quat_b,
            )
        else:
            raise ValueError(
                "reachable_grasp_target_frame_source must be "
                "'sequence_target', 'current_tool', or 'slot_relative', "
                f"got {source!r}"
            )
        self._reachable_grasp_target_frame.visualize(target_pos_w, target_quat_w)

    def _refresh_debug_markers(self) -> None:
        if bool(getattr(self.cfg, "show_target_book_marker", True)) and not sim_utils.is_prim_path_valid(
            getattr(self, "_target_book_marker_path", "/World/envs/env_0/V6TargetBook50")
        ):
            self._create_target_book_marker()
        if bool(getattr(self.cfg, "show_target_ee_marker", True)):
            self._create_target_ee_marker()
        if bool(getattr(self.cfg, "show_current_ee_marker", True)):
            self._create_current_ee_marker()
        if bool(getattr(self.cfg, "show_robot_base_reference_marker", False)):
            self._create_robot_base_reference_marker()
        if bool(getattr(self.cfg, "show_reachable_grasp_target_frame", False)):
            self._create_reachable_grasp_target_frame()

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
        bounded_stage_names = (
            "residual_curriculum_grasp_translation_bounds_1",
            "residual_curriculum_grasp_translation_bounds_2",
            "residual_curriculum_grasp_translation_bounds_3",
            "residual_curriculum_grasp_translation_bounds_final",
        )
        bounded_stage = getattr(self.cfg, bounded_stage_names[stage], None)
        if bounded_stage is not None:
            lower, upper = bounded_stage
            self.cfg.book_grasp_translation_jitter_min = tuple(float(value) for value in lower)
            self.cfg.book_grasp_translation_jitter_max = tuple(float(value) for value in upper)

    def _nominal_release_assist_enabled(self) -> bool:
        if not bool(getattr(self.cfg, "enable_nominal_release_assist", False)):
            return False
        return self._residual_curriculum_progress() < float(getattr(self.cfg, "nominal_release_assist_until_frac", 0.30))

    def _align_gripper_to_sampled_slot(self, env_ids_t: torch.Tensor) -> None:
        """Keep the direct grasp demo at the configured xArm joint pose."""
        if bool(getattr(self.cfg, "debug_robot_target_pose_only", False)):
            return
        if bool(getattr(self.cfg, "debug_reachable_grasp_sequence", False)):
            return
        super()._align_gripper_to_sampled_slot(env_ids_t)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        missing_index_sequence = getattr(
            self.cfg, "debug_forced_missing_book_index_sequence", None
        )
        if missing_index_sequence:
            sequence = tuple(int(index) for index in missing_index_sequence)
            row_count = int(getattr(self.cfg, "row_book_count", 10))
            if not sequence or any(
                index < 0 or index >= row_count for index in sequence
            ):
                raise ValueError(
                    "debug_forced_missing_book_index_sequence must contain "
                    f"indices in [0, {row_count - 1}]"
                )
            cursor = int(
                getattr(self, "_debug_missing_index_sequence_cursor", 0)
            )
            selected_index = sequence[cursor % len(sequence)]
            self.cfg.forced_missing_book_index = selected_index
            self._debug_missing_index_sequence_cursor = cursor + 1
            print(
                "[XARM_SLOT_SEQUENCE] "
                f"episode={cursor + 1} missing_index={selected_index}",
                flush=True,
            )
        cmin, cmax = self._residual_curriculum_clearance_range()
        self.cfg.slot_lateral_clearance_min = cmin
        self.cfg.slot_lateral_clearance_max = cmax
        self._apply_residual_reset_curriculum()
        super()._reset_idx(env_ids)
        env_ids_t = self._env_ids if env_ids is None else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._apply_debug_row_layout_y_offset(env_ids_t)
        # Grasp-only preflights must remove neighboring geometry before the
        # xArm pose/grasp settling sequence advances physics.
        if bool(getattr(self.cfg, "debug_omit_bookshelf_obstacles", False)):
            self._omit_bookshelf_obstacles(env_ids_t)
        if bool(getattr(self.cfg, "reset_to_slot_relative_tool_pose", False)):
            self._reset_to_slot_relative_tool_pose(env_ids_t)
        if bool(getattr(self.cfg, "debug_start_from_default_grasp_pose", False)):
            self._reset_to_default_grasp_start(env_ids_t)
        if bool(getattr(self.cfg, "debug_robot_target_pose_only", False)):
            self._prepare_robot_target_pose_only(env_ids_t)
            if bool(
                getattr(self.cfg, "debug_spawn_book_with_collision_clearance", False)
            ):
                self._spawn_book_with_collision_clearance(env_ids_t)
            elif bool(getattr(self.cfg, "debug_spawn_book_panda_style", False)):
                self._spawn_book_panda_style(env_ids_t)
                if bool(
                    getattr(self.cfg, "debug_robot_nominal_controller_demo", False)
                ):
                    self._prepare_robot_nominal_preinsert_pose(env_ids_t)
        elif bool(getattr(self.cfg, "debug_reachable_grasp_sequence", False)):
            self._prepare_reachable_grasp_sequence(env_ids_t)
        else:
            if bool(getattr(self.cfg, "debug_snap_book_to_grasp_on_reset", False)):
                self._snap_book_to_measured_grasp(env_ids_t)
            self._capture_fixed_tool_to_book_transform(env_ids_t)
        self._clear_debug_rrt_plan()
        self._clear_debug_curobo_plan()
        self._debug_preinsert_hold_buf[env_ids_t] = 0
        if bool(getattr(self.cfg, "debug_spawn_at_target_tool_pose", False)):
            self._spawn_at_planned_tool_pose(env_ids_t)
        _, ee_quat_b = self._ee_pose_in_base()
        self._debug_position_only_ee_quat_b[env_ids_t] = ee_quat_b[env_ids_t].detach().clone()
        self._debug_integrated_target_pos_env[env_ids_t] = (
            self._ee_tool_pos_env()[env_ids_t].detach().clone()
        )
        self._debug_scripted_retreat_start_pos_env[env_ids_t] = (
            self._debug_integrated_target_pos_env[env_ids_t]
        )
        self._debug_scripted_retreat_initialized[env_ids_t] = False
        self._debug_nominal_push_tracking_paused[env_ids_t] = False
        self._debug_nominal_push_alignment_complete[env_ids_t] = False
        self._debug_nominal_push_line_y_env[env_ids_t] = 0.0
        self._debug_nominal_push_target_z_env[env_ids_t] = 0.0
        self._debug_nominal_push_lowering_complete[env_ids_t] = False
        self._debug_nominal_push_line_initialized[env_ids_t] = False
        self._raw_policy_release_request[env_ids_t] = False
        self._blocked_policy_release_request[env_ids_t] = False
        reset_env0 = env_ids is None
        if env_ids is not None:
            reset_env0 = bool(torch.any(torch.as_tensor(env_ids, device=self.device, dtype=torch.long) == 0).item())
        if reset_env0:
            self._create_target_book_marker()
            self._create_target_ee_marker()
            self._create_current_ee_marker()
            self._create_robot_base_reference_marker()
            if bool(getattr(self.cfg, "debug_print_sampled_grasp_joints", False)):
                self._print_env0_joint_values("[Bookshelf v6] sampled grasp joints")
        self._apply_reset_acceptance_gate(env_ids_t)

    def _apply_debug_row_layout_y_offset(self, env_ids_t: torch.Tensor) -> None:
        """Shift a diagnostic row without changing its ten-position schema."""

        offset_m = float(getattr(self.cfg, "debug_row_layout_y_offset_m", 0.0))
        if abs(offset_m) < 1.0e-12:
            return

        self._slot_center_y_env[env_ids_t] += offset_m
        for name in self._row_book_names():
            if name not in self.scene.rigid_objects:
                continue
            obj = self.scene.rigid_objects[name]
            state = obj.data.root_state_w[env_ids_t].clone()
            state[:, 1] += offset_m
            obj.write_root_state_to_sim(state, env_ids=env_ids_t)
        self.scene.write_data_to_sim()

        if bool(torch.any(env_ids_t == 0).item()):
            print(
                "[XARM_BOOKSHELF_DEBUG] shifted the complete side-book row "
                f"by y={offset_m:+.6f} m; slot_center_y="
                f"{float(self._slot_center_y_env[0].item()):+.6f} m",
                flush=True,
            )

    def _prepare_robot_target_pose_only(self, env_ids_t: torch.Tensor) -> None:
        """Write and hold the configured xArm pose with all scene contact removed."""

        robot_default = self.robot.data.default_root_state[env_ids_t].clone()
        robot_default[:, 0:3] += self.scene.env_origins[env_ids_t]
        joint_pos = self.robot.data.default_joint_pos[env_ids_t].clone()
        joint_vel = torch.zeros_like(self.robot.data.default_joint_vel[env_ids_t])

        parked_book_state = self.book.data.default_root_state[env_ids_t].clone()
        parked_book_state[:, 0:3] = self.scene.env_origins[env_ids_t]
        parked_book_state[:, 2] -= 5.0
        parked_book_state[:, 7:] = 0.0

        self.robot.write_root_state_to_sim(robot_default, env_ids=env_ids_t)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids_t)
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids_t)
        self.book.write_root_state_to_sim(parked_book_state, env_ids=env_ids_t)

        target_arm = joint_pos[:, self._arm_joint_ids].clone()
        if not hasattr(self, "_robot_target_pose_only_arm"):
            self._robot_target_pose_only_arm = self._arm_hold_joint_pos.clone()
        self._robot_target_pose_only_arm[env_ids_t] = target_arm
        self._arm_hold_joint_pos[env_ids_t] = target_arm
        self._robot_forward_backward_wait_step[env_ids_t] = 0
        self._robot_forward_backward_motion_step[env_ids_t] = 0
        self._robot_forward_backward_initialized[env_ids_t] = False
        self._robot_nominal_handoff_wait_step[env_ids_t] = 0
        self._robot_nominal_handoff_complete[env_ids_t] = False

        # The isolated grasp demo still labels the fixed, reviewed tool target as
        # ``sequence_target``.  Cache that pose explicitly before the dynamic book
        # is introduced so the marker cannot follow later contact motion.
        ee_body_pos_w = self.robot.data.body_pos_w[env_ids_t, self._ee_body_idx]
        ee_body_quat_w = self.robot.data.body_quat_w[env_ids_t, self._ee_body_idx]
        target_tool_pos_w = ee_body_pos_w + math_utils.quat_apply(
            ee_body_quat_w,
            self._ik_body_offset_pos_b[env_ids_t],
        )
        if not hasattr(self, "_reachable_grasp_sequence_target_pos_w"):
            self._reachable_grasp_sequence_target_pos_w = (
                self.robot.data.root_pos_w.clone()
            )
            self._reachable_grasp_sequence_target_quat_w = (
                self.robot.data.root_quat_w.clone()
            )
        self._reachable_grasp_sequence_target_pos_w[env_ids_t] = target_tool_pos_w
        self._reachable_grasp_sequence_target_quat_w[env_ids_t] = ee_body_quat_w

        self.scene.write_data_to_sim()
        if bool(torch.any(env_ids_t == 0).item()):
            target = {
                self.robot.joint_names[int(joint_id)]: float(target_arm[0, index].item())
                for index, joint_id in enumerate(self._arm_joint_ids)
            }
            print(
                "[XARM_TARGET_POSE] robot state written directly; no physics "
                f"contact was used; arm_target_rad={target}",
                flush=True,
            )

    def _apply_robot_target_pose_only(self) -> bool:
        if not bool(getattr(self.cfg, "debug_robot_target_pose_only", False)):
            return False
        nominal_demo = bool(
            getattr(self.cfg, "debug_robot_nominal_controller_demo", False)
        )
        if nominal_demo and bool(torch.all(self._robot_nominal_handoff_complete).item()):
            return False

        self.robot.set_joint_position_target(
            self._robot_target_pose_only_arm,
            joint_ids=self._arm_joint_ids,
        )
        self._arm_hold_joint_pos[:] = self._robot_target_pose_only_arm
        grasp_demo_active = bool(
            getattr(self.cfg, "debug_spawn_book_with_collision_clearance", False)
        ) or bool(getattr(self.cfg, "debug_spawn_book_panda_style", False))
        if grasp_demo_active:
            final_target = torch.full(
                (self.num_envs, len(self._gripper_command_joint_ids)),
                float(self.cfg.gripper_closed_joint_pos),
                device=self.device,
                dtype=torch.float32,
            )
            ramp_steps = max(
                1,
                int(getattr(self.cfg, "debug_robot_target_gripper_ramp_steps", 1)),
            )
            settle_steps = max(
                0,
                int(getattr(self.cfg, "debug_robot_target_gripper_settle_steps", 0)),
            )
            hold_steps = ramp_steps + settle_steps
            progress = torch.clamp(
                (self._robot_target_gripper_ramp_step.to(torch.float32) + 1.0)
                / float(ramp_steps),
                max=1.0,
            )
            smooth_progress = progress * progress * (3.0 - 2.0 * progress)
            gripper_target = self._robot_target_gripper_ramp_start + (
                final_target - self._robot_target_gripper_ramp_start
            ) * smooth_progress.unsqueeze(-1)
            self.robot.set_joint_position_target(
                gripper_target,
                joint_ids=self._gripper_command_joint_ids,
            )

            hold_mask = self._robot_target_gripper_ramp_step < hold_steps
            if torch.any(hold_mask):
                held_ids = self._env_ids[hold_mask]
                held_book_state = self._robot_target_gripper_held_book_state[
                    held_ids
                ].clone()
                held_book_state[:, 7:] = 0.0
                self.book.write_root_state_to_sim(
                    held_book_state,
                    env_ids=held_ids,
                )

            release_mask = self._robot_target_gripper_ramp_step == hold_steps
            if bool(release_mask[0].item()):
                print(
                    "[XARM_GRASP_DEMO] placement support released; the book is "
                    "now fully dynamic",
                    flush=True,
                )
            self._robot_target_gripper_ramp_step = torch.clamp(
                self._robot_target_gripper_ramp_step + 1,
                max=hold_steps + 1,
            )
            self._apply_robot_forward_backward_motion(hold_steps)
            self._apply_robot_nominal_controller_handoff(hold_steps)
        if nominal_demo and bool(torch.all(self._robot_nominal_handoff_complete).item()):
            return False
        return True

    def _apply_robot_nominal_controller_handoff(self, hold_steps: int) -> None:
        """Hand a settled physical grasp to the existing nominal controller."""

        if not bool(
            getattr(self.cfg, "debug_robot_nominal_controller_demo", False)
        ):
            return

        support_released = self._robot_target_gripper_ramp_step > int(hold_steps)
        self._robot_nominal_handoff_wait_step = torch.where(
            support_released & ~self._robot_nominal_handoff_complete,
            self._robot_nominal_handoff_wait_step + 1,
            self._robot_nominal_handoff_wait_step,
        )
        wait_steps = max(
            0,
            int(getattr(self.cfg, "debug_robot_nominal_handoff_wait_steps", 60)),
        )
        handoff = (
            support_released
            & ~self._robot_nominal_handoff_complete
            & (self._robot_nominal_handoff_wait_step >= wait_steps)
        )
        if not torch.any(handoff):
            return

        handoff_ids = self._env_ids[handoff]
        if bool(getattr(self.cfg, "debug_hold_book_fixed_to_tool", False)):
            # The ideal case starts from the exact centered book target. Do not
            # preserve a small IK/contact offset as the new rigid grasp.
            self._place_ideal_book_at_nominal_target(handoff_ids)
        else:
            # The normal reset path may already have frozen a transform before
            # the debug grasp was established. Measure the settled dynamic
            # grasp anew.
            self._debug_tool_to_book_transform_frozen[handoff_ids] = False
            self._capture_fixed_tool_to_book_transform(handoff_ids)

        corners = self._book_corners_env()[handoff_ids]
        front_x = corners[..., 0].max(dim=-1).values
        rear_x = corners[..., 0].min(dim=-1).values
        mouth_x = float(self._geom_mouth_x)
        self._prev_rear_to_mouth[handoff_ids] = (rear_x - mouth_x).detach()
        self._prev_front_to_back[handoff_ids] = (
            float(self.cfg.slot_x_back) - front_x
        ).detach()
        self._success_steps_buf[handoff_ids] = 0
        self._mode[handoff_ids] = _MODE_INSERT
        self._mode_start[handoff_ids] = _MODE_INSERT
        self._script_step_buf[handoff_ids] = 0
        self._release_request[handoff_ids] = False
        self._release_step_buf[handoff_ids] = -1
        self._push_start_step_buf[handoff_ids] = -1
        self._robot_nominal_handoff_complete[handoff_ids] = True
        self._debug_integrated_target_pos_env[handoff_ids] = (
            self._ee_tool_pos_env()[handoff_ids].detach().clone()
        )
        self._debug_nominal_push_alignment_complete[handoff_ids] = False

        if bool(handoff[0].item()):
            metrics = self._compute_task_metrics()
            print(
                "[XARM_NOMINAL_HANDOFF] dynamic grasp measured; nominal "
                "controller enabled with zero residual; "
                f"rear_to_mouth={float(metrics['rear_to_mouth'][0].item()):+.6f} "
                f"lat_err={float(metrics['lat_err'][0].item()):+.6f} "
                f"z_err={float(metrics['z_err'][0].item()):+.6f} "
                f"yaw_err_deg={math.degrees(float(metrics['yaw_err'][0].item())):+.3f}",
                flush=True,
            )

    def _place_ideal_book_at_nominal_target(
        self, env_ids_t: torch.Tensor
    ) -> torch.Tensor:
        """Center the ideal rigid grasp exactly at the slot-mouth target."""

        desired_book_pos, desired_book_quat = self._target_book_pose_tensors(
            inside_fraction=0.0
        )
        book_state = self.book.data.root_state_w[env_ids_t].clone()
        book_state[:, 0:3] = (
            desired_book_pos[env_ids_t] + self.scene.env_origins[env_ids_t]
        )
        book_state[:, 3:7] = desired_book_quat[env_ids_t]
        book_state[:, 7:] = 0.0
        self.book.write_root_state_to_sim(book_state, env_ids=env_ids_t)

        if not hasattr(self, "_book_offset_tool"):
            self._book_offset_tool = torch.zeros(
                (self.num_envs, 3), device=self.device, dtype=torch.float32
            )
            self._book_rel_quat_tool = torch.zeros(
                (self.num_envs, 4), device=self.device, dtype=torch.float32
            )
            self._book_rel_quat_tool[:, 0] = 1.0

        ee_body_pos_env = (
            self.robot.data.body_pos_w[:, self._ee_body_idx]
            - self.scene.env_origins
        )
        ee_body_quat_w = self.robot.data.body_quat_w[:, self._ee_body_idx]
        tool_pos_env = ee_body_pos_env + math_utils.quat_apply(
            ee_body_quat_w, self._ik_body_offset_pos_b
        )
        self._book_offset_tool[env_ids_t] = math_utils.quat_apply_inverse(
            ee_body_quat_w[env_ids_t],
            desired_book_pos[env_ids_t] - tool_pos_env[env_ids_t],
        )
        self._book_rel_quat_tool[env_ids_t] = math_utils.quat_mul(
            math_utils.quat_inv(ee_body_quat_w[env_ids_t]),
            desired_book_quat[env_ids_t],
        )
        self._debug_tool_to_book_transform_frozen[env_ids_t] = True
        return book_state

    def _prepare_robot_nominal_preinsert_pose(
        self, env_ids_t: torch.Tensor
    ) -> None:
        """Place the xArm grasp using the original book-relative target math."""

        # Capture the just-created grasp, then reuse the original nominal task's
        # desired-book -> tool-pose conversion at the slot mouth.
        self._debug_tool_to_book_transform_frozen[env_ids_t] = False
        self._capture_fixed_tool_to_book_transform(env_ids_t)
        self._spawn_at_planned_tool_pose(env_ids_t)
        ideal_book_state = None
        if bool(getattr(self.cfg, "debug_hold_book_fixed_to_tool", False)):
            ideal_book_state = self._place_ideal_book_at_nominal_target(env_ids_t)

        arm_target = self.robot.data.joint_pos[env_ids_t][
            :, self._arm_joint_ids
        ].clone()
        self._robot_target_pose_only_arm[env_ids_t] = arm_target
        self._arm_hold_joint_pos[env_ids_t] = arm_target
        if ideal_book_state is None:
            ideal_book_state = self.book.data.root_state_w[env_ids_t].clone()
        self._robot_target_gripper_held_book_state[env_ids_t] = ideal_book_state
        self._robot_target_gripper_ramp_start[env_ids_t] = (
            self.robot.data.joint_pos[env_ids_t][
                :, self._gripper_command_joint_ids
            ].clone()
        )

        ee_body_pos_w = self.robot.data.body_pos_w[env_ids_t, self._ee_body_idx]
        ee_body_quat_w = self.robot.data.body_quat_w[env_ids_t, self._ee_body_idx]
        target_tool_pos_w = ee_body_pos_w + math_utils.quat_apply(
            ee_body_quat_w,
            self._ik_body_offset_pos_b[env_ids_t],
        )
        self._reachable_grasp_sequence_target_pos_w[env_ids_t] = target_tool_pos_w
        self._reachable_grasp_sequence_target_quat_w[env_ids_t] = ee_body_quat_w

        desired_book_pos, _ = self._target_book_pose_tensors(inside_fraction=0.0)
        actual_book_pos = self._book_pos_env()[env_ids_t]
        position_error = torch.linalg.norm(
            actual_book_pos - desired_book_pos[env_ids_t], dim=-1
        )
        if bool(torch.any(env_ids_t == 0).item()):
            env0_index = int(
                torch.nonzero(env_ids_t == 0, as_tuple=False)[0, 0].item()
            )
            print(
                "[XARM_NOMINAL_PREINSERT] original book-relative target "
                "applied to xArm; "
                f"book_position_error_m={float(position_error[env0_index].item()):.6f} "
                f"tool_env_m={target_tool_pos_w[env0_index].detach().cpu().tolist()}",
                flush=True,
            )

    def _apply_robot_forward_backward_motion(self, hold_steps: int) -> None:
        """Move a settled dynamic grasp smoothly along +X and back."""

        if not bool(
            getattr(self.cfg, "debug_robot_forward_backward_demo", False)
        ):
            return

        support_released = self._robot_target_gripper_ramp_step > int(hold_steps)
        self._robot_forward_backward_wait_step = torch.where(
            support_released,
            self._robot_forward_backward_wait_step + 1,
            torch.zeros_like(self._robot_forward_backward_wait_step),
        )
        wait_steps = max(
            0,
            int(getattr(self.cfg, "debug_robot_forward_backward_wait_steps", 60)),
        )
        ready = support_released & (
            self._robot_forward_backward_wait_step >= wait_steps
        )
        initialize = ready & ~self._robot_forward_backward_initialized
        if torch.any(initialize):
            initialize_ids = self._env_ids[initialize]
            tool_pos_env = self._ee_tool_pos_env().detach()
            _, tool_quat_b = self._ee_pose_in_base()
            self._robot_forward_backward_origin_pos_env[initialize_ids] = (
                tool_pos_env[initialize_ids]
            )
            self._robot_forward_backward_origin_quat_b[initialize_ids] = (
                tool_quat_b[initialize_ids].detach()
            )
            self._robot_forward_backward_motion_step[initialize_ids] = 0
            self._robot_forward_backward_initialized[initialize_ids] = True
            if bool(initialize[0].item()):
                print(
                    "[XARM_FORWARD_BACKWARD] dynamic grasp settled; starting "
                    "smooth +X motion from the measured tool pose",
                    flush=True,
                )

        moving = ready & self._robot_forward_backward_initialized
        if not torch.any(moving):
            return

        distance_m = float(
            getattr(self.cfg, "debug_robot_forward_backward_distance_m", 0.05)
        )
        half_period_steps = max(
            1,
            int(
                getattr(
                    self.cfg,
                    "debug_robot_forward_backward_half_period_steps",
                    180,
                )
            ),
        )
        cycle_steps = 2 * half_period_steps
        cycle_step = torch.remainder(
            self._robot_forward_backward_motion_step, cycle_steps
        )
        phase = cycle_step.to(torch.float32) * (math.pi / half_period_steps)
        displacement = 0.5 * distance_m * (1.0 - torch.cos(phase))

        target_pos_env = self._ee_tool_pos_env().detach().clone()
        _, target_quat_b = self._ee_pose_in_base()
        target_quat_b = target_quat_b.detach().clone()
        moving_target_pos = self._robot_forward_backward_origin_pos_env[
            moving
        ].clone()
        moving_target_pos[:, 0] += displacement[moving]
        target_pos_env[moving] = moving_target_pos
        target_quat_b[moving] = self._robot_forward_backward_origin_quat_b[moving]

        joint_pos_des = self._compute_ik_joint_targets_from_tool_quat(
            target_pos_env,
            target_quat_b,
        )
        moving_ids = self._env_ids[moving]
        arm_des = joint_pos_des[moving_ids]
        limits = self.robot.data.soft_joint_pos_limits[moving_ids][
            :, self._arm_joint_ids
        ]
        arm_des = torch.maximum(
            torch.minimum(arm_des, limits[..., 1]), limits[..., 0]
        )
        self.robot.set_joint_position_target(
            arm_des,
            joint_ids=self._arm_joint_ids,
            env_ids=moving_ids,
        )
        self._arm_hold_joint_pos[moving_ids] = arm_des

        self._reachable_grasp_sequence_target_pos_w[moving_ids] = (
            target_pos_env[moving_ids] + self.scene.env_origins[moving_ids]
        )
        self._reachable_grasp_sequence_target_quat_w[moving_ids] = (
            math_utils.quat_mul(
                self.robot.data.root_quat_w[moving_ids],
                target_quat_b[moving_ids],
            )
        )

        if bool(moving[0].item()):
            env0_step = int(self._robot_forward_backward_motion_step[0].item())
            report_interval = max(1, half_period_steps // 2)
            if env0_step % report_interval == 0:
                leg = "forward" if int(cycle_step[0].item()) < half_period_steps else "return"
                print(
                    "[XARM_FORWARD_BACKWARD] "
                    f"leg={leg} displacement_m={float(displacement[0].item()):.6f} "
                    f"target_tool_env_m={target_pos_env[0].detach().cpu().tolist()}",
                    flush=True,
                )

        self._robot_forward_backward_motion_step[moving] += 1

    def _spawn_book_panda_style(self, env_ids_t: torch.Tensor) -> None:
        """Reproduce the Panda reset: snap, support briefly, then release."""

        book_state = self._snap_book_to_measured_grasp(env_ids_t)
        self._robot_target_gripper_ramp_start[env_ids_t] = (
            self.robot.data.joint_pos[env_ids_t][:, self._gripper_command_joint_ids]
        )
        self._robot_target_gripper_held_book_state[env_ids_t] = book_state
        self._robot_target_gripper_ramp_step[env_ids_t] = 0
        self.scene.write_data_to_sim()

        if bool(torch.any(env_ids_t == 0).item()):
            print(
                "[XARM_PANDA_RESET] book snapped to measured finger midpoint; "
                f"book_grasp_offset_hand={tuple(self.cfg.book_grasp_offset_hand)}; "
                "velocity reset to zero",
                flush=True,
            )

    def _spawn_book_with_collision_clearance(self, env_ids_t: torch.Tensor) -> None:
        """Place the book for the temporary support-close-release sequence."""

        n = int(env_ids_t.numel())
        dtype = torch.float32
        grasp_pos_w, grasp_quat_w = self._grasp_frame_pose_w(env_ids_t)
        book_quat_grasp = self._book_grasp_relative_quat(n, dtype)
        book_rotation_grasp = math_utils.matrix_from_quat(book_quat_grasp)
        book_half_size = 0.5 * torch.tensor(
            self.cfg.book_size,
            device=self.device,
            dtype=dtype,
        )
        book_half_extent_grasp = torch.matmul(
            torch.abs(book_rotation_grasp),
            book_half_size.view(1, 3, 1).expand(n, 3, 1),
        ).squeeze(-1)

        left_pos_w = self.robot.data.body_pos_w[env_ids_t, self._left_finger_body_idx]
        right_pos_w = self.robot.data.body_pos_w[env_ids_t, self._right_finger_body_idx]
        finger_origin_distance = torch.linalg.norm(left_pos_w - right_pos_w, dim=-1)
        inner_surface_offset = float(self.cfg.debug_finger_inner_surface_offset_m)
        finger_opening = finger_origin_distance - 2.0 * inner_surface_offset
        book_thickness = 2.0 * book_half_extent_grasp[:, 1]
        total_finger_clearance = finger_opening - book_thickness
        minimum_total_clearance = 2.0 * float(self.cfg.debug_book_min_finger_clearance_m)
        numerical_tolerance_m = 1.0e-5
        if torch.any(
            total_finger_clearance
            < minimum_total_clearance - numerical_tolerance_m
        ):
            smallest = float(total_finger_clearance.min().item())
            raise RuntimeError(
                "book would overlap the xArm finger collision surfaces: "
                f"smallest total clearance is {smallest:.6f} m"
            )

        palm_clearance = float(self.cfg.debug_book_palm_clearance_m)
        book_offset_grasp = torch.zeros((n, 3), device=self.device, dtype=dtype)
        book_offset_grasp[:, 2] = book_half_extent_grasp[:, 2] + palm_clearance
        book_pos_w = grasp_pos_w + math_utils.quat_apply(
            grasp_quat_w,
            book_offset_grasp,
        )
        book_quat_w = math_utils.quat_mul(grasp_quat_w, book_quat_grasp)

        book_state = self.book.data.default_root_state[env_ids_t].clone()
        book_state[:, 0:3] = book_pos_w
        book_state[:, 3:7] = book_quat_w
        book_state[:, 7:] = 0.0
        self.book.write_root_state_to_sim(book_state, env_ids=env_ids_t)
        self._robot_target_gripper_ramp_start[env_ids_t] = (
            self.robot.data.joint_pos[env_ids_t][:, self._gripper_command_joint_ids]
        )
        self._robot_target_gripper_held_book_state[env_ids_t] = book_state
        self._robot_target_gripper_ramp_step[env_ids_t] = 0
        self.scene.write_data_to_sim()

        if bool(torch.any(env_ids_t == 0).item()):
            env0_index = int(
                torch.nonzero(env_ids_t == 0, as_tuple=False)[0, 0].item()
            )
            print(
                "[XARM_BOOK_CLEARANCE] book placed without changing "
                "the arm or gripper; "
                f"finger_opening_m={float(finger_opening[env0_index].item()):.6f} "
                f"book_thickness_m={float(book_thickness[env0_index].item()):.6f} "
                f"clearance_per_side_m={0.5 * float(total_finger_clearance[env0_index].item()):.6f} "
                f"book_half_depth_m={float(book_half_extent_grasp[env0_index, 2].item()):.6f} "
                f"palm_clearance_m={palm_clearance:.6f}",
                flush=True,
            )

    def _prepare_reachable_grasp_sequence(self, env_ids_t: torch.Tensor) -> None:
        """Spawn a dynamic book inside a settled, near-contact gripper."""

        target_arm = self.robot.data.joint_pos[env_ids_t][:, self._arm_joint_ids].clone()
        preclose_value = getattr(self.cfg, "debug_reachable_grasp_preclose_joint_pos", None)
        if preclose_value is None:
            preclose_value = float(self.cfg.gripper_open_joint_pos)
        preclose_value = float(preclose_value)

        # Settle the gripper linkage before the book exists at the grasp point.
        # This lets the USD mimic joints find a physically consistent pose.
        parked_book_state = self.book.data.root_state_w[env_ids_t].clone()
        parked_book_state[:, 0:3] = self.scene.env_origins[env_ids_t]
        parked_book_state[:, 2] -= 5.0
        parked_book_state[:, 7:] = 0.0
        self.book.write_root_state_to_sim(parked_book_state, env_ids=env_ids_t)

        preclose_command = torch.full(
            (env_ids_t.numel(), len(self._gripper_command_joint_ids)),
            preclose_value,
            device=self.device,
            dtype=torch.float32,
        )
        preclose_steps = max(
            1, int(getattr(self.cfg, "debug_reachable_grasp_preclose_settle_steps", 30))
        )
        for _ in range(preclose_steps):
            self.robot.set_joint_position_target(
                target_arm, joint_ids=self._arm_joint_ids, env_ids=env_ids_t
            )
            self.robot.set_joint_position_target(
                preclose_command,
                joint_ids=self._gripper_command_joint_ids,
                env_ids=env_ids_t,
            )
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

        ee_body_pos_w = self.robot.data.body_pos_w[env_ids_t, self._ee_body_idx]
        ee_body_quat_w = self.robot.data.body_quat_w[env_ids_t, self._ee_body_idx]
        target_tool_pos_w = ee_body_pos_w + math_utils.quat_apply(
            ee_body_quat_w,
            self._ik_body_offset_pos_b[env_ids_t],
        )

        # Spawn with zero velocity at the exact measured grasp frame. The book
        # is dynamic immediately; no later step rewrites its root state.
        self._snap_book_to_measured_grasp(env_ids_t)

        if not hasattr(self, "_reachable_grasp_sequence_target_arm"):
            self._reachable_grasp_sequence_target_arm = self._arm_hold_joint_pos.clone()
            self._reachable_grasp_sequence_target_pos_w = self.robot.data.root_pos_w.clone()
            self._reachable_grasp_sequence_target_quat_w = self.robot.data.root_quat_w.clone()
        self._reachable_grasp_sequence_target_arm[env_ids_t] = target_arm
        self._reachable_grasp_sequence_target_pos_w[env_ids_t] = target_tool_pos_w
        self._reachable_grasp_sequence_target_quat_w[env_ids_t] = ee_body_quat_w
        self._reachable_grasp_sequence_step[env_ids_t] = 0
        self._reachable_grasp_sequence_complete[env_ids_t] = False

        self._arm_hold_joint_pos[env_ids_t] = target_arm

        if bool(torch.any(env_ids_t == 0).item()):
            print(
                "[XARM_GRASP_SEQUENCE] dynamic book spawned in settled near-contact "
                "gripper; holding the measured grasp width",
                flush=True,
            )

    def _apply_reachable_grasp_sequence(self) -> bool:
        if not bool(getattr(self.cfg, "debug_reachable_grasp_sequence", False)):
            return False

        settle_steps = max(1, int(self.cfg.debug_reachable_grasp_settle_steps))
        step = self._reachable_grasp_sequence_step

        active = ~self._reachable_grasp_sequence_complete
        finishing_mask = active & (step >= settle_steps)

        arm_target = self._reachable_grasp_sequence_target_arm.clone()
        self.robot.set_joint_position_target(arm_target, joint_ids=self._arm_joint_ids)
        self._arm_hold_joint_pos[:] = arm_target

        if len(self._gripper_command_joint_ids) > 0:
            grasp_width = getattr(self.cfg, "debug_reachable_grasp_preclose_joint_pos", None)
            if grasp_width is None:
                grasp_width = float(self.cfg.gripper_open_joint_pos)
            grasp_width = float(grasp_width)
            gripper_target = torch.full(
                (self.num_envs, len(self._gripper_command_joint_ids)),
                grasp_width,
                device=self.device,
                dtype=torch.float32,
            )
            self.robot.set_joint_position_target(
                gripper_target, joint_ids=self._gripper_command_joint_ids
            )

        if torch.any(finishing_mask):
            finishing_ids = self._env_ids[finishing_mask]
            contact = []
            if len(self._gripper_command_joint_ids) > 0:
                contact = (
                    self.robot.data.joint_pos[finishing_ids][
                        :, self._gripper_command_joint_ids
                    ][0]
                    .detach()
                    .cpu()
                    .tolist()
                )
            self._reachable_grasp_sequence_complete[finishing_ids] = True
            if bool(torch.any(finishing_ids == 0).item()):
                print(
                    "[XARM_GRASP_SEQUENCE] constant-width dynamic grasp observation complete; "
                    f"gripper_position_rad={contact}",
                    flush=True,
                )

        self._reachable_grasp_sequence_step[active] += 1
        return True

    def _reset_to_slot_relative_tool_pose(self, env_ids_t: torch.Tensor) -> None:
        """Solve the configured slot-relative TCP pose with the active robot model."""

        offset = torch.tensor(
            self.cfg.reset_tool_offset_slot_xyz,
            device=self.device,
            dtype=torch.float32,
        )
        quat = torch.tensor(
            self.cfg.reset_tool_quaternion_slot_wxyz,
            device=self.device,
            dtype=torch.float32,
        )
        if offset.shape != (3,) or quat.shape != (4,):
            raise ValueError("slot-relative reset tool pose must contain xyz and wxyz")
        quat_norm = torch.linalg.norm(quat)
        if not bool(torch.isfinite(offset).all().item()) or not bool(torch.isfinite(quat_norm).item()):
            raise ValueError("slot-relative reset tool pose must be finite")
        if float(quat_norm.item()) < 1.0e-6:
            raise ValueError("slot-relative reset tool quaternion must be nonzero")
        quat = quat / quat_norm

        target_pos_env = self._ee_tool_pos_env().detach().clone()
        _, target_quat_b = self._ee_pose_in_base()
        target_quat_b = target_quat_b.detach().clone()
        target_z = (
            float(self.cfg.shelf_top_z)
            + float(self.cfg.shelf_thickness)
            + 0.5 * float(self.cfg.book_size[1])
        )
        target_pos_env[env_ids_t, 0] = float(self.cfg.slot_x_open) + offset[0]
        target_pos_env[env_ids_t, 1] = self._slot_center_y()[env_ids_t] + offset[1]
        target_pos_env[env_ids_t, 2] = target_z + offset[2]
        target_quat_b[env_ids_t] = quat

        iterations = max(1, int(self.cfg.reset_tool_ik_iters))
        for _ in range(iterations):
            joint_pos_des = self._compute_ik_joint_targets_from_tool_quat(
                target_pos_env,
                target_quat_b,
            )
            joint_pos = self.robot.data.joint_pos[env_ids_t].clone()
            joint_vel = self.robot.data.joint_vel[env_ids_t].clone()
            arm_des = joint_pos_des[env_ids_t]
            limits = self.robot.data.soft_joint_pos_limits[env_ids_t][:, self._arm_joint_ids]
            arm_des = torch.maximum(torch.minimum(arm_des, limits[..., 1]), limits[..., 0])
            joint_pos[:, self._arm_joint_ids] = arm_des
            joint_vel[:, self._arm_joint_ids] = 0.0
            self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids_t)
            self.robot.set_joint_position_target(joint_pos, env_ids=env_ids_t)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

        # The slot-relative IK above intentionally establishes the nominal
        # xArm pose. Apply reset joint noise once afterwards so IK cannot erase
        # the requested training perturbation, then place the book from the
        # resulting measured grasp frame.
        arm_noise = self._scenario_joint_noise_env[env_ids_t]
        if arm_noise.shape[1] != len(self._arm_joint_ids):
            raise RuntimeError("sampled arm noise does not match the xArm joint count")
        self._scenario_applied_joint_noise_env[env_ids_t] = 0.0
        if bool(torch.any(arm_noise != 0.0).item()):
            joint_pos = self.robot.data.joint_pos[env_ids_t].clone()
            joint_vel = self.robot.data.joint_vel[env_ids_t].clone()
            nominal_arm_pos = joint_pos[:, self._arm_joint_ids].clone()
            arm_des = nominal_arm_pos + arm_noise
            limits = self.robot.data.soft_joint_pos_limits[env_ids_t][
                :, self._arm_joint_ids
            ]
            arm_des = torch.maximum(
                torch.minimum(arm_des, limits[..., 1]), limits[..., 0]
            )
            self._scenario_applied_joint_noise_env[env_ids_t] = (
                arm_des - nominal_arm_pos
            )
            joint_pos[:, self._arm_joint_ids] = arm_des
            joint_vel[:, self._arm_joint_ids] = 0.0
            self.robot.write_joint_state_to_sim(
                joint_pos, joint_vel, env_ids=env_ids_t
            )
            self.robot.set_joint_position_target(joint_pos, env_ids=env_ids_t)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

        self._arm_hold_joint_pos[env_ids_t] = self.robot.data.joint_pos[env_ids_t][
            :, self._arm_joint_ids
        ].clone()
        self.robot.set_joint_position_target(
            self._arm_hold_joint_pos[env_ids_t],
            joint_ids=self._arm_joint_ids,
            env_ids=env_ids_t,
        )

        # The official xArm USD starts at the 34 mm placement gap. Establish
        # the configured 32 mm holding gap while the book pose is supported,
        # then let the rigid book become dynamic after the warmup loop.
        if len(self._gripper_command_joint_ids) > 0:
            finger_des = torch.full(
                (env_ids_t.numel(), len(self._gripper_command_joint_ids)),
                float(self.cfg.gripper_closed_joint_pos),
                device=self.device,
                dtype=torch.float32,
            )
            self.robot.set_joint_position_target(
                finger_des,
                joint_ids=self._gripper_command_joint_ids,
                env_ids=env_ids_t,
            )

        snapped_book_state = self._snap_book_to_measured_grasp(env_ids_t)
        warmup = max(0, int(getattr(self.cfg, "reset_warmup_steps", 0)))
        for _ in range(warmup):
            self.book.write_root_state_to_sim(snapped_book_state, env_ids=env_ids_t)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
        # End the supported settling phase at the exact requested transform
        # with zero velocity. The next normal environment step is the first
        # fully dynamic grasp observation.
        self.book.write_root_state_to_sim(snapped_book_state, env_ids=env_ids_t)
        self.scene.write_data_to_sim()
        self.scene.update(dt=0.0)

        actual_pos = self._ee_tool_pos_env()[env_ids_t]
        _, actual_quat = self._ee_pose_in_base()
        actual_quat = actual_quat[env_ids_t]
        pos_error = torch.linalg.norm(actual_pos - target_pos_env[env_ids_t], dim=-1)
        rot_error = math_utils.quat_error_magnitude(target_quat_b[env_ids_t], actual_quat)
        if bool(torch.any(env_ids_t == 0).item()):
            env0_index = int(torch.nonzero(env_ids_t == 0, as_tuple=False)[0, 0].item())
            sampled_noise_deg = math.degrees(
                float(
                    torch.abs(
                        self._scenario_applied_joint_noise_env[
                            env_ids_t[env0_index]
                        ]
                    ).max().item()
                )
            )
            print(
                "[XARM_RANDOMIZED_RESET] "
                "nominal_target_delta_after_joint_noise_m="
                f"{float(pos_error[env0_index].item()):.6f} "
                "nominal_target_rotation_delta_after_joint_noise_deg="
                f"{math.degrees(float(rot_error[env0_index].item())):.3f} "
                f"applied_joint_noise_max_deg={sampled_noise_deg:.3f} "
                f"target_tcp_env_m={target_pos_env[env_ids_t][env0_index].detach().cpu().tolist()}"
            )

        self._target_pos_env[env_ids_t] = actual_pos
        _, _, actual_yaw = math_utils.euler_xyz_from_quat(actual_quat)
        self._target_yaw[env_ids_t] = actual_yaw

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
        if len(self._gripper_command_joint_ids) > 0:
            finger_des = torch.full(
                (env_ids_t.numel(), len(self._gripper_command_joint_ids)),
                float(self.cfg.gripper_closed_joint_pos),
                device=self.device,
                dtype=torch.float32,
            )
            self.robot.set_joint_position_target(
                finger_des, joint_ids=self._gripper_command_joint_ids, env_ids=env_ids_t
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
        # Shelf collisions are disabled before PhysX starts in _setup_scene.
        # Deleting initialized collision prims here produces detachShape errors.
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
                if len(self._gripper_command_joint_ids) > 0:
                    joint_pos[:, self._gripper_command_joint_ids] = float(
                        self.cfg.gripper_closed_joint_pos
                    )
                    joint_vel[:, self._gripper_command_joint_ids] = 0.0

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

            if len(self._gripper_command_joint_ids) > 0:
                joint_pos[:, self._gripper_command_joint_ids] = float(
                    self.cfg.gripper_closed_joint_pos
                )
                joint_vel[:, self._gripper_command_joint_ids] = 0.0

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

        target_pos_b = self._position_env_to_base(target_pos_env)
        offset_des_b = math_utils.quat_apply(target_quat_b, self._ik_body_offset_pos_b)
        body_pos_des_b = target_pos_b - offset_des_b

        self._ik_cmd[:, 0:3] = body_pos_des_b
        self._ik_cmd[:, 3:7] = target_quat_b
        self._ik.set_command(self._ik_cmd)

        ee_pos_b, ee_quat_b = self._ee_pose_in_base()
        jacobian = self.robot.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, self._jacobi_joint_ids]
        joint_pos = self.robot.data.joint_pos[:, self._arm_joint_ids]

        rotation_weight = getattr(self.cfg, "debug_pose_ik_rotation_weight", None)
        if rotation_weight is not None:
            # Control the offset TCP directly. The xArm Jacobian belongs to
            # link7, while the pushing point is 172 mm away; assuming perfect
            # wrist orientation converts small angular errors into large
            # lateral TCP errors.
            tool_offset_b = math_utils.quat_apply(
                ee_quat_b,
                self._ik_body_offset_pos_b,
            )
            tool_pos_b = ee_pos_b + tool_offset_b
            position_error = target_pos_b - tool_pos_b
            _, rotation_error = math_utils.compute_pose_error(
                ee_pos_b,
                ee_quat_b,
                ee_pos_b,
                target_quat_b,
                rot_error_type="axis_angle",
            )

            offset_x, offset_y, offset_z = tool_offset_b.unbind(dim=-1)
            zeros = torch.zeros_like(offset_x)
            offset_skew = torch.stack(
                (
                    zeros,
                    -offset_z,
                    offset_y,
                    offset_z,
                    zeros,
                    -offset_x,
                    -offset_y,
                    offset_x,
                    zeros,
                ),
                dim=-1,
            ).reshape(self.num_envs, 3, 3)
            tool_position_jacobian = (
                jacobian[:, 0:3, :]
                - offset_skew @ jacobian[:, 3:6, :]
            )
            rotation_weight = float(rotation_weight)
            weighted_error = torch.cat(
                (position_error, rotation_weight * rotation_error), dim=-1
            )
            weighted_jacobian = torch.cat(
                (
                    tool_position_jacobian,
                    rotation_weight * jacobian[:, 3:6, :],
                ),
                dim=1,
            )
            jacobian_t = weighted_jacobian.transpose(1, 2)
            damping = 0.01
            damping_matrix = (damping**2) * torch.eye(
                weighted_jacobian.shape[1],
                device=self.device,
                dtype=weighted_jacobian.dtype,
            )
            delta_joint_pos = jacobian_t @ torch.linalg.solve(
                weighted_jacobian @ jacobian_t + damping_matrix,
                weighted_error.unsqueeze(-1),
            )
            return joint_pos + delta_joint_pos.squeeze(-1)

        return self._ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

    @staticmethod
    def _apply_base_frame_orientation_delta(
        current_quat_b: torch.Tensor,
        delta_yaw: torch.Tensor,
        delta_pitch: torch.Tensor,
    ) -> torch.Tensor:
        """Apply small nominal yaw/pitch corrections without Euler decomposition."""

        zeros = torch.zeros_like(delta_yaw)
        delta_quat_b = math_utils.quat_from_euler_xyz(
            zeros,
            delta_pitch,
            delta_yaw,
        )
        return math_utils.quat_mul(delta_quat_b, current_quat_b)

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

        if len(self._gripper_command_joint_ids) > 0:
            finger_des = torch.full(
                (1, len(self._gripper_command_joint_ids)),
                float(self.cfg.gripper_closed_joint_pos),
                device=self.device,
                dtype=torch.float32,
            )
            self.robot.set_joint_position_target(
                finger_des,
                joint_ids=self._gripper_command_joint_ids,
                env_ids=torch.tensor([0], device=self.device),
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

        if len(self._gripper_command_joint_ids) > 0:
            finger_des = torch.full(
                (1, len(self._gripper_command_joint_ids)),
                float(self.cfg.gripper_closed_joint_pos),
                device=self.device,
                dtype=torch.float32,
            )
            self.robot.set_joint_position_target(
                finger_des,
                joint_ids=self._gripper_command_joint_ids,
                env_ids=torch.tensor([0], device=self.device),
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
        raw_policy_release = self.actions[:, -1] > float(self.cfg.release_trigger_threshold)
        self._raw_policy_release_request = raw_policy_release

        metrics = None
        geometry_ready = None
        if self._policy_release_guard_mode == "observable_geometry":
            metrics = self._compute_task_metrics()
            geometry_ready = self._nominal_release_mask(metrics)
            policy_release = raw_policy_release & geometry_ready
            self._blocked_policy_release_request = (
                raw_policy_release
                & (self._mode == _MODE_INSERT)
                & ~geometry_ready
            )
        else:
            policy_release = raw_policy_release
            self._blocked_policy_release_request = torch.zeros_like(raw_policy_release)

        nominal_release = torch.zeros_like(raw_policy_release)
        if (
            self._nominal_release_assist_enabled()
            and
            bool(getattr(self.cfg, "enable_nominal_controller", True))
            and not bool(getattr(self.cfg, "debug_freeze_nominal_controller", False))
        ):
            if metrics is None:
                metrics = self._compute_task_metrics()
            if geometry_ready is None:
                geometry_ready = self._nominal_release_mask(metrics)
            nominal_release = geometry_ready

        self._release_request = policy_release | nominal_release

    def _get_rewards(self) -> torch.Tensor:
        rew = super()._get_rewards()
        premature_weight = float(
            getattr(self.cfg, "premature_release_penalty", 0.0)
        )
        premature_release = self._blocked_policy_release_request.float()
        premature_penalty = premature_weight * premature_release
        rew = rew - premature_penalty

        self.extras.setdefault("log", {})
        self.extras["log"]["raw_policy_release_fraction"] = (
            self._raw_policy_release_request.float().mean()
        )
        self.extras["log"]["blocked_policy_release_fraction"] = premature_release.mean()
        self.extras["log"]["premature_release_penalty_mean"] = premature_penalty.mean()

        weight = float(getattr(self.cfg, "residual_action_l2_weight", 0.0))
        if weight <= 0.0:
            return rew

        residual_l2 = torch.mean(torch.square(self.actions[:, 0:5]), dim=-1)
        penalty = weight * residual_l2
        rew = rew - penalty
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
        if bool(getattr(self.cfg, "debug_disable_episode_resets", False)):
            terminated = torch.zeros_like(terminated)
            time_out = torch.zeros_like(time_out)
        return terminated, time_out

    def _update_nominal_push_lowering_phase(self, mode: torch.Tensor) -> None:
        """Prepare Panda-style push height before the straight xArm push."""

        enabled = bool(
            getattr(self.cfg, "debug_nominal_push_reuse_insert_forward", False)
        ) and bool(
            getattr(self.cfg, "debug_nominal_push_lower_before_forward", False)
        )
        push_mask = mode == _MODE_PUSH
        self._debug_nominal_push_line_initialized[~push_mask] = False
        self._debug_nominal_push_lowering_complete[~push_mask] = False
        if not enabled or not torch.any(push_mask):
            return

        tool_pos = self._ee_tool_pos_env()
        book_pos = self._book_pos_env()
        entering_push = push_mask & ~self._debug_nominal_push_line_initialized
        if torch.any(entering_push):
            corners = self._book_corners_env()
            book_bottom_z = corners[..., 2].min(dim=-1).values
            book_top_z = corners[..., 2].max(dim=-1).values
            book_height = torch.clamp(book_top_z - book_bottom_z, min=1.0e-4)
            panda_push_z = book_bottom_z + (
                float(self.cfg.nominal_push_z_fraction_from_bottom) * book_height
            )
            align_to_book_center = bool(
                getattr(self.cfg, "debug_nominal_push_align_to_book_center", False)
            )
            push_line_y = (
                book_pos[:, 1]
                if align_to_book_center
                else tool_pos[:, 1]
            )
            self._debug_nominal_push_line_y_env[entering_push] = (
                push_line_y[entering_push].detach()
            )
            self._debug_nominal_push_target_z_env[entering_push] = (
                panda_push_z[entering_push].detach()
            )
            self._debug_nominal_push_lowering_complete[entering_push] = False
            self._debug_nominal_push_line_initialized[entering_push] = True
            self._debug_integrated_target_pos_env[entering_push] = (
                tool_pos[entering_push].detach()
            )
            if bool(entering_push[0].item()):
                print(
                    "[XARM_PUSH_PHASE] CENTERING_AND_LOWERING; "
                    "Panda target is book_bottom + "
                    f"{float(self.cfg.nominal_push_z_fraction_from_bottom):.3f} * book_height; "
                    f"tool_y={float(tool_pos[0, 1].item()):+.6f} "
                    f"book_center_y={float(book_pos[0, 1].item()):+.6f} "
                    f"start_z={float(tool_pos[0, 2].item()):+.6f} "
                    f"target_z={float(panda_push_z[0].item()):+.6f}",
                    flush=True,
                )

        lowering = push_mask & ~self._debug_nominal_push_lowering_complete
        height_error = self._debug_nominal_push_target_z_env - tool_pos[:, 2]
        lateral_error = self._debug_nominal_push_line_y_env - tool_pos[:, 1]
        height_tolerance = float(self.cfg.nominal_push_dz_limit)
        lateral_tolerance = float(self.cfg.nominal_push_dy_limit)
        reached_push_line = (
            lowering
            & (torch.abs(height_error) <= height_tolerance)
            & (torch.abs(lateral_error) <= lateral_tolerance)
        )
        if torch.any(reached_push_line):
            self._debug_nominal_push_lowering_complete[reached_push_line] = True
            # Begin the retained forward target from the measured pose reached
            # by the xArm, not from a stale INSERT or retreat target.
            self._debug_integrated_target_pos_env[reached_push_line] = (
                tool_pos[reached_push_line].detach()
            )
            if bool(reached_push_line[0].item()):
                print(
                    "[XARM_PUSH_PHASE] PUSHING_STRAIGHT_X; book-center Y and "
                    "lower height reached; Y/Z/orientation now held",
                    flush=True,
                )

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

        reuse_insert_forward = bool(
            getattr(self.cfg, "debug_nominal_push_reuse_insert_forward", False)
        )
        if torch.any(push_mask) and reuse_insert_forward:
            lower_before_forward = bool(
                getattr(self.cfg, "debug_nominal_push_lower_before_forward", False)
            )
            push_forward = push_mask.clone()
            if lower_before_forward:
                push_forward = push_mask & self._debug_nominal_push_lowering_complete
                lowering = push_mask & ~self._debug_nominal_push_lowering_complete
                # Advance a smooth commanded-height ramp, as the fixed retreat
                # does. Using measured tool Z here leaves the target only one
                # millimetre ahead and makes the xArm descend extremely slowly.
                push_height_error = self._debug_nominal_push_target_z_env - (
                    self._debug_integrated_target_pos_env[:, 2]
                )
                push_lower_dz = torch.clamp(
                    float(self.cfg.nominal_push_height_gain) * push_height_error,
                    -float(self.cfg.nominal_push_dz_limit),
                    float(self.cfg.nominal_push_dz_limit),
                )
                nominal[:, 2] = torch.where(lowering, push_lower_dz, nominal[:, 2])

            # Forward motion starts only after lowering. The remaining PUSH
            # deltas stay zero, holding Y, Z, and orientation at that pose.
            nominal[:, 0] = torch.where(push_forward, insert_dx, torch.zeros_like(nominal[:, 0]))

        if torch.any(push_mask) and not reuse_insert_forward:
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
            if bool(getattr(self.cfg, "debug_nominal_push_hold_y_only", False)):
                # Keep the push on the release/retreat lateral line while the
                # existing height controller moves to the lower push point.
                push_dy = torch.zeros_like(push_dy)
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

        if self._apply_robot_target_pose_only():
            return

        if self._apply_reachable_grasp_sequence():
            return

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

        self._update_nominal_push_lowering_phase(mode)

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

        normal_mask = mode != _MODE_SCRIPTED
        push_mask = normal_mask & (mode == _MODE_PUSH)
        panda_lower_then_push = bool(
            getattr(self.cfg, "debug_nominal_push_reuse_insert_forward", False)
        ) and bool(
            getattr(self.cfg, "debug_nominal_push_lower_before_forward", False)
        )
        if panda_lower_then_push and torch.any(push_mask):
            lowering_mask = push_mask & ~self._debug_nominal_push_lowering_complete
            pushing_mask = push_mask & self._debug_nominal_push_lowering_complete
            delta[push_mask, 1] = 0.0
            delta[push_mask, 3] = 0.0
            delta[push_mask, 4] = 0.0
            delta[lowering_mask, 0] = 0.0
            delta[pushing_mask, 2] = 0.0
        ee_tool_pos_env = self._ee_tool_pos_env()
        book_pos_env = self._book_pos_env()
        lock_push_y = bool(
            getattr(self.cfg, "debug_nominal_push_lock_y_to_entry", False)
        )
        self._debug_nominal_push_line_initialized[~push_mask] = False
        if lock_push_y and torch.any(push_mask):
            entering_push = push_mask & ~self._debug_nominal_push_line_initialized
            if torch.any(entering_push):
                self._debug_nominal_push_line_y_env[entering_push] = (
                    ee_tool_pos_env[entering_push, 1].detach()
                )
                if (
                    bool(getattr(self.cfg, "debug_integrate_position_target_ee", False))
                    and not bool(
                        getattr(
                            self.cfg,
                            "debug_nominal_push_current_relative_target",
                            False,
                        )
                    )
                ):
                    # Start PUSH from the measured post-retreat pose. The
                    # retained target then builds only within its lead bound.
                    self._debug_integrated_target_pos_env[entering_push] = (
                        ee_tool_pos_env[entering_push].detach()
                    )
                self._debug_nominal_push_line_initialized[entering_push] = True
        push_spine_y_error = ee_tool_pos_env[:, 1] - book_pos_env[:, 1]
        arm_joint_pos = self.robot.data.joint_pos[:, self._arm_joint_ids]
        push_tracking_error = torch.abs(
            _wrap_to_pi(self._arm_hold_joint_pos - arm_joint_pos)
        ).amax(dim=-1)
        joint_tracking_enabled = bool(
            getattr(self.cfg, "debug_nominal_push_tracking_pause_enabled", False)
        )
        spine_tracking_enabled = bool(
            getattr(self.cfg, "debug_nominal_push_spine_tracking_enabled", False)
        )
        if joint_tracking_enabled:
            pause_error = float(
                getattr(self.cfg, "debug_nominal_push_tracking_pause_joint_error_rad", 0.12)
            )
            resume_error = float(
                getattr(self.cfg, "debug_nominal_push_tracking_resume_joint_error_rad", 0.08)
            )

        alignment_complete = self._debug_nominal_push_alignment_complete.clone()
        alignment_complete[~push_mask] = False
        if spine_tracking_enabled:
            pause_lateral_error = float(
                getattr(self.cfg, "debug_nominal_push_spine_pause_lateral_error_m", 0.004)
            )
            resume_lateral_error = float(
                getattr(self.cfg, "debug_nominal_push_spine_resume_lateral_error_m", 0.002)
            )
            absolute_spine_y_error = torch.abs(push_spine_y_error)
            alignment_lost = push_mask & alignment_complete & (
                absolute_spine_y_error >= pause_lateral_error
            )
            alignment_complete = torch.where(
                alignment_lost, torch.zeros_like(alignment_complete), alignment_complete
            )
            alignment_ready = push_mask & ~alignment_complete & (
                absolute_spine_y_error <= resume_lateral_error
            )
            if joint_tracking_enabled:
                alignment_ready &= push_tracking_error <= resume_error
            alignment_complete = torch.where(
                alignment_ready, torch.ones_like(alignment_complete), alignment_complete
            )
        else:
            alignment_complete = push_mask.clone()
        self._debug_nominal_push_alignment_complete[:] = alignment_complete

        if joint_tracking_enabled or spine_tracking_enabled:
            paused = self._debug_nominal_push_tracking_paused.clone() & push_mask
            if joint_tracking_enabled:
                paused = torch.where(
                    push_mask & (push_tracking_error >= pause_error),
                    torch.ones_like(paused),
                    paused,
                )
                paused = torch.where(
                    push_mask
                    & alignment_complete
                    & (push_tracking_error <= resume_error),
                    torch.zeros_like(paused),
                    paused,
                )
            paused |= push_mask & ~alignment_complete
            self._debug_nominal_push_tracking_paused[:] = paused
            delta[paused, 0] = 0.0
            delta[paused, 2] = 0.0
        else:
            self._debug_nominal_push_tracking_paused[:] = False

        if spine_tracking_enabled and torch.any(push_mask):
            # A paused push must still be able to move laterally toward the
            # released book. Otherwise zeroing X/Z while hold-y is enabled
            # produces an all-zero delta and the arm can never leave pause.
            recovery_step = float(
                getattr(self.cfg, "debug_nominal_push_spine_recovery_step_m", 0.002)
            )
            spine_recovery_dy = torch.clamp(
                -push_spine_y_error,
                -recovery_step,
                recovery_step,
            )
            delta[push_mask, 1] = spine_recovery_dy[push_mask]

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
                push_phase = "inactive"
                if mode_i == _MODE_PUSH:
                    lower_first = bool(
                        getattr(self.cfg, "debug_nominal_push_reuse_insert_forward", False)
                    ) and bool(
                        getattr(self.cfg, "debug_nominal_push_lower_before_forward", False)
                    )
                    if lower_first:
                        push_phase = (
                            "PUSHING"
                            if bool(self._debug_nominal_push_lowering_complete[env_i].item())
                            else "LOWERING"
                        )
                    else:
                        push_phase = (
                            "PANDA_CURRENT_RELATIVE"
                            if bool(
                                getattr(
                                    self.cfg,
                                    "debug_nominal_push_current_relative_target",
                                    False,
                                )
                            )
                            else "PANDA_BOUNDED_RETAINED"
                        )
                print(
                    "[residual debug] "
                    f"step={int(self.common_step_counter)} env={env_i} mode={mode_i} "
                    f"push_phase={push_phase} "
                    f"release_action={raw_release:+.3f} release_req={release_req} "
                    f"push_track_err={float(push_tracking_error[env_i].item()):.4f}rad "
                    f"push_spine_y_err={float(push_spine_y_error[env_i].item()):+.4f}m "
                    f"push_z={float(ee_tool_pos_env[env_i, 2].item()):+.4f}->"
                    f"{float(self._debug_nominal_push_target_z_env[env_i].item()):+.4f}m "
                    f"push_aligned={bool(self._debug_nominal_push_alignment_complete[env_i].item())} "
                    f"push_x_paused={bool(self._debug_nominal_push_tracking_paused[env_i].item())} "
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

        _, ee_quat_b = self._ee_pose_in_base()
        _, ee_pitch_b, ee_yaw_b = math_utils.euler_xyz_from_quat(ee_quat_b)

        integrate_position_target = bool(
            getattr(self.cfg, "debug_integrate_position_target_ee", False)
        )
        target_pos_env = (
            self._debug_integrated_target_pos_env.clone()
            if integrate_position_target
            else ee_tool_pos_env.clone()
        )
        scripted_mask = mode == _MODE_SCRIPTED
        scripted_fixed_retreat = bool(
            getattr(self.cfg, "debug_scripted_fixed_retreat_path", False)
        )
        self._debug_scripted_retreat_initialized[~scripted_mask] = False
        if scripted_fixed_retreat and torch.any(scripted_mask):
            retreat_start_mask = scripted_mask & ~self._debug_scripted_retreat_initialized
            if torch.any(retreat_start_mask):
                self._debug_scripted_retreat_start_pos_env[retreat_start_mask] = (
                    ee_tool_pos_env[retreat_start_mask].detach().clone()
                )
                self._debug_scripted_retreat_initialized[retreat_start_mask] = True

            open_steps = int(self.cfg.script_open_steps)
            retreat_steps = max(1, int(self.cfg.script_retreat_steps))
            configured_total_dx = getattr(
                self.cfg, "debug_scripted_fixed_retreat_total_dx", None
            )
            retreat_total_dx = (
                float(configured_total_dx)
                if configured_total_dx is not None
                else retreat_steps * float(self.cfg.script_retreat_dx)
            )
            if not math.isfinite(retreat_total_dx):
                raise ValueError("fixed retreat total X distance must be finite")
            retreat_progress = torch.clamp(
                self._script_step_buf - open_steps + 1,
                min=0,
                max=retreat_steps,
            ).to(dtype=torch.float32) / float(retreat_steps)
            fixed_retreat_target = self._debug_scripted_retreat_start_pos_env.clone()
            fixed_retreat_target[:, 0] += (
                retreat_progress * retreat_total_dx
            )
            fixed_retreat_target[:, 2] += (
                retreat_progress * retreat_steps * float(self.cfg.script_retreat_dz)
            )
            target_pos_env[scripted_mask] = fixed_retreat_target[scripted_mask]

        scripted_current_relative = bool(
            getattr(self.cfg, "debug_scripted_current_relative_target", False)
        )
        if (
            scripted_current_relative
            and not scripted_fixed_retreat
            and torch.any(scripted_mask)
        ):
            # Preserve the original controller's retreat semantics. Its
            # script_retreat_dx is an offset from the measured tool pose, not
            # a delta to accumulate repeatedly onto the INSERT target.
            target_pos_env[scripted_mask] = ee_tool_pos_env[scripted_mask]
        push_current_relative = bool(
            getattr(self.cfg, "debug_nominal_push_current_relative_target", False)
        )
        if push_current_relative and torch.any(push_mask):
            target_pos_env[push_mask] = ee_tool_pos_env[push_mask]
        target_yaw = ee_yaw_b.clone()
        target_pitch = ee_pitch_b.clone()

        if torch.any(normal_mask):
            target_pos_env[normal_mask] = (
                target_pos_env[normal_mask] + delta[normal_mask, 0:3]
            )
            target_yaw[normal_mask] = _wrap_to_pi(
                ee_yaw_b[normal_mask] + delta[normal_mask, 3]
            )
            target_pitch[normal_mask] = _wrap_to_pi(
                ee_pitch_b[normal_mask] + delta[normal_mask, 4]
            )

        push_target_lead = getattr(
            self.cfg, "debug_nominal_push_max_target_lead_m", None
        )
        if push_target_lead is not None and torch.any(push_mask):
            # Retain enough Cartesian error to push against contact, but never
            # let an unreachable IK target accumulate far ahead of the arm.
            lead_m = float(push_target_lead)
            target_pos_env[push_mask, 0] = torch.minimum(
                target_pos_env[push_mask, 0],
                ee_tool_pos_env[push_mask, 0] + lead_m,
            )
            vertical_target_lead = getattr(
                self.cfg,
                "debug_nominal_push_max_vertical_target_lead_m",
                None,
            )
            vertical_lead_m = (
                lead_m
                if vertical_target_lead is None
                else float(vertical_target_lead)
            )
            target_pos_env[push_mask, 2] = torch.clamp(
                target_pos_env[push_mask, 2],
                min=ee_tool_pos_env[push_mask, 2] - vertical_lead_m,
                max=ee_tool_pos_env[push_mask, 2] + vertical_lead_m,
            )

        if lock_push_y and torch.any(push_mask):
            # PUSH is a straight line in the insertion plane. Latch the
            # measured lateral coordinate once at PUSH entry and actively
            # command it thereafter; a zero dy alone does not remove a stale
            # lateral component from an integrated Cartesian target.
            target_pos_env[push_mask, 1] = self._debug_nominal_push_line_y_env[
                push_mask
            ]

            if bool(getattr(self.cfg, "debug_print_residual_components", False)):
                interval = max(
                    1, int(getattr(self.cfg, "debug_print_residual_interval", 30))
                )
                env_i = int(
                    getattr(self.cfg, "debug_print_residual_env_index", 0)
                )
                if (
                    0 <= env_i < self.num_envs
                    and bool(push_mask[env_i].item())
                    and int(self.common_step_counter) % interval == 0
                ):
                    print(
                        "[XARM_PUSH_TARGET] "
                        f"tool_xyz={ee_tool_pos_env[env_i].detach().cpu().tolist()} "
                        f"target_xyz={target_pos_env[env_i].detach().cpu().tolist()} "
                        "lateral_error_m="
                        f"{float((ee_tool_pos_env[env_i, 1] - target_pos_env[env_i, 1]).item()):+.6f}",
                        flush=True,
                    )

        if spine_tracking_enabled and torch.any(push_mask):
            # Hold the measured book spine directly for the complete PUSH.
            # This avoids an incremental Y correction oscillating around it.
            target_pos_env[push_mask, 1] = book_pos_env[push_mask, 1]
            paused = self._debug_nominal_push_tracking_paused
            if torch.any(paused):
                target_pos_env[paused, 0] = ee_tool_pos_env[paused, 0]
                target_pos_env[paused, 2] = ee_tool_pos_env[paused, 2]

        if torch.any(scripted_mask):
            retreat_mask = scripted_mask & (self._script_step_buf >= int(self.cfg.script_open_steps))
            if torch.any(retreat_mask) and not scripted_fixed_retreat:
                target_pos_env[retreat_mask, 0] += float(self.cfg.script_retreat_dx)
                target_pos_env[retreat_mask, 2] += float(self.cfg.script_retreat_dz)

        if integrate_position_target:
            self._debug_integrated_target_pos_env[:] = target_pos_env

        if bool(getattr(self.cfg, "debug_position_only_target_ee", False)):
            joint_pos_des = self._compute_ik_joint_targets_from_tool_quat(
                target_pos_env, self._debug_position_only_ee_quat_b
            )
        elif bool(getattr(self.cfg, "debug_use_base_frame_quat_deltas", False)):
            target_quat = ee_quat_b.clone()
            if torch.any(normal_mask):
                target_quat[normal_mask] = self._apply_base_frame_orientation_delta(
                    ee_quat_b[normal_mask],
                    delta[normal_mask, 3],
                    delta[normal_mask, 4],
                )
            joint_pos_des = self._compute_ik_joint_targets_from_tool_quat(
                target_pos_env,
                target_quat,
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

        if len(self._gripper_command_joint_ids) > 0:
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
            finger_des = finger_cmd.unsqueeze(-1).expand(
                self.num_envs, len(self._gripper_command_joint_ids)
            )
            self.robot.set_joint_position_target(
                finger_des, joint_ids=self._gripper_command_joint_ids
            )
