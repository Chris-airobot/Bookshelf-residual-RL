#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Config for Bookshelf-Residual-Direct-v0.

Residual-RL bookshelf task: v5-style randomization/release mechanics with a
geometric nominal insertion controller added inside the environment.
"""

import math

from isaaclab.utils import configclass

from .bookshelf_env_cfg_v5 import BookshelfEnvCfg as BookshelfEnvCfgV5


@configclass
class BookshelfEnvCfg(BookshelfEnvCfgV5):
    """Residual-RL config with a fast curriculum toward tight 2 mm slots."""

    # Move the whole shelf/slot target farther from the robot for planner debugging.
    bookshelf_x_offset = 0
    slot_x_open = 0.63 + bookshelf_x_offset
    slot_x_back = 0.83 + bookshelf_x_offset

    # --- Debug geometry validation ---
    gripper_closed_joint_pos = 0.0165
    gripper_push_closed_joint_pos = 0.0
    debug_freeze_nominal_controller = False
    debug_disable_nominal_release = True
    debug_spawn_at_target_tool_pose = False
    debug_spawn_with_curobo = False
    debug_spawn_ik_iters = 80
    debug_spawn_inside_fraction = 0.0
    debug_freeze_tool_to_book_transform = True
    debug_omit_bookshelf_obstacles = False
    debug_start_from_default_grasp_pose = False
    debug_print_sampled_grasp_joints = False
    debug_start_joint_pos = {
        "panda_joint1": 0.9659621119499207,
        "panda_joint2": -1.4637905359268188,
        "panda_joint3": -1.1110988855361938,
        "panda_joint4": -2.9047939777374268,
        "panda_joint5": -2.670137405395508,
        "panda_joint6": 2.388262987136841,
        "panda_joint7": 2.357635259628296,
        "panda_finger_joint1": 0.03999944031238556,
        "panda_finger_joint2": 0.03999943286180496,
    }
    debug_hold_book_fixed_to_tool = False
    debug_use_full_target_ee_quat = False
    debug_position_only_target_ee = False
    debug_use_lula_rrt_planner = False
    debug_use_curobo_planner = False
    debug_print_residual_components = False
    debug_print_residual_interval = 30
    debug_print_residual_env_index = 0
    debug_curobo_motion_step_size = 0.01
    debug_rrt_max_iterations = 50000
    debug_rrt_interpolation_max_dist = 0.01
    debug_done_on_preinsert_reached = False
    debug_preinsert_hold_seconds = 1.0
    debug_preinsert_pos_tol = 0.010
    debug_preinsert_rot_tol = math.radians(10.0)
    show_target_book_marker = False
    show_target_ee_marker = False
    show_current_ee_marker = False
    target_ee_marker_axis_length = 0.20
    target_ee_marker_axis_thickness = 0.006

    slot_lateral_clearance_min = 0.0020
    slot_lateral_clearance_max = 0.0020

    # Fast curriculum for residual PPO.  The schedule is based on the global
    # environment step counter, so with N envs it advances every N transitions.
    enable_residual_clearance_curriculum = True
    # 4069 PPO iterations * 32 rollout steps. This counter is independent of num_envs.
    residual_curriculum_total_steps = 130_208
    residual_curriculum_1_frac = 0.10
    residual_curriculum_2_frac = 0.20
    residual_curriculum_3_frac = 0.30
    residual_curriculum_clearance_1 = (0.008, 0.008)
    residual_curriculum_clearance_2 = (0.006, 0.006)
    residual_curriculum_clearance_3 = (0.004, 0.006)
    residual_curriculum_clearance_final = (0.003, 0.003)

    enable_residual_reset_curriculum = True
    residual_curriculum_reset_1 = (math.radians(1.0), 0.002, 0.002, 0.002, math.radians(2.0))
    residual_curriculum_reset_2 = (math.radians(2.0), 0.004, 0.003, 0.002, math.radians(4.0))
    residual_curriculum_reset_3 = (math.radians(3.0), 0.008, 0.006, 0.003, math.radians(8.0))
    residual_curriculum_reset_final = residual_curriculum_reset_3

    enable_residual_action_scale_curriculum = True
    residual_curriculum_action_scale_1 = 0.30
    residual_curriculum_action_scale_2 = 0.50
    residual_curriculum_action_scale_3 = 0.75
    residual_curriculum_action_scale_final = 1.00
    enable_nominal_release_assist = True
    nominal_release_assist_until_frac = 0.30

    # PPO outputs residual corrections, not full motion commands.  Keep these
    # smaller than v5 full-action scales so the nominal controller remains the
    # leading insertion/push intent.
    dx_action_scale = 0.0020
    dy_action_scale = 0.0010
    dz_action_scale = 0.0015
    dyaw_action_scale = math.radians(0.35)
    dpitch_action_scale = math.radians(0.30)
    residual_action_l2_weight = 0.01

    # --- Residual RL nominal controller ---
    # The PPO action remains a bounded residual. These terms provide the base
    # Cartesian motion so delta_final = delta_nominal + delta_policy.
    enable_nominal_controller = True
    nominal_insert_dx = 0.0010
    nominal_insert_dx_near_mouth = 0.0007
    nominal_push_dx = 0.0008
    nominal_lateral_gain = 0.25
    nominal_height_gain = 0.18
    nominal_insert_z_offset = 0.006
    nominal_yaw_gain = 0.14
    nominal_pitch_gain = 0.020
    nominal_push_lateral_gain = 0.35
    nominal_push_height_gain = 0.30
    nominal_push_yaw_gain = 0.20
    nominal_push_pitch_gain = 0.08
    nominal_push_z_fraction_from_bottom = 0.20
    nominal_push_dy_limit = 0.0005
    nominal_push_dz_limit = 0.0010
    nominal_align_lat_thresh = 0.006
    nominal_align_z_thresh = 0.010
    nominal_align_yaw_thresh = math.radians(6.0)
    nominal_align_tilt_x_thresh = 0.10
    nominal_unaligned_dx_scale = 0.0
    nominal_dy_limit = 0.0015
    nominal_dz_limit = 0.0018
    nominal_dyaw_limit = math.radians(0.35)
    nominal_dpitch_limit = math.radians(0.25)
    nominal_slow_rear_to_mouth = -0.035
    nominal_release_inside_fraction = 0.50
    nominal_plan_position_gain = 0.35
    nominal_plan_yaw_gain = 0.50
    nominal_plan_pitch_gain = 0.50
    nominal_plan_orientation_first = True
    nominal_plan_yaw_thresh = math.radians(3.0)
    nominal_plan_pitch_thresh = math.radians(3.0)
    nominal_plan_pos_thresh = 0.006
    nominal_release_rear_to_mouth = -0.030
    nominal_release_front_to_back_min = 0.015
    nominal_release_lat_thresh = 0.010
    nominal_release_z_thresh = 0.018
    nominal_release_yaw_thresh = math.radians(8.0)
    nominal_release_tilt_x_thresh = 0.12
    final_dx_limit = 0.0080
    final_dy_limit = 0.0030
    final_dz_limit = 0.0070
    final_dyaw_limit = math.radians(0.8)
    final_dpitch_limit = math.radians(0.6)
