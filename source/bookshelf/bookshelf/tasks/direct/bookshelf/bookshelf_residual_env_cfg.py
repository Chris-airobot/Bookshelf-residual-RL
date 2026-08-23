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

from .bookshelf_env_cfg_v4 import BOOK_TRUE_GROUND_LOWEST_Z_THRESH
from .bookshelf_env_cfg_v5 import BookshelfEnvCfg as BookshelfEnvCfgV5


@configclass
class BookshelfEnvCfg(BookshelfEnvCfgV5):
    """Residual-RL config with fixed 3 mm clearance and reset-noise curriculum."""

    # Move the whole shelf/slot target farther from the robot for planner debugging.
    bookshelf_x_offset = 0
    slot_x_open = 0.63 + bookshelf_x_offset
    slot_x_back = 0.83 + bookshelf_x_offset

    # --- Debug geometry validation ---
    gripper_closed_joint_pos = 0.015
    gripper_push_closed_joint_pos = 0.0
    reset_arm_joint_pos_noise = math.radians(1.5)
    book_grasp_x_jitter = 0.003
    book_grasp_y_jitter = 0.003
    book_grasp_z_jitter = 0.0015
    book_grasp_yaw_jitter = math.radians(3.0)
    # Panda keeps the legacy finger-midpoint reset. Embodiments with a measured
    # calibration may instead place the book from their end-effector frame.
    book_grasp_pose_source = "finger_midpoint"
    eef_book_translation_xyz = (0.0, 0.0, 0.0)
    eef_book_quaternion_wxyz = (1.0, 0.0, 0.0, 0.0)
    debug_freeze_nominal_controller = False
    debug_disable_nominal_release = True
    debug_spawn_at_target_tool_pose = False
    debug_spawn_with_curobo = False
    debug_spawn_ik_iters = 80
    debug_spawn_inside_fraction = 0.0
    reset_to_slot_relative_tool_pose = False
    reset_tool_offset_slot_xyz = (0.0, 0.0, 0.0)
    reset_tool_quaternion_slot_wxyz = (1.0, 0.0, 0.0, 0.0)
    reset_tool_ik_iters = 120
    debug_freeze_tool_to_book_transform = True
    debug_omit_bookshelf_obstacles = False
    debug_omit_target_book = False
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
    debug_snap_book_to_grasp_on_reset = False
    debug_robot_target_pose_only = False
    debug_spawn_book_with_collision_clearance = False
    debug_spawn_book_panda_style = False
    debug_robot_forward_backward_demo = False
    debug_robot_forward_backward_distance_m = 0.05
    debug_robot_forward_backward_wait_steps = 60
    debug_robot_forward_backward_half_period_steps = 180
    debug_robot_nominal_controller_demo = False
    debug_robot_nominal_handoff_wait_steps = 60
    debug_row_layout_y_offset_m = 0.0
    debug_forced_missing_book_index_sequence = None
    debug_robot_target_gripper_ramp_steps = 1
    debug_robot_target_gripper_settle_steps = 0
    debug_book_palm_clearance_m = 0.005
    debug_book_min_finger_clearance_m = 0.001
    debug_finger_inner_surface_offset_m = 0.0
    debug_reachable_grasp_sequence = False
    debug_reachable_grasp_preclose_joint_pos = None
    debug_reachable_grasp_preclose_settle_steps = 30
    debug_reachable_grasp_settle_steps = 60
    debug_use_full_target_ee_quat = False
    debug_use_base_frame_quat_deltas = False
    debug_action5_as_base_x = False
    debug_position_only_target_ee = False
    debug_pose_ik_rotation_weight = None
    debug_integrate_position_target_ee = False
    debug_scripted_current_relative_target = False
    debug_scripted_fixed_retreat_path = False
    debug_scripted_fixed_retreat_total_dx = None
    debug_nominal_push_current_relative_target = False
    debug_nominal_push_reuse_insert_forward = False
    debug_nominal_push_lower_before_forward = False
    debug_nominal_push_align_to_book_center = False
    debug_nominal_push_hold_y_only = False
    debug_nominal_push_lock_y_to_entry = False
    debug_nominal_push_max_target_lead_m = None
    debug_nominal_push_max_vertical_target_lead_m = None
    debug_nominal_push_tracking_pause_enabled = False
    debug_nominal_push_tracking_pause_joint_error_rad = 0.12
    debug_nominal_push_tracking_resume_joint_error_rad = 0.08
    debug_nominal_push_spine_tracking_enabled = False
    debug_nominal_push_spine_pause_lateral_error_m = 0.004
    debug_nominal_push_spine_resume_lateral_error_m = 0.002
    debug_nominal_push_spine_recovery_step_m = 0.002
    debug_use_lula_rrt_planner = False
    debug_use_curobo_planner = False
    debug_print_residual_components = False
    debug_print_residual_interval = 30
    debug_print_residual_env_index = 0
    debug_curobo_motion_step_size = 0.01
    debug_rrt_max_iterations = 50000
    debug_rrt_interpolation_max_dist = 0.01
    debug_done_on_preinsert_reached = False
    # Inspection only: keep a failed scene intact so grasp diagnostics can be
    # read after the book falls. Normal training and evaluation still reset.
    debug_disable_episode_resets = False
    # Training-only reset filter. Diagnostic and evaluation entrypoints keep
    # this disabled so their reported pass rates include every sampled grasp.
    enable_constructive_grasp_reset = False
    enable_reset_acceptance_gate = False
    reset_acceptance_validation_steps = 12
    reset_acceptance_max_attempts = 50
    reset_acceptance_translation_limit_m = 0.003
    reset_acceptance_rotation_limit_rad = math.radians(3.0)
    reset_acceptance_arm_error_limit_rad = math.radians(8.0)
    reset_acceptance_ground_height_m = BOOK_TRUE_GROUND_LOWEST_Z_THRESH
    debug_preinsert_hold_seconds = 1.0
    debug_preinsert_pos_tol = 0.010
    debug_preinsert_rot_tol = math.radians(10.0)
    show_target_book_marker = False
    show_target_ee_marker = False
    show_current_ee_marker = False
    target_ee_marker_source = "planned_release"
    show_reachable_grasp_target_frame = False
    reachable_grasp_target_frame_source = "slot_relative"
    target_ee_marker_axis_length = 0.20
    target_ee_marker_axis_thickness = 0.006

    slot_lateral_clearance_min = 0.0030
    slot_lateral_clearance_max = 0.0030

    # Curriculum for residual PPO.  The schedule is based on the global
    # environment step counter, so it is independent of the number of envs.
    enable_residual_clearance_curriculum = False
    # 8138 PPO iterations * 32 rollout steps.
    residual_curriculum_total_steps = 260_416
    residual_curriculum_1_frac = 0.20
    residual_curriculum_2_frac = 0.50
    residual_curriculum_3_frac = 1.00
    residual_curriculum_clearance_1 = (0.08, 0.010)
    residual_curriculum_clearance_2 = (0.006, 0.006)
    residual_curriculum_clearance_3 = (0.004, 0.006)
    residual_curriculum_clearance_final = (0.003, 0.003)

    enable_residual_reset_curriculum = True
    # tuple: arm joint noise, grasp x/y/z jitter, grasp yaw jitter
    residual_curriculum_reset_1 = (math.radians(1.5), 0.0030, 0.0030, 0.0015, math.radians(3.0))
    residual_curriculum_reset_2 = (math.radians(2.0), 0.0050, 0.0040, 0.0020, math.radians(5.0))
    residual_curriculum_reset_3 = (math.radians(3.0), 0.0080, 0.0060, 0.0030, math.radians(8.0))
    residual_curriculum_reset_final = residual_curriculum_reset_3

    enable_residual_action_scale_curriculum = False
    residual_curriculum_action_scale_1 = 0.30
    residual_curriculum_action_scale_2 = 0.50
    residual_curriculum_action_scale_3 = 0.75
    residual_curriculum_action_scale_final = 1.00
    enable_nominal_release_assist = False
    nominal_release_assist_until_frac = 0.30
    # Keep "none" as the baseline behavior.  The observable-geometry guard
    # accepts a policy release only after the measured book pose satisfies the
    # same insertion/alignment test used by the nominal release helper.
    policy_release_guard_mode = "none"
    premature_release_penalty = 0.5

    # PPO outputs residual corrections, not full motion commands.  Keep these
    # smaller than v5 full-action scales so the nominal controller remains the
    # leading insertion/push intent.
    dx_action_scale = 0.0020
    dy_action_scale = 0.0010
    dz_action_scale = 0.0015
    dyaw_action_scale = math.radians(0.35)
    dpitch_action_scale = math.radians(0.30)
    enable_base_y_rotation_action = False
    dbase_y_rotation_action_scale = math.radians(0.30)
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
    final_dbase_y_rotation_limit = math.radians(0.6)
