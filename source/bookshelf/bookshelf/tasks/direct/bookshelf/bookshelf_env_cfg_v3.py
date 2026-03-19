#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Config for Bookshelf-Direct-v3: planner-handoff residual pose control with impedance execution."""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.utils import configclass


@configclass
class BookshelfEnvV3SceneCfg(InteractiveSceneCfg):
    """Book must appear before contact sensor (scene entity order)."""

    book: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Book",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[-0.02, 0.0, 0.229 / 2.0],
            rot=(math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.152, 0.229, 0.02),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=2,
                max_angular_velocity=100.0,
                max_linear_velocity=10.0,
                max_depenetration_velocity=2.0,
                disable_gravity=False,
                linear_damping=0.2,
                angular_damping=4.0,
            ),
            mass_props=MassPropertiesCfg(mass=0.45),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            activate_contact_sensors=True,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.2, 0.8)),
        ),
    )
    book_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Book",
        max_contact_data_count_per_prim=24,
        update_period=0.0,
        track_air_time=False,
    )


@configclass
class BookshelfEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 10.0
    # Actions: residual shelf-frame [dx, dy, dyaw] in [-1, 1] (delta target each step).
    action_space = 3
    # forward_err, lateral_err, yaw_err, vx, vy, yaw_rate, fx, fy, mz, prev_ax, prev_ay, prev_az
    observation_space = 12
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    scene: BookshelfEnvV3SceneCfg = BookshelfEnvV3SceneCfg(
        num_envs=4096,
        env_spacing=2.0,
        replicate_physics=True,
    )

    # Book geometry (same standing convention as v0/v1/v2)
    book_size = (0.152, 0.229, 0.02)
    book_standing_quat = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)

    # Slot frame: +X insertion, +Y lateral. Mouth at slot_x_open, back at slot_x_back.
    slot_x_open = 0.11
    slot_x_back = 0.24
    slot_center_y = 0.0
    slot_lateral_clearance = 0.007
    wall_thickness = 0.02

    # Success: insertion, centering, yaw alignment, upright; held N steps.
    success_min_insertion = 0.050
    success_lateral_thresh = 0.010
    success_yaw_thresh = math.radians(4.0)
    success_steps = 10

    # Residual action scaling (delta target update per policy step).
    dx_action_scale = 0.006
    dy_action_scale = 0.004
    dyaw_action_scale = math.radians(2.0)

    # Impedance-to-target gains (applied as external wrench).
    k_pos_x = 260.0
    k_pos_y = 260.0
    d_vel_x = 28.0
    d_vel_y = 28.0
    k_yaw = 28.0
    d_yaw_rate = 6.0

    # Clamp controller wrench (keeps training stable and comparable to v2).
    max_force = 18.0
    max_yaw_torque = 2.4

    obs_clip = 10.0
    max_abs_xy = 0.55
    max_contact_force_norm = 140.0
    upright_dot_thresh = 0.85

    # Jam: strong lateral force and/or yaw torque, little forward progress.
    jam_lateral_force_thresh = 25.0
    jam_yaw_torque_thresh = 2.0
    jam_consecutive_steps = 32

    # Reward scales (same structure as v2).
    progress_scale = 2.5
    center_scale = 1.2
    yaw_scale = 0.9
    # Keep reward compact; avoid dense "be aligned" bonus that can reward orbiting.
    # Sharpen the progress gate so "push" is valuable mainly when aligned.
    progress_lateral_sigma = 0.005
    progress_yaw_sigma = math.radians(4.0)
    misaligned_push_lateral_thresh = 0.008
    misaligned_push_yaw_thresh = math.radians(7.0)
    misaligned_push_scale = 3.0
    action_delta_penalty_scale = 0.0004
    contact_penalty_scale = 0.006
    # In v3, penalize backoff only when well-aligned (gate multiplies this term).
    backward_penalty_scale = 0.8
    # Strong dense penalty for unrecoverable side-slide (outside channel while shallow).
    side_bad_lateral_thresh = 0.03
    side_bad_insertion_thresh = 0.5 * success_min_insertion
    side_bad_penalty = -6.0
    success_bonus = 90.0
    timeout_penalty = -2.0
    jam_penalty = -10.0
    oob_penalty = -12.0
    tipped_penalty = -12.0
    fell_penalty = -12.0
    explode_penalty = -18.0

    contact_obs_force_scale = 40.0
    contact_obs_torque_scale = 3.0

    # Reset: leading face stays left of geometric mouth (side walls’ −X face), not slot_x_open.
    initial_slot_mouth_clearance_x = 0.012
    initial_forward_back_span = 0.09
    initial_lateral_offset_range = (-0.008, 0.008)
    initial_yaw_range = (math.radians(-8.0), math.radians(8.0))

    # Target initialization and bounds (planner-handoff local region).
    target_pos_clip = (0.30, 0.06)  # (max_abs_x_from_open, max_abs_y_from_center)
    target_yaw_clip = math.radians(35.0)
