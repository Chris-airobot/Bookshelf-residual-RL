# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Config for Bookshelf-Direct-v1: contact-rich slot insertion (no yaw)."""

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
class BookshelfEnvV1SceneCfg(InteractiveSceneCfg):
    """Book must appear before contact sensor (scene entity order)."""

    book: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Book",
        # Default: left of geometric slot mouth (~0.08); reset uses same rule.
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
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.8)),
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
    episode_length_s = 8.0
    action_space = 2
    # forward_err, lateral_err, vx, vy, fx, fy (normalized)
    observation_space = 6
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    scene: BookshelfEnvV1SceneCfg = BookshelfEnvV1SceneCfg(
        num_envs=4096,
        env_spacing=2.0,
        replicate_physics=True,
    )

    # Book geometry (same standing convention as v0)
    book_size = (0.152, 0.229, 0.02)
    book_standing_quat = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)

    # Slot frame: +X insertion, +Y lateral. Mouth at slot_x_open, back at slot_x_back.
    slot_x_open = 0.11
    slot_x_back = 0.24
    slot_center_y = 0.0
    # Extra lateral gap (total) added to book thickness for slot width (curriculum knob).
    slot_lateral_clearance = 0.008
    wall_thickness = 0.02

    # Success: COM at least this far past slot mouth, lateral aligned, held N steps.
    success_min_insertion = 0.045
    success_lateral_thresh = 0.012
    success_steps = 8

    # Planar push force from actions (N); scaled by action in [-1,1].
    force_scale = 14.0

    obs_clip = 10.0
    max_abs_xy = 0.55
    max_contact_force_norm = 120.0
    upright_dot_thresh = 0.85

    # Jam: strong lateral contact, little forward progress.
    jam_lateral_force_thresh = 25.0
    jam_consecutive_steps = 28

    # Reward scales
    progress_scale = 2.5
    center_scale = 1.0
    # Gate forward progress by lateral alignment: exp(-(lat_err/sigma)^2)
    progress_lateral_sigma = 0.007
    # Penalize pushing forward while misaligned: -scale * max(0, d_ins) if lat_err > thresh
    misaligned_push_thresh = 0.008
    misaligned_push_scale = 2.0
    action_penalty_scale = 0.0008
    contact_penalty_scale = 0.015
    backward_penalty_scale = 0.8
    success_bonus = 35.0
    timeout_penalty = -2.0
    jam_penalty = -8.0

    contact_obs_force_scale = 40.0

    # Reset: leading face stays left of *geometric* mouth (side walls’ −X face), not slot_x_open.
    initial_slot_mouth_clearance_x = 0.012
    # Sample COM x in [x_max - span, x_max] (further -X = more standoff).
    initial_forward_back_span = 0.07
    # Lateral jitter ~slot channel; wide range hits side walls.
    initial_lateral_offset_range = (-0.006, 0.006)
