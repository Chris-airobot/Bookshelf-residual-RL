#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Config for Bookshelf-Direct-v4 (hybrid insert + scripted release/retreat + push).

This version uses:
- one shared RL policy
- 5D action: [dx, dy, dz, dyaw, g]
- learned INSERT mode
- scripted RELEASE+RETREAT block
- learned PUSH mode

In INSERT, action[4] > release_trigger_threshold switches to scripted release (no geometry gating).
During SCRIPTED and PUSH, the gripper is forced open.
Terminal success uses geometry thresholds only in PUSH mode.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

# Shelf base position (m); add SHELF_OFFSET_X to move the whole shelf away from the robot in +X.
_SHELF_BASE_OPEN = 0.68
_SHELF_BASE_BACK = 0.81
# Choose offset so back panel (slot_x_back + wall_thickness/2) is ~0.81 m.
SHELF_OFFSET_X = -0.01

# Manipuland cuboid (L, H, T) in meters; neighbor kinematic books use the same dimensions.
_BOOK_LWH = (0.152, 0.229, 0.02)

# Pre-insertion arm pose (captured from sim / joint readout).
ROBOT_PRE_INSERTION_JOINT_POS = {
    "panda_joint1": 1.8027729988098145,
    "panda_joint2": 1.4919356107711792,
    "panda_joint3": -1.2221229076385498,
    "panda_joint4": -2.691943883895874,
    "panda_joint5": -2.6905927658081055,
    "panda_joint6": 2.214129686355591,
    "panda_joint7": -1.0659489631652832,
    "panda_finger_joint.*": 0.002,
}


@configclass
class BookshelfEnvV4SceneCfg(InteractiveSceneCfg):
    """Scene container only. Robot and book are spawned in ``BookshelfEnv._setup_scene``."""


@configclass
class BookshelfEnvCfg(DirectRLEnvCfg):
    """Bookshelf v4: one shared policy, hybrid modes (insert / scripted / push)."""

    decimation = 2
    episode_length_s = 10.0

    # Actions (5D): [dx, dy, dz, dyaw, g_trigger] in [-1, 1]
    # In INSERT, g_trigger requests release. In SCRIPTED and PUSH, gripper is forced open.
    action_space = 5

    # Obs (10):
    # [mode,
    #  rear_to_mouth, front_to_back, lat_err, z_err, yaw_err,
    #  tool_to_book_x, tool_to_book_y, tool_to_book_z,
    #  gripper_open01]
    observation_space = 10
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    scene: BookshelfEnvV4SceneCfg = BookshelfEnvV4SceneCfg(
        num_envs=4096,
        env_spacing=2.0,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos=ROBOT_PRE_INSERTION_JOINT_POS,
        ),
        actuators={
            "panda_shoulder": FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_shoulder"].replace(
                stiffness=6000.0,
                damping=100.0,
                effort_limit_sim=120.0,
            ),
            "panda_forearm": FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_forearm"].replace(
                stiffness=6000.0,
                damping=100.0,
                effort_limit_sim=80.0,
            ),
            "panda_hand": FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_hand"].replace(
                stiffness=1.2e5,
                damping=5.0e2,
                effort_limit_sim=500.0,
            ),
        },
    )

    book: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Book",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[-0.02, 0.0, 0.229 / 2.0],
            rot=(math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=_BOOK_LWH,
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
            physics_material=RigidBodyMaterialCfg(
                static_friction=1.8,
                dynamic_friction=1.5,
                friction_combine_mode="max",
            ),
            activate_contact_sensors=True,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.2, 0.8)),
        ),
    )

    book_size = _BOOK_LWH
    book_standing_quat = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)
    neighbor_book_size: tuple[float, float, float] = _BOOK_LWH

    # Shelf / slot geometry
    slot_x_open = _SHELF_BASE_OPEN + SHELF_OFFSET_X
    slot_x_back = _SHELF_BASE_BACK + SHELF_OFFSET_X
    slot_center_y = 0.0
    slot_lateral_clearance = 0.02

    enable_slot_clearance_curriculum = False
    slot_lateral_clearance_start = 0.02
    slot_lateral_clearance_end = 0.004
    slot_clearance_curriculum_steps = 20_000_000

    shelf_extra_books_per_side: int = 2
    neighbor_book_pitch_gap: float = 0.002
    wall_thickness = 0.02
    shelf_top_z = 0.05
    shelf_thickness = 0.02

    # --- Action scales ---
    dx_action_scale = 0.006
    dy_action_scale = 0.002
    dz_action_scale = 0.006
    dyaw_action_scale = math.radians(0.6)

    gripper_closed_joint_pos = 0.008
    gripper_open_joint_pos = 0.04
    # INSERT→SCRIPTED when action[4] > this value (strictly greater).
    release_trigger_threshold = 0.5

    ik_hold_action_epsilon = 1e-5
    reset_arm_joint_pos_noise = math.radians(2.0)
    ik_body_offset_pos = (0.0, 0.0, 0.107)

    # --- Observation scaling ---
    mode_obs_insert = 0.0
    mode_obs_scripted = 0.5
    mode_obs_push = 1.0

    rear_to_mouth_obs_scale = 0.08
    front_to_back_obs_scale = 0.08
    lat_err_obs_scale = 0.05
    z_err_obs_scale = 0.05
    yaw_err_obs_scale = math.radians(30.0)
    tool_to_book_pos_obs_scale = 0.25

    # --- Scripted release + retreat + gripper close ---
    script_open_steps = 3
    script_retreat_steps = 6
    script_close_steps = 5
    script_retreat_dx = -0.015
    script_retreat_dz = 0.000

    # --- Terminal success (checked only in PUSH mode) ---
    success_rear_to_mouth_min = -0.012
    success_rear_to_mouth_max = 0.002
    success_front_clear_min = 0.0
    success_front_clear_max = 0.055
    success_front_clear_eps_m = 2.0e-4
    success_z_thresh = 0.015
    success_yaw_thresh = math.radians(8.0)
    success_steps = 4
    success_lateral_margin = 0.000
    success_lateral_extent_eps_m = 5.0e-4
    success_max_lin_vel = 0.0
    success_max_ang_vel = 0.0
    min_steps_before_success = 5

    # --- Reward (simple) ---
    # INSERT mode
    insert_progress_scale = 5.0
    insert_lat_penalty_scale = 0.8
    insert_z_penalty_scale = 0.8
    insert_yaw_penalty_scale = 0.5

    # PUSH mode
    push_progress_scale = 5.0
    push_lat_penalty_scale = 0.8
    push_z_penalty_scale = 0.8
    push_yaw_penalty_scale = 0.5

    # Global
    step_penalty = -0.001
    success_bonus = 100.0
    drop_penalty = -20.0

    # Debug/safety terminations (optional).
    max_abs_xy = 0.95
    fell_height_thresh = 0.16
    upright_dot_thresh = 0.85
    enable_failure_terminations = False

    # Ground / shelf support
    book_floor_lowest_z_thresh = 0.042
    shelf_footprint_x_pad_m = 0.04
    shelf_footprint_y_pad_m = 0.05
    book_on_shelf_z_slack_m = 0.02

    # Book reset in grasp frame
    book_grasp_offset_hand = (0.0, 0.0, 0.075)

    book_grasp_orientation_in_hand = "franka_axes"
    book_to_hand_quat_franka_axes_wxyz = (0.5, -0.5, -0.5, -0.5)
    book_grasp_rel_quat_wxyz = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)

    book_grasp_x_jitter = 0.003
    book_grasp_y_jitter = 0.002
    book_grasp_yaw_jitter = math.radians(2.0)

    # Extra sim steps after snapping the book into the gripper on reset,
    # allowing contacts to settle so the grasp is stable at episode start.
    reset_warmup_steps = 10