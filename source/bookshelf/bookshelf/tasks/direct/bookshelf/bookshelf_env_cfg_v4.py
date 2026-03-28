#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Config for Bookshelf-Direct-v4 (insert-only).

This cfg keeps only what is needed to spawn:
- the robot articulation
- the book rigid object
- the bookshelf geometry (slot parameters, shelf dimensions)

This version defines an **insertion-only** RL task:
- Robot keeps grasping the book (no release, no push-after-release)
- Success is geometry-only at the slot mouth (dwell a few steps)
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
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

# Pre-insertion arm pose (captured from sim / joint readout). Fingers at 0.005 for closed grasp on the book (not 0.04 cube-open).
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
    """Scene container only. Robot, book, and sensors are spawned in ``BookshelfEnv._setup_scene`` (Factory-style).

    Putting assets in :class:`InteractiveSceneCfg` caused ``InteractiveScene`` to clone/replicate before
    ``_setup_scene`` added the shelf, and a second ``clone_environments`` then broke per-env robot poses.
    """


@configclass
class BookshelfEnvCfg(DirectRLEnvCfg):
    """Insertion-only env (4D Cartesian + yaw actions, geometry success)."""

    decimation = 2
    episode_length_s = 10.0
    # Actions: [dx, dy, dz, dyaw] in [-1, 1]
    action_space = 4
    # Obs (15):
    # [front_to_mouth, lat_err, yaw_err, z_err,
    #  ex, ey, ez, eyaw,
    #  vx, vy, wz,
    #  prev_dx, prev_dy, prev_dz, prev_dyaw]
    observation_space = 15
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # Minimal scene (no assets): only env_0 exists until _setup_scene spawns robot/book/shelf and calls
    # clone_environments once — same pattern as Isaac Lab Factory direct env.
    scene: BookshelfEnvV4SceneCfg = BookshelfEnvV4SceneCfg(
        num_envs=4096,
        env_spacing=2.0,
        replicate_physics=True,
        # Fabric cloning can leave PhysX rigid-body instance counts inconsistent with ContactSensor's
        # (count // num_envs == num_bodies) check — RuntimeError in contact_sensor._initialize_impl.
        clone_in_fabric=False,
    )

    # Explicit env_.* paths (Isaac Lab Factory style); spawned in _setup_scene, not via InteractiveSceneCfg.
    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos=ROBOT_PRE_INSERTION_JOINT_POS,
        ),
        actuators={
            "panda_shoulder": FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_shoulder"].replace(
                stiffness=600.0,
                damping=100.0,
                effort_limit_sim=120.0,
            ),
            "panda_forearm": FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_forearm"].replace(
                stiffness=600.0,
                damping=100.0,
                effort_limit_sim=80.0,
            ),
            "panda_hand": FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_hand"].replace(
                stiffness=1.2e4,
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
            physics_material=RigidBodyMaterialCfg(
                static_friction=1.8,
                dynamic_friction=1.5,
                friction_combine_mode="max",
            ),
            activate_contact_sensors=True,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.2, 0.8)),
        ),
    )
    book_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Book",
        max_contact_data_count_per_prim=24,
        update_period=0.0,
        track_air_time=False,
    )

    book_size = (0.152, 0.229, 0.02)
    book_standing_quat = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)

    # Slot mouth / back; SHELF_OFFSET_X shifts the whole shelf in +X (see module constants above).
    slot_x_open = _SHELF_BASE_OPEN + SHELF_OFFSET_X
    slot_x_back = _SHELF_BASE_BACK + SHELF_OFFSET_X
    slot_center_y = 0.0
    # Slightly narrower than debug-wide setting, while still forgiving for early training.
    slot_lateral_clearance = 0.02
    # Extra kinematic books along ±Y outside the slot-defining pair (visual + tighter shelf layout).
    shelf_extra_books_per_side: int = 2
    # Center-to-center step along Y between adjacent standing books (spine thickness + small gap).
    neighbor_book_pitch_gap: float = 0.002
    wall_thickness = 0.02
    # Shelf height: slightly lower than z=0.2-COM alignment to avoid scraping the shelf during straight inserts.
    shelf_top_z = 0.05
    shelf_thickness = 0.02
    

    # --- insertion-only task parameters ---

    # Action scales (meters / radians per step).
    # NOTE: these are meters; keep them in the mm range for this tight insertion task.
    dx_action_scale = 0.06
    dy_action_scale = 0.04
    dz_action_scale = 0.03
    dyaw_action_scale = math.radians(1.2)

    # If |action| < this on all dims, hold last IK arm setpoint (avoids sag from tracking measured q under load).
    ik_hold_action_epsilon = 1e-5

    # Reset randomization (arm joints). Larger -> more visible book pose variety.
    reset_arm_joint_pos_noise = math.radians(2.0)

    # Isaac Lab reach (ik_abs) uses a fictitious end-effector frame offset from `panda_hand`.
    ik_body_offset_pos = (0.0, 0.0, 0.107)

    # Scale for tracking errors (ex, ey, ez, eyaw) in observations.
    tracking_error_obs_scale = 0.05

    # Light normalization/clipping for raw geometry + velocity obs.
    front_to_mouth_obs_scale = 0.08
    lat_err_obs_scale = 0.05
    z_err_obs_scale = 0.05
    yaw_err_obs_scale = math.radians(30.0)
    lin_vel_obs_scale = 0.5
    ang_vel_obs_scale = 2.0

    # Success thresholds (geometry-only at slot mouth).
    # Insertion-only "release-ready partial insertion corridor" (measured against true `slot_x_open/slot_x_back`):
    # - enter: front face is this far past the mouth plane (m)
    # - back: front face is still this far away from the back panel (m); set <=0 to disable upper bound
    success_enter_margin = 0.08
    success_back_margin = 0.03
    success_z_thresh = 0.015
    success_yaw_thresh = math.radians(8.0)
    success_steps = 4
    # Require a small clearance margin relative to the nominal slot half-width.
    success_lateral_margin = 0.000


    # Optional stability gate for success (set to 0 to disable).
    success_max_lin_vel = 0.0
    success_max_ang_vel = 0.0

    # Prevent immediate success on reset (steps).
    min_steps_before_success = 5

    # Debug printing for success gates.
    debug_print_success = False
    debug_print_success_every = 1

    # Reward shaping (simple, insertion-only).
    progress_scale = 7.0
    # Linear + quadratic depth shaping in [0, success_enter_margin] to reward committing past the mouth.
    depth_reward_scale = 2.5
    depth_corridor_progress_scale = 1.2
    center_progress_scale = 2.0
    yaw_progress_scale = 1.0
    lateral_penalty_scale = 0.9
    yaw_penalty_scale = 0.5
    z_penalty_scale = 0.7
    # Softer base scale; first centimeters past mouth use clearance_ramp (see env).
    clearance_violation_scale = 1.5
    # Meters past mouth plane over which clearance penalty ramps from 50% to 100%.
    clearance_ramp_depth_m = 0.03
    aligned_forward_bonus_scale = 2.0
    # Loose alignment gate for adding forward bonus (not success thresholds).
    aligned_bonus_lat_thresh = 0.01
    aligned_bonus_yaw_thresh = math.radians(10.0)
    # Dense bonus when the book has effectively "reached" the slot mouth region (aligned + close in X).
    # Keep scale well above typical per-step penalties so hovering in this funnel beats oscillating away.
    slot_reach_bonus_scale = 6.0
    slot_reach_lat_thresh = 0.012
    slot_reach_yaw_thresh = math.radians(12.0)
    slot_reach_z_thresh = 0.018
    # front_to_mouth must be in (-slot_reach_mouth_window_m, success_enter_margin) (approach + pre-success insert).
    slot_reach_mouth_window_m = 0.10
    # Archive-inspired anti-dither shaping.
    # Closer to 1.0 => less "comfort" outside the mouth (reduces pre-insert local optimum).
    pre_mouth_penalty_scale = 0.85
    mouth_dither_window = 0.015
    dx_signflip_penalty_scale = 0.4
    dx_signflip_active_thresh = 0.2
    action_delta_penalty_scale = 0.0015
    step_penalty = -0.001
    success_bonus = 50.0

    # Debug/safety terminations (optional).
    max_abs_xy = 0.95
    fell_height_thresh = 0.16
    upright_dot_thresh = 0.85
    enable_failure_terminations = False

    # v4 reset: book COM from grasp (finger midpoint + panda_hand axes). Orientation:
    # - "standing_world": book_standing_quat + world yaw jitter (neighbors / shelf upright in world).
    # - "grasp_relative": q_world = q_panda_hand * q_book_in_hand * yaw_jitter; see book_grasp_orientation_in_hand.
    book_reset_orient_mode = "grasp_relative"

    # Book COM offset in grasp frame (origin = finger midpoint, +X/+Y/+Z = panda_hand body axes).
    # +Z_hand = approach (between fingers). Increase Z to pull the COM out of the palm when using franka_axes orientation.
    # Grasp origin is the finger midpoint (see BookshelfEnv._grasp_frame_pose_w).
    # With franka_axes grasp mapping, this means:
    # - hand +Y ~= book thickness axis (half thickness ~= 0.01) => keep Y offset small
    # - hand +Z ~= book long axis (half length ~= 0.076) => keep Z offset within the long-axis span
    book_grasp_offset_hand = (0.0, 0.0, 0.075)

    # grasp_relative: how q_book_in_hand is chosen.
    # "franka_axes": +X_book || +Z_hand (approach), +Y_book || +X_hand, +Z_book || +Y_hand (close).
    # "manual_quat": use book_grasp_rel_quat_wxyz only.
    book_grasp_orientation_in_hand = "franka_axes"
    # Body B to panda_hand H (wxyz); columns of R_BH are (+Z_h,+X_h,+Y_h) as images of book (+X,+Y,+Z)_b.
    book_to_hand_quat_franka_axes_wxyz = (0.5, -0.5, -0.5, -0.5)
    # Used when book_grasp_orientation_in_hand == "manual_quat".
    book_grasp_rel_quat_wxyz = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)
    # In-hand COM jitter at reset (meters), applied in grasp frame before snapping the book.
    book_grasp_x_jitter = 0.003
    book_grasp_y_jitter = 0.002
    # World +Z yaw after q_book_in_hand (non-zero breaks strict franka_axes grasp lock).
    book_grasp_yaw_jitter = math.radians(2.0)
    debug_print_grasp_frame = False
