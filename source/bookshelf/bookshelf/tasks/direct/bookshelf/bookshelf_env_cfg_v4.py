#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Config for Bookshelf-Direct-v4: robot-held insertion, action [dx,dy,dz,dyaw,dgrip], v3 reward backbone + robot terms.

Differential IK uses absolute pose mode (``use_relative_mode=False``), matching the reach task
``ik_abs_env_cfg.FrankaReachEnvCfg`` pattern: integrated target position + yaw, with roll/pitch frozen at reset.
Policy yaw and observation ``eyaw`` use the same base-frame XYZ-Euler yaw as the IK orientation command.
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
SHELF_OFFSET_X = 0.09

# Pre-insertion arm pose (captured from sim / joint readout). Fingers at 0.005 for closed grasp on the book (not 0.04 cube-open).
ROBOT_PRE_INSERTION_JOINT_POS = {
    "panda_joint1": 0.007454383186995983,
    "panda_joint2": -0.29195407032966614,
    "panda_joint3": 0.20468570291996002,
    "panda_joint4": -2.4609861373901367,
    "panda_joint5": -2.8037502765655518,
    "panda_joint6": 2.483391523361206,
    "panda_joint7": -2.576188802719116,
    "panda_finger_joint.*": 0.005,
}


@configclass
class BookshelfEnvV4SceneCfg(InteractiveSceneCfg):
    """Scene container only. Robot, book, and sensors are spawned in ``BookshelfEnv._setup_scene`` (Factory-style).

    Putting assets in :class:`InteractiveSceneCfg` caused ``InteractiveScene`` to clone/replicate before
    ``_setup_scene`` added the shelf, and a second ``clone_environments`` then broke per-env robot poses.
    """


@configclass
class BookshelfEnvCfg(DirectRLEnvCfg):
    """v4: robot-held, tight feasible slot. Obs: geometry + (ex,ey,ez,eyaw) + motion + contact + prev_action (5)."""

    decimation = 2
    episode_length_s = 10.0
    action_space = 5
    # Geometry(3) + controller state(4: ex,ey,ez,eyaw) + motion(3) + contact(3) + prev_action(5) = 18
    observation_space = 18
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
    # Shelf height: top of shelf at 0.25 m; books (standing) center at shelf_top_z + shelf_thick + book_height/2.
    shelf_top_z = 0.25

    success_min_insertion = 0.040
    success_lateral_thresh = 0.015
    success_yaw_thresh = math.radians(12.0)
    success_steps = 3

    # action: [dx, dy, dz, dyaw, dgrip] in [-1, 1]; each step applies residual deltas on CURRENT EE/tool pose.
    # IK command remains absolute pose in base frame (reach ik_abs style), computed from current pose + delta.
    dx_action_scale = 0.06  # ±6 mm
    dy_action_scale = 0.04  # ±4 mm
    dz_action_scale = 0.05  # ±5 mm vertical (book held off shelf / alignment)
    dyaw_action_scale = math.radians(2.0)  # ±2°
    dgrip_action_scale = 0.002  # per-step finger joint delta (m), clamped to finger soft limits

    # Isaac Lab reach (ik_abs) uses a fictitious end-effector frame offset from `panda_hand`.
    # We mirror that here so "dx forward" moves the tool tip, not the raw hand body.
    ik_body_offset_pos = (0.0, 0.0, 0.107)

    # k_pos_x = 260.0
    # k_pos_y = 260.0
    # d_vel_x = 28.0
    # d_vel_y = 28.0
    # k_yaw = 28.0
    # d_yaw_rate = 6.0
    # max_force = 18.0
    # max_yaw_torque = 2.4

    obs_clip = 10.0
    # Workspace: cover shifted slot_x_back (~0.9 m) + margin.
    max_abs_xy = 0.95
    max_contact_force_norm = 1400.0 # if the contact force is greater than 1400, it is exploding
    # Consider the book "fallen" if COM height drops below this threshold.
    # This catches both flat-on-ground and standing-on-ground cases.
    fell_height_thresh = 0.16
    upright_dot_thresh = 0.85
    # Grasp-relative pose: local +Y (spine) is often horizontal in world; only enforce shelf-style upright
    # at/inside the slot mouth (ins = x - slot_x_open >= threshold).
    tipped_check_min_ins = 0.0

    # jam detection
    jam_lateral_force_thresh = 25.0
    jam_yaw_torque_thresh = 2.0
    jam_consecutive_steps = 32

    # Reward: v3 backbone
    progress_scale = 4.0
    center_scale = 2.0
    yaw_scale = 1.5
    # Keep some progress incentive even when alignment is poor.
    progress_gate_floor = 0.3
    # Dense term on absolute insertion depth (clipped) to discourage static local optima.
    insertion_pos_reward_scale = 0.8
    progress_lateral_sigma = 0.005
    progress_yaw_sigma = math.radians(4.0)
    misaligned_push_lateral_thresh = 0.008
    misaligned_push_yaw_thresh = math.radians(7.0)
    misaligned_push_scale = 3.0
    action_delta_penalty_scale = 0.0003
    # Gripper action smoothness (5th action dim): penalize abrupt open/close commands.
    grip_action_delta_penalty_scale = 0.001
    # Per-step living cost to discourage waiting/stalling.
    step_penalty = -0.001
    contact_penalty_scale = 0.006
    backward_penalty_scale = 0.4
    side_bad_lateral_thresh = 0.03
    side_bad_insertion_thresh = 0.025  # 0.5 * success_min_insertion
    side_bad_penalty = -2.0
    success_bonus = 150.0
    timeout_penalty = -2.0
    jam_penalty = -4.0
    oob_penalty = -12.0
    tipped_penalty = -12.0
    fell_penalty = -25.0
    explode_penalty = -18.0

    # v4: backoff allowance — when misaligned, reduce backward penalty so policy can retreat then reinsert.
    backoff_allowance_lateral_thresh = 0.012
    backoff_allowance_yaw_thresh = math.radians(10.0)
    backoff_backward_penalty_scale = 0.2

    # Gripper release shaping (used only now that action includes dgrip).
    release_open_action_deadband = 0.2
    release_align_lateral_thresh = 0.01
    release_align_yaw_thresh = math.radians(4.0)
    release_min_insertion = 0.045
    release_bonus_scale = 0.1
    premature_open_penalty_scale = 1.0
    # Penalize dropping while gripper is substantially open and insertion is still shallow.
    drop_height_thresh = 0.08
    gripper_open_norm_thresh = 0.7
    drop_after_open_penalty_scale = 20.0

    # v4: gripper/tool collision (no-op until robot/gripper bodies and sensor exist).
    # gripper_collision_penalty_scale = -3.0

    contact_obs_force_scale = 40.0
    contact_obs_torque_scale = 3.0
    # Scale for ex, ey, eyaw in obs so they sit in a similar range as other obs.
    tracking_error_obs_scale = 0.05

    initial_slot_mouth_clearance_x = 0.012
    # initial_forward_back_span = 0.09
    # initial_lateral_offset_range = (-0.008, 0.008)
    # initial_yaw_range = (math.radians(-8.0), math.radians(8.0))

    target_pos_clip = (0.30, 0.06)
    # EE target z in env frame (m): keep above ground / below ceiling for lifted grasp.
    target_pos_env_z_min = 0.20
    target_pos_env_z_max = 0.55
    target_yaw_clip = math.radians(35.0)
    # If False, do not clip integrated Cartesian/yaw targets in _apply_action() / reset initialization.
    enable_target_clamp = False

    # Differential IK: if |action| < this on all dims, hold current arm joint pos (reduces idle wobble).
    ik_hold_action_epsilon = 1e-5

    # After we place the book during reset, allow a short PhysX settling window so the closed fingers
    # can establish stable contact before the first user action (e.g., scripts/manual_step.py).
    reset_book_contact_settle_steps = 5

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
    book_grasp_x_jitter = 0.0
    book_grasp_y_jitter = 0.0
    # World +Z yaw after q_book_in_hand (non-zero breaks strict franka_axes grasp lock).
    book_grasp_yaw_jitter = 0.0
    debug_print_grasp_frame = False
    # If True, print per-env breakdown when terminated=True (finds explode vs fell vs tipped vs oob vs success).
    debug_print_terminate_reason = False
    # If False, terminate only on success (plus timeout), not on explode/oob/fell/tipped/jammed.
    enable_failure_terminations = False
