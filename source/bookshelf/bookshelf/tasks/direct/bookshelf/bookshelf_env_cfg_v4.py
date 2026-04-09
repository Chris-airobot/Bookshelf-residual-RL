#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Config for Bookshelf-Direct-v4 (phase-aware placement).

Spawns the robot articulation, book rigid object, and bookshelf geometry (slot parameters, shelf dimensions).

Single-policy episode with internal phases: **held insert → release → retreat → push**, then terminal
success only when the book is fully placed and stable (same geometry gates as before).

- Actions: Cartesian residual + pitch + yaw + gripper (gripper forced closed in insert/retreat/push as configured).
- Termination: final placement success, book dropped to floor, optional OOB/fall, or timeout.
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
    """Bookshelf v4: 6D actions; phase FSM + full-placement terminal success."""

    decimation = 2
    episode_length_s = 10.0
    # Actions (6D): [dx, dy, dz, dpitch, dyaw, g_gripper] in [-1, 1]
    # g=-1/low -> closed, g=+1/high -> open (threshold applies in RELEASE phase only).
    action_space = 6
    # Obs (16): phase_norm,
    #   front_to_mouth, lat_err, yaw_err, z_err, front_to_back, rear_to_mouth,
    #   gripper_open, supported_on_shelf, release_ready, hand_clearance_scaled,
    #   tool_to_book_xyz (3), tool yaw/pitch minus book (2)
    observation_space = 16
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
    # Slot-defining neighbor cuboids: same dimensions as the manipuland (see ``_BOOK_LWH``).
    neighbor_book_size: tuple[float, float, float] = _BOOK_LWH

    # Shelf depth anchors (env X): used with ``mid_x = 0.5*(open+back)`` for shelf, neighbors, and back panel.
    # The *metric* insertion mouth plane is **not** ``slot_x_open``; it is derived in the env from neighbor
    # cuboid geometry (``neighbor_book_size`` / ``book_standing_quat``).
    slot_x_open = _SHELF_BASE_OPEN + SHELF_OFFSET_X
    slot_x_back = _SHELF_BASE_BACK + SHELF_OFFSET_X
    slot_center_y = 0.0
    # Stage-1 (wide-slot) training: set the physical slot clearance here.
    slot_lateral_clearance = 0.02
    # Optional slot-clearance curriculum (linear schedule over global env steps).
    # NOTE: shelf geometry is spawned once at startup using the curriculum start value;
    # runtime success/support checks anneal from start -> end.
    enable_slot_clearance_curriculum = False
    slot_lateral_clearance_start = 0.02
    slot_lateral_clearance_end = 0.004
    slot_clearance_curriculum_steps = 20_000_000
    # Extra kinematic books along ±Y outside the slot-defining pair (visual + tighter shelf layout).
    shelf_extra_books_per_side: int = 2
    # Center-to-center step along Y between adjacent standing books (spine thickness + small gap).
    neighbor_book_pitch_gap: float = 0.002
    wall_thickness = 0.02
    # Shelf height: slightly lower than z=0.2-COM alignment to avoid scraping the shelf during straight inserts.
    shelf_top_z = 0.05
    shelf_thickness = 0.02
    

    # --- Action scales (meters / radians per step) ---

    dx_action_scale = 0.006
    # Extra forward authority in INSERT only (multiplies dx residual; dy/dz unchanged).
    phase_insert_dx_multiplier = 1.75
    dy_action_scale = 0.002
    dz_action_scale = 0.0015
    dyaw_action_scale = math.radians(0.6)
    dpitch_action_scale = dyaw_action_scale
    gripper_closed_joint_pos = 0.002
    gripper_open_joint_pos = 0.04
    # Command threshold for opening in RELEASE phase only (g >= threshold → open target).
    gripper_open_action_threshold = 0.95

    # Per-phase arm authority multipliers (applied to dx..dyaw residuals before IK).
    phase_insert_arm_scale = 1.0
    phase_release_arm_scale = 0.35
    phase_retreat_arm_scale = 1.2
    phase_push_arm_scale = 1.0

    ik_hold_action_epsilon = 1e-5
    # If True, never freeze arm on near-zero actions during INSERT (reduces mouth-hover stickiness).
    ik_hold_disable_in_insert = True
    reset_arm_joint_pos_noise = math.radians(2.0)
    ik_body_offset_pos = (0.0, 0.0, 0.107)

    # --- Observation scaling / clipping ---

    front_to_mouth_obs_scale = 0.08
    rear_to_mouth_obs_scale = 0.08
    front_to_back_obs_scale = 0.08
    lat_err_obs_scale = 0.05
    z_err_obs_scale = 0.05
    yaw_err_obs_scale = math.radians(30.0)
    hand_clearance_obs_scale = 0.25
    tool_to_book_pos_obs_scale = 0.25
    tool_yaw_minus_book_yaw_obs_scale = math.radians(30.0)
    tool_pitch_minus_book_pitch_obs_scale = math.radians(30.0)

    # --- Phase FSM: transition gates ---

    # INSERT → RELEASE: release_ready must hold this many consecutive env steps.
    phase_insert_min_release_ready_hold_steps = 3
    # release_ready geometry (looser than terminal success).
    # Convention: front_x = max book corner X (leading edge into +X / into slot); mouth = slot mouth plane.
    # Then front_to_mouth = front_x - mouth is > 0 once that leading edge is past the mouth (inside the slot).
    phase_release_ready_requires_past_mouth = True
    # Require front_to_mouth > this (use a small negative value, e.g. -1e-4, if you need numerical slack).
    phase_release_ready_min_front_to_mouth = 0.0
    phase_release_ready_rear_to_mouth_min = -0.033
    phase_release_ready_max_abs_lat_err = 0.012
    phase_release_ready_max_abs_yaw_err = math.radians(12.0)
    phase_release_ready_max_abs_z_err = 0.022
    phase_release_ready_requires_supported = True

    # RELEASE → RETREAT: gripper open fraction and low book motion, held consecutively.
    phase_release_gripper_open_frac = 0.82
    phase_release_max_lin_vel = 0.12
    phase_release_max_ang_vel = 0.85
    phase_release_min_open_stable_steps = 3

    # RETREAT → PUSH: minimum tool–book clearance (m) and hold steps.
    # Clearance is min distance from the IK tool point to book corners (proxy, not full finger mesh clearance).
    phase_retreat_min_hand_clearance_m = 0.09
    phase_retreat_min_clear_hold_steps = 3

    # --- Full placement success (terminal, only counted in PUSH phase) ---
    # rear_to_mouth = rear_x - mouth (m). Acceptable band for terminal success (inclusive).
    success_rear_to_mouth_min = -0.012
    success_rear_to_mouth_max = 0.002
    # front_to_back = slot_x_back - front_x: clearance from deepest face to inner back (0 = flush).
    success_front_clear_min = 0.0
    success_front_clear_max = 0.055
    # Allow small numerical/pose tolerance at the success gate.
    # (Used in `_get_dones()` to turn "very close" placements into success.)
    success_front_clear_eps_m = 2.0e-4

    success_z_thresh = 0.015
    success_yaw_thresh = math.radians(8.0)
    success_steps = 4
    success_lateral_margin = 0.000
    # Max allowed deviation of the book from the slot center in terms of lateral extent.
    success_lateral_extent_eps_m = 5.0e-4


    # Optional stability gate for success (set to 0 to disable).
    success_max_lin_vel = 0.0
    success_max_ang_vel = 0.0

    # Prevent immediate success on reset (steps).
    min_steps_before_success = 5

    debug_print_success = False
    debug_print_success_every = 1

    # --- Global shaping / terminal ---
    step_penalty = -0.001
    success_bonus = 550.0
    drop_penalty = -3.0

    # INSERT phase rewards / penalties
    rew_insert_rear_progress_scale = 5.0
    rew_insert_pre_entry_forward_scale = 1.0
    # Post-entry shaping only when leading edge is truly inside (front_to_mouth > 0); no pre-mouth window.
    rew_insert_post_entry_push_scale = 1.0
    rew_insert_stall_penalty_scale = 0.5
    rew_insert_stall_thresh_m = 2.0e-4
    rew_insert_bottom_reward_scale = 0.5
    # Full alignment weights; pre-entry uses `* rew_insert_align_penalty_pre_mul` (milder before crossing mouth).
    rew_insert_lat_penalty_scale = 0.9
    rew_insert_yaw_penalty_scale = 0.5
    rew_insert_z_penalty_scale = 1.0
    rew_insert_align_penalty_pre_mul = 0.55
    # Continuous milestone + one-time bonuses (see env latches / pending flags).
    rew_insert_release_ready_bonus_scale = 2.5
    rew_insert_cross_mouth_bonus = 12.0
    rew_insert_to_release_transition_bonus = 35.0
    # Absolute depth (in addition to delta progress): linear in clamped rear_to_mouth.
    rew_insert_abs_rear_to_mouth_scale = 1.2
    rew_insert_abs_rear_to_mouth_clip_lo = -0.07
    rew_insert_abs_rear_to_mouth_clip_hi = 0.012
    # Hover trap: near mouth, negligible rear progress, not release-ready.
    rew_insert_hover_penalty_scale = 0.4
    rew_insert_hover_mouth_low_m = -0.01
    rew_insert_hover_mouth_high_m = 0.005
    rew_insert_hover_d_rear_thresh_m = 3.0e-4

    # RELEASE phase
    rew_release_open_progress_scale = 0.8
    rew_release_speed_penalty_scale = 0.6
    rew_release_lat_penalty_scale = 0.25
    rew_release_z_penalty_scale = 0.25
    rew_release_yaw_penalty_scale = 0.2

    # RETREAT phase
    rew_retreat_clearance_delta_scale = 2.5
    rew_retreat_book_speed_penalty_scale = 0.8
    rew_retreat_rear_pull_penalty_scale = 4.0

    # PUSH phase (final seating)
    rew_push_rear_progress_scale = 5.0
    rew_push_post_push_scale = 1.0
    rew_push_stall_penalty_scale = 0.5
    rew_push_stall_thresh_m = 2.0e-4
    rew_push_bottom_reward_scale = 0.5
    rew_push_lat_penalty_scale = 0.9
    rew_push_yaw_penalty_scale = 0.5
    rew_push_z_penalty_scale = 1.0

    # Debug/safety terminations (optional).
    max_abs_xy = 0.95
    fell_height_thresh = 0.16
    upright_dot_thresh = 0.85
    enable_failure_terminations = False

    # Ground failure uses **lowest book corner** z (not COM): standing book on floor can have COM ~0.11 m.
    book_floor_lowest_z_thresh = 0.042
    # Extra padding (m) around shelf cuboid footprint for "still on shelf" exemption (see env).
    shelf_footprint_x_pad_m = 0.04
    shelf_footprint_y_pad_m = 0.05
    # Lowest corner must be at/above deck minus slack to count as supported on shelf (deck = shelf_top_z + shelf_thickness).
    book_on_shelf_z_slack_m = 0.02

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
