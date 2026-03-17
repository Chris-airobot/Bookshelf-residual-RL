# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass


@configclass
class BookshelfEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    # - spaces definition
    action_space = 2
    observation_space = 4
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.0, replicate_physics=True)

    # rigid object used as "book" (simple cuboid)
    # visual / physical size of the book in (x, y, z): width, height (spine), thickness
    book_size = (0.152, 0.229, 0.02)
    # 90° around X so the book stands vertically (height 0.229 along world Z). Quat (w, x, y, z).
    book_standing_quat = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)

    book_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Book",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.15, 0.0, book_size[1] / 2.0],
            rot=book_standing_quat,
        ),
        spawn=sim_utils.CuboidCfg(
            size=book_size,
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.8)),
        ),
    )

    # target slot position in env frame (no geometry yet)
    target_depth = 0.20  # [m] along +X from env origin
    target_lateral = 0.0  # [m] along +Y from env origin

    # action scaling (interpreted as planar velocities)
    forward_action_scale = 0.05  # [m/s] for +X
    lateral_action_scale = 0.05  # [m/s] for +Y

    # numeric safety (helps prevent PPO NaNs on long runs)
    obs_clip = 10.0
    max_abs_xy = 1.0  # terminate if |x| or |y| exceeds this in env frame

    # reward scales
    progress_scale = 1.0
    center_scale = 0.5
    action_penalty_scale = 0.01
    success_bonus = 5.0
    timeout_penalty = -1.0

    # success thresholds
    success_forward_thresh = 0.01  # [m]
    success_lateral_thresh = 0.01  # [m]
    success_steps = 5  # consecutive steps inside success region

    # reset randomization ranges (in env frame)
    initial_forward_offset_range = (-0.03, 0.03)  # [m]
    initial_lateral_offset_range = (-0.03, 0.03)  # [m]