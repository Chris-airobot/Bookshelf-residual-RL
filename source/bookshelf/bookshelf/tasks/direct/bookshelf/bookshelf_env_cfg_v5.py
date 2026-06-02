#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Config for Bookshelf-Direct-v5.

V5 keeps the v4 task structure and adds fixed-row reset randomization:
- 6D action: [dx, dy, dz, dyaw, dpitch, g]
- 12D observation: v4 obs plus upright tilt x/y
- ten possible book positions in a fixed shelf row
- one sampled missing-book slot
- side books can be one-slot or two-slot books
- side-book height variants are randomized across row positions
- physical gap width: 23-26 mm
- book grasp pose jitter: x/y/z/yaw
- pre-insertion arm joint noise: +/- 3 deg
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass

from .bookshelf_env_cfg_v4 import BookshelfEnvCfg as BookshelfEnvCfgV4
from .bookshelf_env_cfg_v4 import _BOOK_LWH


_SIDE_BOOK_SINGLE_BASE_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/SideBookSingleBase",
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=[0.0, 0.0, 0.0],
        rot=(math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0),
    ),
    spawn=sim_utils.CuboidCfg(
        size=_BOOK_LWH,
        rigid_props=RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            disable_gravity=True,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=2,
            max_depenetration_velocity=2.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.25, 0.15)),
    ),
)

_SIDE_BOOK_DOUBLE_BASE_CFG = _SIDE_BOOK_SINGLE_BASE_CFG.replace(
    prim_path="/World/envs/env_.*/SideBookDoubleBase",
    spawn=_SIDE_BOOK_SINGLE_BASE_CFG.spawn.replace(size=(_BOOK_LWH[0], _BOOK_LWH[1], 2.0 * _BOOK_LWH[2])),
)

_SIDE_BOOK_HEIGHTS = (0.229, 0.205, 0.185, 0.218, 0.195, 0.229, 0.202, 0.175, 0.214)
_WIDE_SIDE_BOOK_HEIGHTS = (0.215, 0.190, 0.229, 0.200)


@configclass
class BookshelfEnvCfg(BookshelfEnvCfgV4):
    """Bookshelf v5: v4 with randomized rigid slot and grasp reset."""

    action_space = 6
    observation_space = 12

    dpitch_action_scale = math.radians(0.6)

    # Keep v4 reward/success. The reset geometry and upright-tilt control change.
    row_book_count = 10
    side_book_height_min = 0.175
    side_book_height_max = _BOOK_LWH[1]
    side_book_merge_probability = 0.35
    top_shelf_clearance = 0.025
    slot_lateral_clearance_min = 0.003
    slot_lateral_clearance_max = 0.006
    forced_missing_book_index = -1

    # V5 allows small over-insertion past the nominal back reference.
    success_front_clear_min = -0.003
    success_lateral_extent_eps_m = 0.0015

    reset_arm_joint_pos_noise = math.radians(3.0)

    book_grasp_x_jitter = 0.008
    book_grasp_y_jitter = 0.006
    book_grasp_z_jitter = 0.003
    book_grasp_yaw_jitter = math.radians(8.0)

    # Object pool: enough one-slot and two-slot books to fill ten row positions around one missing slot.

    side_book_0: RigidObjectCfg = _SIDE_BOOK_SINGLE_BASE_CFG.replace(
        prim_path="/World/envs/env_.*/SideBook0",
        spawn=_SIDE_BOOK_SINGLE_BASE_CFG.spawn.replace(
            size=(_BOOK_LWH[0], _SIDE_BOOK_HEIGHTS[0], _BOOK_LWH[2]),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.40, 0.25, 0.15))
        ),
    )
    side_book_1: RigidObjectCfg = _SIDE_BOOK_SINGLE_BASE_CFG.replace(
        prim_path="/World/envs/env_.*/SideBook1",
        spawn=_SIDE_BOOK_SINGLE_BASE_CFG.spawn.replace(
            size=(_BOOK_LWH[0], _SIDE_BOOK_HEIGHTS[1], _BOOK_LWH[2]),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.46, 0.32, 0.18))
        ),
    )
    side_book_2: RigidObjectCfg = _SIDE_BOOK_SINGLE_BASE_CFG.replace(
        prim_path="/World/envs/env_.*/SideBook2",
        spawn=_SIDE_BOOK_SINGLE_BASE_CFG.spawn.replace(
            size=(_BOOK_LWH[0], _SIDE_BOOK_HEIGHTS[2], _BOOK_LWH[2]),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.52, 0.36, 0.20))
        ),
    )
    side_book_3: RigidObjectCfg = _SIDE_BOOK_SINGLE_BASE_CFG.replace(
        prim_path="/World/envs/env_.*/SideBook3",
        spawn=_SIDE_BOOK_SINGLE_BASE_CFG.spawn.replace(
            size=(_BOOK_LWH[0], _SIDE_BOOK_HEIGHTS[3], _BOOK_LWH[2]),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.58, 0.43, 0.24))
        ),
    )
    side_book_4: RigidObjectCfg = _SIDE_BOOK_SINGLE_BASE_CFG.replace(
        prim_path="/World/envs/env_.*/SideBook4",
        spawn=_SIDE_BOOK_SINGLE_BASE_CFG.spawn.replace(
            size=(_BOOK_LWH[0], _SIDE_BOOK_HEIGHTS[4], _BOOK_LWH[2]),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.48, 0.30, 0.16))
        ),
    )
    side_book_5: RigidObjectCfg = _SIDE_BOOK_SINGLE_BASE_CFG.replace(
        prim_path="/World/envs/env_.*/SideBook5",
        spawn=_SIDE_BOOK_SINGLE_BASE_CFG.spawn.replace(
            size=(_BOOK_LWH[0], _SIDE_BOOK_HEIGHTS[5], _BOOK_LWH[2]),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.62, 0.48, 0.28))
        ),
    )
    side_book_6: RigidObjectCfg = _SIDE_BOOK_SINGLE_BASE_CFG.replace(
        prim_path="/World/envs/env_.*/SideBook6",
        spawn=_SIDE_BOOK_SINGLE_BASE_CFG.spawn.replace(
            size=(_BOOK_LWH[0], _SIDE_BOOK_HEIGHTS[6], _BOOK_LWH[2]),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.44, 0.28, 0.18))
        ),
    )
    side_book_7: RigidObjectCfg = _SIDE_BOOK_SINGLE_BASE_CFG.replace(
        prim_path="/World/envs/env_.*/SideBook7",
        spawn=_SIDE_BOOK_SINGLE_BASE_CFG.spawn.replace(
            size=(_BOOK_LWH[0], _SIDE_BOOK_HEIGHTS[7], _BOOK_LWH[2]),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.56, 0.38, 0.22))
        ),
    )
    side_book_8: RigidObjectCfg = _SIDE_BOOK_SINGLE_BASE_CFG.replace(
        prim_path="/World/envs/env_.*/SideBook8",
        spawn=_SIDE_BOOK_SINGLE_BASE_CFG.spawn.replace(
            size=(_BOOK_LWH[0], _SIDE_BOOK_HEIGHTS[8], _BOOK_LWH[2]),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.50, 0.34, 0.19))
        ),
    )

    wide_side_book_0: RigidObjectCfg = _SIDE_BOOK_DOUBLE_BASE_CFG.replace(
        prim_path="/World/envs/env_.*/WideSideBook0",
        spawn=_SIDE_BOOK_DOUBLE_BASE_CFG.spawn.replace(
            size=(_BOOK_LWH[0], _WIDE_SIDE_BOOK_HEIGHTS[0], 2.0 * _BOOK_LWH[2]),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.44, 0.28, 0.16))
        ),
    )
    wide_side_book_1: RigidObjectCfg = _SIDE_BOOK_DOUBLE_BASE_CFG.replace(
        prim_path="/World/envs/env_.*/WideSideBook1",
        spawn=_SIDE_BOOK_DOUBLE_BASE_CFG.spawn.replace(
            size=(_BOOK_LWH[0], _WIDE_SIDE_BOOK_HEIGHTS[1], 2.0 * _BOOK_LWH[2]),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.56, 0.38, 0.20))
        ),
    )
    wide_side_book_2: RigidObjectCfg = _SIDE_BOOK_DOUBLE_BASE_CFG.replace(
        prim_path="/World/envs/env_.*/WideSideBook2",
        spawn=_SIDE_BOOK_DOUBLE_BASE_CFG.spawn.replace(
            size=(_BOOK_LWH[0], _WIDE_SIDE_BOOK_HEIGHTS[2], 2.0 * _BOOK_LWH[2]),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.50, 0.34, 0.18))
        ),
    )
    wide_side_book_3: RigidObjectCfg = _SIDE_BOOK_DOUBLE_BASE_CFG.replace(
        prim_path="/World/envs/env_.*/WideSideBook3",
        spawn=_SIDE_BOOK_DOUBLE_BASE_CFG.spawn.replace(
            size=(_BOOK_LWH[0], _WIDE_SIDE_BOOK_HEIGHTS[3], 2.0 * _BOOK_LWH[2]),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.62, 0.48, 0.28))
        ),
    )
