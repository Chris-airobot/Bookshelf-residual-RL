"""Residual bookshelf task using the physical xArm7 embodiment."""

from __future__ import annotations

import math

from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from .bookshelf_env_cfg_v4 import BOOK_TRUE_GROUND_LOWEST_Z_THRESH
from .bookshelf_residual_env_cfg import BookshelfEnvCfg as PandaResidualEnvCfg
from .xarm7_asset_cfg import (
    XARM7_FINGER_INNER_SURFACE_OFFSET_M,
    XARM7_GRIPPER_FULLY_CLOSED_JOINT_POS,
    XARM7_SIM_BOOK_HOLD_JOINT_POS,
    XARM7_SIM_BOOK_SPAWN_JOINT_POS,
    XARM7_WITH_GRIPPER_CFG,
    XARM_GRIPPER_COMMAND_JOINT_EXPR,
    XARM_GRIPPER_STATE_JOINT_EXPR,
)


XARM7_PRETARGET_JOINT_POS = {
    "joint1": 1.2342693425054612,
    "joint2": 1.5322427671441177,
    # The physical xArm reports these continuous joints on a 0..2*pi branch,
    # while the official Isaac USD uses the equivalent -pi..pi branch.
    "joint3": 4.904658882462919 - 2.0 * math.pi,
    "joint4": 1.302429752118059,
    "joint5": 3.302595179623167 - 2.0 * math.pi,
    "joint6": 0.6839448116011184,
    "joint7": 4.4791192150828865 - 2.0 * math.pi,
    "drive_joint": XARM7_SIM_BOOK_SPAWN_JOINT_POS,
    "left_finger_joint": XARM7_SIM_BOOK_SPAWN_JOINT_POS,
    "left_inner_knuckle_joint": XARM7_SIM_BOOK_SPAWN_JOINT_POS,
    "right_outer_knuckle_joint": XARM7_SIM_BOOK_SPAWN_JOINT_POS,
    "right_finger_joint": XARM7_SIM_BOOK_SPAWN_JOINT_POS,
    "right_inner_knuckle_joint": XARM7_SIM_BOOK_SPAWN_JOINT_POS,
}

# Keep the approved physical depth and measured pre-target offset, while
# centering the simulated ten-slot row laterally on the robot base. The
# physical approved slot is offset in Y because it describes one particular
# slot; carrying that offset into a full row makes one edge artificially hard.
XARM7_APPROVED_SLOT_CENTER_BASE = (
    0.8554391824906817,
    0.08412625750748041,
    0.17092253331755783,
)
XARM7_REVIEWED_PRETARGET_TCP_BASE = (
    0.7381960526120858,
    0.0793841964784254,
    0.1668091453908952,
)
XARM7_REVIEWED_PRETARGET_TCP_QUAT_WXYZ = (
    -0.02665888195916604,
    0.727445396248154,
    0.025591244894331722,
    0.6851697509922721,
)
XARM7_SIM_SLOT_CENTER = (
    0.63,
    0.0,
    XARM7_APPROVED_SLOT_CENTER_BASE[2],
)
XARM7_SCENE_BASE_POS = (
    XARM7_SIM_SLOT_CENTER[0] - XARM7_APPROVED_SLOT_CENTER_BASE[0],
    XARM7_SIM_SLOT_CENTER[1],
    0.0,
)
XARM7_PRETARGET_OFFSET_SLOT = tuple(
    tcp - slot
    for tcp, slot in zip(
        XARM7_REVIEWED_PRETARGET_TCP_BASE,
        XARM7_APPROVED_SLOT_CENTER_BASE,
    )
)

# The inherited book is 236 mm tall and stands on a 20 mm shelf deck.
XARM7_SHELF_TOP_Z = XARM7_SIM_SLOT_CENTER[2] - 0.02 - 0.5 * 0.236


@configclass
class BookshelfEnvCfg(PandaResidualEnvCfg):
    """xArm7 task with measured pre-target and configurable grasp-depth bounds."""

    robot = XARM7_WITH_GRIPPER_CFG.replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=XARM7_SCENE_BASE_POS,
            joint_pos=XARM7_PRETARGET_JOINT_POS,
        )
    )

    robot_arm_joint_names_expr = "joint[1-7]"
    robot_finger_joint_names_expr = XARM_GRIPPER_STATE_JOINT_EXPR
    robot_gripper_command_joint_names_expr = XARM_GRIPPER_COMMAND_JOINT_EXPR
    robot_left_finger_body_name = "left_finger"
    robot_right_finger_body_name = "right_finger"
    robot_hand_body_name = "xarm_gripper_base_link"
    # link_eef is coincident with link7 and link_tcp is a massless fixed link.
    # Using rigid links here avoids depending on how the URDF importer handles
    # massless fixed links while preserving the same frames through offsets.
    robot_ee_body_name = "link7"
    # Preserve the original task's grasp convention: finger midpoint position
    # with the hand orientation.
    robot_grasp_frame_body_name = ""

    # link_tcp is 172 mm along +Z from link_eef in the official xArm URDF.
    ik_body_offset_pos = (0.0, 0.0, 0.172)

    # The official USD couples the linkage to drive_joint. This joint value
    # places the collision-pad surfaces at the 34 mm book thickness.
    gripper_closed_joint_pos = XARM7_SIM_BOOK_HOLD_JOINT_POS
    gripper_open_joint_pos = 0.0
    gripper_push_closed_joint_pos = XARM7_GRIPPER_FULLY_CLOSED_JOINT_POS
    # Panda's six-frame retreat and five-frame close are too short for the
    # official xArm actuator. Retreat 120 mm over one second so the fingertips
    # clear the released book before closing, then hold that target while the
    # gripper reaches its fully closed PUSH configuration.
    # Keep the legacy 1 mm per-step value separate from the fixed-path endpoint.
    script_retreat_steps = 60
    script_close_steps = 30
    script_retreat_dx = -0.001
    debug_scripted_fixed_retreat_total_dx = -0.120
    debug_reachable_grasp_preclose_joint_pos = XARM7_SIM_BOOK_SPAWN_JOINT_POS
    debug_robot_target_gripper_ramp_steps = 30
    debug_robot_target_gripper_settle_steps = 30
    # The negative-Y extent of the left finger and positive-Y extent of the
    # right finger are the inward pad surfaces. The old value used the opposite
    # (outer) mesh extent and missed a severe initial collision.
    debug_finger_inner_surface_offset_m = XARM7_FINGER_INNER_SURFACE_OFFSET_M
    debug_book_min_finger_clearance_m = 0.0

    # Keep the original simulated grasp and upright-book convention. These
    # values are shared by the held book, neighboring books, and target math.
    book_grasp_offset_hand = (0.0, 0.0, 0.075)
    book_standing_quat = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)
    # The reviewed xArm wrist is a few degrees from the world-standing book
    # orientation. Spawn the rigid book from the configured book-in-hand axes
    # so its 34 mm thickness, rather than a tilted projection of its height,
    # lies across the gripper pads.
    book_grasp_orientation_source = "grasp_relative"

    # Every reset samples a new rigid grasp. Recompute the controller's
    # tool-to-book transform from that sample instead of retaining episode 1.
    debug_freeze_tool_to_book_transform = False

    # Initialize through the official USD's own kinematics.  These values are
    # the reviewed physical pre-target TCP expressed in the approved slot
    # frame, so they remain valid when the simulated shelf is repositioned.
    reset_to_slot_relative_tool_pose = True
    reset_tool_offset_slot_xyz = XARM7_PRETARGET_OFFSET_SLOT
    reset_tool_quaternion_slot_wxyz = XARM7_REVIEWED_PRETARGET_TCP_QUAT_WXYZ
    reset_tool_ik_iters = 160
    # Applied only by scripts/sb3/train.py. Direct task creation and evaluation
    # retain the reviewed physical pre-target geometry.
    xarm_training_reset_standoff_m = 0.030

    # Visual references for checking the physical layout at a glance. The
    # orange base footprint is the intended robot mounting point; the green
    # book is the desired straight insertion pose.
    show_robot_base_reference_marker = True
    robot_base_reference_pos = XARM7_SCENE_BASE_POS
    show_target_book_marker = True
    show_target_ee_marker = True

    # Preserve the complete original residual-RL bookshelf scene. Only the
    # robot embodiment changes from Panda to xArm7.
    slot_center_y = 0.0
    slot_x_open = 0.63
    slot_x_back = 0.83
    shelf_top_z = XARM7_SHELF_TOP_Z
    shelf_thickness = 0.02

    # Provisional, deliberately conservative bounds until the shallowest and
    # deepest reliable physical grasps are measured. In the finger-midpoint
    # grasp frame, Z is the grasp-depth direction for this mounted gripper.
    book_grasp_translation_jitter_min = (-0.002, -0.002, -0.002)
    book_grasp_translation_jitter_max = (0.002, 0.002, 0.002)

    residual_curriculum_grasp_translation_bounds_1 = (
        (-0.001, -0.001, -0.001),
        (0.001, 0.001, 0.001),
    )
    residual_curriculum_grasp_translation_bounds_2 = (
        (-0.002, -0.002, -0.003),
        (0.002, 0.002, 0.003),
    )
    residual_curriculum_grasp_translation_bounds_3 = (
        (-0.003, -0.003, -0.005),
        (0.003, 0.003, 0.005),
    )
    residual_curriculum_grasp_translation_bounds_final = (
        (-0.003, -0.003, -0.005),
        (0.003, 0.003, 0.005),
    )

    # Keep angular variation modest until physical grasp-angle measurements are available.
    residual_curriculum_reset_1 = (math.radians(1.0), 0.0, 0.0, 0.0, math.radians(1.0))
    residual_curriculum_reset_2 = (math.radians(1.5), 0.0, 0.0, 0.0, math.radians(2.0))
    residual_curriculum_reset_3 = (math.radians(2.0), 0.0, 0.0, 0.0, math.radians(3.0))
    residual_curriculum_reset_final = residual_curriculum_reset_3

    # Hold the randomized book pose while the official linkage reaches the
    # 32 mm grasp target; afterwards the book is an ordinary dynamic rigid body.
    reset_warmup_steps = 60

    # scripts/sb3/train.py enables this gate for xArm training. Keeping the
    # task default false prevents direct evaluation tools from filtering out
    # difficult randomization samples.
    enable_reset_acceptance_gate = False
    reset_acceptance_validation_steps = 12
    reset_acceptance_max_attempts = 50
    reset_acceptance_translation_limit_m = 0.003
    reset_acceptance_rotation_limit_rad = math.radians(3.0)
    reset_acceptance_arm_error_limit_rad = math.radians(8.0)
    reset_acceptance_ground_height_m = BOOK_TRUE_GROUND_LOWEST_Z_THRESH
