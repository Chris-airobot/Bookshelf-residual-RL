"""Isaac Lab articulation configuration for NVIDIA's official xArm7 asset."""

from __future__ import annotations

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.spawners.from_files import spawn_from_usd
from isaaclab.sim.utils import clone, get_current_stage
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from pxr import PhysxSchema, UsdPhysics


XARM_GRIPPER_STATE_JOINT_EXPR = (
    "(drive_joint|left_finger_joint|left_inner_knuckle_joint|"
    "right_outer_knuckle_joint|right_finger_joint|right_inner_knuckle_joint)"
)
XARM_GRIPPER_COMMAND_JOINT_EXPR = "drive_joint"

# Official URDF linkage dimensions and the inward extent of the finger meshes.
XARM7_OUTER_KNUCKLE_PIVOT_Y_M = 0.035
XARM7_FINGER_JOINT_Y_M = 0.035465
XARM7_FINGER_JOINT_Z_M = 0.042039
XARM7_FINGER_INNER_SURFACE_OFFSET_M = 0.0260032
XARM7_SIM_BOOK_THICKNESS_M = 0.034
XARM7_GRIPPER_FULLY_CLOSED_JOINT_POS = 0.85


def xarm7_gripper_joint_for_pad_gap(pad_gap_m: float) -> float:
    """Convert a desired inner-pad gap to the official drive-joint angle."""

    pad_gap_m = float(pad_gap_m)
    if not math.isfinite(pad_gap_m) or pad_gap_m <= 0.0:
        raise ValueError("xArm gripper pad gap must be positive and finite")

    def gap_at(joint_pos: float) -> float:
        finger_origin_half_gap = (
            XARM7_OUTER_KNUCKLE_PIVOT_Y_M
            + math.cos(joint_pos) * XARM7_FINGER_JOINT_Y_M
            - math.sin(joint_pos) * XARM7_FINGER_JOINT_Z_M
        )
        return (
            2.0 * finger_origin_half_gap
            - 2.0 * XARM7_FINGER_INNER_SURFACE_OFFSET_M
        )

    lower_joint = 0.0
    upper_joint = XARM7_GRIPPER_FULLY_CLOSED_JOINT_POS
    maximum_gap = gap_at(lower_joint)
    minimum_gap = gap_at(upper_joint)
    if not minimum_gap <= pad_gap_m <= maximum_gap:
        raise ValueError(
            "requested xArm pad gap is outside the official linkage range: "
            f"{pad_gap_m:.6f} m not in [{minimum_gap:.6f}, {maximum_gap:.6f}] m"
        )

    for _ in range(64):
        midpoint = 0.5 * (lower_joint + upper_joint)
        if gap_at(midpoint) > pad_gap_m:
            lower_joint = midpoint
        else:
            upper_joint = midpoint
    return 0.5 * (lower_joint + upper_joint)


# Spawn without overlap, then add 1 mm of preload per side for friction.
# The other five joints follow drive_joint through the USD's mimic APIs.
XARM7_SIM_BOOK_SPAWN_JOINT_POS = xarm7_gripper_joint_for_pad_gap(
    XARM7_SIM_BOOK_THICKNESS_M
)
XARM7_SIM_BOOK_HOLD_JOINT_POS = xarm7_gripper_joint_for_pad_gap(0.032)


def _xarm7_usd_path() -> str:
    official = f"{ISAAC_NUCLEUS_DIR}/Robots/Ufactory/xarm7/xarm7.usd"
    return os.environ.get("BOOKSHELF_XARM7_USD_PATH", official).strip() or official


@clone
def _spawn_xarm7_from_usd(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn the official asset as one arm-plus-gripper articulation."""

    prim = spawn_from_usd(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )
    nested_root_path = f"{prim_path}/gripper/root_joint"
    nested_root = get_current_stage().GetPrimAtPath(nested_root_path)
    if not nested_root.IsValid():
        raise RuntimeError(
            "The official xArm7 USD no longer has the expected nested gripper root: "
            f"{nested_root_path}"
        )
    nested_root.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    nested_root.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
    return prim


XARM7_WITH_GRIPPER_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        func=_spawn_xarm7_from_usd,
        usd_path=_xarm7_usd_path(),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=2,
        ),
    ),
    articulation_root_prim_path="/root_joint",
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "joint1": 1.2342693425054612,
            "joint2": 1.5322427671441177,
            # Match the official USD's -pi..pi branch without changing the
            # physical joint rotations recorded on xArm's 0..2*pi branch.
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
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-7]"],
            effort_limit_sim={
                "joint[1-2]": 50.0,
                "joint[3-5]": 30.0,
                "joint[6-7]": 20.0,
            },
            velocity_limit_sim=3.14,
            stiffness={
                "joint[1-5]": 400.0,
                "joint[6-7]": 1200.0,
            },
            damping={
                "joint[1-5]": 80.0,
                "joint[6-7]": 120.0,
            },
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[XARM_GRIPPER_COMMAND_JOINT_EXPR],
            effort_limit_sim=50.0,
            velocity_limit_sim=2.0,
            stiffness=200.0,
            damping=10.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
