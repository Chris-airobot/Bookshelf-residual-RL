#!/usr/bin/env python3
"""Staged 3MT presentation scene for the bookshelf placement thesis story.

This script opens the Bookshelf-Direct-v4 environment and stages one clean
case for screenshot capture:
0 = blue start book in the gripper
1 = green successful pre-release book grasped near the shelf slot
2 = red failed/reversed book in the gripper
3 = red failed/reversed book with a top-side grasp
4 = red failed/reversed book with a bottom-side grasp

Example:
    ~/isaacsim/python.sh scripts/3MT.py 0 --task Bookshelf-Direct-v4 --num_envs 1
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import math
import random
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Create a clean 3MT bookshelf placement scene.")
parser.add_argument(
    "case",
    nargs="?",
    type=int,
    choices=(0, 1, 2, 3, 4),
    default=None,
    help=(
        "Screenshot case: 0=start blue in gripper, 1=green pre-release grasp near slot, "
        "2=failed red reversed in gripper, 3=failed red top grasp, 4=failed red bottom grasp."
    ),
)
parser.add_argument(
    "--case",
    dest="case_opt",
    type=int,
    choices=(0, 1, 2, 3, 4),
    default=None,
    help="Same as positional case. Overrides the positional value when provided.",
)
parser.add_argument("--task", type=str, default="Bookshelf-Direct-v4", help="Isaac Lab task id.")
parser.add_argument("--num_envs", type=int, default=1, help="Keep this at 1 for the staged presentation scene.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--paths",
    action="store_true",
    default=False,
    help="Show subtle blue/red candidate approach paths. Hidden by default for slide clarity.",
)
parser.add_argument(
    "--grasp_frame",
    action="store_true",
    default=False,
    help="Show an RGB frame at the computed gripper grasp target.",
)
parser.add_argument(
    "--no_grasp_frame",
    action="store_false",
    dest="grasp_frame",
    help="Hide the computed gripper grasp target frame.",
)
parser.add_argument(
    "--screenshot_dir",
    type=str,
    default=None,
    help="If set, capture one PNG from PresentationCamera into this directory and exit.",
)
parser.add_argument(
    "--manual_drag",
    action="store_true",
    default=False,
    help="Pause after staging and select robot/hand prims so you can manually drag them in Isaac Sim.",
)
parser.add_argument("--screenshot_width", type=int, default=1920, help="Screenshot width in pixels.")
parser.add_argument("--screenshot_height", type=int, default=1080, help="Screenshot height in pixels.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import carb
import gymnasium as gym
import omni.timeline
import omni.usd
import torch

import isaaclab_tasks  # noqa: F401
import isaaclab.utils.math as math_utils
from isaaclab_tasks.utils import parse_env_cfg

from isaaclab.sim.utils.stage import get_current_stage
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade


_REPO_ROOT = Path(__file__).resolve().parents[1]
_BOOKSHELF_SRC = _REPO_ROOT / "source" / "bookshelf"
if str(_BOOKSHELF_SRC) not in sys.path:
    sys.path.insert(0, str(_BOOKSHELF_SRC))

import bookshelf.tasks  # noqa: F401,E402


GRASP_CLEARANCE_M = 0.016
CASE2_ROBOT_BASE_POS = (0.02, -0.8, 0.0)
CASE3_TOP_ROBOT_BASE_POS = (0.06, -0.24, 0.0)
PRESENTATION_VISIBLE_BOOKS_PER_SECTION = 7
PRESENTATION_EXTRA_RIGHT_VISUAL_BOOKS = 5
PRESENTATION_SHELF_EXTRA_BOOKS_PER_SIDE = 19
PRESENTATION_SECOND_SLOT_EXTRA_INDEX = 7
PRESENTATION_SIDE_BOOK_SEED = 314159


def _quat_mul(q1: tuple[float, float, float, float], q2: tuple[float, float, float, float]):
    """Multiply wxyz quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _dot3(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross3(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _normalize3(v: tuple[float, float, float]) -> tuple[float, float, float]:
    n = math.sqrt(max(_dot3(v, v), 1.0e-12))
    return (v[0] / n, v[1] / n, v[2] / n)


def _normalize_quat(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    n = math.sqrt(max(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3], 1.0e-12))
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


def _quat_apply_tuple(
    quat_wxyz: tuple[float, float, float, float],
    vec: tuple[float, float, float],
) -> tuple[float, float, float]:
    q_vec = (0.0, vec[0], vec[1], vec[2])
    q_conj = (quat_wxyz[0], -quat_wxyz[1], -quat_wxyz[2], -quat_wxyz[3])
    out = _quat_mul(_quat_mul(quat_wxyz, q_vec), q_conj)
    return (out[1], out[2], out[3])


def _quat_from_axes(
    x_axis: tuple[float, float, float],
    y_axis: tuple[float, float, float],
    z_axis: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    """Quaternion from world-space hand axes stored as rotation-matrix columns."""
    m00, m01, m02 = x_axis[0], y_axis[0], z_axis[0]
    m10, m11, m12 = x_axis[1], y_axis[1], z_axis[1]
    m20, m21, m22 = x_axis[2], y_axis[2], z_axis[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return ((0.25 * s), (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s)
    if m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        return ((m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s)
    if m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        return ((m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s)
    s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
    return ((m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s)


def _top_down_hand_quat_for_book(book_quat_wxyz: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """World top-down hand pose: +Z_hand points down, +Y_hand closes across book thickness."""
    z_axis = (0.0, 0.0, -1.0)
    book_thickness_axis = _quat_apply_tuple(book_quat_wxyz, (0.0, 0.0, 1.0))
    y_axis = (
        book_thickness_axis[0] - _dot3(book_thickness_axis, z_axis) * z_axis[0],
        book_thickness_axis[1] - _dot3(book_thickness_axis, z_axis) * z_axis[1],
        book_thickness_axis[2] - _dot3(book_thickness_axis, z_axis) * z_axis[2],
    )
    if _dot3(y_axis, y_axis) < 1.0e-8:
        y_axis = (0.0, 1.0, 0.0)
    y_axis = _normalize3(y_axis)
    x_axis = _normalize3(_cross3(y_axis, z_axis))
    y_axis = _normalize3(_cross3(z_axis, x_axis))
    return _normalize_quat(_quat_from_axes(x_axis, y_axis, z_axis))


def _world_yaw_quat(yaw_rad: float) -> tuple[float, float, float, float]:
    return (math.cos(0.5 * yaw_rad), 0.0, 0.0, math.sin(0.5 * yaw_rad))


def _local_y_quat(angle_rad: float) -> tuple[float, float, float, float]:
    return (math.cos(0.5 * angle_rad), 0.0, math.sin(0.5 * angle_rad), 0.0)


def _local_z_quat(angle_rad: float) -> tuple[float, float, float, float]:
    return (math.cos(0.5 * angle_rad), 0.0, 0.0, math.sin(0.5 * angle_rad))


def _make_preview_material(
    stage,
    path: str,
    color: tuple[float, float, float],
    opacity: float = 1.0,
    roughness: float = 0.65,
):
    """Create a simple USD PreviewSurface material."""
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    if opacity < 1.0:
        shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _bind_material(stage, prim_path: str, material) -> None:
    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid():
        UsdShade.MaterialBindingAPI(prim).Bind(material)


def _bind_material_tree(stage, prim_path: str, material) -> None:
    """Bind material to a prim and all descendants, useful for spawned assets."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    for p in Usd.PrimRange(prim):
        if p.IsA(UsdGeom.Imageable):
            UsdShade.MaterialBindingAPI(p).Bind(material)


def _hide_imageable_tree(stage, prim_path: str) -> None:
    """Hide the default visual tree while keeping the simulated rigid body alive."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    for p in Usd.PrimRange(prim):
        path_lower = str(p.GetPath()).lower()
        if "/collisions" in path_lower or "/collision" in path_lower:
            continue
        if p.IsA(UsdGeom.Imageable):
            UsdGeom.Imageable(p).MakeInvisible()


def _show_imageable_tree(stage, prim_path: str) -> None:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    for p in Usd.PrimRange(prim):
        path_lower = str(p.GetPath()).lower()
        if "/collisions" in path_lower or "/collision" in path_lower:
            continue
        if p.IsA(UsdGeom.Imageable):
            UsdGeom.Imageable(p).MakeVisible()


def _deactivate_prim(stage, prim_path: str) -> None:
    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid():
        prim.SetActive(False)


def _presentation_visible_neighbor_paths(base: str) -> list[str]:
    left_paths = [
        f"{base}/LeftNeighborExtra{i}" for i in range(PRESENTATION_VISIBLE_BOOKS_PER_SECTION - 1, 0, -1)
    ]
    left_paths.append(f"{base}/LeftNeighborBook")

    right_first_section = [f"{base}/RightNeighborBook"]
    right_first_section.extend(
        f"{base}/RightNeighborExtra{i}" for i in range(1, PRESENTATION_VISIBLE_BOOKS_PER_SECTION)
    )
    right_second_section = [
        f"{base}/RightNeighborExtra{i}"
        for i in range(
            PRESENTATION_SECOND_SLOT_EXTRA_INDEX + 1,
            PRESENTATION_SECOND_SLOT_EXTRA_INDEX + 1 + PRESENTATION_VISIBLE_BOOKS_PER_SECTION,
        )
    ]
    return left_paths + right_first_section + right_second_section


def _hide_presentation_gap_books(stage, base: str) -> None:
    for i in range(PRESENTATION_VISIBLE_BOOKS_PER_SECTION, PRESENTATION_SHELF_EXTRA_BOOKS_PER_SIDE + 1):
        _deactivate_prim(stage, f"{base}/LeftNeighborExtra{i}")
    _deactivate_prim(stage, f"{base}/RightNeighborExtra{PRESENTATION_SECOND_SLOT_EXTRA_INDEX}")


def _hide_presentation_neighbor_visuals(stage, base: str) -> None:
    neighbor_paths = [f"{base}/LeftNeighborBook", f"{base}/RightNeighborBook"]
    for i in range(1, PRESENTATION_SHELF_EXTRA_BOOKS_PER_SIDE + 1):
        neighbor_paths.append(f"{base}/LeftNeighborExtra{i}")
        neighbor_paths.append(f"{base}/RightNeighborExtra{i}")
    for prim_path in neighbor_paths:
        _hide_imageable_tree(stage, prim_path)


def _darker_color(color: tuple[float, float, float], scale: float = 0.58) -> tuple[float, float, float]:
    return (
        max(0.0, min(1.0, color[0] * scale)),
        max(0.0, min(1.0, color[1] * scale)),
        max(0.0, min(1.0, color[2] * scale)),
    )


def _create_randomized_side_book_visuals(
    stage,
    prim_root: str,
    materials_root: str,
    logical_sections_y: list[list[float]],
    mid_x: float,
    shelf_top_z: float,
    shelf_thickness: float,
    standing_quat: tuple[float, float, float, float],
    book_l: float,
    base_book_h: float,
    base_book_t: float,
    pitch_y: float,
    page_material,
    label_material,
) -> None:
    """Draw varied presentation-only side books while leaving the physical slot layout fixed."""
    rng = random.Random(PRESENTATION_SIDE_BOOK_SEED)
    cover_color = (0.66, 0.49, 0.31)
    cover_mat = _make_preview_material(stage, f"{materials_root}/SideBookCover", cover_color, 1.0)
    spine_mat = _make_preview_material(stage, f"{materials_root}/SideBookSpine", _darker_color(cover_color, 0.72), 1.0)

    visual_idx = 0
    for section_y in logical_sections_y:
        i = 0
        while i < len(section_y):
            slots_left = len(section_y) - i
            merge = slots_left >= 2 and rng.random() < 0.28
            span = 2 if merge else 1
            y_slice = section_y[i : i + span]
            center_y = sum(y_slice) / float(span)
            visual_t = max(base_book_t * 0.72, span * base_book_t + (span - 1) * (pitch_y - base_book_t))
            visual_h = rng.uniform(base_book_h * 0.78, base_book_h * 1.02)
            visual_l = rng.uniform(book_l * 0.92, book_l * 1.04)
            center_x = mid_x + rng.uniform(-0.004, 0.004)
            center_z = shelf_top_z + shelf_thickness + 0.5 * visual_h

            _create_book_visual(
                stage,
                f"{prim_root}/SideBook{visual_idx:02d}",
                (visual_l, visual_h, visual_t),
                (center_x, center_y, center_z),
                standing_quat,
                cover_mat,
                page_material,
                spine_mat,
                label_material if rng.random() < 0.55 else None,
                label_material if rng.random() < 0.35 else None,
            )
            i += span
            visual_idx += 1


def _show_only_failed_case_robot_hand(stage) -> None:
    """For case 2, keep only the wrist/hand/fingers visible so the red book is not blocked."""
    robot_base = "/World/envs/env_0/Robot"
    for link_name in (
        "panda_link0",
        "panda_link1",
        "panda_link2",
        "panda_link3",
        "panda_link4",
        "panda_link5",
        "panda_link6",
    ):
        link_path = f"{robot_base}/{link_name}"
        link_prim = stage.GetPrimAtPath(link_path)
        if not link_prim.IsValid():
            continue
        for p in Usd.PrimRange(link_prim):
            path = str(p.GetPath())
            path_lower = path.lower()
            if path == link_path or "/collisions" in path_lower or "/collision" in path_lower:
                continue
            if p.IsA(UsdGeom.Imageable):
                UsdGeom.Imageable(p).MakeInvisible()

    for link_name in ("panda_link7", "panda_hand", "panda_leftfinger", "panda_rightfinger"):
        _show_imageable_tree(stage, f"{robot_base}/{link_name}")


def _enter_manual_drag_mode(case_id: int) -> None:
    """Pause sim playback and select useful prims for manual presentation posing."""
    omni.timeline.get_timeline_interface().pause()
    if case_id in (2, 3, 4):
        selected_paths = [
            "/World/envs/env_0/Robot/panda_link7",
            "/World/envs/env_0/Robot/panda_hand",
            "/World/envs/env_0/Robot/panda_leftfinger",
            "/World/envs/env_0/Robot/panda_rightfinger",
        ]
    else:
        selected_paths = ["/World/envs/env_0/Robot"]
    omni.usd.get_context().get_selection().set_selected_prim_paths(selected_paths, True)
    print("[INFO] Manual drag mode enabled. Simulation playback is paused.")
    print("[INFO] Use the Isaac Sim Move/Rotate tool on the selected prims, then capture manually.")
    print("[INFO] Selected prims:")
    for path in selected_paths:
        print(f"[INFO]   {path}")


def _set_xform(
    prim,
    translation: tuple[float, float, float],
    orientation_wxyz: tuple[float, float, float, float],
    scale: tuple[float, float, float],
) -> None:
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
    qw, qx, qy, qz = orientation_wxyz
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(qw, Gf.Vec3f(qx, qy, qz)))
    xform.AddScaleOp().Set(Gf.Vec3f(*scale))


def _create_visual_cuboid(
    stage,
    prim_path: str,
    size: tuple[float, float, float],
    translation: tuple[float, float, float],
    orientation_wxyz: tuple[float, float, float, float],
    material,
) -> None:
    cube = UsdGeom.Cube.Define(stage, prim_path)
    cube.CreateSizeAttr(1.0)
    _set_xform(cube.GetPrim(), translation, orientation_wxyz, size)
    UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(material)


def _create_book_visual(
    stage,
    prim_root: str,
    size: tuple[float, float, float],
    translation: tuple[float, float, float],
    orientation_wxyz: tuple[float, float, float, float],
    cover_material,
    page_material,
    spine_material,
    title_material=None,
    label_material=None,
    create_body: bool = True,
) -> None:
    """Create a clean book visual with cover, spine, page-opening side, and top pages."""
    book_l, book_h, book_t = size
    detail_thick = 0.0012
    face_overlay_thick = 0.001

    if create_body:
        _create_visual_cuboid(
            stage,
            f"{prim_root}/BookBody",
            (book_l, book_h, book_t),
            translation,
            orientation_wxyz,
            cover_material,
        )

    # This matches scripts/book_visual_test.py: -local X is the spine/back,
    # +local X is the clean page-opening side.
    spine_width = 0.010
    spine_center = _local_to_world(translation, orientation_wxyz, (-0.5 * book_l + 0.5 * spine_width, 0.0, 0.0))
    opening_center = _local_to_world(
        translation, orientation_wxyz, (0.5 * book_l + 0.5 * face_overlay_thick, 0.0, 0.0)
    )
    top_center = _local_to_world(
        translation, orientation_wxyz, (0.0, 0.5 * book_h + 0.5 * face_overlay_thick, 0.0)
    )
    _create_visual_cuboid(
        stage,
        f"{prim_root}/Spine",
        (spine_width, book_h, book_t * 1.002),
        spine_center,
        orientation_wxyz,
        spine_material,
    )
    _create_visual_cuboid(
        stage,
        f"{prim_root}/ForeEdgePages",
        (face_overlay_thick, book_h, book_t * 0.92),
        opening_center,
        orientation_wxyz,
        page_material,
    )
    _create_visual_cuboid(
        stage,
        f"{prim_root}/TopPages",
        (book_l * 0.98, face_overlay_thick, book_t * 0.92),
        top_center,
        orientation_wxyz,
        page_material,
    )

    if title_material is not None:
        title_center = _local_to_world(translation, orientation_wxyz, (0.014, 0.020, 0.5 * book_t + 0.5 * detail_thick))
        _create_visual_cuboid(
            stage,
            f"{prim_root}/FrontTitleBlock",
            (book_l * 0.44, book_h * 0.26, detail_thick),
            title_center,
            orientation_wxyz,
            title_material,
        )

    if label_material is not None:
        label_center = _local_to_world(translation, orientation_wxyz, (-0.5 * book_l + 0.0008, -0.010, 0.0))
        _create_visual_cuboid(
            stage,
            f"{prim_root}/SpineLabel",
            (detail_thick, book_h * 0.36, book_t * 0.50),
            label_center,
            orientation_wxyz,
            label_material,
        )


def _local_to_world(
    origin: tuple[float, float, float],
    quat_wxyz: tuple[float, float, float, float],
    offset: tuple[float, float, float],
) -> tuple[float, float, float]:
    qw, qx, qy, qz = quat_wxyz
    ox, oy, oz = offset
    tx = 2.0 * (qy * oz - qz * oy)
    ty = 2.0 * (qz * ox - qx * oz)
    tz = 2.0 * (qx * oy - qy * ox)
    rx = ox + qw * tx + (qy * tz - qz * ty)
    ry = oy + qw * ty + (qz * tx - qx * tz)
    rz = oz + qw * tz + (qx * ty - qy * tx)
    return (origin[0] + rx, origin[1] + ry, origin[2] + rz)


def _create_side_grasp_cues(
    stage,
    prim_root: str,
    size: tuple[float, float, float],
    translation: tuple[float, float, float],
    orientation_wxyz: tuple[float, float, float, float],
    material,
) -> None:
    """Small pads on the book faces where a two-sided grasp contacts it."""
    book_l, book_h, book_t = size
    for name, zsgn in (("NearFace", 1.0), ("FarFace", -1.0)):
        pad_pos = _local_to_world(translation, orientation_wxyz, (-0.030, 0.0, zsgn * (0.5 * book_t + 0.004)))
        _create_visual_cuboid(
            stage,
            f"{prim_root}/{name}",
            (book_l * 0.34, book_h * 0.42, 0.006),
            pad_pos,
            orientation_wxyz,
            material,
        )


def _create_world_top_page_cap(
    stage,
    prim_path: str,
    size: tuple[float, float, float],
    translation: tuple[float, float, float],
    orientation_wxyz: tuple[float, float, float, float],
    material,
    thickness: float = 0.0012,
) -> None:
    """Add a white page cap on the visible world-up edge of a posed standing book."""
    book_l, book_h, book_t = size
    local_top_candidates = ((0.0, 0.5 * book_h, 0.0), (0.0, -0.5 * book_h, 0.0))
    top_offset = max(local_top_candidates, key=lambda offset: _local_to_world((0.0, 0.0, 0.0), orientation_wxyz, offset)[2])
    cap_center = _local_to_world(
        translation,
        orientation_wxyz,
        (
            top_offset[0],
            top_offset[1] + math.copysign(0.5 * thickness, top_offset[1]),
            top_offset[2],
        ),
    )
    _create_visual_cuboid(
        stage,
        prim_path,
        (book_l * 0.98, thickness, book_t * 0.92),
        cap_center,
        orientation_wxyz,
        material,
    )


def _current_book_pose_w(uw) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    pos = uw.book.data.root_link_pos_w[0].detach().cpu().tolist()
    quat = uw.book.data.root_link_quat_w[0].detach().cpu().tolist()
    return (float(pos[0]), float(pos[1]), float(pos[2])), (
        float(quat[0]),
        float(quat[1]),
        float(quat[2]),
        float(quat[3]),
    )


def _create_curve(
    stage,
    prim_path: str,
    points: list[tuple[float, float, float]],
    width: float,
    material,
) -> None:
    curve = UsdGeom.BasisCurves.Define(stage, prim_path)
    curve.CreateTypeAttr("linear")
    curve.CreateCurveVertexCountsAttr([len(points)])
    curve.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
    curve.CreateWidthsAttr([width for _ in points])
    UsdShade.MaterialBindingAPI(curve.GetPrim()).Bind(material)


def _create_frame_axes(
    stage,
    prim_root: str,
    origin: tuple[float, float, float],
    quat_wxyz: tuple[float, float, float, float],
    x_material,
    y_material,
    z_material,
    length: float = 0.075,
    width: float = 0.006,
) -> None:
    """Create RGB local frame axes: X red, Y green, Z blue."""
    axes = (
        ("X", (length, 0.0, 0.0), x_material),
        ("Y", (0.0, length, 0.0), y_material),
        ("Z", (0.0, 0.0, length), z_material),
    )
    for name, offset, material in axes:
        end = _local_to_world(origin, quat_wxyz, offset)
        curve = UsdGeom.BasisCurves.Define(stage, f"{prim_root}/{name}")
        curve.CreateTypeAttr("linear")
        curve.CreateCurveVertexCountsAttr([2])
        curve.CreatePointsAttr([Gf.Vec3f(*origin), Gf.Vec3f(*end)])
        curve.CreateWidthsAttr([width, width])
        UsdShade.MaterialBindingAPI(curve.GetPrim()).Bind(material)


def _normalize_vec3(v: tuple[float, float, float]) -> tuple[float, float, float]:
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if n < 1e-9:
        return (0.0, 0.0, 0.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def _cross_vec3(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _create_camera_looking_at(
    stage,
    prim_path: str,
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
) -> None:
    """Create a real USD camera prim looking at target."""
    camera = UsdGeom.Camera.Define(stage, prim_path)
    camera.CreateFocalLengthAttr(20.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))

    forward = _normalize_vec3((target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]))
    world_up = (0.0, 0.0, 1.0)
    right = _normalize_vec3(_cross_vec3(forward, world_up))
    up = _normalize_vec3(_cross_vec3(right, forward))
    back = (-forward[0], -forward[1], -forward[2])

    matrix = Gf.Matrix4d(
        right[0], right[1], right[2], 0.0,
        up[0], up[1], up[2], 0.0,
        back[0], back[1], back[2], 0.0,
        eye[0], eye[1], eye[2], 1.0,
    )
    xform = UsdGeom.Xformable(camera.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(matrix)


def _set_active_viewport_camera(camera_path: str) -> None:
    """Point the active viewport at a camera prim when Kit viewport APIs are available."""
    try:
        from omni.kit.viewport.utility import get_active_viewport

        viewport = get_active_viewport()
        if viewport is not None:
            viewport.camera_path = camera_path
    except Exception as exc:
        print(f"[WARN] Could not set active viewport camera to {camera_path}: {exc}")


def _capture_viewport_png(
    camera_path: str,
    output_dir: Path,
    file_name: str,
    width: int,
    height: int,
    warmup_updates: int = 24,
    timeout_updates: int = 900,
) -> list[str]:
    """Capture one PNG from the active viewport camera and return output paths."""
    output_dir.mkdir(parents=True, exist_ok=True)

    import omni.kit.app

    app = omni.kit.app.get_app()
    try:
        app.get_extension_manager().set_extension_enabled_immediate("omni.kit.capture.viewport", True)
    except Exception as exc:
        print(f"[WARN] Could not explicitly enable viewport capture extension: {exc}")

    from omni.kit.capture.viewport import CaptureExtension, CaptureOptions, CaptureRenderPreset
    from omni.kit.viewport.utility import get_active_viewport

    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("No active viewport is available for screenshot capture.")
    viewport.camera_path = camera_path

    for _ in range(max(0, warmup_updates)):
        simulation_app.update()

    capture = CaptureExtension.get_instance()
    capture.show_default_progress_window = False

    options = CaptureOptions()
    options.camera = camera_path
    options.output_folder = str(output_dir)
    options.file_name = file_name
    options.file_type = ".png"
    options.res_width = int(width)
    options.res_height = int(height)
    options.render_preset = CaptureRenderPreset.RAY_TRACE
    options.hdr_output = False
    options.overwrite_existing_frames = True
    options.rt_wait_for_render_resolve_in_seconds = 1
    capture.options = options

    existing_outputs = {p.resolve() for p in output_dir.glob(f"{file_name}*.png")}
    if not capture.start():
        raise RuntimeError("Viewport capture could not start. Check that the Isaac viewport is active.")

    updates = 0
    while not capture.done and updates < timeout_updates:
        simulation_app.update()
        updates += 1
        candidates = [
            p.resolve()
            for p in sorted(output_dir.glob(f"{file_name}*.png"))
            if p.resolve() not in existing_outputs and p.stat().st_size > 0
        ]
        if candidates:
            for _ in range(12):
                simulation_app.update()
            return [str(p) for p in candidates]

    if not capture.done:
        capture.cancel()
        raise TimeoutError(f"Viewport capture did not finish after {timeout_updates} updates.")

    outputs = capture.get_outputs()
    for _ in range(120):
        if outputs and all(Path(p).is_file() for p in outputs):
            break
        simulation_app.update()

    return outputs


def _configure_fast_rendering() -> None:
    """Use normal interactive rendering for fast solid-color screenshot staging."""
    settings = carb.settings.get_settings()
    for key, value in {
        "/renderer/active": "rtx",
        "/rtx/rendermode": "RayTracedLighting",
        "/rtx/translucency/enabled": False,
        "/rtx/pathtracing/spp": 1,
        "/rtx/pathtracing/totalSpp": 1,
    }.items():
        settings.set(key, value)


def _stage_book_in_gripper(uw) -> None:
    """Keep the target book visually attached to the robot gripper."""
    env_id = torch.tensor([0], device=uw.device, dtype=torch.long)
    if hasattr(uw, "_snap_book_to_measured_grasp"):
        uw._snap_book_to_measured_grasp(env_id)

    if hasattr(uw, "_finger_joint_ids") and len(uw._finger_joint_ids) > 0:
        finger_des = torch.full(
            (1, len(uw._finger_joint_ids)),
            float(uw.cfg.gripper_closed_joint_pos),
            device=uw.device,
            dtype=torch.float32,
        )
        uw.robot.set_joint_position_target(finger_des, joint_ids=uw._finger_joint_ids, env_ids=env_id)


def _q_body_to_hand_grasp(uw) -> torch.Tensor:
    fw, fx, fy, fz = uw.cfg.book_to_hand_quat_franka_axes_wxyz
    return torch.tensor([fw, fx, fy, fz], device=uw.device, dtype=torch.float32).view(1, 4)


def _grasp_mid_from_hand_body_hand(uw) -> torch.Tensor:
    env_id = torch.tensor([0], device=uw.device, dtype=torch.long)
    grasp_pos_w, _ = uw._grasp_frame_pose_w(env_id)
    hand_pos_w = uw.robot.data.body_pos_w[env_id, uw._ee_body_idx]
    hand_quat_w = uw.robot.data.body_quat_w[env_id, uw._ee_body_idx]
    return math_utils.quat_apply(math_utils.quat_conjugate(hand_quat_w), grasp_pos_w - hand_pos_w)


def _move_robot_hand_pose_to(
    uw,
    hand_pos_env: torch.Tensor,
    hand_quat_w: torch.Tensor,
    steps: int = 80,
) -> None:
    """Move the real robot hand body to a full pose target."""
    hand_pos_w = hand_pos_env + uw.scene.env_origins[0:1]
    hand_pos_b, hand_quat_b = math_utils.subtract_frame_transforms(
        uw.robot.data.root_pos_w,
        uw.robot.data.root_quat_w,
        hand_pos_w,
        hand_quat_w,
    )

    uw._ik_cmd[:, 0:3] = hand_pos_b
    uw._ik_cmd[:, 3:7] = hand_quat_b
    uw._ik.set_command(uw._ik_cmd)

    for _ in range(max(1, steps)):
        ee_pos_b, ee_quat_b = uw._ee_pose_in_base()
        jacobian = uw.robot.root_physx_view.get_jacobians()[:, uw._jacobi_body_idx, :, uw._jacobi_joint_ids]
        current_arm_pos = uw.robot.data.joint_pos[:, uw._arm_joint_ids]
        joint_pos_des = uw._ik.compute(ee_pos_b, ee_quat_b, jacobian, current_arm_pos)
        joint_vel = uw.robot.data.joint_vel.clone()
        joint_pos = uw.robot.data.joint_pos.clone()
        joint_pos[:, uw._arm_joint_ids] = joint_pos_des
        joint_vel[:, uw._arm_joint_ids] = 0.0

        if hasattr(uw, "_finger_joint_ids") and len(uw._finger_joint_ids) > 0:
            joint_pos[:, uw._finger_joint_ids] = float(uw.cfg.gripper_closed_joint_pos)
            joint_vel[:, uw._finger_joint_ids] = 0.0

        uw.robot.write_joint_state_to_sim(joint_pos, joint_vel)
        uw.robot.set_joint_position_target(joint_pos)
        uw.scene.write_data_to_sim()
        uw.sim.step(render=False)
        uw.scene.update(dt=uw.physics_dt)


def _hand_pose_for_book_grasp(
    uw,
    book_pos_env: tuple[float, float, float],
    book_quat_wxyz: tuple[float, float, float, float],
    reverse_side: bool = False,
    grasp_side: str = "default",
    clearance_m: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return panda_hand and finger-midpoint poses for the environment's book-in-hand grasp mapping."""
    book_pos = torch.tensor(book_pos_env, device=uw.device, dtype=torch.float32).view(1, 3)
    book_quat = torch.tensor(book_quat_wxyz, device=uw.device, dtype=torch.float32).view(1, 4)
    if grasp_side == "top":
        q_hand = torch.tensor(
            _top_down_hand_quat_for_book(book_quat_wxyz),
            device=uw.device,
            dtype=torch.float32,
        ).view(1, 4)
    else:
        q_b2h = _q_body_to_hand_grasp(uw)
        q_hand = math_utils.quat_mul(book_quat, math_utils.quat_inv(q_b2h))
        grasp_rot = None
        if grasp_side == "page":
            grasp_rot = _local_y_quat(math.pi)
        elif grasp_side == "bottom":
            grasp_rot = _local_y_quat(0.5 * math.pi)
        elif reverse_side:
            grasp_rot = _local_y_quat(math.pi)
        elif grasp_side != "default":
            raise ValueError(f"Unknown grasp_side: {grasp_side}")

        if grasp_rot is not None:
            q_grasp = torch.tensor(grasp_rot, device=uw.device, dtype=torch.float32).view(1, 4)
            q_hand = math_utils.quat_mul(q_hand, q_grasp)

    hand_offset = torch.tensor(uw.cfg.book_grasp_offset_hand, device=uw.device, dtype=torch.float32).view(1, 3)
    if grasp_side == "top":
        hand_offset[:, :] = 0.0
        hand_offset[:, 2] = 0.17
    elif grasp_side == "bottom":
        book_h = float(uw.cfg.book_size[1])
        hand_offset[:, :] = 0.0
        hand_offset[:, 2] = 0.5 * book_h + float(clearance_m)
    else:
        hand_offset[:, 2] += float(clearance_m)
    grasp_mid_from_hand = _grasp_mid_from_hand_body_hand(uw)
    grasp_pos = book_pos - math_utils.quat_apply(q_hand, hand_offset)
    hand_pos = grasp_pos - math_utils.quat_apply(q_hand, grasp_mid_from_hand)
    return hand_pos, q_hand, grasp_pos, q_hand


def _move_robot_to_grasp_book(
    uw,
    book_pos_env: tuple[float, float, float],
    book_quat_wxyz: tuple[float, float, float, float],
    reverse_side: bool = False,
    grasp_side: str = "default",
    clearance_m: float = 0.0,
    steps: int = 80,
) -> None:
    """Move panda_hand to the inverse of the environment's book-in-hand grasp mapping."""
    hand_pos, q_hand, _, _ = _hand_pose_for_book_grasp(
        uw,
        book_pos_env,
        book_quat_wxyz,
        reverse_side=reverse_side,
        grasp_side=grasp_side,
        clearance_m=clearance_m,
    )
    _move_robot_hand_pose_to(uw, hand_pos, q_hand, steps=steps)


def _failed_case_grasp_side(case_id: int) -> str:
    if case_id == 3:
        return "top"
    if case_id == 4:
        return "bottom"
    return "page"


def _stage_presentation_scene(env, case_id: int, create_visuals: bool = True) -> str:
    """Stage one screenshot case for the 3MT placement/grasp story."""
    uw = env.unwrapped
    cfg = uw.cfg
    stage = get_current_stage()
    _configure_fast_rendering()

    book_l, book_h, book_t = [float(v) for v in cfg.book_size]
    visual_book_size = (book_l, book_h, max(book_t, 0.026))
    slot_x_open = float(cfg.slot_x_open)
    slot_x_back = float(cfg.slot_x_back)
    slot_center_y = float(cfg.slot_center_y)
    neighbor_book_size = getattr(cfg, "neighbor_book_size", cfg.book_size)
    neighbor_book_t = float(neighbor_book_size[2])
    neighbor_pitch_y = neighbor_book_t + float(getattr(cfg, "neighbor_book_pitch_gap", 0.0))
    left_neighbor_y = slot_center_y - neighbor_book_t - 0.5 * float(cfg.slot_lateral_clearance)
    right_neighbor_y = slot_center_y + neighbor_book_t + 0.5 * float(cfg.slot_lateral_clearance)
    right_slot_center_y = right_neighbor_y + PRESENTATION_SECOND_SLOT_EXTRA_INDEX * neighbor_pitch_y
    visual_left_slot_center_y = right_slot_center_y
    visual_right_slot_center_y = slot_center_y
    side_book_sections_y = [
        [left_neighbor_y - i * neighbor_pitch_y for i in range(PRESENTATION_VISIBLE_BOOKS_PER_SECTION - 1, -1, -1)],
        [right_neighbor_y + i * neighbor_pitch_y for i in range(PRESENTATION_VISIBLE_BOOKS_PER_SECTION)],
        [
            right_neighbor_y + i * neighbor_pitch_y
            for i in range(
                PRESENTATION_SECOND_SLOT_EXTRA_INDEX + 1,
                PRESENTATION_SECOND_SLOT_EXTRA_INDEX
                + 1
                + PRESENTATION_VISIBLE_BOOKS_PER_SECTION
                + PRESENTATION_EXTRA_RIGHT_VISUAL_BOOKS,
            )
        ],
    ]
    z_center = float(cfg.shelf_top_z + cfg.shelf_thickness + 0.5 * book_h)
    shelf_center_x = 0.5 * (slot_x_open + slot_x_back)
    standing_quat = (
        float(cfg.book_standing_quat[0]),
        float(cfg.book_standing_quat[1]),
        float(cfg.book_standing_quat[2]),
        float(cfg.book_standing_quat[3]),
    )
    start_pos_x = slot_x_open - 0.145
    start_pos = (start_pos_x, slot_center_y, z_center)
    story_start_pos = (slot_x_open - 0.205, slot_center_y - 0.012, z_center)
    start_quat = standing_quat
    # Partially inserted: book center sits 35% of slot depth from the opening so it visually
    # sticks out of the shelf slot without reaching the back.
    success_pos_x = slot_x_open + 0.1 * (slot_x_back - slot_x_open)
    success_pos = (success_pos_x, visual_left_slot_center_y, z_center)
    success_quat = standing_quat
    failed_yaw = math.radians(35.0)
    failed_quat = _quat_mul(_world_yaw_quat(failed_yaw), standing_quat)
    failed_visual_quat = _quat_mul(failed_quat, _local_z_quat(math.pi))
    story_failed_pos = (slot_x_open - 0.025, visual_right_slot_center_y - 0.070, z_center + 0.010)
    failed_pos = story_failed_pos

    _stage_book_in_gripper(uw)
    if create_visuals and case_id == 0:
        _move_robot_to_grasp_book(uw, start_pos, start_quat, clearance_m=GRASP_CLEARANCE_M)
    elif create_visuals and case_id == 1:
        _move_robot_to_grasp_book(uw, success_pos, success_quat, clearance_m=GRASP_CLEARANCE_M)
    elif create_visuals and case_id in (2, 3, 4):
        _move_robot_to_grasp_book(
            uw,
            failed_pos,
            failed_visual_quat,
            grasp_side=_failed_case_grasp_side(case_id),
            clearance_m=GRASP_CLEARANCE_M,
        )

    if create_visuals:
        base = "/World/envs/env_0/Bookshelf"
        materials = "/World/Looks/ThreeMT"
        mat_blue = _make_preview_material(stage, f"{materials}/grasped_blue", (0.02, 0.16, 0.90), 1.0)
        mat_blue_spine = _make_preview_material(stage, f"{materials}/grasped_blue_spine", (0.00, 0.05, 0.38), 1.0)
        mat_pages = _make_preview_material(stage, f"{materials}/warm_pages", (1.0, 0.98, 0.90), 1.0)
        mat_book_label = _make_preview_material(stage, f"{materials}/book_label", (0.94, 0.92, 0.78), 1.0)
        mat_green = _make_preview_material(stage, f"{materials}/success_green", (0.0, 0.78, 0.20), 1.0)
        mat_green_spine = _make_preview_material(stage, f"{materials}/success_green_spine", (0.0, 0.34, 0.10), 1.0)
        mat_red = _make_preview_material(stage, f"{materials}/failed_red", (0.95, 0.02, 0.00), 1.0)
        mat_red_spine = _make_preview_material(stage, f"{materials}/failed_red_spine", (0.42, 0.0, 0.0), 1.0)
        mat_green_ghost = _make_preview_material(stage, f"{materials}/success_green_ghost", (0.34, 0.95, 0.50), 1.0)
        mat_green_ghost_spine = _make_preview_material(
            stage, f"{materials}/success_green_ghost_spine", (0.10, 0.58, 0.20), 1.0
        )
        mat_red_ghost = _make_preview_material(stage, f"{materials}/failed_red_ghost", (1.0, 0.34, 0.28), 1.0)
        mat_red_ghost_spine = _make_preview_material(
            stage, f"{materials}/failed_red_ghost_spine", (0.72, 0.06, 0.03), 1.0
        )
        mat_pages_ghost = _make_preview_material(stage, f"{materials}/warm_pages_ghost", (1.0, 0.99, 0.92), 1.0)
        mat_path_blue = _make_preview_material(stage, f"{materials}/path_blue", (0.05, 0.45, 1.0), 0.55)
        mat_path_red = _make_preview_material(stage, f"{materials}/path_red", (1.0, 0.05, 0.02), 0.35)
        mat_wood = _make_preview_material(stage, f"{materials}/quiet_wood", (0.50, 0.42, 0.32), 1.0)
        mat_axis_x = _make_preview_material(stage, f"{materials}/axis_x", (1.0, 0.0, 0.0), 1.0)
        mat_axis_y = _make_preview_material(stage, f"{materials}/axis_y", (0.0, 0.85, 0.0), 1.0)
        mat_axis_z = _make_preview_material(stage, f"{materials}/axis_z", (0.0, 0.25, 1.0), 1.0)

        _hide_presentation_gap_books(stage, base)
        _hide_presentation_neighbor_visuals(stage, base)
        _create_randomized_side_book_visuals(
            stage,
            "/World/ThreeMT/RandomizedShelfBooks",
            f"{materials}/RandomizedShelfBooks",
            side_book_sections_y,
            shelf_center_x,
            float(cfg.shelf_top_z),
            float(cfg.shelf_thickness),
            standing_quat,
            book_l,
            book_h,
            neighbor_book_t,
            neighbor_pitch_y,
            mat_pages,
            mat_book_label,
        )

        for prim_path in (f"{base}/Shelf", f"{base}/BackPanel"):
            _bind_material(stage, prim_path, mat_wood)

        _hide_imageable_tree(stage, "/World/envs/env_0/Book")

        grasped_pos, grasped_quat = _current_book_pose_w(uw)
        frame_pos = None
        frame_quat = None
        if case_id == 0:
            blue_pos = start_pos
            _create_book_visual(
                stage,
                "/World/ThreeMT/StartBookBlue",
                visual_book_size,
                blue_pos,
                start_quat,
                mat_blue,
                mat_pages,
                mat_blue_spine,
                mat_book_label,
                mat_book_label,
            )
            _, _, grasp_pos, grasp_quat = _hand_pose_for_book_grasp(
                uw, blue_pos, start_quat, clearance_m=GRASP_CLEARANCE_M
            )
            frame_pos = tuple(float(v) for v in grasp_pos[0].detach().cpu().tolist())
            frame_quat = tuple(float(v) for v in grasp_quat[0].detach().cpu().tolist())
        elif case_id == 1:
            _create_book_visual(
                stage,
                "/World/ThreeMT/PreReleaseBookGreen",
                visual_book_size,
                success_pos,
                success_quat,
                mat_green,
                mat_pages,
                mat_green_spine,
                mat_book_label,
                mat_book_label,
            )
            _, _, grasp_pos, grasp_quat = _hand_pose_for_book_grasp(
                uw, success_pos, success_quat, clearance_m=GRASP_CLEARANCE_M
            )
            frame_pos = tuple(float(v) for v in grasp_pos[0].detach().cpu().tolist())
            frame_quat = tuple(float(v) for v in grasp_quat[0].detach().cpu().tolist())
        elif case_id in (2, 3, 4):
            failed_name = {
                2: "FailedBookRedReversed",
                3: "FailedBookRedTopGrasp",
                4: "FailedBookRedBottomGrasp",
            }[case_id]
            _create_book_visual(
                stage,
                f"/World/ThreeMT/{failed_name}",
                visual_book_size,
                failed_pos,
                failed_visual_quat,
                mat_red,
                mat_pages,
                mat_red_spine,
                mat_book_label,
                mat_book_label,
            )
            _create_world_top_page_cap(
                stage,
                f"/World/ThreeMT/{failed_name}/VisibleTopPages",
                visual_book_size,
                failed_pos,
                failed_visual_quat,
                mat_pages,
            )
            _, _, grasp_pos, grasp_quat = _hand_pose_for_book_grasp(
                uw,
                failed_pos,
                failed_visual_quat,
                grasp_side=_failed_case_grasp_side(case_id),
                clearance_m=GRASP_CLEARANCE_M,
            )
            frame_pos = tuple(float(v) for v in grasp_pos[0].detach().cpu().tolist())
            frame_quat = tuple(float(v) for v in grasp_quat[0].detach().cpu().tolist())

        if args_cli.grasp_frame and frame_pos is not None and frame_quat is not None:
            print(
                "[INFO] Grasp frame target "
                f"pos=({frame_pos[0]:.4f}, {frame_pos[1]:.4f}, {frame_pos[2]:.4f}) "
                f"quat_wxyz=({frame_quat[0]:.4f}, {frame_quat[1]:.4f}, {frame_quat[2]:.4f}, {frame_quat[3]:.4f})"
            )
            _create_frame_axes(
                stage,
                "/World/ThreeMT/GraspTargetFrame",
                frame_pos,
                frame_quat,
                mat_axis_x,
                mat_axis_y,
                mat_axis_z,
            )

        if args_cli.paths:
            path_z = z_center + 0.035
            _create_curve(
                stage,
                "/World/ThreeMT/SuccessfulApproachPath",
                [
                    (slot_x_open - 0.22, slot_center_y - 0.08, path_z + 0.08),
                    (slot_x_open - 0.13, slot_center_y - 0.04, path_z + 0.04),
                    (slot_x_open - 0.05, slot_center_y - 0.015, path_z + 0.015),
                    (shelf_center_x - 0.02, slot_center_y, path_z),
                ],
                0.006,
                mat_path_blue,
            )
            _create_curve(
                stage,
                "/World/ThreeMT/FailedApproachPath",
                [
                    (slot_x_open - 0.20, slot_center_y + 0.10, path_z + 0.07),
                    (slot_x_open - 0.11, slot_center_y + 0.06, path_z + 0.04),
                    (slot_x_open - 0.03, slot_center_y + 0.035, path_z + 0.015),
                    (shelf_center_x - 0.025, slot_center_y + 0.03, path_z),
                ],
                0.005,
                mat_path_red,
            )

        if case_id in (2, 3, 4):
            _show_only_failed_case_robot_hand(stage)

        # A presentation-friendly 45-degree front view, slightly above the shelf.
        camera_eye = (slot_x_open - 0.76, -0.48, z_center + 0.30)
        camera_target = (slot_x_open - 0.02, slot_center_y + 0.09, z_center + 0.025)
        camera_path = "/World/ThreeMT/PresentationCamera"
        _create_camera_looking_at(stage, camera_path, camera_eye, camera_target)
        _set_active_viewport_camera(camera_path)
        if hasattr(uw.sim, "set_camera_view"):
            uw.sim.set_camera_view(
                eye=camera_eye,
                target=camera_target,
            )
        return camera_path
    return ""


def main():
    case_id = args_cli.case_opt if args_cli.case_opt is not None else args_cli.case
    if case_id is None:
        case_id = 0

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 2.0
    env_cfg.episode_length_s = 1_000_000.0
    env_cfg.book_grasp_x_jitter = 0.0
    env_cfg.book_grasp_y_jitter = 0.0
    env_cfg.book_grasp_yaw_jitter = 0.0
    if hasattr(env_cfg, "shelf_extra_books_per_side"):
        env_cfg.shelf_extra_books_per_side = PRESENTATION_SHELF_EXTRA_BOOKS_PER_SIDE
    if case_id == 3:
        env_cfg.robot.init_state.pos = CASE3_TOP_ROBOT_BASE_POS
    elif case_id in (2, 4):
        env_cfg.robot.init_state.pos = CASE2_ROBOT_BASE_POS

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    camera_path = _stage_presentation_scene(env, case_id)
    if args_cli.manual_drag:
        _enter_manual_drag_mode(case_id)

    case_names = {
        0: "start: blue book grasped by the robot",
        1: "success: green book grasped near the slot before release",
        2: "failed: red book grasped reversed, with the page-opening side inside the gripper",
        3: "failed: red book in the same wrong pose, grasped from the top",
        4: "failed: red book in the same wrong pose, grasped from the bottom",
    }
    print(f"[INFO] 3MT screenshot case {case_id} staged ({case_names[case_id]}).")

    if args_cli.screenshot_dir and not args_cli.manual_drag:
        output_dir = Path(args_cli.screenshot_dir).expanduser().resolve()
        output_name = f"3mt_case_{case_id}"
        outputs = _capture_viewport_png(
            camera_path,
            output_dir,
            output_name,
            args_cli.screenshot_width,
            args_cli.screenshot_height,
        )
        for output in outputs:
            print(f"[INFO] Saved screenshot: {output}")
        sys.stdout.flush()
        sys.stderr.flush()
        import os

        os._exit(0)

    print("[INFO] Capture this case, then rerun with 0, 1, 2, 3, or 4 for the other source images.")
    print("[INFO] Close the Isaac Sim window to exit.")

    while simulation_app.is_running():
        simulation_app.update()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
