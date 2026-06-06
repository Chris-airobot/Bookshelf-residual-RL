#!/usr/bin/env python3
"""Standalone visual test for a presentation-style book asset."""

import argparse
import math

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Preview a simple generated book visual.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.sim.utils.stage import get_current_stage
from pxr import Gf, Sdf, UsdGeom, UsdShade


def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _world_yaw_quat(yaw_rad):
    return (math.cos(0.5 * yaw_rad), 0.0, 0.0, math.sin(0.5 * yaw_rad))


def _quat_apply(quat_wxyz, offset):
    qw, qx, qy, qz = quat_wxyz
    ox, oy, oz = offset
    tx = 2.0 * (qy * oz - qz * oy)
    ty = 2.0 * (qz * ox - qx * oz)
    tz = 2.0 * (qx * oy - qy * ox)
    return (
        ox + qw * tx + (qy * tz - qz * ty),
        oy + qw * ty + (qz * tx - qx * tz),
        oz + qw * tz + (qx * ty - qy * tx),
    )


def _local_to_world(origin, quat_wxyz, offset):
    rx, ry, rz = _quat_apply(quat_wxyz, offset)
    return (origin[0] + rx, origin[1] + ry, origin[2] + rz)


def _make_material(stage, path, color, roughness=0.62):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _set_xform(prim, translation, orientation_wxyz, scale):
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
    qw, qx, qy, qz = orientation_wxyz
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(qw, Gf.Vec3f(qx, qy, qz)))
    xform.AddScaleOp().Set(Gf.Vec3f(*scale))


def _cuboid(stage, path, size, translation, orientation_wxyz, material):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    _set_xform(cube.GetPrim(), translation, orientation_wxyz, size)
    UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(material)


def create_book(stage, root, center=(0.0, 0.0, 0.13), yaw_deg=-18.0):
    """Create one book with clear cover, opening/page edge, and spine/back."""
    size = (0.152, 0.229, 0.026)
    width, height, thickness = size
    quat = _quat_mul(_world_yaw_quat(math.radians(yaw_deg)), (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0))

    cover = _make_material(stage, "/World/Looks/Book/CoverBlue", (0.02, 0.16, 0.90))
    cover_dark = _make_material(stage, "/World/Looks/Book/CoverDarkBlue", (0.00, 0.05, 0.38))
    pages = _make_material(stage, "/World/Looks/Book/Pages", (1.0, 0.98, 0.90))
    title = _make_material(stage, "/World/Looks/Book/CoverTitle", (0.94, 0.92, 0.78))
    label = _make_material(stage, "/World/Looks/Book/SpineLabel", (0.95, 0.92, 0.78))

    detail_thick = 0.0012

    # Main book body stays a single clean cuboid. Details are shallow surface patches.
    _cuboid(stage, f"{root}/BookBody", (width, height, thickness), center, quat, cover)

    # A real book has a spine on one long edge (-X) and an opening/page edge on the opposite edge (+X).
    spine_width = 0.010
    face_overlay_thick = 0.001
    spine_center = _local_to_world(center, quat, (-0.5 * width + 0.5 * spine_width, 0.0, 0.0))
    fore_edge = _local_to_world(center, quat, (0.5 * width + 0.5 * face_overlay_thick, 0.0, 0.0))
    top_edge = _local_to_world(center, quat, (0.0, 0.5 * height + 0.5 * face_overlay_thick, 0.0))
    _cuboid(stage, f"{root}/Spine", (spine_width, height, thickness * 1.002), spine_center, quat, cover_dark)
    _cuboid(stage, f"{root}/ForeEdgePages", (face_overlay_thick, height, thickness * 0.92), fore_edge, quat, pages)
    _cuboid(stage, f"{root}/TopPages", (width * 0.98, face_overlay_thick, thickness * 0.92), top_edge, quat, pages)

    # Minimal cover/title and spine label so the two cover faces read as covers.
    title_center = _local_to_world(center, quat, (0.014, 0.020, 0.5 * thickness + 0.5 * detail_thick))
    _cuboid(stage, f"{root}/FrontTitleBlock", (width * 0.44, height * 0.26, detail_thick), title_center, quat, title)

    spine_label_center = _local_to_world(center, quat, (-0.5 * width + 0.0008, -0.010, 0.0))
    _cuboid(stage, f"{root}/SpineLabel", (detail_thick, height * 0.36, thickness * 0.50), spine_label_center, quat, label)


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0)
    sim = SimulationContext(sim_cfg)
    stage = get_current_stage()

    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/Ground", ground)
    light = sim_utils.DomeLightCfg(intensity=1800.0, color=(0.78, 0.78, 0.78))
    light.func("/World/Light", light)

    create_book(stage, "/World/TestBook")
    sim.set_camera_view(eye=(0.34, -0.42, 0.30), target=(0.0, 0.0, 0.13))
    sim.reset()

    print("[INFO] Book visual test scene loaded.")
    print("[INFO] Inspect: pale page-opening side, dark spine/back side, top page lines.")
    while simulation_app.is_running():
        sim.step()


if __name__ == "__main__":
    main()
    simulation_app.close()
