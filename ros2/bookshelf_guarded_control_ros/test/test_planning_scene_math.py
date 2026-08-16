from pathlib import Path

import numpy as np
import pytest
import yaml

from bookshelf_guarded_control_ros.planning_scene_math import (
    GLOBAL_APPROACH,
    LOCAL_INSERTION,
    global_scene_status_error,
    local_handoff_error,
    scene_status_error,
    shelf_box_from_slot,
    shelf_front_plane_error_m,
)
from bookshelf_guarded_control_ros.policy_tool_control_math import make_transform


ROOT = Path(__file__).resolve().parents[1]
SCENE_CONFIG = ROOT / "config" / "bookshelf_scene_physical.yaml"
OVERLAY_CONFIG = (
    ROOT.parent
    / "bookshelf_shadow_ros"
    / "config"
    / "offline_physical_scene_visualization.yaml"
)


def _parameters(path, node_name):
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    return document[node_name]["ros__parameters"]


def test_shelf_box_front_face_starts_at_slot_mouth():
    transform_base_slot = make_transform(
        [0.80, 0.10, 0.25],
        [0.0, 0.0, 0.0, 1.0],
    )
    box = shelf_box_from_slot(
        transform_base_slot,
        base_frame="link_base",
        size_xyz=[0.20, 0.70, 0.40],
        center_offset_slot_xyz=[0.10, 0.0, 0.0],
    )

    assert box.frame_id == "link_base"
    assert box.size_xyz == (0.20, 0.70, 0.40)
    assert np.allclose(box.transform_frame_box[:3, 3], [0.90, 0.10, 0.25])
    assert shelf_front_plane_error_m(
        box.size_xyz,
        [0.10, 0.0, 0.0],
    ) == 0.0


def test_level_shelf_matches_validated_overlay_geometry():
    scene = _parameters(SCENE_CONFIG, "bookshelf_scene_manager")
    overlay = _parameters(OVERLAY_CONFIG, "offline_scene_visualizer")
    transform_base_slot = make_transform(
        overlay["slot_translation_xyz"],
        overlay["slot_quaternion_xyzw"],
    )

    assert scene["shelf_box_size_xyz"] == overlay["shelf_size_xyz"]
    assert scene["shelf_box_center_offset_slot_xyz"] == (
        overlay["shelf_center_offset_slot_xyz"]
    )
    assert scene["shelf_bottom_height_base_m"] == pytest.approx(
        overlay["shelf_bottom_height_base_m"]
    )
    assert scene["table_box_size_xyz"] == overlay["table_size_xyz"]
    assert scene["table_box_center_base_xyz"] == overlay["table_center_base_xyz"]
    assert scene["table_box_quaternion_base_xyzw"] == (
        overlay["table_quaternion_base_xyzw"]
    )
    assert scene["held_book_size_xyz"] == overlay["held_book_size_xyz"]
    assert scene["held_book_center_tcp_xyz"] == overlay["held_book_center_tcp_xyz"]
    assert scene["held_book_quaternion_tcp_xyzw"] == (
        overlay["held_book_quaternion_tcp_xyzw"]
    )

    box = shelf_box_from_slot(
        transform_base_slot,
        base_frame="link_base",
        size_xyz=scene["shelf_box_size_xyz"],
        center_offset_slot_xyz=scene["shelf_box_center_offset_slot_xyz"],
        level_with_base=scene["shelf_level_with_base"],
        bottom_height_base_m=scene["shelf_bottom_height_base_m"],
    )

    slot_heading = transform_base_slot[:2, 0]
    slot_heading /= np.linalg.norm(slot_heading)
    assert box.transform_frame_box[:2, 0] == pytest.approx(slot_heading)
    assert box.transform_frame_box[:3, 2] == pytest.approx([0.0, 0.0, 1.0])
    assert box.transform_frame_box[2, 3] == pytest.approx(0.215)
    assert shelf_front_plane_error_m(
        box.size_xyz,
        scene["shelf_box_center_offset_slot_xyz"],
    ) == pytest.approx(0.0)


def test_level_shelf_rejects_missing_bottom_height():
    transform_base_slot = make_transform([0.8, 0.1, 0.25])

    with pytest.raises(ValueError, match="shelf_bottom_height_base_m"):
        shelf_box_from_slot(
            transform_base_slot,
            base_frame="link_base",
            size_xyz=[0.30, 0.95, 0.40],
            center_offset_slot_xyz=[0.15, 0.0, 0.0],
            level_with_base=True,
            bottom_height_base_m=None,
        )


def test_local_handoff_is_fail_closed():
    common = dict(
        hardware_measurements_confirmed=True,
        allow_local_insertion=True,
        activation_ready=True,
        activation_fresh=True,
        global_scene_applied=True,
        shelf_front_plane_error=0.0,
        maximum_front_plane_error_m=0.005,
    )
    assert local_handoff_error(**common) is None

    for key in (
        "hardware_measurements_confirmed",
        "allow_local_insertion",
        "activation_ready",
        "activation_fresh",
        "global_scene_applied",
    ):
        blocked = dict(common)
        blocked[key] = False
        assert local_handoff_error(**blocked)

    misaligned = dict(common)
    misaligned["shelf_front_plane_error"] = 0.010
    assert "front face" in local_handoff_error(**misaligned)


def test_runtime_scene_status_requires_local_mode_table_and_book():
    status = {
        "mode": LOCAL_INSERTION,
        "scene_applied": True,
        "hardware_measurements_confirmed": True,
        "held_book_pose_check_passed": True,
        "held_book_pose_check_fresh": True,
        "objects": {
            "bookshelf_keepout": False,
            "table": True,
            "held_book": True,
        },
    }
    assert scene_status_error(status) is None

    global_status = dict(status, mode=GLOBAL_APPROACH)
    assert "expected" in scene_status_error(global_status)

    missing_table = {**status, "objects": {**status["objects"], "table": False}}
    assert "table" in scene_status_error(missing_table)

    shelf_active = {
        **status,
        "objects": {**status["objects"], "bookshelf_keepout": True},
    }
    assert "still active" in scene_status_error(shelf_active)


def test_global_scene_status_requires_keepout_table_and_held_book():
    status = {
        "mode": GLOBAL_APPROACH,
        "scene_applied": True,
        "hardware_measurements_confirmed": True,
        "held_book_pose_check_passed": True,
        "held_book_pose_check_fresh": True,
        "objects": {
            "bookshelf_keepout": True,
            "table": True,
            "held_book": True,
        },
    }
    assert global_scene_status_error(status) is None

    local = dict(status, mode=LOCAL_INSERTION)
    assert "expected" in global_scene_status_error(local)

    no_keepout = {
        **status,
        "objects": {**status["objects"], "bookshelf_keepout": False},
    }
    assert "keep-out" in global_scene_status_error(no_keepout)

    failed_book_check = {**status, "held_book_pose_check_passed": False}
    assert "held-book" in global_scene_status_error(failed_book_check)

    stale_book_check = {**status, "held_book_pose_check_fresh": False}
    assert "stale" in global_scene_status_error(stale_book_check)
