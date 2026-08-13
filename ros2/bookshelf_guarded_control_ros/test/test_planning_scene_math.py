import numpy as np

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
