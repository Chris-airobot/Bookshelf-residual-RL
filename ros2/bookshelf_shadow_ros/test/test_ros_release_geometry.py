from pathlib import Path
import struct

import numpy as np
import pytest

from bookshelf_shadow_ros.policy_observation_math import make_transform
from bookshelf_shadow_ros.ros_release_geometry import (
    aabb_distance,
    book_snapshot,
    gripper_collision_snapshot,
    stl_local_bounds,
)


def test_book_snapshot_uses_slot_frame_depth_convention():
    slot = np.eye(4)
    book = make_transform([0.10, 0.0, 0.0])
    snapshot = book_snapshot(book, slot, [0.156, 0.034, 0.236], 0.20)

    assert snapshot["trailing_edge_depth_from_mouth_m"] == pytest.approx(0.022)
    assert snapshot["leading_edge_penetration_from_mouth_m"] == pytest.approx(0.178)
    assert snapshot["front_to_back_remaining_m"] == pytest.approx(0.022)


def test_aabb_distance_is_zero_for_overlap_and_euclidean_for_separation():
    assert aabb_distance([0, 0, 0], [1, 1, 1], [0.5, 0.5, 0.5], [2, 2, 2]) == 0.0
    assert aabb_distance([0, 0, 0], [1, 1, 1], [2, 3, 1], [3, 4, 2]) == pytest.approx(
        np.sqrt(5.0)
    )


def test_binary_stl_bounds_are_read_without_external_mesh_package(tmp_path: Path):
    path = tmp_path / "triangle.stl"
    values = (0.0, 0.0, 1.0, -1.0, 2.0, 3.0, 4.0, -5.0, 6.0, 0.5, 1.0, -2.0)
    path.write_bytes(b"x" * 80 + struct.pack("<I", 1) + struct.pack("<12fH", *values, 0))

    minimum, maximum = stl_local_bounds(path)

    assert minimum.tolist() == pytest.approx([-1.0, -5.0, -2.0])
    assert maximum.tolist() == pytest.approx([4.0, 2.0, 6.0])


def test_gripper_snapshot_reports_closest_proxy_obstacle():
    snapshot = gripper_collision_snapshot(
        transform_base_slot=np.eye(4),
        link_transforms_base={"left_finger": make_transform([-0.01, 0.0, 0.0])},
        mesh_bounds={
            "left_finger": (np.array([-0.005, -0.005, -0.005]), np.array([0.005] * 3))
        },
        slot_depth_m=0.20,
        slot_width_m=0.04,
        slot_height_m=0.25,
    )

    assert snapshot["bodies"][0]["name"] == "left_finger"
    assert snapshot["closest_body_obstacle_pair"]["distance_m"] > 0.0
    assert "official xarm_description" in snapshot["method"]
