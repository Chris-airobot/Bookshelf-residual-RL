from pathlib import Path

import numpy as np
import pytest
import yaml

from bookshelf_shadow_ros.offline_scene_visualization import (
    build_offline_scene_geometry,
    shelf_bottom_height_m,
    shelf_front_plane_error_m,
    table_top_height_m,
    validated_joint_state,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "offline_physical_scene_visualization.yaml"
LAUNCH = ROOT / "launch" / "offline_physical_scene_visualization.launch.py"
XACRO = ROOT / "urdf" / "offline_xarm7_visualization.urdf.xacro"
NODE = (
    ROOT
    / "bookshelf_shadow_ros"
    / "offline_scene_visualizer_node.py"
)


def _parameters():
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    return document["offline_scene_visualizer"]["ros__parameters"]


def _geometry():
    values = _parameters()
    return build_offline_scene_geometry(
        slot_translation_xyz=values["slot_translation_xyz"],
        slot_quaternion_xyzw=values["slot_quaternion_xyzw"],
        slot_width_m=values["slot_width_m"],
        slot_visual_height_m=values["slot_visual_height_m"],
        shelf_size_xyz=values["shelf_size_xyz"],
        shelf_center_offset_slot_xyz=values["shelf_center_offset_slot_xyz"],
        shelf_bottom_height_base_m=values["shelf_bottom_height_base_m"],
        table_size_xyz=values["table_size_xyz"],
        table_center_base_xyz=values["table_center_base_xyz"],
        table_quaternion_base_xyzw=values["table_quaternion_base_xyzw"],
        held_book_size_xyz=values["held_book_size_xyz"],
        preinsert_book_center_slot_xyz=values[
            "preinsert_book_center_slot_xyz"
        ],
        anchor_slot_to_shelf_support_height=values[
            "anchor_slot_to_shelf_support_height"
        ],
    )


def test_default_scene_has_expected_coarse_dimensions_and_planes():
    assert _parameters()["scene_configuration_confirmed"] is True
    geometry = _geometry()

    assert geometry.shelf_size_xyz == pytest.approx((0.30, 0.95, 0.40))
    assert geometry.table_size_xyz == pytest.approx((1.50, 0.60, 0.05))
    assert geometry.transform_base_table[:3, 3] == pytest.approx(
        (0.75, 0.0, -0.025)
    )
    assert geometry.held_book_size_xyz == pytest.approx((0.156, 0.034, 0.236))
    assert geometry.slot_width_m == pytest.approx(0.03970380872488022)
    assert geometry.slot_support_anchored is True
    slot_up = geometry.transform_base_slot[:3, 2]
    slot_lower_edge = (
        geometry.transform_base_slot[:3, 3] - 0.5 * geometry.slot_visual_height_m * slot_up
    )
    assert slot_lower_edge[2] == pytest.approx(0.015, abs=1e-12)
    assert shelf_front_plane_error_m(geometry) == pytest.approx(0.0, abs=1e-12)
    assert shelf_bottom_height_m(geometry) == pytest.approx(0.015, abs=1e-12)
    assert table_top_height_m(geometry) == pytest.approx(0.0, abs=1e-12)


def test_shelf_is_level_and_preserves_the_approved_slot_heading():
    geometry = _geometry()
    slot_heading = geometry.transform_base_slot[:2, 0]
    slot_heading /= np.linalg.norm(slot_heading)

    assert geometry.transform_base_shelf[:2, 0] == pytest.approx(slot_heading)
    assert geometry.transform_base_shelf[:3, 2] == pytest.approx([0.0, 0.0, 1.0])
    assert geometry.transform_base_shelf[2, 3] == pytest.approx(0.215)


def test_intended_joint_state_is_complete_and_unique():
    values = _parameters()
    names, positions = validated_joint_state(
        values["joint_names"], values["joint_positions"]
    )

    assert names[:7] == tuple(f"joint{index}" for index in range(1, 8))
    assert names[-1] == "drive_joint"
    assert len(positions) == 8


def test_invalid_scene_vectors_fail_closed():
    values = _parameters()
    values["shelf_size_xyz"] = [0.30, -0.95, 0.40]

    with pytest.raises(ValueError, match="shelf_size_xyz"):
        build_offline_scene_geometry(
            slot_translation_xyz=values["slot_translation_xyz"],
            slot_quaternion_xyzw=values["slot_quaternion_xyzw"],
            slot_width_m=values["slot_width_m"],
            slot_visual_height_m=values["slot_visual_height_m"],
            shelf_size_xyz=values["shelf_size_xyz"],
            shelf_center_offset_slot_xyz=values[
                "shelf_center_offset_slot_xyz"
            ],
            shelf_bottom_height_base_m=values[
                "shelf_bottom_height_base_m"
            ],
            table_size_xyz=values["table_size_xyz"],
            table_center_base_xyz=values["table_center_base_xyz"],
            table_quaternion_base_xyzw=values[
                "table_quaternion_base_xyzw"
            ],
            held_book_size_xyz=values["held_book_size_xyz"],
            preinsert_book_center_slot_xyz=values[
                "preinsert_book_center_slot_xyz"
            ],
        )


def test_launch_and_node_have_no_motion_or_planning_interface():
    launch_source = LAUNCH.read_text(encoding="utf-8")
    xacro_source = XACRO.read_text(encoding="utf-8")
    node_source = NODE.read_text(encoding="utf-8")

    assert "offline_xarm7_visualization.urdf.xacro" in launch_source
    assert 'package="robot_state_publisher"' in launch_source
    assert "$(find xarm_description)" in xacro_source
    assert 'load_gazebo_plugin="false"' in xacro_source
    assert 'package="bookshelf_shadow_ros"' in launch_source
    assert 'package="rviz2"' in launch_source
    for forbidden in (
        "xarm_api",
        "".join(("xarm_", "planner")),
        "".join(("move_", "group")),
        "ros2_control_node",
        "controller_manager",
        "guarded_policy_tool_executor",
        "".join(("policy_", "to_robot")),
        "".join(("cartesian_", "action_", "executor")),
        "ActionClient",
        "create_client(",
        "create_service(",
    ):
        assert forbidden not in launch_source
        assert forbidden not in xacro_source
        assert forbidden not in node_source

    assert "create_publisher(" in node_source
    assert '"hardware_commanded": False' in node_source
    assert '"execution_authorized": False' in node_source


def test_visualizer_can_show_marker_without_publishing_robot_state():
    node_source = NODE.read_text(encoding="utf-8")
    assert 'self.declare_parameter("show_coarse_bookshelf", True)' in node_source
    assert "if self.show_coarse_bookshelf:" in node_source
    assert 'self.declare_parameter("marker_enabled", False)' in node_source
    assert 'self.declare_parameter("target_book_frame"' in node_source
    assert 'marker.ns = "book_aruco"' in node_source
    assert "if self.marker_enabled:" in node_source
