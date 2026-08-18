from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCENE_CONFIG = ROOT / "config" / "offline_physical_scene_visualization.yaml"
CHECK_CONFIG = ROOT / "config" / "real_scene_detection_overlay.yaml"
LAUNCH = ROOT / "launch" / "real_scene_detection_overlay.launch.py"
RVIZ = ROOT / "rviz" / "real_scene_detection_overlay.rviz"
VISUALIZER = ROOT / "bookshelf_shadow_ros" / "offline_scene_visualizer_node.py"
CHECK_NODE = ROOT / "bookshelf_shadow_ros" / "static_slot_environment_check_node.py"
DETECTOR = ROOT / "bookshelf_shadow_ros" / "rgbd_slot_detector.py"


def _parameters(path, node_name):
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    return document[node_name]["ros__parameters"]


def test_approved_slot_is_identical_in_scene_and_live_check_configs():
    scene = _parameters(SCENE_CONFIG, "offline_scene_visualizer")
    check = _parameters(CHECK_CONFIG, "static_slot_environment_check")

    assert check["base_frame"] == scene["base_frame"] == "link_base"
    assert check["static_slot_translation_xyz"] == pytest.approx(
        scene["slot_translation_xyz"]
    )
    assert check["static_slot_quaternion_xyzw"] == pytest.approx(
        scene["slot_quaternion_xyzw"]
    )
    assert check["static_slot_width_m"] == pytest.approx(scene["slot_width_m"])
    assert "human_approved" in check["static_slot_transform_status"]


def test_live_check_keeps_fixed_fail_closed_tolerances():
    check = _parameters(CHECK_CONFIG, "static_slot_environment_check")

    assert check["required_matching_samples"] == 30
    assert check["minimum_confidence"] == pytest.approx(0.60)
    assert check["maximum_translation_error_m"] == pytest.approx(0.010)
    assert check["maximum_rotation_error_deg"] == pytest.approx(5.0)
    assert check["maximum_width_error_m"] == pytest.approx(0.005)


def test_replay_hides_stale_reference_and_includes_lower_shelf_edge():
    check = _parameters(CHECK_CONFIG, "static_slot_environment_check")
    check_source = CHECK_NODE.read_text(encoding="utf-8")
    detector_source = DETECTOR.read_text(encoding="utf-8")

    assert check["show_static_reference_markers"] is False
    assert check["anchor_live_slot_to_support_height"] is True
    assert check["support_height_base_m"] == pytest.approx(0.015)
    assert 'self.declare_parameter("show_static_reference_markers", True)' in (
        check_source
    )
    assert 'self.declare_parameter("roi_y_max", 0.98)' in detector_source
    assert 'self.declare_parameter("anchor_live_slot_to_support_height", False)' in (
        check_source
    )


def test_overlay_uses_live_robot_state_and_has_no_execution_interface():
    launch_source = LAUNCH.read_text(encoding="utf-8")
    visualizer_source = VISUALIZER.read_text(encoding="utf-8")
    check_source = CHECK_NODE.read_text(encoding="utf-8")

    assert 'name="offline_scene_visualizer"' in launch_source
    assert 'name="real_coarse_scene_overlay"' not in launch_source
    assert '"publish_joint_states": False' in launch_source
    assert 'package="rqt_image_view"' in launch_source
    assert 'arguments=["/slot_detector/debug_image"]' in launch_source
    assert 'executable="slot_orientation_audit"' in launch_source
    assert 'condition=IfCondition(enable_orientation_audit)' in launch_source
    assert 'DeclareLaunchArgument(\n                "show_debug_image"' in launch_source
    assert "robot_state_publisher" not in launch_source
    assert "xarm_description" not in launch_source
    assert "create_subscription(" in check_source
    assert "create_publisher(" in visualizer_source
    for forbidden in (
        "ActionClient",
        "create_service(",
        "create_client(",
        "execute_trajectory",
        "follow_joint_trajectory",
        "guarded_policy_tool_executor",
    ):
        assert forbidden not in launch_source
        assert forbidden not in visualizer_source
        assert forbidden not in check_source


def test_overlay_scene_configuration_is_explicit_and_fail_closed():
    scene = _parameters(SCENE_CONFIG, "offline_scene_visualizer")
    visualizer_source = VISUALIZER.read_text(encoding="utf-8")

    assert scene["scene_configuration_confirmed"] is True
    assert 'self.declare_parameter("scene_configuration_confirmed", False)' in (
        visualizer_source
    )
    assert "scene_configuration_confirmed must be true" in visualizer_source
    assert "scene YAML was not applied" in visualizer_source


def test_rviz_combines_robot_scene_slot_and_optional_point_cloud():
    source = RVIZ.read_text(encoding="utf-8")

    assert "/robot_description" in source
    assert "/bookshelf_offline_scene/markers" in source
    assert "/bookshelf_environment/slot_markers" in source
    assert "/camera/depth/color/points" in source
    assert "rviz_default_plugins/PointCloud2" in source
