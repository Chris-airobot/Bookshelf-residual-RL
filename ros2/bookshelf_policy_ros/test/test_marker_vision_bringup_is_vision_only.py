from pathlib import Path


def test_marker_vision_bringup_has_manual_robot_control_but_no_policy_executor():
    source = (
        Path(__file__).parents[1] / "launch" / "marker_vision_bringup.launch.py"
    ).read_text(encoding="utf-8")
    assert "robot_setup.launch.py" in source
    assert '"enable_robot_control"' in source
    assert 'default_value="true"' in source
    assert '"robot_ip"' in source
    assert '"robot_ip": LaunchConfiguration("robot_ip")' in source
    assert '"show_rviz"' in source
    assert '"show_rviz": LaunchConfiguration("show_rviz")' in source
    assert '"enable_rviz": "false"' in source
    assert 'package="rviz2"' not in source
    assert '"enable_calibrated_book_detection"' in source
    assert '"enable_legacy_three_book_detection"' in source
    assert 'default_value="false"' in source
    assert "marker_book_bag_calibration.launch.py" in source
    assert '"detected_marker_frame": "target_book_marker"' in source
    assert '"detected_book_frame": "target_book_center"' in source
    for camera_argument in (
        "camera_name",
        "camera_namespace",
        "serial_no",
        "color_profile",
        "depth_profile",
        "align_depth",
        "enable_sync",
        "enable_pointcloud",
    ):
        assert f'"{camera_argument}"' in source
    forbidden = (
        "guarded_policy_tool_executor",
        "policy_to_robot_node",
        "cartesian_action_executor_node",
        "action_executor_node",
    )
    for token in forbidden:
        assert token not in source


def test_robot_setup_forwards_headless_rviz_option():
    source = (
        Path(__file__).parents[1] / "launch" / "robot_setup.launch.py"
    ).read_text(encoding="utf-8")
    assert "'show_rviz'" in source
    assert "default_value='false'" in source
    assert "'show_rviz':   show_rviz" in source


def test_physical_hardware_bringup_owns_hardware_but_no_policy():
    source = (
        Path(__file__).parents[1]
        / "launch"
        / "physical_hardware_bringup.launch.py"
    ).read_text(encoding="utf-8")
    required = (
        '"robot_ip"',
        '"marker_vision_bringup.launch.py"',
        '"enable_robot_control": "true"',
        '"enable_calibrated_book_detection": "true"',
        '"enable_legacy_three_book_detection": "false"',
        '"show_rviz"',
    )
    for token in required:
        assert token in source
    forbidden = (
        "policy_hardware_shadow",
        "policy_shadow_inference",
        "guarded_policy_tool_executor",
        "guarded_preinsert_executor",
        "policy_to_robot",
    )
    for token in forbidden:
        assert token not in source
