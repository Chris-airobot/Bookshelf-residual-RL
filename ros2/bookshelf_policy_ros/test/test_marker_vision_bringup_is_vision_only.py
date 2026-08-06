from pathlib import Path


def test_marker_vision_bringup_has_manual_robot_control_but_no_policy_executor():
    source = (
        Path(__file__).parents[1] / "launch" / "marker_vision_bringup.launch.py"
    ).read_text(encoding="utf-8")
    assert "robot_setup.launch.py" in source
    assert '"enable_robot_control"' in source
    assert 'default_value="true"' in source
    assert '"enable_rviz": "false"' in source
    assert 'package="rviz2"' not in source
    assert '"enable_calibrated_book_detection"' in source
    assert '"enable_legacy_three_book_detection"' in source
    assert 'default_value="false"' in source
    assert "marker_book_bag_calibration.launch.py" in source
    forbidden = (
        "guarded_policy_tool_executor",
        "policy_to_robot_node",
        "cartesian_action_executor_node",
        "action_executor_node",
    )
    for token in forbidden:
        assert token not in source
