from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / "launch" / "policy_servo_software_simulation.launch.py"
NODE = (
    ROOT
    / "bookshelf_guarded_control_ros"
    / "policy_servo_simulator_node.py"
)


def test_simulation_reuses_real_policy_and_direct_servo_controller():
    source = LAUNCH.read_text(encoding="utf-8")
    required = (
        'executable="policy_servo_simulator"',
        '"physical_policy_deployment.launch.py"',
        '"operation": "control"',
        '"maximum_total_translation_m"',
        "validate_shadow_rehearsal_assets",
        "guarded_policy_tool_overrides",
        '"command_target_is_hardware": "false"',
    )
    for token in required:
        assert token in source


def test_simulation_has_no_hardware_or_planning_interfaces():
    source = LAUNCH.read_text(encoding="utf-8")
    node_source = NODE.read_text(encoding="utf-8")
    combined = source + node_source
    forbidden = (
        "/xarm/",
        "xarm_msgs",
        "MoveCartesian",
        "FollowJointTrajectory",
        "ExecuteTrajectory",
        "GetMotionPlan",
        "ActionClient",
        "move_group",
        "controller_manager",
        "xarm_gripper",
        "realsense2_camera",
    )
    for token in forbidden:
        assert token not in combined

    assert '"hardware_commanded": False' in node_source
    assert '"execution_authorized": False' in node_source


def test_simulation_uses_isolated_frames_topics_and_bounded_shutdown():
    source = LAUNCH.read_text(encoding="utf-8")
    required = (
        '"sim_link_base"',
        '"sim_link_eef"',
        '"sim_link_tcp"',
        '"sim_target_book_center"',
        '"/bookshelf_sim/servo/start"',
        '"/bookshelf_sim/servo/delta_twist_cmds"',
        '"command_target_is_hardware": "false"',
        'default_value="0.005"',
        'DeclareLaunchArgument("duration_s"',
        "TimerAction",
        "Shutdown",
    )
    for token in required:
        assert token in source


def test_simulation_visualizes_reviewed_scene_and_marker_mount():
    launch_source = LAUNCH.read_text(encoding="utf-8")
    node_source = NODE.read_text(encoding="utf-8")
    for token in (
        '"real_book_aruco0_mount.yaml"',
        '"shelf_box_size_xyz"',
        '"table_box_size_xyz"',
        '"book_marker_translation_xyz"',
    ):
        assert token in launch_source
    for token in (
        'name="bookshelf"',
        'name="table"',
        'name="aruco_marker_0"',
        'name="gripper"',
        'name="xarm_base"',
    ):
        assert token in node_source


def test_simulation_separates_pipeline_pass_from_direction_diagnostic():
    node_source = NODE.read_text(encoding="utf-8")
    assert '"bounded_stop_reached": bounded_stop_reached' in node_source
    assert '"forward_progress_check_passed"' in node_source
    assert '"latest_policy_delta"' in node_source
    passed_block = node_source.split("passed = bool(", 1)[1].split(")", 1)[0]
    assert "bounded_stop_reached" in passed_block
    assert "forward_progress >= minimum_progress" not in passed_block
