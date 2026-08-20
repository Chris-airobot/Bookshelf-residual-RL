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
        'executable="policy_observation_adapter"',
        'executable="policy_shadow_inference"',
        'executable="direct_policy_servo"',
        'executable="policy_servo_simulator"',
        "validate_shadow_rehearsal_assets",
        "guarded_policy_tool_overrides",
        '"book_pose_source": "marker"',
        '"slot_pose_source": "configured_static"',
        '"block_on_activation_checks": False',
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
        '"command_target_is_hardware": False',
        'DeclareLaunchArgument("duration_s"',
        "TimerAction",
        "Shutdown",
    )
    for token in required:
        assert token in source
