from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / "launch" / "xarm7_policy_software_simulation.launch.py"
RVIZ = ROOT / "rviz" / "xarm7_policy_software_simulation.rviz"


def test_combined_simulation_uses_official_xarm_fake_moveit_and_servo():
    source = LAUNCH.read_text(encoding="utf-8")
    required = (
        '"_robot_moveit_fake.launch.py"',
        '"xarm7_moveit_servo_server.launch.py"',
        '"use_fake_hardware": "true"',
        '"physical_policy_deployment.launch.py"',
        '"command_target_is_hardware": "false"',
        '"enforce_translation_budget": "false"',
        '"start_servo_service": "/servo_server/start_servo"',
        '"twist_command_topic": "/servo_server/delta_twist_cmds"',
        'executable="fake_pretarget_initializer"',
        'executable="fake_release_retreat_sequence"',
        'executable="experiment_logger"',
        '"require_control_enable": "true"',
        '"/bookshelf_sim/policy_control_enabled"',
        '"yield_when_control_disabled": "true"',
    )
    for token in required:
        assert token in source


def test_combined_simulation_automatically_writes_monitor_logs():
    source = LAUNCH.read_text(encoding="utf-8")
    for token in (
        '"monitor_output_root"',
        '"xarm7_policy_software_simulation"',
        '"SIMULATION MONITOR DIRECTORY:',
        '"run_dir": str(run_dir)',
        '"capture_condition": "official_xarm_fake_hardware"',
    ):
        assert token in source


def test_combined_simulation_has_one_robot_state_source_and_no_robot_api():
    source = LAUNCH.read_text(encoding="utf-8")
    assert '"publish_joint_states": False' in source
    assert '"joint_states_topic": "/joint_states"' in source
    assert "xarm_api" not in source
    assert "UFRobotSystemHardware" not in source
    assert "robot_ip:=192" not in source
    assert '"command_target_is_hardware": "false"' in source


def test_combined_simulation_starts_at_reviewed_pretarget_and_runs_bounded_control():
    source = LAUNCH.read_text(encoding="utf-8")
    for value in (
        "1.2342693425054612",
        "1.5322427671441177",
        "4.904658882462919",
        "1.302429752118059",
        "3.302595179623167",
        "0.6839448116011184",
        "4.4791192150828865",
    ):
        assert value in source
    assert 'default_value="control"' in source
    assert 'default_value="0.30"' in source
    assert '"move_duration_s": 0.5' in source


def test_combined_simulation_runs_release_retreat_and_push_on_fake_hardware():
    source = LAUNCH.read_text(encoding="utf-8")
    required = (
        '"/xarm_gripper_traj_controller/follow_joint_trajectory"',
        '"retreat_direction_base_xyz": retreat_direction',
        '"tcp_frame": "link_tcp"',
        '"book_size_xyz": scene["held_book_size_xyz"]',
        '"scripted_retreat_distance_m"',
        '"scripted_retreat_speed_m_s"',
        '"policy_push_book_distance_m"',
        '"policy_push_timeout_s"',
        '"gripper_open_position": 0.0',
        '"gripper_closed_position": 0.85',
        '"tcp_frame": "target_book_center"',
        '"held_book_center_tcp_xyz": [0.0, 0.0, 0.0]',
    )
    for token in required:
        assert token in source


def test_combined_simulation_can_set_back_physical_grasp_without_changing_policy_tool():
    source = LAUNCH.read_text(encoding="utf-8")
    for token in (
        '"physical_grasp_setback_m"',
        "derive_simulation_grasp_setback",
        '"simulation_grasp_config.yaml"',
        '"simulation_only_derived_config": True',
        '"source_approved_config_sha256"',
        '"approved_config": str(runtime_config)',
        '"initial_grasp_alignment_enabled": grasp_setback_m > 0.0',
        '"servo_already_started": (',
        'executable="ros_release_geometry"',
        '"xarm_release_geometry.json"',
        '"physical_release_guard_enabled"',
        '"physical_release_tcp_x_limit_m"',
        '"minimum_book_leading_penetration_m"',
        '"push_to_target_trailing_depth_enabled": True',
        '"policy_push_target_trailing_depth_m"',
        '"capture_condition": "task_release"',
    ):
        assert token in source


def test_combined_simulation_publishes_reviewed_scene_book_and_marker():
    source = LAUNCH.read_text(encoding="utf-8")
    for token in (
        '"target_book_center"',
        '"real_book_aruco0_mount.yaml"',
        '"marker_enabled": True',
        '"show_coarse_bookshelf": False',
        '"shelf_box_size_xyz"',
        '"held_book_center_tcp_xyz"',
        '"/bookshelf_sim/markers"',
    ):
        assert token in source


def test_combined_rviz_has_moveit_robot_and_bookshelf_scene():
    source = RVIZ.read_text(encoding="utf-8")
    assert "moveit_rviz_plugin/MotionPlanning" in source
    assert "Planning Group: xarm7" in source
    assert "rviz_default_plugins/RobotModel" in source
    assert "/bookshelf_sim/markers" in source
    assert "Fixed Frame: link_base" in source
