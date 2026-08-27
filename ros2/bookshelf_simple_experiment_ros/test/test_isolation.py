from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]


def test_runtime_code_does_not_import_existing_deployment_packages():
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (PACKAGE / "bookshelf_simple_experiment_ros").glob("*.py")
    )
    assert "from bookshelf_shadow_ros" not in text
    assert "from bookshelf_guarded_control_ros" not in text
    assert "import bookshelf_shadow_ros" not in text
    assert "import bookshelf_guarded_control_ros" not in text


def test_launch_defaults_to_execution_disabled():
    launch = (PACKAGE / "launch" / "simple_preinsert.launch.py").read_text()
    assert 'DeclareLaunchArgument("allow_execution", default_value="false")' in launch
    assert "/bookshelf_simple/plan_and_execute_preinsert" in launch


def test_virtual_launch_uses_only_official_fake_xarm_and_saved_slot():
    launch = (
        PACKAGE / "launch" / "virtual_saved_slot_preinsert.launch.py"
    ).read_text()
    assert 'FindPackageShare("xarm_moveit_config")' in launch
    assert '"_robot_moveit_fake.launch.py"' in launch
    assert '"execute_virtual", default_value="false"' in launch
    assert 'executable="saved_slot"' in launch
    assert "robot_ip" not in launch
    assert "bookshelf_guarded_control_ros" not in launch
    assert "bookshelf_shadow_ros" not in launch


def test_virtual_launch_sets_far_start_pose_before_planning():
    launch = (
        PACKAGE / "launch" / "virtual_saved_slot_preinsert.launch.py"
    ).read_text()
    trigger = (
        PACKAGE
        / "bookshelf_simple_experiment_ros"
        / "virtual_trigger_node.py"
    ).read_text()
    assert '"initial_joint_positions": [' in launch
    assert "0.4342693425054612" in launch
    assert 'self.declare_parameter("initial_move_duration_s", 2.0)' in trigger
    assert "fake xArm reached the far virtual starting pose" in trigger


def test_virtual_auto_start_plans_without_automatic_execution():
    launch = (
        PACKAGE / "launch" / "virtual_saved_slot_preinsert.launch.py"
    ).read_text()
    trigger = (
        PACKAGE
        / "bookshelf_simple_experiment_ros"
        / "virtual_trigger_node.py"
    ).read_text()
    assert 'Trigger, "/bookshelf_simple/plan_preinsert"' in trigger
    assert '"separate_execution_confirmation": True' in launch
    assert '"allow_execution": ParameterValue(execute_virtual, value_type=bool)' in launch


def test_real_launch_has_frozen_slot_and_two_operator_confirmations():
    launch = (PACKAGE / "launch" / "real_preinsert_workflow.launch.py").read_text()
    assert '"require_slot_acceptance": True' in launch
    assert '"separate_execution_confirmation": True' in launch
    assert '"allow_execution": True' in launch
    assert "/bookshelf_simple/accept_slot" in launch
    assert "/bookshelf_simple/plan_preinsert" in launch
    assert "/bookshelf_simple/execute_preinsert" in launch
    assert "robot_ip" not in launch
    assert "xarm_moveit_config" not in launch


def test_simple_policy_defaults_to_shadow_and_keeps_rosbag_optional():
    launch = (PACKAGE / "launch" / "simple_policy_one_step.launch.py").read_text()
    node = (
        PACKAGE
        / "bookshelf_simple_experiment_ros"
        / "simple_policy_control_node.py"
    ).read_text()
    assert 'DeclareLaunchArgument("execute", default_value="false")' in launch
    assert 'DeclareLaunchArgument("rollout", default_value="false")' in launch
    assert 'DeclareLaunchArgument("max_steps", default_value="150")' in launch
    assert 'DeclareLaunchArgument("translation_tolerance_m", default_value="0.0005")' in launch
    assert '"rotation_tolerance_rad", default_value="0.004363323129985824"' in launch
    assert 'DeclareLaunchArgument("record_bag", default_value="false")' in launch
    assert 'DeclareLaunchArgument("visualization_hold_s", default_value="60.0")' in launch
    assert "if self.execute:" in node
    assert "self.servo_start_client = self.create_client" in node
    assert "release_executed\": False" in node
    assert "GripperCommand" in node


def test_virtual_policy_launch_is_software_only_and_has_two_modes():
    launch = (PACKAGE / "launch" / "virtual_policy_one_step.launch.py").read_text()
    assert 'FindPackageShare("xarm_moveit_config")' in launch
    assert '"_robot_moveit_fake.launch.py"' in launch
    assert '"simple_xarm7_servo_server.launch.py"' in launch
    assert 'condition=IfCondition(execute)' in launch
    assert 'DeclareLaunchArgument("execute", default_value="false")' in launch
    assert 'DeclareLaunchArgument("rollout", default_value="false")' in launch
    assert 'DeclareLaunchArgument("max_steps", default_value="150")' in launch
    assert 'DeclareLaunchArgument("translation_tolerance_m", default_value="0.0005")' in launch
    assert '"rotation_tolerance_rad", default_value="0.004363323129985824"' in launch
    assert 'executable="fake_policy_start"' in launch
    assert "robot_ip" not in launch
    assert "bookshelf_guarded_control_ros" not in launch
    assert "bookshelf_shadow_ros" not in launch
    assert "bookshelf_policy_ros" not in launch


def test_post_insert_uses_real_xarm_gripper_command_action():
    node = (
        PACKAGE
        / "bookshelf_simple_experiment_ros"
        / "simple_policy_control_node.py"
    ).read_text()
    assert "GripperCommand" in node
    assert "FollowJointTrajectory" not in node
    assert '"/xarm_gripper/gripper_action"' in node
    assert 'self.declare_parameter("gripper_max_effort", 0.0)' in node
    assert "release_executed\": False" in node
    assert "bookshelf_guarded_control_ros" not in node


def test_policy_rviz_subscribes_to_every_snapshot_pose():
    rviz = (PACKAGE / "rviz" / "simple_policy_one_step.rviz").read_text()
    for topic in (
        "/bookshelf_simple/policy/markers",
        "/bookshelf_simple/policy/slot_pose",
        "/bookshelf_simple/policy/current_book_pose",
        "/bookshelf_simple/policy/current_tcp_pose",
        "/bookshelf_simple/policy/current_policy_tool_pose",
        "/bookshelf_simple/policy/target_tcp",
        "/bookshelf_simple/policy/target_policy_tool_pose",
    ):
        assert topic in rviz
