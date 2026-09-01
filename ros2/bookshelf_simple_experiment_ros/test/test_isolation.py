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


def test_offline_preinsert_visualization_is_fake_plan_only():
    launch = (
        PACKAGE / "launch" / "offline_preinsert_visualization.launch.py"
    ).read_text()
    assert '"virtual_saved_slot_preinsert.launch.py"' in launch
    assert '"execute_virtual": "false"' in launch
    assert '"show_rviz": "false"' in launch
    assert '"real_preinsert_workflow.rviz"' in launch
    assert 'executable="offline_slot_debug_image"' in launch
    assert "robot_ip" not in launch
    assert "slot_detector" not in launch
    assert 'FindPackageShare("realsense2_camera")' not in launch
    assert 'package="realsense2_camera"' not in launch
    assert 'FindPackageShare("xarm_moveit_servo")' not in launch
    assert 'executable="simple_policy_control"' not in launch


def test_offline_debug_image_is_static_and_publish_only():
    node = (
        PACKAGE
        / "bookshelf_simple_experiment_ros"
        / "offline_slot_debug_image_node.py"
    ).read_text()
    assert '"/slot_detector/debug_image"' in node
    assert "create_publisher" in node
    assert "create_subscription" not in node
    assert "ActionClient" not in node
    assert "create_client" not in node
    assert "OFFLINE PREVIEW" in node


def test_offline_rosbag_preview_uses_real_detector_and_fake_moveit_only():
    launch = (
        PACKAGE / "launch" / "offline_rosbag_preinsert_visualization.launch.py"
    ).read_text()
    assert '"_robot_moveit_fake.launch.py"' in launch
    assert 'executable="slot_detector"' in launch
    assert 'executable="saved_slot"' not in launch
    assert 'executable="simple_preinsert"' in launch
    assert 'DeclareLaunchArgument("allow_execution", default_value="false")' in launch
    assert '"allow_execution": ParameterValue(allow_execution, value_type=bool)' in launch
    assert 'DeclareLaunchArgument("require_slot_acceptance", default_value="false")' in launch
    assert '"require_slot_acceptance": ParameterValue(' in launch
    assert 'executable="virtual_trigger"' in launch
    assert "1.283901572227478" in launch
    assert '"ros2", "bag", "play", bag_path' in launch
    assert '"--loop"' in launch
    assert '"/camera/color/image_raw"' in launch
    assert '"/camera/aligned_depth_to_color/image_raw"' in launch
    assert '"/camera/color/camera_info:=/slot_detector/camera_info"' in launch
    assert '"publish_handeye_camera_link.launch.py"' in launch
    assert '"camera_color_frame", "camera_color_optical_frame"' in launch
    assert "robot_ip" not in launch
    assert "realsense2_camera" not in launch
    assert "moveit_servo" not in launch
    assert 'executable="simple_policy_control"' not in launch
    assert 'executable="offline_slot_debug_image"' not in launch
    assert '"offline_rosbag_preinsert_workflow.rviz"' in launch
    assert '"show_rviz": preview_rviz' in launch
    assert '"rviz_config": rviz_config' in launch
    assert 'package="rviz2"' not in launch

    rviz = (
        PACKAGE / "rviz" / "offline_rosbag_preinsert_workflow.rviz"
    ).read_text()
    assert "Class: moveit_rviz_plugin/MotionPlanning" in rviz
    assert "Class: bookshelf_rviz_image_panel/DebugImagePanel" in rviz
    assert "Class: rviz_default_plugins/Image" not in rviz
    assert "Class: rviz_default_plugins/Camera" not in rviz


def test_real_launch_has_frozen_slot_and_two_operator_confirmations():
    launch = (PACKAGE / "launch" / "real_preinsert_workflow.launch.py").read_text()
    assert '"require_slot_acceptance": True' in launch
    assert '"separate_execution_confirmation": True' in launch
    assert 'DeclareLaunchArgument("allow_execution", default_value="true")' in launch
    assert '"allow_execution": ParameterValue(allow_execution, value_type=bool)' in launch
    assert "/bookshelf_simple/accept_slot" in launch
    assert "/bookshelf_simple/plan_preinsert" in launch
    assert "/bookshelf_simple/execute_preinsert" in launch
    assert "robot_ip" not in launch
    assert "xarm_moveit_config" not in launch


def test_real_preinsert_rviz_restores_slot_detector_debug_image():
    rviz = (PACKAGE / "rviz" / "real_preinsert_workflow.rviz").read_text()
    assert "Class: rviz_default_plugins/RobotModel" in rviz
    assert "Class: moveit_rviz_plugin/MotionPlanning" in rviz
    assert "Name: MotionPlanning" in rviz
    assert "Class: rviz_default_plugins/MarkerArray" in rviz
    assert "Class: rviz_default_plugins/Image" in rviz
    assert "Name: Slot Detector Debug" in rviz
    assert "Topic: /slot_detector/debug_image" in rviz
    assert "Class: rviz_default_plugins/TF" in rviz
    assert "Slot Detector Debug:\n    collapsed: false" in rviz
    assert "Hide Right Dock: false" in rviz
    for existing_topic in (
        "/bookshelf_simple/markers",
        "/bookshelf_simple/target_tcp_pose",
    ):
        assert existing_topic in rviz


def test_experiment_rviz_uses_compact_target_tcp_and_hides_moveit_goal_ring():
    for name in (
        "offline_rosbag_preinsert_workflow.rviz",
        "real_preinsert_workflow.rviz",
    ):
        rviz = (PACKAGE / "rviz" / name).read_text()
        assert "Class: moveit_rviz_plugin/MotionPlanning" in rviz
        assert "Axes Length: 0.08" in rviz
        assert "Axes Radius: 0.008" in rviz
        assert "Interactive Marker Size: 0" in rviz
        assert "Query Goal State: false" in rviz


def test_real_operator_owns_one_rviz_using_real_preinsert_config():
    operator_launch = (
        PACKAGE / "launch" / "real_experiment_operator.launch.py"
    ).read_text()
    preinsert_launch = (
        PACKAGE / "launch" / "real_preinsert_workflow.launch.py"
    ).read_text()
    assert '"real_preinsert_workflow.launch.py"' in operator_launch
    assert '"show_rviz": "false"' in operator_launch
    assert '"show_rviz": LaunchConfiguration("show_rviz")' in operator_launch
    assert '"real_preinsert_workflow.rviz"' in preinsert_launch
    assert 'executable="rviz2"' in preinsert_launch


def test_operator_pose_helper_is_read_only_and_names_both_snapshots():
    helper = (
        PACKAGE.parents[1] / "scripts" / "ros2" / "capture_operator_joint_pose.sh"
    ).read_text()
    assert "ros2 topic echo --once /joint_states" in helper
    assert '"scan"' in helper
    assert '"loading"' in helper
    assert '${pose_name}_joint_state.yaml' in helper
    assert "ros2 action" not in helper
    assert "ros2 service" not in helper


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


def test_post_insert_defaults_to_real_xarm_gripper_and_supports_fake_rehearsal():
    node = (
        PACKAGE
        / "bookshelf_simple_experiment_ros"
        / "simple_policy_control_node.py"
    ).read_text()
    assert "GripperCommand" in node
    assert "FollowJointTrajectory" in node
    assert '"/xarm_gripper/gripper_action"' in node
    assert 'self.declare_parameter("gripper_action_type", GRIPPER_COMMAND)' in node
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
