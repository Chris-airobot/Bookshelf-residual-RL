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
