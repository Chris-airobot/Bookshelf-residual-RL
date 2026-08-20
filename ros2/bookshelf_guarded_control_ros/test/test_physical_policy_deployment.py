from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / "launch" / "physical_policy_deployment.launch.py"


def test_policy_deployment_reuses_hardware_and_validates_approved_assets():
    source = LAUNCH.read_text(encoding="utf-8")
    required = (
        '"approved_config"',
        "validate_shadow_rehearsal_assets",
        '"experiment_logging.launch.py"',
        '"static_slot_environment_check.launch.py"',
        '"policy_hardware_shadow.launch.py"',
        '"start_live_detector": "false"',
        '"require_activation_envelope": "true"',
        '"adapter_config": LaunchConfiguration("approved_config")',
    )
    for token in required:
        assert token in source


def test_policy_deployment_does_not_start_hardware_owners():
    source = LAUNCH.read_text(encoding="utf-8")
    forbidden = (
        "marker_vision_bringup.launch.py",
        "robot_setup.launch.py",
        "realsense2_camera",
        "xarm_planner",
        "ActionClient",
        "ExecuteTrajectory",
        "FollowJointTrajectory",
        "guarded_preinsert_executor",
        "guarded_policy_tool_executor",
        "policy_tool_plan_checker",
        "bookshelf_scene_manager",
        "held_book_pose_check",
        "policy_to_robot",
        "send_goal",
    )
    for token in forbidden:
        assert token not in source

    assert '"book_pose_required_stable_samples"' not in source


def test_policy_deployment_uses_live_marker_book_without_recorded_grasp_gate():
    source = LAUNCH.read_text(encoding="utf-8")
    assert 'executable="held_book_pose_check"' not in source
    assert '"adapter_config": LaunchConfiguration("approved_config")' in source
    assert '"start_live_detector": "false"' in source


def test_policy_deployment_operations_are_simple_and_bounded():
    source = LAUNCH.read_text(encoding="utf-8")

    assert '"operation"' in source
    assert 'default_value="calculate"' in source
    assert 'operation not in ("calculate", "control")' in source
    assert 'if operation == "calculate"' in source
    assert 'executable="direct_policy_servo"' in source
    assert 'overrides.pop("require_scene_status", None)' in source
    assert '"execution_approval_token"' not in source
    assert 'executable="policy_tool_plan_checker"' not in source
    assert 'executable="guarded_policy_tool_executor"' not in source
    assert '"block_on_activation_checks": "false"' in source
    assert "guarded_policy_tool_overrides" in source
    assert '"maximum_total_translation_m"' in source
    assert 'default_value="0.0"' in source
    assert "validate_maximum_total_translation_m" in source
    assert 'overrides["maximum_total_translation_m"]' in source


def test_policy_deployment_has_one_slot_detector_owner():
    source = LAUNCH.read_text(encoding="utf-8")
    assert source.count('"start_live_detector": "true"') == 1
    assert source.count('"start_live_detector": "false"') == 1
