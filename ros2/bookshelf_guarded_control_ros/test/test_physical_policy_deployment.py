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
        'executable="held_book_pose_check"',
        '"policy_hardware_shadow.launch.py"',
        '"start_live_detector": "false"',
        '"require_activation_envelope": "true"',
        '"adapter_config": LaunchConfiguration("approved_config")',
    )
    for token in required:
        assert token in source


def test_policy_deployment_has_no_hardware_interface():
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
        "policy_to_robot",
        "send_goal",
    )
    for token in forbidden:
        assert token not in source

    assert 'executable="held_book_pose_check"' in source


def test_policy_deployment_modes_are_explicit_and_fail_closed_by_default():
    source = LAUNCH.read_text(encoding="utf-8")

    assert '"execution_mode"' in source
    assert 'default_value="shadow"' in source
    assert 'mode not in ("shadow", "plan_only", "single_step")' in source
    assert 'if mode == "shadow"' in source
    assert 'if mode == "plan_only"' in source
    assert 'executable="policy_tool_plan_checker"' in source
    assert 'executable="guarded_policy_tool_executor"' in source
    assert '"permit_local_scene_handoff"' in source
    assert '"execution_approval_token"' in source
    assert 'default_value="DISABLED"' in source
    assert '"dry_run": False' in source
    assert '"allow_execution": True' in source
    assert "guarded_policy_tool_overrides" in source


def test_policy_deployment_has_one_slot_detector_owner():
    source = LAUNCH.read_text(encoding="utf-8")
    assert source.count('"start_live_detector": "true"') == 1
    assert source.count('"start_live_detector": "false"') == 1
