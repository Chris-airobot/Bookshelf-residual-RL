from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / "launch" / "physical_experiment_observation_bringup.launch.py"
PACKAGE = ROOT / "package.xml"


def test_observation_bringup_combines_required_nonexecuting_launches():
    source = LAUNCH.read_text(encoding="utf-8")
    required = (
        '"trial_name"',
        '"trial_slot_config"',
        '"scene_config"',
        '"marker_vision_bringup.launch.py"',
        '"experiment_logging.launch.py"',
        '"static_slot_environment_check.launch.py"',
        '"enable_calibrated_book_detection"',
        'executable="held_book_pose_check"',
        '"book_pose_required_stable_samples"',
        '"enable_legacy_three_book_detection": "false"',
        '"start_live_detector": "true"',
        '"show_rviz"',
    )
    for token in required:
        assert token in source


def test_observation_bringup_has_no_planning_or_execution_interface():
    source = LAUNCH.read_text(encoding="utf-8")
    forbidden = (
        "ActionClient",
        "ExecuteTrajectory",
        "FollowJointTrajectory",
        "send_goal",
        "guarded_policy_tool_executor",
        "calibrated_preinsert_plan_only",
        "bookshelf_scene_manager",
        "policy_calibrated_static_shadow",
    )
    for token in forbidden:
        assert token not in source


def test_observation_bringup_declares_cross_package_runtime_dependency():
    source = PACKAGE.read_text(encoding="utf-8")
    assert "<exec_depend>bookshelf_policy_ros</exec_depend>" in source
