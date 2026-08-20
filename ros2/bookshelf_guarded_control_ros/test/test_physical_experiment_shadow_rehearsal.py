from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / "launch" / "physical_experiment_shadow_rehearsal.launch.py"


def test_rehearsal_uses_one_approved_config_and_one_detector_owner():
    source = LAUNCH.read_text(encoding="utf-8")
    required = (
        '"approved_config"',
        '"physical_experiment_observation_bringup.launch.py"',
        '"policy_hardware_shadow.launch.py"',
        '"trial_slot_config": LaunchConfiguration("approved_config")',
        '"scene_config": LaunchConfiguration("approved_config")',
        '"adapter_config": LaunchConfiguration("approved_config")',
        '"start_live_detector": "false"',
        '"require_activation_envelope": "true"',
        "validate_shadow_rehearsal_assets",
    )
    for token in required:
        assert token in source


def test_rehearsal_launch_has_no_motion_interface():
    source = LAUNCH.read_text(encoding="utf-8")
    forbidden = (
        "ActionClient",
        "ExecuteTrajectory",
        "FollowJointTrajectory",
        "send_goal",
        'executable="guarded_policy_tool_executor"',
        'executable="guarded_preinsert_executor"',
        'executable="calibrated_preinsert_plan_only"',
        'executable="bookshelf_scene_manager"',
        "Node(",
    )
    for token in forbidden:
        assert token not in source
