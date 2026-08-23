from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / "launch" / "policy_hardware_shadow.launch.py"


def test_shadow_detector_can_be_disabled_by_composed_owner():
    source = LAUNCH.read_text(encoding="utf-8")
    assert '"start_live_detector"' in source
    assert 'default_value="true"' in source
    assert 'condition=IfCondition(LaunchConfiguration("start_live_detector"))' in source


def test_shadow_launch_remains_read_only():
    source = LAUNCH.read_text(encoding="utf-8")
    for forbidden in (
        "ActionClient",
        "ExecuteTrajectory",
        "FollowJointTrajectory",
        "send_goal",
        "guarded_policy_tool_executor",
    ):
        assert forbidden not in source


def test_activation_checks_can_be_made_diagnostic_only():
    source = LAUNCH.read_text(encoding="utf-8")
    assert '"block_on_activation_checks"' in source
    assert 'LaunchConfiguration("block_on_activation_checks")' in source


def test_observation_frames_can_be_shared_with_software_simulation():
    source = LAUNCH.read_text(encoding="utf-8")
    for argument in (
        '"base_frame"',
        '"ee_frame"',
        '"target_book_frame"',
        '"joint_states_topic"',
        '"message_max_age_s"',
        '"tf_max_age_s"',
    ):
        assert argument in source
