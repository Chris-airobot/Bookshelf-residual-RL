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
