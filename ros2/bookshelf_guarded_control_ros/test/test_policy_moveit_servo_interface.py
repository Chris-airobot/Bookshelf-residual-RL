from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NODE = ROOT / "bookshelf_guarded_control_ros" / "direct_policy_servo_node.py"


def test_policy_controller_uses_moveit_servo_without_direct_xarm_services():
    source = NODE.read_text(encoding="utf-8")

    required = (
        "TwistStamped",
        '"/servo_server/start_servo"',
        '"/servo_server/delta_twist_cmds"',
        '"/servo_server/status"',
        "compute_policy_tool_target",
        "eef_target_from_tcp_target",
        "bounded_error_twist",
        "target_safety_error",
        "provenance_error",
        "_publish_zero_twist",
        "observe_position",
        '"total_measured_translation_m"',
        '"translation_budget_enforced"',
        '"yield_when_control_disabled"',
        'error == "control enable is false"',
    )
    for token in required:
        assert token in source

    forbidden = (
        "MoveCartesian",
        "SetInt16",
        "/xarm/set_mode",
        "/xarm/set_state",
        "/xarm/set_servo_cartesian_aa",
        "GetMotionPlan",
        "ExecuteTrajectory",
        "FollowJointTrajectory",
        "ActionClient",
        "xarm_gripper",
    )
    for token in forbidden:
        assert token not in source


def test_unlimited_translation_is_restricted_to_fake_hardware():
    source = NODE.read_text(encoding="utf-8")
    assert 'self.declare_parameter("enforce_translation_budget", True)' in source
    assert "enforce_translation_budget=false is allowed only for fake hardware" in source
    assert "if self.enforce_translation_budget:" in source
    assert "servo_already_started=true is allowed only for fake hardware" in source
