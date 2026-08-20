from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NODE = ROOT / "bookshelf_guarded_control_ros" / "direct_policy_servo_node.py"


def test_policy_controller_uses_moveit_servo_without_direct_xarm_services():
    source = NODE.read_text(encoding="utf-8")

    required = (
        "TwistStamped",
        '"/servo_server/start_servo"',
        '"/servo_server/delta_twist_cmds"',
        "compute_policy_tool_target",
        "eef_target_from_tcp_target",
        "bounded_error_twist",
        "target_safety_error",
        "provenance_error",
        "_publish_zero_twist",
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
