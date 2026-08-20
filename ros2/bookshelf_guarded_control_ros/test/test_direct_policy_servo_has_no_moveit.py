from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NODE = (
    ROOT
    / "bookshelf_guarded_control_ros"
    / "direct_policy_servo_node.py"
)


def test_direct_servo_uses_xarm_service_without_moveit_or_gripper():
    source = NODE.read_text(encoding="utf-8")

    required = (
        "MoveCartesian",
        '"/xarm/set_servo_cartesian_aa"',
        "compute_policy_tool_target",
        "target_safety_error",
        "provenance_error",
        "control_rate_hz",
        "policy_command_duration_s",
        "request.relative = False",
        "request.is_tool_coord = False",
    )
    for token in required:
        assert token in source

    forbidden = (
        "moveit_msgs",
        "GetMotionPlan",
        "ExecuteTrajectory",
        "FollowJointTrajectory",
        "ActionClient",
        "xarm_gripper",
    )
    for token in forbidden:
        assert token not in source
