from moveit_msgs.msg import RobotTrajectory
from rclpy.serialization import deserialize_message, serialize_message
from trajectory_msgs.msg import JointTrajectoryPoint

from bookshelf_guarded_control_ros.policy_tool_control_math import (
    canonical_ros_message_sha256,
)


def _trajectory():
    message = RobotTrajectory()
    message.joint_trajectory.header.frame_id = "link_base"
    message.joint_trajectory.joint_names = ["joint1", "joint2"]
    point = JointTrajectoryPoint()
    point.positions = [1.25, -0.5]
    point.velocities = [0.01, -0.02]
    point.time_from_start.sec = 2
    point.time_from_start.nanosec = 500_000_000
    message.joint_trajectory.points = [point]
    return message


def test_hash_is_stable_for_same_message_and_dds_round_trip():
    message = _trajectory()
    expected = canonical_ros_message_sha256(message)

    assert canonical_ros_message_sha256(message) == expected
    received = deserialize_message(serialize_message(message), RobotTrajectory)
    assert canonical_ros_message_sha256(received) == expected


def test_hash_changes_when_a_commanded_trajectory_field_changes():
    original = _trajectory()
    changed = _trajectory()
    changed.joint_trajectory.points[0].positions[1] = -0.500001

    assert canonical_ros_message_sha256(changed) != canonical_ros_message_sha256(
        original
    )
