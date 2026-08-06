"""Audit policy-tool TF candidates without planning or robot commands."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config = PathJoinSubstitution(
        [
            FindPackageShare("bookshelf_shadow_ros"),
            "config",
            "policy_tool_frame_audit.yaml",
        ]
    )
    arguments = [
        DeclareLaunchArgument(
            "output_dir",
            default_value="/tmp/bookshelf_policy_tool_audit",
        ),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument(
            "tf_max_age_s",
            default_value="0.50",
            description="Set to 0 for offline rosbag replay.",
        ),
    ]
    node = Node(
        package="bookshelf_shadow_ros",
        executable="policy_tool_frame_audit",
        name="policy_tool_frame_audit",
        output="screen",
        parameters=[
            config,
            {
                "output_dir": LaunchConfiguration("output_dir"),
                "use_sim_time": ParameterValue(
                    LaunchConfiguration("use_sim_time"), value_type=bool
                ),
                "tf_max_age_s": ParameterValue(
                    LaunchConfiguration("tf_max_age_s"), value_type=float
                ),
            },
        ],
    )
    return LaunchDescription(
        arguments
        + [
            LogInfo(
                msg=(
                    "Starting READ-ONLY policy tool-frame audit. Candidate "
                    "ranking does not select a frame or authorize motion."
                )
            ),
            node,
        ]
    )
