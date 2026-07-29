"""Attach the subscriber-only auditor to an already running shadow pipeline."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution(
        [
            FindPackageShare("bookshelf_shadow_ros"),
            "config",
            "policy_stream_audit.yaml",
        ]
    )
    config_argument = DeclareLaunchArgument(
        "audit_config",
        default_value=default_config,
        description="Complete shadow-stream audit parameter file.",
    )
    output_argument = DeclareLaunchArgument(
        "output_dir",
        default_value="/tmp/bookshelf_policy_stream_audit",
        description="Directory for policy stream CSV and JSON reports.",
    )
    samples_argument = DeclareLaunchArgument(
        "target_samples",
        default_value="1200",
        description="Number of policy-debug cycles to audit.",
    )
    reference_width_argument = DeclareLaunchArgument(
        "reference_slot_width_m",
        default_value="0.0",
        description="Optional manually measured physical slot width in metres.",
    )

    audit = Node(
        package="bookshelf_shadow_ros",
        executable="policy_stream_audit",
        name="policy_stream_audit",
        output="screen",
        parameters=[
            LaunchConfiguration("audit_config"),
            {
                "output_dir": LaunchConfiguration("output_dir"),
                "target_samples": ParameterValue(
                    LaunchConfiguration("target_samples"),
                    value_type=int,
                ),
                "reference_slot_width_m": ParameterValue(
                    LaunchConfiguration("reference_slot_width_m"),
                    value_type=float,
                ),
            },
        ],
    )
    return LaunchDescription(
        [
            config_argument,
            output_argument,
            samples_argument,
            reference_width_argument,
            LogInfo(
                msg=(
                    "Attaching subscriber-only policy stream audit. "
                    "No detector, policy, IK, trajectory, or robot-command node is started."
                )
            ),
            audit,
        ]
    )
