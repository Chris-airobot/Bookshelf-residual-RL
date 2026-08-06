"""Launch the disabled-by-default, token-gated single-step executor."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution(
        [
            FindPackageShare("bookshelf_guarded_control_ros"),
            "config",
            "guarded_policy_tool_executor.yaml",
        ]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "executor_config",
                default_value=default_config,
                description="Explicitly reviewed guarded-executor parameters.",
            ),
            LogInfo(
                msg=(
                    "Starting GUARDED executor with committed fail-closed defaults: "
                    "dry_run=true, allow_execution=false, unverified transform rejected, "
                    "planning scene unconfirmed, and approval token disabled."
                )
            ),
            Node(
                package="bookshelf_guarded_control_ros",
                executable="guarded_policy_tool_executor",
                name="guarded_policy_tool_executor",
                output="screen",
                parameters=[LaunchConfiguration("executor_config")],
            ),
        ]
    )
