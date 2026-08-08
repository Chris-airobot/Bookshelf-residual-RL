"""Launch the disabled-by-default, token-gated single-step executor."""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
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
    logging_launch = PathJoinSubstitution(
        [
            FindPackageShare("bookshelf_shadow_ros"),
            "launch",
            "experiment_logging.launch.py",
        ]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "executor_config",
                default_value=default_config,
                description="Explicitly reviewed guarded-executor parameters.",
            ),
            DeclareLaunchArgument(
                "enable_logging",
                default_value="true",
                description="Automatically record every guarded physical trial.",
            ),
            DeclareLaunchArgument(
                "trial_name",
                default_value="guarded_single_step",
                description="Short experiment-trial label.",
            ),
            DeclareLaunchArgument(
                "experiment_output_root",
                description="Persistent root directory for automatic trial logs.",
            ),
            DeclareLaunchArgument(
                "repository_path",
                description="Unified source checkout recorded in the manifest.",
            ),
            DeclareLaunchArgument(
                "policy_bundle",
                description="Exact portable actor bundle used by the shadow pipeline.",
            ),
            DeclareLaunchArgument(
                "activation_envelope",
                description="Exact activation-envelope JSON used by the shadow pipeline.",
            ),
            DeclareLaunchArgument(
                "record_camera",
                default_value="true",
                description="Record compressed RGB-D topics with the trial.",
            ),
            LogInfo(
                msg=(
                    "Starting GUARDED executor with committed fail-closed defaults: "
                    "dry_run=true, allow_execution=false, unverified transform rejected, "
                    "planning scene unconfirmed, and approval token disabled."
                )
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(logging_launch),
                condition=IfCondition(LaunchConfiguration("enable_logging")),
                launch_arguments={
                    "trial_name": LaunchConfiguration("trial_name"),
                    "output_root": LaunchConfiguration("experiment_output_root"),
                    "repository_path": LaunchConfiguration("repository_path"),
                    "policy_bundle": LaunchConfiguration("policy_bundle"),
                    "activation_envelope": LaunchConfiguration(
                        "activation_envelope"
                    ),
                    "record_camera": LaunchConfiguration("record_camera"),
                }.items(),
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
