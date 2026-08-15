"""Launch the disabled-by-default one-shot global pre-insertion executor."""

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
            "guarded_preinsert_executor.yaml",
        ]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "executor_config",
                default_value=default_config,
                description="Separately reviewed global pre-insertion executor YAML.",
            ),
            LogInfo(
                msg=(
                    "Starting one-shot GLOBAL PRE-INSERT executor. The committed "
                    "configuration is dry-run, has no approval token, and cannot "
                    "create an execution action client."
                )
            ),
            Node(
                package="bookshelf_guarded_control_ros",
                executable="guarded_preinsert_executor",
                name="guarded_preinsert_executor",
                output="screen",
                parameters=[LaunchConfiguration("executor_config")],
            ),
        ]
    )
