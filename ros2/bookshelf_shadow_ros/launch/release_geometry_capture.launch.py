"""Launch one subscriber-only xArm release-geometry capture."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "approved_config",
                description="Reviewed combined trial configuration used by the running adapter.",
            ),
            DeclareLaunchArgument(
                "output_path",
                default_value="/tmp/" + "xarm_ros_release_geometry.json",
            ),
            DeclareLaunchArgument(
                "capture_condition",
                default_value="release_requested",
                description=(
                    "release_requested is the exact INSERT release event; "
                    "first_valid is available for stationary geometry debugging."
                ),
            ),
            LogInfo(
                msg=(
                    "Starting READ-ONLY xArm release geometry capture. This launch "
                    "contains no publishers or robot-command interfaces."
                )
            ),
            Node(
                package="bookshelf_shadow_ros",
                executable="ros_release_geometry",
                name="ros_release_geometry",
                output="screen",
                parameters=[
                    {
                        "approved_config_path": LaunchConfiguration("approved_config"),
                        "output_path": LaunchConfiguration("output_path"),
                        "capture_condition": LaunchConfiguration("capture_condition"),
                    }
                ],
            ),
        ]
    )
