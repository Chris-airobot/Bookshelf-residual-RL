"""Apply coarse bookshelf/table/held-book MoveIt collision geometry."""

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
            "bookshelf_scene_physical.yaml",
        ]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "scene_config",
                default_value=default_config,
                description=(
                    "Single YAML containing all physical shelf, table, held-book "
                    "and scene-handoff settings."
                ),
            ),
            LogInfo(
                msg=(
                    "Starting planning-scene-only bookshelf manager. It has no "
                    "trajectory, controller, gripper or robot-command interface."
                )
            ),
            Node(
                package="bookshelf_guarded_control_ros",
                executable="bookshelf_scene_manager",
                name="bookshelf_scene_manager",
                output="screen",
                parameters=[
                    LaunchConfiguration("scene_config"),
                    {"scene_config_path": LaunchConfiguration("scene_config")},
                ],
            ),
        ]
    )
