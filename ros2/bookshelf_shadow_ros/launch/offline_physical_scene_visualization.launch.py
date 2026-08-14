"""Visualize the intended xArm pose and coarse physical scene without motion."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    scene_config = LaunchConfiguration("scene_config")
    rviz_config = LaunchConfiguration("rviz_config")
    show_rviz = LaunchConfiguration("show_rviz")

    robot_description = ParameterValue(
        Command(
            [
                FindExecutable(name="xacro"),
                " ",
                PathJoinSubstitution(
                    [
                        FindPackageShare("bookshelf_shadow_ros"),
                        "urdf",
                        "offline_xarm7_visualization.urdf.xacro",
                    ]
                ),
            ]
        ),
        value_type=str,
    )
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="offline_xarm7_robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description}],
    )

    visualizer = Node(
        package="bookshelf_shadow_ros",
        executable="offline_scene_visualizer",
        name="offline_scene_visualizer",
        output="screen",
        parameters=[scene_config],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="offline_physical_scene_rviz",
        output="screen",
        arguments=["-d", rviz_config],
        condition=IfCondition(show_rviz),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "scene_config",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("bookshelf_shadow_ros"),
                        "config",
                        "offline_physical_scene_visualization.yaml",
                    ]
                ),
                description="Visual-only joint pose and coarse scene parameters.",
            ),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("bookshelf_shadow_ros"),
                        "rviz",
                        "offline_physical_scene.rviz",
                    ]
                ),
            ),
            DeclareLaunchArgument("show_rviz", default_value="true"),
            LogInfo(
                msg=(
                    "Starting OFFLINE VISUALIZATION ONLY. No hardware driver, "
                    "MoveIt planner, controller, trajectory, gripper command, "
                    "or executor is launched."
                )
            ),
            robot_state_publisher,
            visualizer,
            rviz,
        ]
    )
