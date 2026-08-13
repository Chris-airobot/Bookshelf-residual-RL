"""Calculate and plan the global pre-insertion target without execution."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    scene_launch = PathJoinSubstitution(
        [
            FindPackageShare("bookshelf_guarded_control_ros"),
            "launch",
            "bookshelf_scene_manager.launch.py",
        ]
    )
    target_launch = PathJoinSubstitution(
        [
            FindPackageShare("bookshelf_shadow_ros"),
            "launch",
            "calibrated_preinsert_target.launch.py",
        ]
    )
    planner_config = PathJoinSubstitution(
        [
            FindPackageShare("bookshelf_guarded_control_ros"),
            "config",
            "calibrated_preinsert_plan_only.yaml",
        ]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "target_config",
                description="Approved trial_static_slot.yaml parameter file.",
            ),
            DeclareLaunchArgument(
                "scene_config",
                description=(
                    "Reviewed physical shelf, table, and held-book scene YAML."
                ),
            ),
            DeclareLaunchArgument(
                "planner_config",
                default_value=planner_config,
                description="Fail-closed global pre-insertion planner parameters.",
            ),
            DeclareLaunchArgument(
                "output_dir",
                default_value="/tmp/bookshelf_preinsert_plan",
                description="Directory for target and plan-only JSON reports.",
            ),
            DeclareLaunchArgument(
                "maximum_preserved_book_orientation_error_deg",
                default_value="15.0",
            ),
            LogInfo(
                msg=(
                    "Starting automatic GLOBAL PRE-INSERTION PLAN-ONLY bridge. "
                    "It calculates the target and requests a MoveIt path, but "
                    "creates no execution, controller, gripper, or robot-command client."
                )
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(scene_launch),
                launch_arguments={
                    "scene_config": LaunchConfiguration("scene_config"),
                }.items(),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(target_launch),
                launch_arguments={
                    "target_config": LaunchConfiguration("target_config"),
                    "target_orientation_mode": "preserve_current_tcp",
                    "maximum_preserved_book_orientation_error_deg": LaunchConfiguration(
                        "maximum_preserved_book_orientation_error_deg"
                    ),
                    "output_dir": LaunchConfiguration("output_dir"),
                }.items(),
            ),
            Node(
                package="bookshelf_guarded_control_ros",
                executable="calibrated_preinsert_plan_only",
                name="calibrated_preinsert_plan_only",
                output="screen",
                parameters=[
                    LaunchConfiguration("planner_config"),
                    {"output_dir": LaunchConfiguration("output_dir")},
                ],
            ),
        ]
    )
