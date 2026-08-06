"""Launch only the virtual-policy-tool MoveIt path checker."""

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
            "policy_tool_plan_checker.yaml",
        ]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "checker_config",
                default_value=default_config,
                description="Fail-closed policy-tool plan checker parameters.",
            ),
            LogInfo(
                msg=(
                    "Starting PLAN-ONLY policy-tool checker. This launch has no "
                    "trajectory execution, controller, gripper, or robot-command client."
                )
            ),
            Node(
                package="bookshelf_guarded_control_ros",
                executable="policy_tool_plan_checker",
                name="policy_tool_plan_checker",
                output="screen",
                parameters=[LaunchConfiguration("checker_config")],
            ),
        ]
    )

