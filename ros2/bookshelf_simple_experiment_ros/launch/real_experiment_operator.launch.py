"""One-command physical bookshelf bringup with reviewed operator controls."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    physical = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare("bookshelf_policy_ros"),
            "launch",
            "physical_hardware_bringup.launch.py",
        ])),
        launch_arguments={
            "robot_ip": LaunchConfiguration("robot_ip"),
            # The preinsert workflow below owns the sole RViz instance.
            "show_rviz": "false",
        }.items(),
    )
    preinsert = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare("bookshelf_simple_experiment_ros"),
            "launch",
            "real_preinsert_workflow.launch.py",
        ])),
        launch_arguments={
            "show_rviz": LaunchConfiguration("show_rviz"),
            "frozen_slot_output": LaunchConfiguration("frozen_slot_output"),
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument("robot_ip", default_value="192.168.1.209"),
        DeclareLaunchArgument("show_rviz", default_value="true"),
        DeclareLaunchArgument(
            "frozen_slot_output",
            default_value="/tmp/bookshelf_simple_frozen_slot.yaml",
        ),
        LogInfo(msg=(
            "BOOKSHELF REAL EXPERIMENT: one physical hardware/MoveIt/Servo stack, "
            "one preinsert workflow, one RViz, and the reviewed operator console. "
            "No motion, gripper, policy, or execution goal is sent automatically."
        )),
        physical,
        preinsert,
        Node(
            package="bookshelf_simple_experiment_ros",
            executable="real_experiment_operator",
            name="bookshelf_real_experiment_operator",
            output="screen",
            emulate_tty=True,
        ),
    ])
