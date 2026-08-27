"""Software-only saved-slot pre-insertion with official xArm fake hardware."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare("bookshelf_simple_experiment_ros")
    config = PathJoinSubstitution([package_share, "config", "simple_preinsert.yaml"])
    rviz_config = PathJoinSubstitution([
        package_share, "rviz", "virtual_saved_slot_preinsert.rviz"
    ])
    slot_config = LaunchConfiguration("slot_config")
    execute_virtual = LaunchConfiguration("execute_virtual")
    auto_start = LaunchConfiguration("auto_start")
    show_rviz = LaunchConfiguration("show_rviz")

    fake_moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare("xarm_moveit_config"),
            "launch",
            "_robot_moveit_fake.launch.py",
        ])),
        launch_arguments={
            "dof": "7",
            "robot_type": "xarm",
            "limited": "false",
            "add_gripper": "true",
            "no_gui_ctrl": "false",
            "show_rviz": show_rviz,
            "rviz_config": rviz_config,
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "slot_config",
            default_value=PathJoinSubstitution([
                EnvironmentVariable("HOME"),
                "BookshelfFiles",
                "experiment_configs",
                "stationary_approved_53e7fe80d56d_20260819_142355",
                "trial_static_slot.yaml",
            ]),
        ),
        DeclareLaunchArgument("execute_virtual", default_value="false"),
        DeclareLaunchArgument("auto_start", default_value="true"),
        DeclareLaunchArgument("show_rviz", default_value="true"),
        LogInfo(msg=[
            "SOFTWARE ONLY: official xArm7 fake hardware + saved slot. ",
            "Virtual trajectory execution enabled: ", execute_virtual,
        ]),
        fake_moveit,
        Node(
            package="bookshelf_simple_experiment_ros",
            executable="saved_slot",
            name="saved_slot_publisher",
            output="screen",
            parameters=[{"slot_config": slot_config}],
        ),
        Node(
            package="bookshelf_simple_experiment_ros",
            executable="simple_preinsert",
            name="simple_preinsert",
            output="screen",
            parameters=[config, {
                "allow_execution": ParameterValue(execute_virtual, value_type=bool),
                "separate_execution_confirmation": True,
                "print_target_diagnostics": True,
                "maximum_goal_joint_delta_rad": 3.2,
            }],
        ),
        Node(
            package="bookshelf_simple_experiment_ros",
            executable="virtual_trigger",
            name="virtual_preinsert_trigger",
            output="screen",
            condition=IfCondition(auto_start),
            parameters=[{
                "initial_joint_positions": [
                    0.4342693425054612,
                    1.5322427671441177,
                    4.904658882462919,
                    1.302429752118059,
                    3.302595179623167,
                    0.6839448116011184,
                    4.4791192150828865,
                ],
                "initial_move_duration_s": 2.0,
            }],
        ),
    ])
