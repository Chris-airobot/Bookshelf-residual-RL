"""Software-only real-workflow preview using official xArm7 fake MoveIt hardware."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare("bookshelf_simple_experiment_ros")
    slot_config = LaunchConfiguration("slot_config")
    rviz_config = PathJoinSubstitution([
        package_share,
        "rviz",
        "real_preinsert_workflow.rviz",
    ])

    virtual_preinsert = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            package_share,
            "launch",
            "virtual_saved_slot_preinsert.launch.py",
        ])),
        launch_arguments={
            "slot_config": slot_config,
            "execute_virtual": "false",
            "auto_start": "true",
            "show_rviz": "false",
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
        LogInfo(msg=(
            "OFFLINE PREVIEW ONLY: official xArm7 fake MoveIt hardware, saved "
            "slot, plan-only preinsert, synthetic camera image, and RViz. No "
            "physical hardware, RealSense, Servo, gripper, or PPO is started."
        )),
        virtual_preinsert,
        Node(
            package="bookshelf_simple_experiment_ros",
            executable="offline_slot_debug_image",
            name="offline_slot_debug_image",
            output="screen",
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="bookshelf_offline_preinsert_rviz",
            output="screen",
            arguments=["-d", rviz_config],
        ),
    ])
