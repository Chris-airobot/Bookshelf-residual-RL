"""Real camera-to-preinsert workflow; expects xArm MoveIt already running."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare("bookshelf_simple_experiment_ros")
    config = PathJoinSubstitution([package_share, "config", "simple_preinsert.yaml"])
    image_topic = LaunchConfiguration("image_topic")
    depth_topic = LaunchConfiguration("depth_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")
    frozen_slot_output = LaunchConfiguration("frozen_slot_output")
    show_rviz = LaunchConfiguration("show_rviz")
    allow_execution = LaunchConfiguration("allow_execution")
    shadow_full_sequence = LaunchConfiguration("shadow_full_sequence")

    return LaunchDescription([
        DeclareLaunchArgument("image_topic", default_value="/camera/color/image_raw"),
        DeclareLaunchArgument(
            "depth_topic", default_value="/camera/aligned_depth_to_color/image_raw"
        ),
        DeclareLaunchArgument(
            "camera_info_topic", default_value="/camera/color/camera_info"
        ),
        DeclareLaunchArgument(
            "frozen_slot_output",
            default_value="/tmp/bookshelf_simple_frozen_slot.yaml",
        ),
        DeclareLaunchArgument("show_rviz", default_value="true"),
        DeclareLaunchArgument("robot_ip", default_value="192.168.1.209"),
        DeclareLaunchArgument("allow_execution", default_value="true"),
        DeclareLaunchArgument("shadow_full_sequence", default_value="false"),
        DeclareLaunchArgument("scan_joint_state_path", default_value=PathJoinSubstitution([
            EnvironmentVariable("HOME"), "BookshelfFiles", "experiment_configs",
            "operator_joint_poses", "scan_joint_state.yaml",
        ])),
        DeclareLaunchArgument("loading_joint_state_path", default_value=PathJoinSubstitution([
            EnvironmentVariable("HOME"), "BookshelfFiles", "experiment_configs",
            "operator_joint_poses", "loading_joint_state.yaml",
        ])),
        LogInfo(msg=(
            "REAL PREINSERT: manually position the empty xArm, inspect the slot, "
            "then confirm in order with /bookshelf_simple/accept_slot, "
            "/bookshelf_simple/plan_preinsert, and "
            "/bookshelf_simple/execute_preinsert. This launch expects the camera "
            "and real xArm MoveIt stack to already be running."
        )),
        Node(
            package="bookshelf_simple_experiment_ros",
            executable="slot_detector",
            name="rgbd_slot_detector",
            output="screen",
            parameters=[config, {
                "image_topic": image_topic,
                "depth_topic": depth_topic,
                "camera_info_topic": camera_info_topic,
            }],
        ),
        Node(
            package="bookshelf_simple_experiment_ros",
            executable="simple_preinsert",
            name="simple_preinsert",
            output="screen",
            parameters=[config, {
                "allow_execution": ParameterValue(allow_execution, value_type=bool),
                "shadow_full_sequence": ParameterValue(
                    shadow_full_sequence, value_type=bool
                ),
                "require_slot_acceptance": True,
                "separate_execution_confirmation": True,
                "print_target_diagnostics": True,
                "frozen_slot_output_path": frozen_slot_output,
                "scan_joint_state_path": LaunchConfiguration("scan_joint_state_path"),
                "loading_joint_state_path": LaunchConfiguration("loading_joint_state_path"),
            }],
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([
                package_share, "launch", "real_preinsert_rviz.launch.py",
            ])),
            launch_arguments={
                "robot_ip": LaunchConfiguration("robot_ip"),
                "show_rviz": show_rviz,
            }.items(),
        ),
    ])
