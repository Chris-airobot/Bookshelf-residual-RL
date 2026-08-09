"""Capture an unapproved static-slot candidate without robot commands."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution(
        [
            FindPackageShare("bookshelf_shadow_ros"),
            "config",
            "static_slot_capture.yaml",
        ]
    )
    arguments = [
        DeclareLaunchArgument("capture_config", default_value=default_config),
        DeclareLaunchArgument(
            "output_dir", default_value="/tmp/bookshelf_static_slot_capture"
        ),
        DeclareLaunchArgument(
            "repository_path", default_value="/home/riot/Chris/bookshelf-unified"
        ),
        DeclareLaunchArgument("target_samples", default_value="120"),
        DeclareLaunchArgument(
            "start_live_detector",
            default_value="true",
            description="Start the read-only RGB-D slot detector.",
        ),
    ]
    detector = Node(
        package="bookshelf_shadow_ros",
        executable="rgbd_slot_detector",
        name="rgbd_slot_detector",
        output="screen",
        condition=IfCondition(LaunchConfiguration("start_live_detector")),
        parameters=[
            {
                "image_topic": "/camera/color/image_raw",
                "depth_topic": "/camera/aligned_depth_to_color/image_raw",
                "camera_info_topic": "/camera/color/camera_info",
                "debug_image_topic": "/slot_detector/debug_image",
            }
        ],
    )
    capture = Node(
        package="bookshelf_shadow_ros",
        executable="static_slot_capture",
        name="static_slot_capture",
        output="screen",
        parameters=[
            LaunchConfiguration("capture_config"),
            {
                "output_dir": LaunchConfiguration("output_dir"),
                "repository_path": LaunchConfiguration("repository_path"),
                "target_samples": ParameterValue(
                    LaunchConfiguration("target_samples"), value_type=int
                ),
            },
        ],
    )
    return LaunchDescription(
        arguments
        + [
            LogInfo(
                msg=(
                    "Starting READ-ONLY static-slot capture. It writes an "
                    "unapproved candidate only; no active configuration, policy, "
                    "IK, planner, executor, trajectory, gripper, or robot command "
                    "is created."
                )
            ),
            detector,
            capture,
        ]
    )
