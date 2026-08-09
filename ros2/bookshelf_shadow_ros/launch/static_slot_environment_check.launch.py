"""Run a read-only live check of the immutable configured bookshelf slot."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution(
        [
            FindPackageShare("bookshelf_shadow_ros"),
            "config",
            "static_slot_environment_check.yaml",
        ]
    )
    arguments = [
        DeclareLaunchArgument("check_config", default_value=default_config),
        DeclareLaunchArgument("output_dir", default_value="/tmp/bookshelf_static_slot_check"),
        DeclareLaunchArgument(
            "start_live_detector",
            default_value="true",
            description="Start the read-only RGB-D detector on the real camera topics.",
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
    check = Node(
        package="bookshelf_shadow_ros",
        executable="static_slot_environment_check",
        name="static_slot_environment_check",
        output="screen",
        parameters=[
            LaunchConfiguration("check_config"),
            {"output_dir": LaunchConfiguration("output_dir")},
        ],
    )
    return LaunchDescription(
        arguments
        + [
            LogInfo(
                msg=(
                    "Starting READ-ONLY static-slot environment check. No policy, "
                    "IK, planner, executor, trajectory, controller, gripper, or "
                    "robot-command node is launched."
                )
            ),
            detector,
            check,
        ]
    )
