"""Launch only slot detection and stability logging for RGB-D rosbag replay."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    output_argument = DeclareLaunchArgument(
        "output_dir",
        default_value="/tmp/bookshelf_slot_audit",
        description="Directory for slot detector CSV and JSON stability reports.",
    )
    samples_argument = DeclareLaunchArgument(
        "target_samples",
        default_value="1200",
        description="Number of detector confidence frames to collect.",
    )
    detector = Node(
        package="bookshelf_shadow_ros",
        executable="rgbd_slot_detector",
        name="rgbd_slot_detector",
        output="screen",
        parameters=[
            {
                "image_topic": "/camera/color/image_raw",
                "depth_topic": "/camera/aligned_depth_to_color/image_raw",
                "camera_info_topic": "/camera/color/camera_info",
                "debug_image_topic": "/slot_detector/debug_image",
            }
        ],
    )
    audit = Node(
        package="bookshelf_shadow_ros",
        executable="slot_detection_audit",
        name="slot_detection_audit",
        output="screen",
        parameters=[
            {
                "output_dir": LaunchConfiguration("output_dir"),
                "target_samples": ParameterValue(
                    LaunchConfiguration("target_samples"),
                    value_type=int,
                ),
            }
        ],
    )
    return LaunchDescription(
        [
            output_argument,
            samples_argument,
            LogInfo(
                msg=(
                    "Starting BAG-ONLY detector audit. No robot state, policy, "
                    "IK, trajectory, or control node is launched."
                )
            ),
            detector,
            audit,
        ]
    )
