"""Launch real RGB-D slot detection and the read-only policy adapter."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution(
        [FindPackageShare("bookshelf_shadow_ros"), "config", "policy_observation_adapter.yaml"]
    )

    config_argument = DeclareLaunchArgument(
        "adapter_config",
        default_value=default_config,
        description="Policy observation adapter parameter file.",
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

    adapter = Node(
        package="bookshelf_shadow_ros",
        executable="policy_observation_adapter",
        name="policy_observation_adapter",
        output="screen",
        parameters=[LaunchConfiguration("adapter_config")],
    )

    return LaunchDescription(
        [
            config_argument,
            LogInfo(
                msg=(
                    "Starting SHADOW-ONLY bookshelf perception and observation. "
                    "No policy execution or robot command node is launched."
                )
            ),
            detector,
            adapter,
        ]
    )
