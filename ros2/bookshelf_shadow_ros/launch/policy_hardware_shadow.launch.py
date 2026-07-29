"""Launch markerless perception, observation adaptation, and read-only PPO inference."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare("bookshelf_shadow_ros")
    default_adapter_config = PathJoinSubstitution(
        [package_share, "config", "policy_observation_adapter_markerless_smoke.yaml"]
    )
    default_inference_config = PathJoinSubstitution(
        [package_share, "config", "policy_shadow_inference.yaml"]
    )

    adapter_config_argument = DeclareLaunchArgument(
        "adapter_config",
        default_value=default_adapter_config,
        description="Markerless policy observation adapter parameter file.",
    )
    inference_config_argument = DeclareLaunchArgument(
        "inference_config",
        default_value=default_inference_config,
        description="Shadow policy inference parameter file.",
    )
    bundle_argument = DeclareLaunchArgument(
        "policy_bundle",
        description="Verified portable .npz actor and VecNormalize bundle.",
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
    inference = Node(
        package="bookshelf_shadow_ros",
        executable="policy_shadow_inference",
        name="policy_shadow_inference",
        output="screen",
        parameters=[
            LaunchConfiguration("inference_config"),
            {"policy_bundle_path": LaunchConfiguration("policy_bundle")},
        ],
    )

    return LaunchDescription(
        [
            adapter_config_argument,
            inference_config_argument,
            bundle_argument,
            LogInfo(
                msg=(
                    "Starting FULL SHADOW pipeline: RGB-D detector -> markerless 12D adapter "
                    "-> VecNormalize -> PPO actor diagnostics. No robot-command node is launched."
                )
            ),
            detector,
            adapter,
            inference,
        ]
    )
