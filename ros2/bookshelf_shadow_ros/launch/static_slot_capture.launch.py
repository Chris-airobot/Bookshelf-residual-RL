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
            "minimum_confidence",
            default_value="0.60",
            description="Minimum detector confidence accepted by the capture.",
        ),
        DeclareLaunchArgument(
            "detector_roi_x_min",
            default_value="0.12",
            description="Normalized left edge of the detector search region.",
        ),
        DeclareLaunchArgument(
            "detector_roi_x_max",
            default_value="0.88",
            description="Normalized right edge of the detector search region.",
        ),
        DeclareLaunchArgument(
            "detector_minimum_slot_width_m",
            default_value="0.020",
            description="Smallest metric opening considered by the detector.",
        ),
        DeclareLaunchArgument(
            "detector_maximum_slot_width_m",
            default_value="0.090",
            description="Largest metric opening considered by the detector.",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Use rosbag /clock for offline replay.",
        ),
        DeclareLaunchArgument(
            "capture_use_latest_tf",
            default_value="false",
            description=(
                "Use the latest available TF only for an explicitly stationary replay."
            ),
        ),
        DeclareLaunchArgument(
            "start_live_detector",
            default_value="true",
            description="Start the read-only RGB-D slot detector.",
        ),
        DeclareLaunchArgument(
            "show_debug_image",
            default_value="false",
            description="Open the annotated RGB detector image.",
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
                "roi_x_min": ParameterValue(
                    LaunchConfiguration("detector_roi_x_min"), value_type=float
                ),
                "roi_x_max": ParameterValue(
                    LaunchConfiguration("detector_roi_x_max"), value_type=float
                ),
                "minimum_slot_width_m": ParameterValue(
                    LaunchConfiguration("detector_minimum_slot_width_m"),
                    value_type=float,
                ),
                "maximum_slot_width_m": ParameterValue(
                    LaunchConfiguration("detector_maximum_slot_width_m"),
                    value_type=float,
                ),
                "use_sim_time": ParameterValue(
                    LaunchConfiguration("use_sim_time"), value_type=bool
                ),
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
                "minimum_confidence": ParameterValue(
                    LaunchConfiguration("minimum_confidence"), value_type=float
                ),
                "use_sim_time": ParameterValue(
                    LaunchConfiguration("use_sim_time"), value_type=bool
                ),
                "use_latest_tf": ParameterValue(
                    LaunchConfiguration("capture_use_latest_tf"), value_type=bool
                ),
            },
        ],
    )
    debug_image_view = Node(
        package="rqt_image_view",
        executable="rqt_image_view",
        name="static_slot_capture_debug_image_view",
        output="screen",
        arguments=["/slot_detector/debug_image"],
        condition=IfCondition(LaunchConfiguration("show_debug_image")),
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
            debug_image_view,
        ]
    )
