"""Start the physical xArm, camera, TF, MoveIt, and book detector once."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    marker_vision_launch = PathJoinSubstitution(
        [
            FindPackageShare("bookshelf_policy_ros"),
            "launch",
            "marker_vision_bringup.launch.py",
        ]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_ip", default_value="192.168.1.209"),
            DeclareLaunchArgument("show_rviz", default_value="false"),
            DeclareLaunchArgument(
                "calibration_output_dir",
                default_value="/tmp/bookshelf_marker_book_live_check",
            ),
            DeclareLaunchArgument("calibration_target_samples", default_value="250"),
            DeclareLaunchArgument("camera_name", default_value="camera"),
            DeclareLaunchArgument("camera_namespace", default_value=""),
            DeclareLaunchArgument("serial_no", default_value=""),
            DeclareLaunchArgument("color_profile", default_value="640x480x30"),
            DeclareLaunchArgument("depth_profile", default_value="640x480x30"),
            DeclareLaunchArgument("align_depth", default_value="true"),
            DeclareLaunchArgument("enable_sync", default_value="true"),
            DeclareLaunchArgument("enable_pointcloud", default_value="true"),
            LogInfo(
                msg=(
                    "Starting the sole PHYSICAL HARDWARE stack: xArm/MoveIt, "
                    "RealSense, hand-eye TF, and calibrated marker-book "
                    "detection. This launch creates hardware-capable MoveIt "
                    "interfaces but sends no motion, gripper, or policy goal. "
                    "Do not start a second hardware bringup."
                )
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(marker_vision_launch),
                launch_arguments={
                    "robot_ip": LaunchConfiguration("robot_ip"),
                    "enable_robot_control": "true",
                    "show_rviz": LaunchConfiguration("show_rviz"),
                    "enable_calibrated_book_detection": "true",
                    "enable_legacy_three_book_detection": "false",
                    "calibration_output_dir": LaunchConfiguration(
                        "calibration_output_dir"
                    ),
                    "calibration_target_samples": LaunchConfiguration(
                        "calibration_target_samples"
                    ),
                    "camera_name": LaunchConfiguration("camera_name"),
                    "camera_namespace": LaunchConfiguration("camera_namespace"),
                    "serial_no": LaunchConfiguration("serial_no"),
                    "color_profile": LaunchConfiguration("color_profile"),
                    "depth_profile": LaunchConfiguration("depth_profile"),
                    "align_depth": LaunchConfiguration("align_depth"),
                    "enable_sync": LaunchConfiguration("enable_sync"),
                    "enable_pointcloud": LaunchConfiguration("enable_pointcloud"),
                }.items(),
            ),
        ]
    )
