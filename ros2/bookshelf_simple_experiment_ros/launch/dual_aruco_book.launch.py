"""Standalone dual-ArUco calibration/runtime book-pose launch."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    reference_mount = PathJoinSubstitution([
        FindPackageShare("bookshelf_simple_experiment_ros"), "config",
        "reference_marker0_book_mount.yaml"])
    defaults = {
        "mode": "runtime", "image_topic": "/camera/color/image_raw",
        "camera_info_topic": "/camera/color/camera_info", "camera_frame": "",
        "dictionary": "DICT_ARUCO_ORIGINAL", "reference_marker_id": "0",
        "reference_marker_size_m": "0.039", "secondary_marker_id": "10",
        "secondary_marker_size_m": "0.039", "reference_mount_yaml": reference_mount,
        "secondary_mount_yaml": "~/BookshelfFiles/experiment_configs/simple_dual_aruco/secondary_marker_book_mount.yaml",
        "target_samples": "200"}
    arguments = [DeclareLaunchArgument(name, default_value=value) for name, value in defaults.items()]
    node = Node(package="bookshelf_simple_experiment_ros", executable="dual_aruco_book",
                name="dual_aruco_book", output="screen",
                parameters=[{name: LaunchConfiguration(name) for name in defaults}])
    return LaunchDescription(arguments + [node])
