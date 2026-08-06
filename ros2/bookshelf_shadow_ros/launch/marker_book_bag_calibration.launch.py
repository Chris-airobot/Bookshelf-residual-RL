"""Launch read-only ArUco-to-book calibration for rosbag replay."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument(
            "mount_yaml",
            default_value=PathJoinSubstitution(
                [
                    FindPackageShare("bookshelf_shadow_ros"),
                    "config",
                    "real_book_aruco0_mount.yaml",
                ]
            ),
            description="Measured marker mount YAML for the grasped book.",
        ),
        DeclareLaunchArgument(
            "output_dir",
            default_value="/tmp/bookshelf_marker_book_calibration",
            description="Directory for CSV, JSON, YAML, and debug images.",
        ),
        DeclareLaunchArgument(
            "target_samples",
            default_value="250",
            description="Number of accepted marker poses to collect.",
        ),
        DeclareLaunchArgument(
            "enable_rviz",
            default_value="true",
            description="Start diagnostics-only RViz book visualization.",
        ),
    ]
    calibrator = Node(
        package="bookshelf_shadow_ros",
        executable="marker_book_calibrator",
        name="marker_book_calibration",
        output="screen",
        parameters=[
            {
                "mount_yaml": LaunchConfiguration("mount_yaml"),
                "output_dir": LaunchConfiguration("output_dir"),
                "target_samples": ParameterValue(
                    LaunchConfiguration("target_samples"), value_type=int
                ),
            }
        ],
    )
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="marker_book_calibration_rviz",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_rviz")),
        arguments=[
            "-d",
            PathJoinSubstitution(
                [
                    FindPackageShare("bookshelf_shadow_ros"),
                    "rviz",
                    "marker_book_calibration.rviz",
                ]
            ),
        ],
    )
    return LaunchDescription(
        arguments
        + [
            LogInfo(
                msg=(
                    "Starting BAG-ONLY marker/book calibration. No policy, IK, "
                    "trajectory, gripper, or robot-command node is launched."
                )
            ),
            calibrator,
            rviz,
        ]
    )
