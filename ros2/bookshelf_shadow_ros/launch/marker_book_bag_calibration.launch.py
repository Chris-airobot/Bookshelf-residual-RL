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
        DeclareLaunchArgument(
            "enable_frame_audit",
            default_value="false",
            description=(
                "Publish current/candidate book frames for human bag-only review."
            ),
        ),
        DeclareLaunchArgument(
            "capture_eef_tcp_context",
            default_value="true",
            description="Capture the fixed link_eef to link_tcp TF context.",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Use rosbag /clock for offline replay.",
        ),
        DeclareLaunchArgument(
            "detected_marker_frame",
            default_value="calibration_aruco0_marker",
            description="Dynamic TF child frame for the detected marker.",
        ),
        DeclareLaunchArgument(
            "detected_book_frame",
            default_value="calibration_detected_book",
            description="Dynamic TF child frame for the marker-derived book centre.",
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
                "enable_frame_audit": ParameterValue(
                    LaunchConfiguration("enable_frame_audit"), value_type=bool
                ),
                "detected_marker_frame": LaunchConfiguration(
                    "detected_marker_frame"
                ),
                "detected_book_frame": LaunchConfiguration(
                    "detected_book_frame"
                ),
                "use_sim_time": ParameterValue(
                    LaunchConfiguration("use_sim_time"), value_type=bool
                ),
            }
        ],
    )
    eef_tcp_context = Node(
        package="bookshelf_shadow_ros",
        executable="eef_tcp_context_capture",
        name="eef_tcp_context_capture",
        output="screen",
        condition=IfCondition(LaunchConfiguration("capture_eef_tcp_context")),
        parameters=[
            {
                "output_path": PathJoinSubstitution(
                    [LaunchConfiguration("output_dir"), "eef_tcp_context.json"]
                ),
                "use_sim_time": ParameterValue(
                    LaunchConfiguration("use_sim_time"), value_type=bool
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
            eef_tcp_context,
            rviz,
        ]
    )
