from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    LogInfo,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


BOOK_QXYZW = [0.5, 0.5, -0.5, 0.5]


def marker_to_book_center_tf(name, parent_frame, child_frame, xyz, condition=None):
    return Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name=name,
        output="screen",
        condition=condition,
        arguments=[
            "--x", str(xyz[0]),
            "--y", str(xyz[1]),
            "--z", str(xyz[2]),
            "--qx", str(BOOK_QXYZW[0]),
            "--qy", str(BOOK_QXYZW[1]),
            "--qz", str(BOOK_QXYZW[2]),
            "--qw", str(BOOK_QXYZW[3]),
            "--frame-id", parent_frame,
            "--child-frame-id", child_frame,
        ],
    )


def generate_launch_description():
    launch_dir = Path(__file__).resolve().parent
    pkg_dir = launch_dir.parent

    robot_launch = launch_dir / "robot_setup.launch.py"
    camera_launch = launch_dir / "camera_setup.launch.py"
    handeye_tf_launch = launch_dir / "publish_handeye_camera_link.launch.py"
    aruco_script = pkg_dir / "scripts" / "multi_aruco_tf_pub.py"
    legacy_condition = IfCondition(
        LaunchConfiguration("enable_legacy_three_book_detection")
    )
    calibrated_condition = IfCondition(
        LaunchConfiguration("enable_calibrated_book_detection")
    )
    robot_condition = IfCondition(LaunchConfiguration("enable_robot_control"))

    calibrated_book_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("bookshelf_shadow_ros"),
                    "launch",
                    "marker_book_bag_calibration.launch.py",
                ]
            )
        ),
        condition=calibrated_condition,
        launch_arguments={
            "output_dir": LaunchConfiguration("calibration_output_dir"),
            "target_samples": LaunchConfiguration("calibration_target_samples"),
            "detected_marker_frame": "target_book_marker",
            "detected_book_frame": "target_book_center",
            # The real-xArm bringup owns the single old-style MoveIt RViz.
            "enable_rviz": "false",
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "enable_robot_control",
            default_value="true",
            description=(
                "Start the existing real-xArm MoveIt and manual planner bringup. "
                "This does not start the bookshelf policy executor."
            ),
        ),
        DeclareLaunchArgument(
            "show_rviz",
            default_value="false",
            description="Start MoveIt RViz. Keep false for SSH/headless bringup.",
        ),
        DeclareLaunchArgument(
            "enable_calibrated_book_detection",
            default_value="true",
            description="Detect calibrated ArUco Original ID 0 on the held book.",
        ),
        DeclareLaunchArgument(
            "enable_legacy_three_book_detection",
            default_value="false",
            description=(
                "Enable the old ID 0/1/2 shelf-book detector. Keep false while "
                "using calibrated held-book ID 0 detection."
            ),
        ),
        DeclareLaunchArgument(
            "calibration_output_dir",
            default_value="/tmp/bookshelf_marker_book_live_check",
        ),
        DeclareLaunchArgument(
            "calibration_target_samples",
            default_value="250",
        ),
        LogInfo(
            msg=(
                "Starting marker vision and optional manual xArm/MoveIt bringup. "
                "MoveIt RViz is disabled by default. No bookshelf policy executor "
                "is launched."
            )
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(robot_launch)),
            condition=robot_condition,
            launch_arguments={
                "show_rviz": LaunchConfiguration("show_rviz"),
            }.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(camera_launch))
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(handeye_tf_launch))
        ),
        calibrated_book_launch,
        marker_to_book_center_tf(
            "target_book_center_static_tf",
            "target_book_marker",
            "target_book_center",
            [0.0, -0.097, -0.078],
            condition=legacy_condition,
        ),
        marker_to_book_center_tf(
            "left_side_book_center_static_tf",
            "left_side_book_marker",
            "left_side_book_center",
            [0.0, 0.0, -0.0895],
            condition=legacy_condition,
        ),
        marker_to_book_center_tf(
            "right_side_book_center_static_tf",
            "right_side_book_marker",
            "right_side_book_center",
            [0.0, 0.0, -0.0895],
            condition=legacy_condition,
        ),
        Node(
            package="bookshelf_policy_ros",
            executable="book_center_tf_node",
            name="book_center_tf_node",
            output="screen",
            condition=legacy_condition,
            arguments=[
                "--target_marker_frame", "target_book_marker",
                "--left_marker_frame", "left_side_book_marker",
                "--right_marker_frame", "right_side_book_marker",
                "--target_center_frame", "target_book_center",
                "--left_center_frame", "left_side_book_center",
                "--right_center_frame", "right_side_book_center",
                "--target_offset_x", "0.0",
                "--target_offset_y", "-0.097",
                "--target_offset_z", "-0.078",
                "--left_offset_x", "0.0",
                "--left_offset_y", "0.0",
                "--left_offset_z", "-0.0895",
                "--right_offset_x", "0.0",
                "--right_offset_y", "0.0",
                "--right_offset_z", "-0.0895",
                "--marker_topic", "/bookshelf_policy/book_boxes",
                "--publish_static_tfs", "false",
                "--book_qx", str(BOOK_QXYZW[0]),
                "--book_qy", str(BOOK_QXYZW[1]),
                "--book_qz", str(BOOK_QXYZW[2]),
                "--book_qw", str(BOOK_QXYZW[3]),
                "--target_size_x", "0.156",
                "--target_size_y", "0.034",
                "--target_size_z", "0.236",
                "--left_size_x", "0.179",
                "--left_size_y", "0.050",
                "--left_size_z", "0.230",
                "--right_size_x", "0.179",
                "--right_size_y", "0.065",
                "--right_size_z", "0.230",
            ],
        ),
        TimerAction(
            period=3.0,
            condition=legacy_condition,
            actions=[
                ExecuteProcess(
                    cmd=[
                        "python3",
                        str(aruco_script),
                        "--image_topic", "/camera/color/image_raw",
                        "--camera_info_topic", "/camera/color/camera_info",
                        "--debug_image_topic", "/bookshelf_policy/aruco_debug_image",
                        "--camera_frame", "camera_color_optical_frame",
                        "--left_id", "0",
                        "--right_id", "1",
                        "--target_id", "2",
                        "--left_marker_length", "0.040",
                        "--right_marker_length", "0.040",
                        "--target_marker_length", "0.030",
                        "--left_frame", "left_side_book_marker",
                        "--right_frame", "right_side_book_marker",
                        "--target_frame", "target_book_marker",
                    ],
                    output="screen",
                )
            ],
        ),
    ])
