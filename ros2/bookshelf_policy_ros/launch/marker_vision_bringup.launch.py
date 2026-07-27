from pathlib import Path

from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


BOOK_QXYZW = [0.5, 0.5, -0.5, 0.5]


def marker_to_book_center_tf(name, parent_frame, child_frame, xyz):
    return Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name=name,
        output="screen",
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

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(robot_launch))
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(camera_launch))
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(handeye_tf_launch))
        ),
        marker_to_book_center_tf(
            "target_book_center_static_tf",
            "target_book_marker",
            "target_book_center",
            [0.0, -0.097, -0.078],
        ),
        marker_to_book_center_tf(
            "left_side_book_center_static_tf",
            "left_side_book_marker",
            "left_side_book_center",
            [0.0, 0.0, -0.0895],
        ),
        marker_to_book_center_tf(
            "right_side_book_center_static_tf",
            "right_side_book_marker",
            "right_side_book_center",
            [0.0, 0.0, -0.0895],
        ),
        Node(
            package="bookshelf_policy_ros",
            executable="book_center_tf_node",
            name="book_center_tf_node",
            output="screen",
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
