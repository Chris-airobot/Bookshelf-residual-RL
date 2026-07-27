from pathlib import Path

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


def generate_launch_description():
    launch_dir = Path(__file__).resolve().parent
    pkg_dir = launch_dir.parent
    apriltag_script = pkg_dir / "scripts" / "apriltag_tf_pub.py"

    return LaunchDescription([
        ExecuteProcess(
            cmd=[
                "python3",
                str(apriltag_script),
                "--image_topic", "/camera/color/image_raw",
                "--camera_info_topic", "/camera/color/camera_info",
                "--camera_frame", "camera_color_optical_frame",
                "--dictionary", "tag36h11",
                "--tag_id", "0",
                "--tag_length", "0.019",
                "--tag_frame", "apriltag_36h11_0",
            ],
            output="screen",
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="apriltag0_to_calibrated_book_center_tf",
            output="screen",
            arguments=[
                "--x", "-0.044188468734586754",
                "--y", "-0.019233968272364746",
                "--z", "-0.2167544560075749",
                "--qx", "0.7235523907813395",
                "--qy", "-0.012252765490307429",
                "--qz", "-0.6901601779617551",
                "--qw", "0.0008580724014047088",
                "--frame-id", "apriltag_36h11_0",
                "--child-frame-id", "calibrated_book_center",
            ],
        ),
    ])
