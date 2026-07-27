from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
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
