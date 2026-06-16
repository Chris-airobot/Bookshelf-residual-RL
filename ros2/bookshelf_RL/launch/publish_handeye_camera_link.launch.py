from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="link_eef_to_camera_link_tf",
            arguments=[
                # translation: x y z
                "0.064694", "-0.017286", "0.018312",

                # rotation quaternion: qx qy qz qw
                "0.711047", "-0.001225", "0.703143", "-0.000110",

                # parent child
                "link_eef",
                "camera_link",
            ],
        )
    ])
