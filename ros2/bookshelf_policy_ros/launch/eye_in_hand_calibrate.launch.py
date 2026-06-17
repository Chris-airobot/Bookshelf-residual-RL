from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare("easy_handeye2"),
                    "launch",
                    "calibrate.launch.py",
                ])
            ),
            launch_arguments={
                "name": "xarm7_eye_in_hand",
                "calibration_type": "eye_in_hand",

                "tracking_base_frame": "camera_color_optical_frame",
                "tracking_marker_frame": "charuco_board",

                "robot_base_frame": "link_base",
                "robot_effector_frame": "link_eef",
            }.items(),
        )
    ])
