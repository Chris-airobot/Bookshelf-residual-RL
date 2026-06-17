from pathlib import Path

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    launch_dir = Path(__file__).resolve().parent
    pkg_dir = launch_dir.parent

    robot_launch = launch_dir / "robot_setup.launch.py"
    camera_launch = launch_dir / "camera_setup.launch.py"
    handeye_tf_launch = launch_dir / "publish_handeye_camera_link.launch.py"
    charuco_script = pkg_dir / "scripts" / "charuco_tf_pub.py"

    return LaunchDescription([
        # 1. Robot driver / robot TF / MoveIt
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(robot_launch))
        ),

        # 2. RealSense camera
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(camera_launch))
        ),

        # 3. Calibrated static TF:
        # link_eef -> camera_link
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(handeye_tf_launch))
        ),

        # 4. ChArUco board live TF:
        # camera_color_optical_frame -> charuco_board
        #
        # Delay a little so camera topics have time to start.
        TimerAction(
            period=3.0,
            actions=[
                ExecuteProcess(
                    cmd=[
                        "python3",
                        str(charuco_script),
                        "--image_topic", "/camera/color/image_raw",
                        "--camera_info_topic", "/camera/color/camera_info",
                        "--camera_frame", "camera_color_optical_frame",
                        "--board_frame", "charuco_board",
                    ],
                    output="screen",
                )
            ],
        ),
    ])
