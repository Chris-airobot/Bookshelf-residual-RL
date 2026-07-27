from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from pathlib import Path


def generate_launch_description():
    launch_dir = Path(__file__).resolve().parent
    camera_launch = launch_dir / "camera_setup.launch.py"
    robot_launch = launch_dir / "robot_setup.launch.py"

    wrist_serial_arg = DeclareLaunchArgument(
        "wrist_serial",
        default_value="242322078188",
        description="Serial number for the eye-in-hand wrist camera",
    )
    external_serial_arg = DeclareLaunchArgument(
        "external_serial",
        default_value="332322070806",
        description="Serial number for the external eye-on-base camera",
    )

    wrist_camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(camera_launch)),
        launch_arguments={
            "camera_name": "wrist_camera",
            "camera_namespace": "",
            "serial_no": LaunchConfiguration("wrist_serial"),
        }.items(),
    )

    external_camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(camera_launch)),
        launch_arguments={
            "camera_name": "external_camera",
            "camera_namespace": "",
            "serial_no": LaunchConfiguration("external_serial"),
        }.items(),
    )

    robot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(robot_launch))
    )

    wrist_handeye_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="link_eef_to_wrist_camera_link_tf",
        arguments=[
            "0.064694", "-0.017286", "0.018312",
            "0.711047", "-0.001225", "0.703143", "-0.000110",
            "link_eef", "wrist_camera_link",
        ],
    )

    return LaunchDescription([
        wrist_serial_arg,
        external_serial_arg,
        robot,
        wrist_camera,
        external_camera,
        wrist_handeye_tf,
    ])
