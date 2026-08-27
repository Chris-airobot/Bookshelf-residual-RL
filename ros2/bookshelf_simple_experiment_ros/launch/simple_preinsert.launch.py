"""Start only the isolated detector and simple pre-insertion node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config = PathJoinSubstitution([
        FindPackageShare("bookshelf_simple_experiment_ros"),
        "config",
        "simple_preinsert.yaml",
    ])
    allow_execution = LaunchConfiguration("allow_execution")
    image_topic = LaunchConfiguration("image_topic")
    depth_topic = LaunchConfiguration("depth_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")
    return LaunchDescription([
        DeclareLaunchArgument("allow_execution", default_value="false"),
        DeclareLaunchArgument("image_topic", default_value="/camera/color/image_raw"),
        DeclareLaunchArgument(
            "depth_topic", default_value="/camera/aligned_depth_to_color/image_raw"
        ),
        DeclareLaunchArgument(
            "camera_info_topic", default_value="/camera/color/camera_info"
        ),
        LogInfo(msg=[
            "Simple pre-insertion workflow. Execution enabled: ", allow_execution,
            ". Detection never commands motion; call ",
            "/bookshelf_simple/plan_and_execute_preinsert explicitly.",
        ]),
        Node(
            package="bookshelf_simple_experiment_ros",
            executable="slot_detector",
            name="rgbd_slot_detector",
            output="screen",
            parameters=[config, {
                "image_topic": image_topic,
                "depth_topic": depth_topic,
                "camera_info_topic": camera_info_topic,
            }],
        ),
        Node(
            package="bookshelf_simple_experiment_ros",
            executable="simple_preinsert",
            name="simple_preinsert",
            output="screen",
            parameters=[config, {
                "allow_execution": ParameterValue(allow_execution, value_type=bool),
            }],
        ),
    ])
