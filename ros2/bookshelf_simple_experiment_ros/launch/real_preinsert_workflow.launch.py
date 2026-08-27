"""Real camera-to-preinsert workflow; expects xArm MoveIt already running."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare("bookshelf_simple_experiment_ros")
    config = PathJoinSubstitution([package_share, "config", "simple_preinsert.yaml"])
    rviz_config = PathJoinSubstitution([
        package_share, "rviz", "real_preinsert_workflow.rviz"
    ])
    image_topic = LaunchConfiguration("image_topic")
    depth_topic = LaunchConfiguration("depth_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")
    frozen_slot_output = LaunchConfiguration("frozen_slot_output")
    show_rviz = LaunchConfiguration("show_rviz")

    return LaunchDescription([
        DeclareLaunchArgument("image_topic", default_value="/camera/color/image_raw"),
        DeclareLaunchArgument(
            "depth_topic", default_value="/camera/aligned_depth_to_color/image_raw"
        ),
        DeclareLaunchArgument(
            "camera_info_topic", default_value="/camera/color/camera_info"
        ),
        DeclareLaunchArgument(
            "frozen_slot_output",
            default_value="/tmp/bookshelf_simple_frozen_slot.yaml",
        ),
        DeclareLaunchArgument("show_rviz", default_value="true"),
        LogInfo(msg=(
            "REAL PREINSERT: manually position the empty xArm, inspect the slot, "
            "then confirm in order with /bookshelf_simple/accept_slot, "
            "/bookshelf_simple/plan_preinsert, and "
            "/bookshelf_simple/execute_preinsert. This launch expects the camera "
            "and real xArm MoveIt stack to already be running."
        )),
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
                "allow_execution": True,
                "require_slot_acceptance": True,
                "separate_execution_confirmation": True,
                "print_target_diagnostics": True,
                "frozen_slot_output_path": frozen_slot_output,
            }],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="bookshelf_simple_real_preinsert_rviz",
            output="screen",
            arguments=["-d", rviz_config],
            condition=IfCondition(show_rviz),
        ),
    ])
