"""Replay real RGB-D into the plan-only preinsert workflow with fake xArm7."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare("bookshelf_simple_experiment_ros")
    config = PathJoinSubstitution([package_share, "config", "simple_preinsert.yaml"])
    bag_path = LaunchConfiguration("bag_path")
    rviz_config = LaunchConfiguration("rviz_config")
    preview_rviz = LaunchConfiguration("preview_rviz")

    fake_moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare("xarm_moveit_config"),
            "launch",
            "_robot_moveit_fake.launch.py",
        ])),
        launch_arguments={
            "dof": "7",
            "robot_type": "xarm",
            "limited": "false",
            "add_gripper": "true",
            "no_gui_ctrl": "false",
            "show_rviz": preview_rviz,
            "rviz_config": rviz_config,
        }.items(),
    )

    handeye_tf = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare("bookshelf_policy_ros"),
            "launch",
            "publish_handeye_camera_link.launch.py",
        ]))
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "bag_path",
            default_value=PathJoinSubstitution([
                EnvironmentVariable("HOME"),
                "BookshelfFiles",
                "real_rgbd",
                "slot_view_01_complete",
                "slot_view_01",
            ]),
        ),
        DeclareLaunchArgument(
            "rviz_config",
            default_value=PathJoinSubstitution([
                package_share,
                "rviz",
                "offline_rosbag_preinsert_workflow.rviz",
            ]),
        ),
        DeclareLaunchArgument("preview_rviz", default_value="true"),
        LogInfo(msg=(
            "SOFTWARE-ONLY ROSBAG PREVIEW: recorded RGB-D, unchanged slot "
            "detector and preinsert planner, official fake xArm7 MoveIt, "
            "plan-only triggering, and one RViz. No physical xArm, RealSense "
            "hardware, Servo, gripper command, PPO, or execution is started."
        )),
        fake_moveit,
        handeye_tf,
        # Exact RealSense static transforms recovered from the reviewed
        # slot-view hardware capture. The hand-eye transform above remains the
        # canonical calibrated link_eef -> camera_link transform.
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="offline_camera_link_to_color_frame_tf",
            arguments=[
                "-0.000413252", "0.014934985", "-0.000171136",
                "0.013329396", "0.002292214", "0.002319336", "0.999905825",
                "camera_link", "camera_color_frame",
            ],
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="offline_camera_color_optical_tf",
            arguments=[
                "0.0", "0.0", "0.0",
                "-0.5", "0.5", "-0.5", "0.5",
                "camera_color_frame", "camera_color_optical_frame",
            ],
        ),
        Node(
            package="bookshelf_simple_experiment_ros",
            executable="slot_detector",
            name="rgbd_slot_detector",
            output="screen",
            parameters=[config, {
                "image_topic": "/camera/color/image_raw",
                "depth_topic": "/camera/aligned_depth_to_color/image_raw",
                "camera_info_topic": "/slot_detector/camera_info",
            }],
        ),
        Node(
            package="bookshelf_simple_experiment_ros",
            executable="simple_preinsert",
            name="simple_preinsert",
            output="screen",
            parameters=[config, {
                "allow_execution": False,
                "require_slot_acceptance": False,
                "separate_execution_confirmation": True,
                "print_target_diagnostics": True,
                "maximum_goal_joint_delta_rad": 3.2,
            }],
        ),
        Node(
            package="bookshelf_simple_experiment_ros",
            executable="virtual_trigger",
            name="offline_rosbag_preinsert_trigger",
            output="screen",
            parameters=[{
                # Recorded xArm scan pose associated with the real slot-view capture.
                "initial_joint_positions": [
                    1.283901572227478,
                    1.654488205909729,
                    5.026281833648682,
                    1.0371201038360596,
                    3.4988553524017334,
                    0.9406297206878662,
                    4.46493673324585,
                ],
                "initial_move_duration_s": 2.0,
            }],
        ),
        ExecuteProcess(
            cmd=[
                "ros2", "bag", "play", bag_path,
                "--loop",
                "--delay", "4.0",
                "--disable-keyboard-controls",
                "--topics",
                "/camera/color/image_raw",
                "/camera/color/camera_info",
                "/camera/aligned_depth_to_color/image_raw",
                "/camera/aligned_depth_to_color/camera_info",
                "--remap",
                "/camera/color/camera_info:=/slot_detector/camera_info",
            ],
            name="slot_view_01_player",
            output="screen",
        ),
    ])
