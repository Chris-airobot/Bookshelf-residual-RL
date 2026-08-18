"""Overlay approved geometry and current RGB-D detection without robot motion."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    scene_config = LaunchConfiguration("scene_config")
    check_config = LaunchConfiguration("check_config")
    output_dir = LaunchConfiguration("output_dir")
    start_live_detector = LaunchConfiguration("start_live_detector")
    show_rviz = LaunchConfiguration("show_rviz")
    show_debug_image = LaunchConfiguration("show_debug_image")
    enable_orientation_audit = LaunchConfiguration("enable_orientation_audit")
    orientation_audit_samples = LaunchConfiguration("orientation_audit_samples")
    use_sim_time = LaunchConfiguration("use_sim_time")
    image_topic = LaunchConfiguration("image_topic")
    depth_topic = LaunchConfiguration("depth_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")

    detector = Node(
        package="bookshelf_shadow_ros",
        executable="rgbd_slot_detector",
        name="rgbd_slot_detector",
        output="screen",
        condition=IfCondition(start_live_detector),
        parameters=[
            {
                "image_topic": image_topic,
                "depth_topic": depth_topic,
                "camera_info_topic": camera_info_topic,
                "debug_image_topic": "/slot_detector/debug_image",
                "use_sim_time": ParameterValue(use_sim_time, value_type=bool),
            }
        ],
    )

    coarse_scene = Node(
        package="bookshelf_shadow_ros",
        executable="offline_scene_visualizer",
        # Keep this name aligned with the node-scoped scene YAML. Renaming it
        # causes ROS 2 to ignore the measured scene parameters silently.
        name="offline_scene_visualizer",
        output="screen",
        parameters=[
            scene_config,
            {
                "publish_joint_states": False,
                "use_sim_time": ParameterValue(use_sim_time, value_type=bool),
            },
        ],
    )

    slot_check = Node(
        package="bookshelf_shadow_ros",
        executable="static_slot_environment_check",
        name="static_slot_environment_check",
        output="screen",
        parameters=[
            check_config,
            {
                "output_dir": output_dir,
                "use_sim_time": ParameterValue(use_sim_time, value_type=bool),
            },
        ],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="real_scene_detection_overlay_rviz",
        output="screen",
        arguments=["-d", LaunchConfiguration("rviz_config")],
        parameters=[
            {"use_sim_time": ParameterValue(use_sim_time, value_type=bool)}
        ],
        condition=IfCondition(show_rviz),
    )

    debug_image_view = Node(
        package="rqt_image_view",
        executable="rqt_image_view",
        name="slot_detector_debug_image_view",
        output="screen",
        arguments=["/slot_detector/debug_image"],
        condition=IfCondition(show_debug_image),
    )

    orientation_audit = Node(
        package="bookshelf_shadow_ros",
        executable="slot_orientation_audit",
        name="slot_orientation_audit",
        output="screen",
        condition=IfCondition(enable_orientation_audit),
        parameters=[
            {
                "output_dir": output_dir,
                "target_samples": ParameterValue(
                    orientation_audit_samples,
                    value_type=int,
                ),
                "use_sim_time": ParameterValue(use_sim_time, value_type=bool),
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "scene_config",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("bookshelf_shadow_ros"),
                        "config",
                        "offline_physical_scene_visualization.yaml",
                    ]
                ),
            ),
            DeclareLaunchArgument(
                "check_config",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("bookshelf_shadow_ros"),
                        "config",
                        "real_scene_detection_overlay.yaml",
                    ]
                ),
            ),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("bookshelf_shadow_ros"),
                        "rviz",
                        "real_scene_detection_overlay.rviz",
                    ]
                ),
            ),
            DeclareLaunchArgument(
                "output_dir", default_value="/tmp/bookshelf_real_scene_overlay"
            ),
            DeclareLaunchArgument("start_live_detector", default_value="true"),
            DeclareLaunchArgument("show_rviz", default_value="true"),
            DeclareLaunchArgument(
                "show_debug_image",
                default_value="true",
                description="Open the annotated RGB slot-detection image.",
            ),
            DeclareLaunchArgument(
                "enable_orientation_audit",
                default_value="false",
                description=(
                    "Write read-only live-versus-reference slot orientation statistics."
                ),
            ),
            DeclareLaunchArgument(
                "orientation_audit_samples",
                default_value="700",
                description="Number of paired live orientation samples to audit.",
            ),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument(
                "image_topic", default_value="/camera/color/image_raw"
            ),
            DeclareLaunchArgument(
                "depth_topic",
                default_value="/camera/aligned_depth_to_color/image_raw",
            ),
            DeclareLaunchArgument(
                "camera_info_topic", default_value="/camera/color/camera_info"
            ),
            LogInfo(
                msg=(
                    "Starting READ-ONLY real-scene detection overlay. It expects "
                    "the hardware bringup to own robot_description, TF, and "
                    "joint_states. No planner, controller, trajectory, gripper, "
                    "executor, or robot-command interface is created."
                )
            ),
            detector,
            coarse_scene,
            slot_check,
            orientation_audit,
            rviz,
            debug_image_view,
        ]
    )
