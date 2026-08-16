"""Start the physical observation stack and automatic logging without planning."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    hardware_launch = PathJoinSubstitution(
        [
            FindPackageShare("bookshelf_policy_ros"),
            "launch",
            "marker_vision_bringup.launch.py",
        ]
    )
    logging_launch = PathJoinSubstitution(
        [
            FindPackageShare("bookshelf_shadow_ros"),
            "launch",
            "experiment_logging.launch.py",
        ]
    )
    slot_check_launch = PathJoinSubstitution(
        [
            FindPackageShare("bookshelf_shadow_ros"),
            "launch",
            "static_slot_environment_check.launch.py",
        ]
    )
    frozen_check_output = PathJoinSubstitution(
        [
            LaunchConfiguration("environment_check_output_root"),
            LaunchConfiguration("trial_name"),
            "frozen_check",
        ]
    )
    held_book_check_output = PathJoinSubstitution(
        [
            LaunchConfiguration("environment_check_output_root"),
            LaunchConfiguration("trial_name"),
            "held_book_pose_check",
        ]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "trial_name",
                description="Unique physical trial identifier used by all logs.",
            ),
            DeclareLaunchArgument(
                "trial_slot_config",
                description="Human-approved trial_static_slot.yaml.",
            ),
            DeclareLaunchArgument(
                "scene_config",
                description=(
                    "Reviewed physical scene YAML containing the fixed "
                    "T_link_tcp_book used by MoveIt."
                ),
            ),
            DeclareLaunchArgument(
                "repository_path",
                default_value="/home/riot/Chris/bookshelf-unified",
            ),
            DeclareLaunchArgument(
                "policy_bundle",
                default_value=(
                    "/home/riot/BookshelfFiles/trained_models/"
                    "bookshelf_residual_2026-07-08_shadow_actor.npz"
                ),
            ),
            DeclareLaunchArgument(
                "activation_envelope",
                default_value=(
                    "/home/riot/BookshelfFiles/policy_activation_envelopes/"
                    "simulator_local_2026-08-08.json"
                ),
            ),
            DeclareLaunchArgument(
                "experiment_output_root",
                default_value="/home/riot/BookshelfFiles/experiment_logs",
            ),
            DeclareLaunchArgument(
                "environment_check_output_root",
                default_value=(
                    "/home/riot/BookshelfFiles/experiment_logs/environment_checks"
                ),
            ),
            DeclareLaunchArgument("record_camera", default_value="true"),
            DeclareLaunchArgument("minimum_free_space_gb", default_value="5.0"),
            DeclareLaunchArgument(
                "enable_calibrated_book_detection",
                default_value="true",
                description=(
                    "Detect ArUco Original ID 0 and publish the measured book frame."
                ),
            ),
            DeclareLaunchArgument(
                "book_detection_target_samples", default_value="250"
            ),
            DeclareLaunchArgument(
                "book_pose_required_stable_samples", default_value="30"
            ),
            DeclareLaunchArgument(
                "show_rviz",
                default_value="false",
                description="Keep false over SSH; opt in only on the Riot desktop.",
            ),
            LogInfo(
                msg=(
                    "Starting PHYSICAL EXPERIMENT OBSERVATION BRINGUP: xArm, "
                    "camera, TF, MoveIt, automatic logging, RGB-D slot detection, "
                    "frozen-slot verification, and live held-book pose checking. "
                    "This launch starts no policy executor, plan request, gripper "
                    "command, or trajectory command."
                )
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(hardware_launch),
                launch_arguments={
                    "enable_robot_control": "true",
                    "enable_calibrated_book_detection": LaunchConfiguration(
                        "enable_calibrated_book_detection"
                    ),
                    "enable_legacy_three_book_detection": "false",
                    "calibration_output_dir": held_book_check_output,
                    "calibration_target_samples": LaunchConfiguration(
                        "book_detection_target_samples"
                    ),
                    "show_rviz": LaunchConfiguration("show_rviz"),
                }.items(),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(logging_launch),
                launch_arguments={
                    "trial_name": LaunchConfiguration("trial_name"),
                    "output_root": LaunchConfiguration("experiment_output_root"),
                    "repository_path": LaunchConfiguration("repository_path"),
                    "policy_bundle": LaunchConfiguration("policy_bundle"),
                    "activation_envelope": LaunchConfiguration(
                        "activation_envelope"
                    ),
                    "record_camera": LaunchConfiguration("record_camera"),
                    "minimum_free_space_gb": LaunchConfiguration(
                        "minimum_free_space_gb"
                    ),
                }.items(),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(slot_check_launch),
                launch_arguments={
                    "check_config": LaunchConfiguration("trial_slot_config"),
                    "output_dir": frozen_check_output,
                    "start_live_detector": "true",
                }.items(),
            ),
            Node(
                package="bookshelf_guarded_control_ros",
                executable="held_book_pose_check",
                name="held_book_pose_check",
                output="screen",
                parameters=[
                    {
                        "scene_config_path": LaunchConfiguration("scene_config"),
                        "required_stable_samples": ParameterValue(
                            LaunchConfiguration(
                                "book_pose_required_stable_samples"
                            ),
                            value_type=int,
                        ),
                        "output_dir": held_book_check_output,
                    }
                ],
            ),
        ]
    )
