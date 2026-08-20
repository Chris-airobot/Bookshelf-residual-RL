"""Start the physical observation stack and automatic logging without planning."""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    TimerAction,
)
from launch.events import Shutdown
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


CAPTURE_CONDITIONS = {"unspecified", "no_book", "book_attached"}


def _bounded_capture(context):
    condition = LaunchConfiguration("capture_condition").perform(context).strip()
    if condition not in CAPTURE_CONDITIONS:
        expected = ", ".join(sorted(CAPTURE_CONDITIONS))
        raise RuntimeError(
            f"capture_condition must be one of {expected}; got {condition!r}"
        )

    duration_s = float(
        LaunchConfiguration("capture_duration_s").perform(context)
    )
    if duration_s < 0.0:
        raise RuntimeError("capture_duration_s must be non-negative")
    if duration_s == 0.0:
        return [LogInfo(msg=f"Stationary capture condition: {condition}; manual stop")]
    return [
        LogInfo(
            msg=(
                f"Stationary capture condition: {condition}; "
                f"automatic clean shutdown after {duration_s:.1f} seconds"
            )
        ),
        TimerAction(
            period=duration_s,
            actions=[
                EmitEvent(
                    event=Shutdown(
                        reason=f"stationary {condition} capture duration complete"
                    )
                )
            ],
        ),
    ]


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
            DeclareLaunchArgument(
                "record_raw_replay_inputs",
                default_value="false",
                description=(
                    "Record raw RGB-D inputs for direct offline replay. Enable "
                    "for bounded stationary dataset captures."
                ),
            ),
            DeclareLaunchArgument(
                "capture_condition",
                default_value="unspecified",
                description="unspecified, no_book, or book_attached.",
            ),
            DeclareLaunchArgument(
                "capture_duration_s",
                default_value="0.0",
                description="Automatic shutdown delay; 0 keeps the launch running.",
            ),
            DeclareLaunchArgument("minimum_free_space_gb", default_value="5.0"),
            DeclareLaunchArgument(
                "start_hardware_bringup",
                default_value="true",
                description=(
                    "Start xArm, MoveIt, camera, hand-eye TF, and calibrated "
                    "book detection. Disable when the dedicated hardware "
                    "launch already owns them."
                ),
            ),
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
                    "Starting PHYSICAL EXPERIMENT OBSERVATION BRINGUP: automatic "
                    "logging, RGB-D slot detection, frozen-slot verification, and "
                    "live held-book pose checking. Hardware bringup is controlled "
                    "by start_hardware_bringup. This launch starts no policy "
                    "executor, plan request, gripper command, or trajectory command."
                )
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(hardware_launch),
                condition=IfCondition(
                    LaunchConfiguration("start_hardware_bringup")
                ),
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
                    "record_raw_replay_inputs": LaunchConfiguration(
                        "record_raw_replay_inputs"
                    ),
                    "capture_condition": LaunchConfiguration("capture_condition"),
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
                        "detected_book_frame": "target_book_center",
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
            OpaqueFunction(function=_bounded_capture),
        ]
    )
