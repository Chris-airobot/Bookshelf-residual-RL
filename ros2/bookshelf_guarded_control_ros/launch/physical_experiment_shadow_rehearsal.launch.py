"""Start one duplicate-free, read-only physical policy rehearsal."""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

from bookshelf_guarded_control_ros.rehearsal_configuration import (
    validate_shadow_rehearsal_assets,
)


def _validate_inputs(context):
    result = validate_shadow_rehearsal_assets(
        LaunchConfiguration("approved_config").perform(context),
        LaunchConfiguration("policy_bundle").perform(context),
        LaunchConfiguration("activation_envelope").perform(context),
    )
    return [
        LogInfo(
            msg=(
                "REHEARSAL INPUTS VERIFIED: candidate="
                f"{result['candidate_id']}; slot={result['slot_pose_source']}; "
                f"book={result['book_pose_source']}; execution authorized=false."
            )
        )
    ]


def generate_launch_description():
    observation_launch = PathJoinSubstitution(
        [
            FindPackageShare("bookshelf_guarded_control_ros"),
            "launch",
            "physical_experiment_observation_bringup.launch.py",
        ]
    )
    shadow_launch = PathJoinSubstitution(
        [
            FindPackageShare("bookshelf_shadow_ros"),
            "launch",
            "policy_hardware_shadow.launch.py",
        ]
    )
    audit_output = PathJoinSubstitution(
        [
            LaunchConfiguration("environment_check_output_root"),
            LaunchConfiguration("trial_name"),
            "policy_shadow_audit",
        ]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "trial_name",
                description="Unique read-only rehearsal identifier.",
            ),
            DeclareLaunchArgument(
                "approved_config",
                description=(
                    "One promoted trial_static_slot.yaml used for slot, book, "
                    "scene, and policy observation parameters."
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
                "record_raw_replay_inputs", default_value="false"
            ),
            DeclareLaunchArgument(
                "capture_condition", default_value="book_attached"
            ),
            DeclareLaunchArgument("capture_duration_s", default_value="0.0"),
            DeclareLaunchArgument("minimum_free_space_gb", default_value="5.0"),
            DeclareLaunchArgument(
                "book_detection_target_samples", default_value="250"
            ),
            DeclareLaunchArgument(
                "book_pose_required_stable_samples", default_value="30"
            ),
            DeclareLaunchArgument("show_rviz", default_value="false"),
            DeclareLaunchArgument("enable_policy_audit", default_value="true"),
            DeclareLaunchArgument("policy_audit_samples", default_value="1200"),
            DeclareLaunchArgument(
                "reference_slot_width_m",
                default_value="0.0",
                description="Optional independently measured slot width.",
            ),
            OpaqueFunction(function=_validate_inputs),
            LogInfo(
                msg=(
                    "Starting UNIFIED READ-ONLY PHYSICAL SHADOW REHEARSAL. "
                    "One RGB-D slot detector is owned by observation bringup; "
                    "policy shadow reuses its topics. No planner, executor, "
                    "gripper command, or trajectory command is launched."
                )
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(observation_launch),
                launch_arguments={
                    "trial_name": LaunchConfiguration("trial_name"),
                    "trial_slot_config": LaunchConfiguration("approved_config"),
                    "scene_config": LaunchConfiguration("approved_config"),
                    "repository_path": LaunchConfiguration("repository_path"),
                    "policy_bundle": LaunchConfiguration("policy_bundle"),
                    "activation_envelope": LaunchConfiguration(
                        "activation_envelope"
                    ),
                    "experiment_output_root": LaunchConfiguration(
                        "experiment_output_root"
                    ),
                    "environment_check_output_root": LaunchConfiguration(
                        "environment_check_output_root"
                    ),
                    "record_camera": LaunchConfiguration("record_camera"),
                    "record_raw_replay_inputs": LaunchConfiguration(
                        "record_raw_replay_inputs"
                    ),
                    "capture_condition": LaunchConfiguration("capture_condition"),
                    "capture_duration_s": LaunchConfiguration("capture_duration_s"),
                    "minimum_free_space_gb": LaunchConfiguration(
                        "minimum_free_space_gb"
                    ),
                    "enable_calibrated_book_detection": "true",
                    "book_detection_target_samples": LaunchConfiguration(
                        "book_detection_target_samples"
                    ),
                    "book_pose_required_stable_samples": LaunchConfiguration(
                        "book_pose_required_stable_samples"
                    ),
                    "show_rviz": LaunchConfiguration("show_rviz"),
                }.items(),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(shadow_launch),
                launch_arguments={
                    "adapter_config": LaunchConfiguration("approved_config"),
                    "policy_bundle": LaunchConfiguration("policy_bundle"),
                    "activation_envelope": LaunchConfiguration(
                        "activation_envelope"
                    ),
                    "require_activation_envelope": "true",
                    "enable_audit": LaunchConfiguration("enable_policy_audit"),
                    "audit_output_dir": audit_output,
                    "audit_samples": LaunchConfiguration("policy_audit_samples"),
                    "reference_slot_width_m": LaunchConfiguration(
                        "reference_slot_width_m"
                    ),
                    "start_live_detector": "false",
                }.items(),
            ),
        ]
    )
