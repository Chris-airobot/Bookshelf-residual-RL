"""Deploy policy calculation or direct local Cartesian control."""

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
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

from bookshelf_guarded_control_ros.rehearsal_configuration import (
    guarded_policy_tool_overrides,
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
                "POLICY INPUTS VERIFIED: candidate="
                f"{result['candidate_id']}; slot={result['slot_pose_source']}; "
                f"book={result['book_pose_source']}."
            )
        )
    ]


def _control_actions(context):
    operation = LaunchConfiguration("operation").perform(context).strip().lower()
    if operation not in ("calculate", "control"):
        raise RuntimeError("operation must be calculate or control")
    if operation == "calculate":
        return [
            LogInfo(
                msg=(
                    "Operation=calculate: policy commands are calculated and "
                    "logged, but no robot-command client is created."
                )
            )
        ]

    approved_config = LaunchConfiguration("approved_config").perform(context)
    policy_bundle = LaunchConfiguration("policy_bundle").perform(context)
    overrides = guarded_policy_tool_overrides(approved_config, policy_bundle)
    overrides.pop("require_scene_status", None)
    overrides.pop("required_scene_mode", None)
    return [
        LogInfo(
            msg=(
                "Operation=control: policy deltas are sent directly to the "
                "xArm Cartesian servo service. MoveIt and the gripper are not "
                "used by this local-control process."
            )
        ),
        Node(
            package="bookshelf_guarded_control_ros",
            executable="direct_policy_servo",
            name="direct_policy_servo",
            output="screen",
            parameters=[
                LaunchConfiguration("servo_config"),
                overrides,
            ],
        ),
    ]


def _bounded_capture(context):
    duration_s = float(LaunchConfiguration("capture_duration_s").perform(context))
    if duration_s < 0.0:
        raise RuntimeError("capture_duration_s must be non-negative")
    if duration_s == 0.0:
        return []
    return [
        TimerAction(
            period=duration_s,
            actions=[
                EmitEvent(
                    event=Shutdown(
                        reason=(
                            "policy deployment duration complete after "
                            f"{duration_s:.1f}s"
                        )
                    )
                )
            ],
        )
    ]


def generate_launch_description():
    package_share = FindPackageShare("bookshelf_guarded_control_ros")
    default_servo_config = PathJoinSubstitution(
        [package_share, "config", "direct_policy_servo.yaml"]
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
            DeclareLaunchArgument("trial_name"),
            DeclareLaunchArgument(
                "approved_config",
                description=(
                    "Promoted trial_static_slot.yaml used by every policy and "
                    "safety component."
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
            DeclareLaunchArgument("record_camera", default_value="false"),
            DeclareLaunchArgument(
                "record_raw_replay_inputs", default_value="false"
            ),
            DeclareLaunchArgument(
                "capture_condition", default_value="book_attached"
            ),
            DeclareLaunchArgument("capture_duration_s", default_value="0.0"),
            DeclareLaunchArgument("minimum_free_space_gb", default_value="5.0"),
            DeclareLaunchArgument(
                "book_pose_required_stable_samples", default_value="30"
            ),
            DeclareLaunchArgument("enable_policy_audit", default_value="true"),
            DeclareLaunchArgument("policy_audit_samples", default_value="1200"),
            DeclareLaunchArgument("reference_slot_width_m", default_value="0.0"),
            DeclareLaunchArgument(
                "operation",
                default_value="calculate",
                description="calculate or control.",
            ),
            DeclareLaunchArgument(
                "servo_config", default_value=default_servo_config
            ),
            OpaqueFunction(function=_validate_inputs),
            LogInfo(
                msg=(
                    "Starting POLICY DEPLOYMENT. It reuses the "
                    "existing robot, camera, TF, and target_book_center from "
                    "physical_hardware_bringup.launch.py. It never starts an "
                    "xArm driver, MoveIt stack, camera driver, or gripper."
                )
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
                    "check_config": LaunchConfiguration("approved_config"),
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
                        "scene_config_path": LaunchConfiguration("approved_config"),
                        "detected_book_frame": "target_book_center",
                        "required_stable_samples": ParameterValue(
                            LaunchConfiguration("book_pose_required_stable_samples"),
                            value_type=int,
                        ),
                        "output_dir": held_book_check_output,
                    }
                ],
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
                    "block_on_activation_checks": "false",
                    "enable_audit": LaunchConfiguration("enable_policy_audit"),
                    "audit_output_dir": audit_output,
                    "audit_samples": LaunchConfiguration("policy_audit_samples"),
                    "reference_slot_width_m": LaunchConfiguration(
                        "reference_slot_width_m"
                    ),
                    "start_live_detector": "false",
                }.items(),
            ),
            OpaqueFunction(function=_control_actions),
            OpaqueFunction(function=_bounded_capture),
        ]
    )
