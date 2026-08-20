"""Deploy the approved physical policy pipeline without starting hardware."""

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


def _as_bool(value):
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _execution_actions(context):
    mode = LaunchConfiguration("execution_mode").perform(context).strip().lower()
    if mode not in ("shadow", "plan_only", "single_step"):
        raise RuntimeError(
            "execution_mode must be shadow, plan_only, or single_step"
        )
    if mode == "shadow":
        return [
            LogInfo(
                msg=(
                    "Execution mode=shadow: no planning-scene manager, planner, "
                    "or execution action client is created."
                )
            )
        ]

    approved_config = LaunchConfiguration("approved_config").perform(context)
    policy_bundle = LaunchConfiguration("policy_bundle").perform(context)
    overrides = guarded_policy_tool_overrides(approved_config, policy_bundle)
    scene_handoff_allowed = _as_bool(
        LaunchConfiguration("permit_local_scene_handoff").perform(context)
    )
    overrides["planning_scene_complete"] = scene_handoff_allowed

    actions = [
        Node(
            package="bookshelf_guarded_control_ros",
            executable="bookshelf_scene_manager",
            name="bookshelf_scene_manager",
            output="screen",
            parameters=[
                approved_config,
                {
                    "scene_config_path": approved_config,
                    "allow_local_insertion": scene_handoff_allowed,
                },
            ],
        )
    ]
    if mode == "plan_only":
        actions.extend(
            [
                LogInfo(
                    msg=(
                        "Execution mode=plan_only: MoveIt plans and trajectory "
                        "checks are enabled, but no execution client exists."
                    )
                ),
                Node(
                    package="bookshelf_guarded_control_ros",
                    executable="policy_tool_plan_checker",
                    name="policy_tool_plan_checker",
                    output="screen",
                    parameters=[
                        LaunchConfiguration("plan_checker_config"),
                        overrides,
                    ],
                ),
            ]
        )
        return actions

    overrides.update(
        {
            "dry_run": False,
            "allow_execution": True,
            "approval_token": LaunchConfiguration("execution_approval_token").perform(
                context
            ),
        }
    )
    actions.extend(
        [
            LogInfo(
                msg=(
                    "Execution mode=single_step: at most one recent checked "
                    "trajectory may run after a matching approval token."
                )
            ),
            Node(
                package="bookshelf_guarded_control_ros",
                executable="guarded_policy_tool_executor",
                name="guarded_policy_tool_executor",
                output="screen",
                parameters=[
                    LaunchConfiguration("executor_config"),
                    overrides,
                ],
            ),
        ]
    )
    return actions


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
    default_plan_checker_config = PathJoinSubstitution(
        [package_share, "config", "policy_tool_plan_checker.yaml"]
    )
    default_executor_config = PathJoinSubstitution(
        [package_share, "config", "guarded_policy_tool_executor.yaml"]
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
                "execution_mode",
                default_value="shadow",
                description="shadow, plan_only, or single_step.",
            ),
            DeclareLaunchArgument(
                "permit_local_scene_handoff",
                default_value="false",
                description=(
                    "Allow the explicit local-insertion scene service handoff."
                ),
            ),
            DeclareLaunchArgument(
                "execution_approval_token",
                default_value="DISABLED",
                description="One-shot token required only by single_step mode.",
            ),
            DeclareLaunchArgument(
                "plan_checker_config", default_value=default_plan_checker_config
            ),
            DeclareLaunchArgument(
                "executor_config", default_value=default_executor_config
            ),
            OpaqueFunction(function=_validate_inputs),
            LogInfo(
                msg=(
                    "Starting POLICY-ONLY DEPLOYMENT. It reuses the "
                    "existing robot, camera, TF, and target_book_center from "
                    "physical_hardware_bringup.launch.py. It never starts an xArm "
                    "driver, MoveIt stack, camera driver, or gripper command."
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
                    "enable_audit": LaunchConfiguration("enable_policy_audit"),
                    "audit_output_dir": audit_output,
                    "audit_samples": LaunchConfiguration("policy_audit_samples"),
                    "reference_slot_width_m": LaunchConfiguration(
                        "reference_slot_width_m"
                    ),
                    "start_live_detector": "false",
                }.items(),
            ),
            OpaqueFunction(function=_execution_actions),
            OpaqueFunction(function=_bounded_capture),
        ]
    )
