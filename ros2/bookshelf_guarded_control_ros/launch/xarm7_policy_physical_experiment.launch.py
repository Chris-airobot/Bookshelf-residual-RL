"""Bring up one fail-closed physical xArm bookshelf episode."""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

from bookshelf_guarded_control_ros.physical_episode_coordinator_math import (
    HARDWARE_AUTHORIZATION_TOKEN,
    validate_episode_operation,
)
from bookshelf_guarded_control_ros.rehearsal_configuration import (
    physical_episode_geometry_overrides,
)


def _as_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise RuntimeError(f"expected a boolean launch value, got {value!r}")


def _coordinator_actions(context):
    operation = validate_episode_operation(
        LaunchConfiguration("operation").perform(context),
        LaunchConfiguration("authorization_token").perform(context),
    )
    boundary_confirmed = _as_bool(
        LaunchConfiguration("physical_release_boundary_confirmed").perform(context)
    )
    if operation == "control" and not boundary_confirmed:
        raise RuntimeError(
            "control requires physical_release_boundary_confirmed:=true after "
            "reviewing a calculate-only geometry report"
        )

    geometry = physical_episode_geometry_overrides(
        LaunchConfiguration("approved_config").perform(context),
        expected_base_frame=LaunchConfiguration("base_frame").perform(context),
        expected_eef_frame=LaunchConfiguration("eef_frame").perform(context),
        expected_tcp_frame=LaunchConfiguration("tcp_frame").perform(context),
        expected_book_frame=LaunchConfiguration("target_book_frame").perform(
            context
        ),
    )
    geometry.update(
        {
            "operation": operation,
            "authorization_token": LaunchConfiguration("authorization_token"),
            "start_immediately": ParameterValue(
                LaunchConfiguration("start_immediately"), value_type=bool
            ),
            "base_frame": LaunchConfiguration("base_frame"),
            "eef_frame": LaunchConfiguration("eef_frame"),
            "tcp_frame": LaunchConfiguration("tcp_frame"),
            "physical_release_tcp_x_limit_m": ParameterValue(
                LaunchConfiguration("physical_release_tcp_x_limit_m"),
                value_type=float,
            ),
            "minimum_book_leading_penetration_m": ParameterValue(
                LaunchConfiguration("minimum_book_leading_penetration_m"),
                value_type=float,
            ),
            "push_target_trailing_depth_m": ParameterValue(
                LaunchConfiguration("push_target_trailing_depth_m"),
                value_type=float,
            ),
            "push_target_tolerance_m": ParameterValue(
                LaunchConfiguration("push_target_tolerance_m"), value_type=float
            ),
            "retreat_distance_m": ParameterValue(
                LaunchConfiguration("retreat_distance_m"), value_type=float
            ),
            "retreat_speed_m_s": ParameterValue(
                LaunchConfiguration("retreat_speed_m_s"), value_type=float
            ),
            "retreat_timeout_s": ParameterValue(
                LaunchConfiguration("retreat_timeout_s"), value_type=float
            ),
            "insert_timeout_s": ParameterValue(
                LaunchConfiguration("insert_timeout_s"), value_type=float
            ),
            "maximum_push_tcp_travel_m": ParameterValue(
                LaunchConfiguration("maximum_push_tcp_travel_m"), value_type=float
            ),
            "push_timeout_s": ParameterValue(
                LaunchConfiguration("push_timeout_s"), value_type=float
            ),
            "message_max_age_s": ParameterValue(
                LaunchConfiguration("message_max_age_s"), value_type=float
            ),
            "tf_max_age_s": ParameterValue(
                LaunchConfiguration("tf_max_age_s"), value_type=float
            ),
            "policy_control_enable_topic": LaunchConfiguration(
                "control_enable_topic"
            ),
            "twist_command_topic": LaunchConfiguration("twist_command_topic"),
            "gripper_action": LaunchConfiguration("gripper_action"),
            "gripper_joint_name": LaunchConfiguration("gripper_joint_name"),
            "gripper_open_position": ParameterValue(
                LaunchConfiguration("gripper_open_position"), value_type=float
            ),
            "gripper_closed_position": ParameterValue(
                LaunchConfiguration("gripper_closed_position"), value_type=float
            ),
            "gripper_move_duration_s": ParameterValue(
                LaunchConfiguration("gripper_move_duration_s"), value_type=float
            ),
        }
    )
    if operation == "calculate":
        message = (
            "CALCULATE-ONLY full-episode bringup: hardware, camera, marker, and "
            "policy observations run, but the coordinator creates no gripper or "
            "motion command interfaces."
        )
    else:
        message = (
            "AUTHORIZED PHYSICAL FULL EPISODE: policy INSERT, measured release, "
            "straight retreat, real gripper close, and live-marker-stopped PUSH."
        )
    return [
        LogInfo(msg=message),
        Node(
            package="bookshelf_guarded_control_ros",
            executable="physical_episode_coordinator",
            name="physical_episode_coordinator",
            output="screen",
            parameters=[geometry],
        ),
    ]


def generate_launch_description():
    hardware_launch = PathJoinSubstitution(
        [
            FindPackageShare("bookshelf_policy_ros"),
            "launch",
            "physical_hardware_bringup.launch.py",
        ]
    )
    policy_launch = PathJoinSubstitution(
        [
            FindPackageShare("bookshelf_guarded_control_ros"),
            "launch",
            "physical_policy_deployment.launch.py",
        ]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("trial_name"),
            DeclareLaunchArgument("approved_config"),
            DeclareLaunchArgument("policy_bundle"),
            DeclareLaunchArgument("activation_envelope"),
            DeclareLaunchArgument("robot_ip", default_value="192.168.1.209"),
            DeclareLaunchArgument("show_rviz", default_value="true"),
            DeclareLaunchArgument(
                "repository_path",
                default_value="/home/riot/Chris/bookshelf-unified",
            ),
            DeclareLaunchArgument(
                "experiment_output_root",
                default_value="/home/riot/BookshelfFiles/experiment_logs",
            ),
            DeclareLaunchArgument("enable_logging", default_value="true"),
            DeclareLaunchArgument("record_camera", default_value="true"),
            DeclareLaunchArgument("operation", default_value="calculate"),
            DeclareLaunchArgument("authorization_token", default_value=""),
            DeclareLaunchArgument("start_immediately", default_value="false"),
            DeclareLaunchArgument(
                "physical_release_boundary_confirmed", default_value="false"
            ),
            DeclareLaunchArgument("base_frame", default_value="link_base"),
            DeclareLaunchArgument("eef_frame", default_value="link_eef"),
            DeclareLaunchArgument("tcp_frame", default_value="link_tcp"),
            DeclareLaunchArgument(
                "target_book_frame", default_value="target_book_center"
            ),
            DeclareLaunchArgument(
                "control_enable_topic",
                default_value="/bookshelf_control/episode_enable",
            ),
            DeclareLaunchArgument(
                "twist_command_topic",
                default_value="/servo_server/delta_twist_cmds",
            ),
            DeclareLaunchArgument(
                "gripper_action",
                default_value=(
                    "/xarm_gripper_traj_controller/follow_joint_trajectory"
                ),
            ),
            DeclareLaunchArgument("gripper_joint_name", default_value="drive_joint"),
            DeclareLaunchArgument("gripper_open_position", default_value="0.0"),
            DeclareLaunchArgument("gripper_closed_position", default_value="0.85"),
            DeclareLaunchArgument("gripper_move_duration_s", default_value="1.5"),
            DeclareLaunchArgument(
                "physical_release_tcp_x_limit_m", default_value="-0.006"
            ),
            DeclareLaunchArgument(
                "minimum_book_leading_penetration_m", default_value="0.08"
            ),
            DeclareLaunchArgument(
                "push_target_trailing_depth_m", default_value="-0.012"
            ),
            DeclareLaunchArgument("push_target_tolerance_m", default_value="0.001"),
            DeclareLaunchArgument("retreat_distance_m", default_value="0.09"),
            DeclareLaunchArgument("retreat_speed_m_s", default_value="0.025"),
            DeclareLaunchArgument("retreat_timeout_s", default_value="10.0"),
            DeclareLaunchArgument("insert_timeout_s", default_value="120.0"),
            DeclareLaunchArgument(
                "maximum_push_tcp_travel_m", default_value="0.14"
            ),
            DeclareLaunchArgument("push_timeout_s", default_value="90.0"),
            DeclareLaunchArgument("message_max_age_s", default_value="0.5"),
            DeclareLaunchArgument("tf_max_age_s", default_value="0.5"),
            DeclareLaunchArgument(
                "maximum_total_translation_m", default_value="0.25"
            ),
            DeclareLaunchArgument(
                "authorization_token_help",
                default_value=HARDWARE_AUTHORIZATION_TOKEN,
                description="Informational exact token required by operation=control.",
            ),
            OpaqueFunction(function=_coordinator_actions),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(hardware_launch),
                launch_arguments={
                    "robot_ip": LaunchConfiguration("robot_ip"),
                    "show_rviz": LaunchConfiguration("show_rviz"),
                }.items(),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(policy_launch),
                launch_arguments={
                    "trial_name": LaunchConfiguration("trial_name"),
                    "approved_config": LaunchConfiguration("approved_config"),
                    "repository_path": LaunchConfiguration("repository_path"),
                    "policy_bundle": LaunchConfiguration("policy_bundle"),
                    "activation_envelope": LaunchConfiguration(
                        "activation_envelope"
                    ),
                    "experiment_output_root": LaunchConfiguration(
                        "experiment_output_root"
                    ),
                    "enable_logging": LaunchConfiguration("enable_logging"),
                    "record_camera": LaunchConfiguration("record_camera"),
                    "operation": LaunchConfiguration("operation"),
                    "maximum_total_translation_m": LaunchConfiguration(
                        "maximum_total_translation_m"
                    ),
                    "base_frame": LaunchConfiguration("base_frame"),
                    "eef_frame": LaunchConfiguration("eef_frame"),
                    "tcp_frame": LaunchConfiguration("tcp_frame"),
                    "target_book_frame": LaunchConfiguration(
                        "target_book_frame"
                    ),
                    "twist_command_topic": LaunchConfiguration(
                        "twist_command_topic"
                    ),
                    "command_target_is_hardware": "true",
                    "enforce_translation_budget": "true",
                    "require_control_enable": "true",
                    "yield_when_control_disabled": "true",
                    "control_enable_topic": LaunchConfiguration(
                        "control_enable_topic"
                    ),
                    "message_max_age_s": LaunchConfiguration("message_max_age_s"),
                    "tf_max_age_s": LaunchConfiguration("tf_max_age_s"),
                }.items(),
            ),
        ]
    )
