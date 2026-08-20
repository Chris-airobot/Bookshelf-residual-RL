"""Run the deployed policy and Servo controller against software-only state."""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    LogInfo,
    OpaqueFunction,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import yaml

from bookshelf_guarded_control_ros.policy_servo_simulation_math import (
    initial_eef_from_slot_book,
)
from bookshelf_guarded_control_ros.policy_tool_control_math import (
    make_transform,
    matrix_to_quaternion_xyzw,
)
from bookshelf_guarded_control_ros.rehearsal_configuration import (
    guarded_policy_tool_overrides,
    validate_shadow_rehearsal_assets,
)


def _parameters(document: dict, node_name: str) -> dict:
    try:
        value = document[node_name]["ros__parameters"]
    except (KeyError, TypeError) as error:
        raise RuntimeError(f"approved configuration is missing {node_name}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid parameters for {node_name}")
    return dict(value)


def _simulation_actions(context):
    duration_s = float(LaunchConfiguration("duration_s").perform(context))
    if duration_s <= 0.0:
        raise RuntimeError("duration_s must be positive")
    approved_config = Path(
        LaunchConfiguration("approved_config").perform(context)
    ).expanduser().resolve()
    policy_bundle = Path(
        LaunchConfiguration("policy_bundle").perform(context)
    ).expanduser().resolve()
    activation_envelope = Path(
        LaunchConfiguration("activation_envelope").perform(context)
    ).expanduser().resolve()
    result = validate_shadow_rehearsal_assets(
        approved_config, policy_bundle, activation_envelope
    )
    document = yaml.safe_load(approved_config.read_text(encoding="utf-8"))
    adapter = _parameters(document, "policy_observation_adapter")
    target = _parameters(document, "calibrated_preinsert_target")
    scene = _parameters(document, "bookshelf_scene_manager")

    transform_base_slot = make_transform(
        adapter["configured_static_slot_translation_xyz"],
        adapter["configured_static_slot_quaternion_xyzw"],
    )
    transform_eef_book = make_transform(
        adapter["eef_book_translation_xyz"],
        adapter["eef_book_quaternion_xyzw"],
    )
    transform_slot_book = make_transform(
        [
            float(LaunchConfiguration("initial_book_x_slot_m").perform(context)),
            float(LaunchConfiguration("initial_book_y_slot_m").perform(context)),
            float(LaunchConfiguration("initial_book_z_slot_m").perform(context)),
        ],
        [0.0, 0.0, 0.0, 1.0],
    )
    transform_base_eef = initial_eef_from_slot_book(
        transform_base_slot,
        transform_slot_book,
        transform_eef_book,
    )
    control_overrides = guarded_policy_tool_overrides(
        approved_config, policy_bundle
    )
    control_overrides.update(
        {
            "base_frame": "sim_link_base",
            "eef_frame": "sim_link_eef",
            "tcp_frame": "sim_link_tcp",
            "start_servo_service": "/bookshelf_sim/servo/start",
            "twist_command_topic": "/bookshelf_sim/servo/delta_twist_cmds",
            "command_target_is_hardware": False,
            "maximum_total_translation_m": 1.0,
            "tf_max_age_s": 1.0,
        }
    )
    adapter_overrides = {
        "base_frame": "sim_link_base",
        "ee_frame": "sim_link_eef",
        "target_book_frame": "sim_target_book_center",
        "book_pose_source": "marker",
        "slot_pose_source": "configured_static",
        "joint_states_topic": "/bookshelf_sim/joint_states",
        "message_max_age_s": 1.0,
        "tf_max_age_s": 1.0,
    }
    simulator_parameters = {
        "candidate_id": result["candidate_id"],
        "output_dir": LaunchConfiguration("output_dir"),
        "initial_eef_translation_xyz": transform_base_eef[:3, 3].tolist(),
        "initial_eef_quaternion_xyzw": matrix_to_quaternion_xyzw(
            transform_base_eef[:3, :3]
        ).tolist(),
        "eef_tcp_translation_xyz": control_overrides[
            "eef_tcp_translation_xyz"
        ],
        "eef_tcp_quaternion_xyzw": control_overrides[
            "eef_tcp_quaternion_xyzw"
        ],
        "eef_book_translation_xyz": adapter["eef_book_translation_xyz"],
        "eef_book_quaternion_xyzw": adapter["eef_book_quaternion_xyzw"],
        "slot_translation_xyz": adapter[
            "configured_static_slot_translation_xyz"
        ],
        "slot_quaternion_xyzw": adapter[
            "configured_static_slot_quaternion_xyzw"
        ],
        "slot_width_m": float(adapter["configured_static_slot_width_m"]),
        "slot_depth_m": float(adapter["slot_depth_m"]),
        "book_size_xyz": adapter["book_size_xyz"],
    }
    package_share = FindPackageShare("bookshelf_guarded_control_ros")
    shadow_share = FindPackageShare("bookshelf_shadow_ros")
    return [
        LogInfo(
            msg=(
                "Starting SOFTWARE-ONLY closed-loop policy rehearsal for "
                f"candidate {result['candidate_id']}. The approved frozen slot, "
                "book-to-EEF transform, PPO actor, and direct Servo controller "
                "are real; robot state and Servo motion are simulated."
            )
        ),
        Node(
            package="bookshelf_guarded_control_ros",
            executable="policy_servo_simulator",
            name="policy_servo_simulator",
            output="screen",
            parameters=[simulator_parameters],
        ),
        Node(
            package="bookshelf_shadow_ros",
            executable="policy_observation_adapter",
            name="policy_observation_adapter",
            output="screen",
            parameters=[str(approved_config), adapter_overrides],
        ),
        Node(
            package="bookshelf_shadow_ros",
            executable="policy_shadow_inference",
            name="policy_shadow_inference",
            output="screen",
            parameters=[
                PathJoinSubstitution(
                    [shadow_share, "config", "policy_shadow_inference.yaml"]
                ),
                {
                    "policy_bundle_path": str(policy_bundle),
                    "activation_envelope_path": str(activation_envelope),
                    "require_activation_envelope": True,
                    "block_on_activation_checks": False,
                },
            ],
        ),
        Node(
            package="bookshelf_guarded_control_ros",
            executable="direct_policy_servo",
            name="direct_policy_servo",
            output="screen",
            parameters=[
                PathJoinSubstitution(
                    [package_share, "config", "direct_policy_servo.yaml"]
                ),
                control_overrides,
            ],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="policy_servo_simulation_rviz",
            output="screen",
            condition=IfCondition(LaunchConfiguration("enable_rviz")),
            arguments=[
                "-d",
                PathJoinSubstitution(
                    [package_share, "rviz", "policy_servo_software_simulation.rviz"]
                ),
            ],
        ),
        TimerAction(
            period=duration_s,
            actions=[
                EmitEvent(
                    event=Shutdown(
                        reason="software-only policy rehearsal duration complete"
                    )
                )
            ],
        ),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("approved_config"),
            DeclareLaunchArgument("policy_bundle"),
            DeclareLaunchArgument("activation_envelope"),
            DeclareLaunchArgument(
                "output_dir", default_value="/tmp/bookshelf_policy_servo_sim"
            ),
            DeclareLaunchArgument("duration_s", default_value="20.0"),
            DeclareLaunchArgument("enable_rviz", default_value="true"),
            DeclareLaunchArgument("initial_book_x_slot_m", default_value="-0.10"),
            DeclareLaunchArgument("initial_book_y_slot_m", default_value="0.0"),
            DeclareLaunchArgument("initial_book_z_slot_m", default_value="0.006"),
            OpaqueFunction(function=_simulation_actions),
        ]
    )
