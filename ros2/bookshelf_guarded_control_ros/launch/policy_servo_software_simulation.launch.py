"""Run the deployed policy and Servo controller against software-only state."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
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
    marker_mount_path = (
        Path(get_package_share_directory("bookshelf_shadow_ros"))
        / "config"
        / "real_book_aruco0_mount.yaml"
    )
    marker_mount = yaml.safe_load(marker_mount_path.read_text(encoding="utf-8"))
    marker_center = marker_mount["marker_center_in_book_m"]
    marker_rotation = marker_mount["rotation_book_marker"]

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
        "marker_size_m": float(marker_mount["marker_black_size_m"]),
        "marker_thickness_m": float(marker_mount["cardboard_thickness_m"]),
        "book_marker_translation_xyz": [
            float(marker_center["x"]),
            float(marker_center["y"]),
            float(marker_center["z"]),
        ],
        "book_marker_quaternion_xyzw": matrix_to_quaternion_xyzw(
            marker_rotation
        ).tolist(),
        "shelf_size_xyz": scene["shelf_box_size_xyz"],
        "shelf_center_offset_slot_xyz": scene[
            "shelf_box_center_offset_slot_xyz"
        ],
        "shelf_bottom_height_base_m": float(
            scene["shelf_bottom_height_base_m"]
        ),
        "table_size_xyz": scene["table_box_size_xyz"],
        "table_center_base_xyz": scene["table_box_center_base_xyz"],
        "table_quaternion_base_xyzw": scene[
            "table_box_quaternion_base_xyzw"
        ],
    }
    package_share = FindPackageShare("bookshelf_guarded_control_ros")
    deployment_launch = PathJoinSubstitution(
        [package_share, "launch", "physical_policy_deployment.launch.py"]
    )
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
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(deployment_launch),
            launch_arguments={
                "trial_name": LaunchConfiguration("trial_name"),
                "approved_config": str(approved_config),
                "repository_path": LaunchConfiguration("repository_path"),
                "policy_bundle": str(policy_bundle),
                "activation_envelope": str(activation_envelope),
                "enable_logging": "false",
                "enable_policy_audit": "false",
                "operation": "control",
                "maximum_total_translation_m": LaunchConfiguration(
                    "maximum_total_translation_m"
                ),
                "base_frame": "sim_link_base",
                "eef_frame": "sim_link_eef",
                "tcp_frame": "sim_link_tcp",
                "target_book_frame": "sim_target_book_center",
                "joint_states_topic": "/bookshelf_sim/joint_states",
                "start_servo_service": "/bookshelf_sim/servo/start",
                "twist_command_topic": "/bookshelf_sim/servo/delta_twist_cmds",
                "command_target_is_hardware": "false",
                "message_max_age_s": "1.0",
                "tf_max_age_s": "1.0",
            }.items(),
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
                "trial_name", default_value="software_policy_servo_simulation"
            ),
            DeclareLaunchArgument(
                "repository_path", default_value="/home/chris/Chris/bookshelf-unified"
            ),
            DeclareLaunchArgument(
                "output_dir", default_value="/tmp/bookshelf_policy_servo_sim"
            ),
            DeclareLaunchArgument("duration_s", default_value="20.0"),
            DeclareLaunchArgument(
                "maximum_total_translation_m", default_value="0.005"
            ),
            DeclareLaunchArgument("enable_rviz", default_value="true"),
            DeclareLaunchArgument("initial_book_x_slot_m", default_value="-0.10"),
            DeclareLaunchArgument("initial_book_y_slot_m", default_value="0.0"),
            DeclareLaunchArgument("initial_book_z_slot_m", default_value="0.006"),
            OpaqueFunction(function=_simulation_actions),
        ]
    )
