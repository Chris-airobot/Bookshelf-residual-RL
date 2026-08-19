"""Replay Bag C through frozen-slot, live-book policy diagnostics only."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare("bookshelf_shadow_ros")
    adapter_defaults = PathJoinSubstitution(
        [package_share, "config", "policy_observation_adapter.yaml"]
    )
    inference_config = PathJoinSubstitution(
        [package_share, "config", "policy_shadow_inference.yaml"]
    )
    rviz_config = PathJoinSubstitution(
        [package_share, "rviz", "stationary_shadow_replay.rviz"]
    )
    arguments = [
        DeclareLaunchArgument(
            "adapter_config",
            description="Generated offline frozen-slot/live-marker adapter YAML.",
        ),
        DeclareLaunchArgument(
            "mount_yaml",
            description="Reviewed physical marker-to-book mount YAML.",
        ),
        DeclareLaunchArgument(
            "policy_bundle",
            description="Portable residual actor and VecNormalize bundle.",
        ),
        DeclareLaunchArgument(
            "activation_envelope",
            description="Reviewed simulator-local activation envelope.",
        ),
        DeclareLaunchArgument(
            "output_dir",
            default_value="/tmp/bookshelf_stationary_shadow_replay",
        ),
        DeclareLaunchArgument("candidate_id", default_value="unknown"),
        DeclareLaunchArgument("minimum_valid_samples", default_value="30"),
        DeclareLaunchArgument("enable_rviz", default_value="false"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
    ]
    common = {
        "use_sim_time": ParameterValue(
            LaunchConfiguration("use_sim_time"), value_type=bool
        )
    }
    detector = Node(
        package="bookshelf_shadow_ros",
        executable="marker_book_calibrator",
        name="marker_book_calibration",
        output="screen",
        parameters=[
            {
                "mount_yaml": LaunchConfiguration("mount_yaml"),
                "output_dir": PathJoinSubstitution(
                    [LaunchConfiguration("output_dir"), "marker"]
                ),
                "target_samples": 1000000,
                "enable_frame_audit": False,
                "detected_marker_frame": "target_book_marker",
                "detected_book_frame": "target_book_center",
                **common,
            }
        ],
    )
    adapter = Node(
        package="bookshelf_shadow_ros",
        executable="policy_observation_adapter",
        name="policy_observation_adapter",
        output="screen",
        parameters=[
            adapter_defaults,
            LaunchConfiguration("adapter_config"),
            common,
        ],
    )
    inference = Node(
        package="bookshelf_shadow_ros",
        executable="policy_shadow_inference",
        name="policy_shadow_inference",
        output="screen",
        parameters=[
            inference_config,
            {
                "policy_bundle_path": LaunchConfiguration("policy_bundle"),
                "activation_envelope_path": LaunchConfiguration(
                    "activation_envelope"
                ),
                "require_activation_envelope": True,
                **common,
            },
        ],
    )
    audit = Node(
        package="bookshelf_shadow_ros",
        executable="stationary_shadow_replay_audit",
        name="stationary_shadow_replay_audit",
        output="screen",
        parameters=[
            {
                "output_dir": LaunchConfiguration("output_dir"),
                "candidate_id": LaunchConfiguration("candidate_id"),
                "minimum_valid_samples": ParameterValue(
                    LaunchConfiguration("minimum_valid_samples"),
                    value_type=int,
                ),
                **common,
            }
        ],
    )
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="stationary_shadow_replay_rviz",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_rviz")),
        arguments=["-d", rviz_config],
        parameters=[common],
    )
    return LaunchDescription(
        arguments
        + [
            LogInfo(
                msg=(
                    "Starting OFFLINE stationary shadow replay: frozen View A "
                    "slot, continuous Bag C marker book, 12D adapter, and PPO "
                    "diagnostics. No planner, executor, controller, gripper, "
                    "or robot-command client is launched."
                )
            ),
            detector,
            adapter,
            inference,
            audit,
            rviz,
        ]
    )
