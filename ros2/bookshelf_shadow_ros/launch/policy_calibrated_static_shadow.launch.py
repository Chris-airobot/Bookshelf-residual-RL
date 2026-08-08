"""Run calibrated static-slot observation and PPO diagnostics without motion."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare("bookshelf_shadow_ros")
    default_adapter_config = PathJoinSubstitution(
        [
            package_share,
            "config",
            "policy_observation_adapter_calibrated_static.yaml",
        ]
    )
    inference_config = PathJoinSubstitution(
        [package_share, "config", "policy_shadow_inference.yaml"]
    )
    audit_config = PathJoinSubstitution(
        [package_share, "config", "policy_stream_audit.yaml"]
    )

    arguments = [
        DeclareLaunchArgument(
            "adapter_config",
            default_value=default_adapter_config,
            description=(
                "Observation-adapter parameter file. The default remains the "
                "fail-closed identity-tool configuration."
            ),
        ),
        DeclareLaunchArgument(
            "policy_bundle",
            description="Verified portable .npz actor and VecNormalize bundle.",
        ),
        DeclareLaunchArgument(
            "activation_envelope",
            default_value="",
            description="Reviewed simulator-local activation-envelope JSON.",
        ),
        DeclareLaunchArgument(
            "require_activation_envelope",
            default_value="true",
            description="Fail closed unless the simulator activation envelope is loaded.",
        ),
        DeclareLaunchArgument(
            "enable_audit",
            default_value="true",
            description="Record the subscriber-only shadow stream.",
        ),
        DeclareLaunchArgument(
            "audit_output_dir",
            default_value="/tmp/bookshelf_calibrated_static_audit",
            description="Directory for policy-stream CSV and JSON reports.",
        ),
        DeclareLaunchArgument(
            "audit_samples",
            default_value="1200",
            description="Number of complete policy cycles to audit.",
        ),
        DeclareLaunchArgument(
            "reference_slot_width_m",
            default_value="0.0",
            description="Optional manually measured slot width in metres.",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Use /clock for offline rosbag replay.",
        ),
    ]

    common_parameters = {
        "use_sim_time": ParameterValue(
            LaunchConfiguration("use_sim_time"), value_type=bool
        )
    }

    adapter = Node(
        package="bookshelf_shadow_ros",
        executable="policy_observation_adapter",
        name="policy_observation_adapter",
        output="screen",
        parameters=[LaunchConfiguration("adapter_config"), common_parameters],
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
                "require_activation_envelope": ParameterValue(
                    LaunchConfiguration("require_activation_envelope"),
                    value_type=bool,
                ),
            },
            common_parameters,
        ],
    )
    audit = Node(
        package="bookshelf_shadow_ros",
        executable="policy_stream_audit",
        name="policy_stream_audit",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_audit")),
        parameters=[
            audit_config,
            {
                "output_dir": LaunchConfiguration("audit_output_dir"),
                "target_samples": ParameterValue(
                    LaunchConfiguration("audit_samples"), value_type=int
                ),
                "reference_slot_width_m": ParameterValue(
                    LaunchConfiguration("reference_slot_width_m"),
                    value_type=float,
                ),
                **common_parameters,
            },
        ],
    )

    return LaunchDescription(
        arguments
        + [
            LogInfo(
                msg=(
                    "Starting CALIBRATED STATIC SHADOW pipeline. The saved "
                    "slot pose and measured rigid-grasp transform are used; no "
                    "RGB-D detector, policy executor, IK, trajectory, gripper, "
                    "or robot-command node is launched."
                )
            ),
            adapter,
            inference,
            audit,
        ]
    )
