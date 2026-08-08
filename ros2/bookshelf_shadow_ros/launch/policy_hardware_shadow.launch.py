"""Launch markerless perception, observation adaptation, and read-only PPO inference."""

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
        [package_share, "config", "policy_observation_adapter_markerless_smoke.yaml"]
    )
    default_inference_config = PathJoinSubstitution(
        [package_share, "config", "policy_shadow_inference.yaml"]
    )
    default_audit_config = PathJoinSubstitution(
        [package_share, "config", "policy_stream_audit.yaml"]
    )

    adapter_config_argument = DeclareLaunchArgument(
        "adapter_config",
        default_value=default_adapter_config,
        description="Markerless policy observation adapter parameter file.",
    )
    inference_config_argument = DeclareLaunchArgument(
        "inference_config",
        default_value=default_inference_config,
        description="Shadow policy inference parameter file.",
    )
    bundle_argument = DeclareLaunchArgument(
        "policy_bundle",
        description="Verified portable .npz actor and VecNormalize bundle.",
    )
    activation_envelope_argument = DeclareLaunchArgument(
        "activation_envelope",
        default_value="",
        description="Reviewed simulator-local activation-envelope JSON.",
    )
    require_activation_envelope_argument = DeclareLaunchArgument(
        "require_activation_envelope",
        default_value="true",
        description="Fail closed unless the simulator activation envelope is loaded.",
    )
    audit_config_argument = DeclareLaunchArgument(
        "audit_config",
        default_value=default_audit_config,
        description="Complete shadow-stream audit parameter file.",
    )
    enable_audit_argument = DeclareLaunchArgument(
        "enable_audit",
        default_value="true",
        description="Record the read-only observation and policy stream.",
    )
    audit_output_argument = DeclareLaunchArgument(
        "audit_output_dir",
        default_value="/tmp/bookshelf_policy_stream_audit",
        description="Directory for policy stream CSV and JSON reports.",
    )
    audit_samples_argument = DeclareLaunchArgument(
        "audit_samples",
        default_value="1200",
        description="Number of policy-debug cycles to audit.",
    )
    reference_width_argument = DeclareLaunchArgument(
        "reference_slot_width_m",
        default_value="0.0",
        description="Optional manually measured physical slot width in metres.",
    )

    detector = Node(
        package="bookshelf_shadow_ros",
        executable="rgbd_slot_detector",
        name="rgbd_slot_detector",
        output="screen",
        parameters=[
            {
                "image_topic": "/camera/color/image_raw",
                "depth_topic": "/camera/aligned_depth_to_color/image_raw",
                "camera_info_topic": "/camera/color/camera_info",
                "debug_image_topic": "/slot_detector/debug_image",
            }
        ],
    )
    adapter = Node(
        package="bookshelf_shadow_ros",
        executable="policy_observation_adapter",
        name="policy_observation_adapter",
        output="screen",
        parameters=[LaunchConfiguration("adapter_config")],
    )
    inference = Node(
        package="bookshelf_shadow_ros",
        executable="policy_shadow_inference",
        name="policy_shadow_inference",
        output="screen",
        parameters=[
            LaunchConfiguration("inference_config"),
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
        ],
    )
    audit = Node(
        package="bookshelf_shadow_ros",
        executable="policy_stream_audit",
        name="policy_stream_audit",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_audit")),
        parameters=[
            LaunchConfiguration("audit_config"),
            {
                "output_dir": LaunchConfiguration("audit_output_dir"),
                "target_samples": ParameterValue(
                    LaunchConfiguration("audit_samples"),
                    value_type=int,
                ),
                "reference_slot_width_m": ParameterValue(
                    LaunchConfiguration("reference_slot_width_m"),
                    value_type=float,
                ),
            },
        ],
    )

    return LaunchDescription(
        [
            adapter_config_argument,
            inference_config_argument,
            bundle_argument,
            activation_envelope_argument,
            require_activation_envelope_argument,
            audit_config_argument,
            enable_audit_argument,
            audit_output_argument,
            audit_samples_argument,
            reference_width_argument,
            LogInfo(
                msg=(
                    "Starting FULL SHADOW pipeline: RGB-D detector -> markerless 12D adapter "
                    "-> VecNormalize -> PPO actor diagnostics -> subscriber-only audit. "
                    "No robot-command node is launched."
                )
            ),
            detector,
            adapter,
            inference,
            audit,
        ]
    )
