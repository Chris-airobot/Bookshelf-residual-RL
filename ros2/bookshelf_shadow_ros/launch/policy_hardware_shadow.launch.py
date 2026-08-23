"""Launch observation adaptation and read-only PPO inference."""

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
        description="Policy inference parameter file.",
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
    block_on_activation_checks_argument = DeclareLaunchArgument(
        "block_on_activation_checks",
        default_value="true",
        description=(
            "When false, activation checks are reported but do not block policy "
            "calculation."
        ),
    )
    audit_config_argument = DeclareLaunchArgument(
        "audit_config",
        default_value=default_audit_config,
        description="Policy-stream audit parameter file.",
    )
    enable_audit_argument = DeclareLaunchArgument(
        "enable_audit",
        default_value="true",
        description="Record the observation and calculated policy stream.",
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
    base_frame_argument = DeclareLaunchArgument(
        "base_frame",
        default_value="link_base",
        description="Robot base frame used by the observation adapter.",
    )
    ee_frame_argument = DeclareLaunchArgument(
        "ee_frame",
        default_value="link_eef",
        description="Robot end-effector frame used by the observation adapter.",
    )
    target_book_frame_argument = DeclareLaunchArgument(
        "target_book_frame",
        default_value="target_book_center",
        description="Live semantic book frame used by the observation adapter.",
    )
    joint_states_topic_argument = DeclareLaunchArgument(
        "joint_states_topic",
        default_value="/joint_states",
        description="Joint-state input used for the gripper observation.",
    )
    message_max_age_argument = DeclareLaunchArgument(
        "message_max_age_s",
        default_value="0.5",
        description="Maximum accepted age for policy observation inputs.",
    )
    tf_max_age_argument = DeclareLaunchArgument(
        "tf_max_age_s",
        default_value="0.5",
        description="Maximum accepted age for observation transforms.",
    )
    detector_argument = DeclareLaunchArgument(
        "start_live_detector",
        default_value="true",
        description=(
            "Start the RGB-D slot detector. Disable when another composed "
            "launch already owns the detector topics."
        ),
    )

    detector = Node(
        package="bookshelf_shadow_ros",
        executable="rgbd_slot_detector",
        name="rgbd_slot_detector",
        output="screen",
        condition=IfCondition(LaunchConfiguration("start_live_detector")),
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
        parameters=[
            LaunchConfiguration("adapter_config"),
            {
                "base_frame": LaunchConfiguration("base_frame"),
                "ee_frame": LaunchConfiguration("ee_frame"),
                "target_book_frame": LaunchConfiguration("target_book_frame"),
                "joint_states_topic": LaunchConfiguration("joint_states_topic"),
                "message_max_age_s": ParameterValue(
                    LaunchConfiguration("message_max_age_s"), value_type=float
                ),
                "tf_max_age_s": ParameterValue(
                    LaunchConfiguration("tf_max_age_s"), value_type=float
                ),
            },
        ],
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
                "block_on_activation_checks": ParameterValue(
                    LaunchConfiguration("block_on_activation_checks"),
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
            block_on_activation_checks_argument,
            audit_config_argument,
            enable_audit_argument,
            audit_output_argument,
            audit_samples_argument,
            reference_width_argument,
            base_frame_argument,
            ee_frame_argument,
            target_book_frame_argument,
            joint_states_topic_argument,
            message_max_age_argument,
            tf_max_age_argument,
            detector_argument,
            LogInfo(
                msg=(
                    "Starting policy calculation: optional RGB-D detector -> 12D "
                    "adapter -> VecNormalize -> PPO actor -> optional audit. "
                    "No robot-command node is launched."
                )
            ),
            detector,
            adapter,
            inference,
            audit,
        ]
    )
