"""Evaluate the regenerated spine-mount book calibration without motion."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_target = PathJoinSubstitution(
        [
            FindPackageShare("bookshelf_shadow_ros"),
            "config",
            "calibrated_preinsert_target.yaml",
        ]
    )
    default_candidate = PathJoinSubstitution(
        [
            FindPackageShare("bookshelf_shadow_ros"),
            "config",
            "spine_mount_book_calibration_candidate.yaml",
        ]
    )
    arguments = [
        DeclareLaunchArgument(
            "target_config",
            default_value=default_target,
            description=(
                "Base target configuration; an approved trial_static_slot.yaml "
                "may be supplied here."
            ),
        ),
        DeclareLaunchArgument(
            "candidate_config",
            default_value=default_candidate,
            description="Unapproved EEF-book and policy-tool override.",
        ),
        DeclareLaunchArgument(
            "output_dir",
            default_value="/tmp/bookshelf_spine_mount_candidate",
        ),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("tf_max_age_s", default_value="0.50"),
        DeclareLaunchArgument(
            "maximum_preserved_book_orientation_error_deg",
            default_value="15.0",
        ),
    ]
    node = Node(
        package="bookshelf_shadow_ros",
        executable="calibrated_preinsert_target",
        name="calibrated_preinsert_target",
        output="screen",
        parameters=[
            LaunchConfiguration("target_config"),
            LaunchConfiguration("candidate_config"),
            {
                "output_dir": LaunchConfiguration("output_dir"),
                "use_sim_time": ParameterValue(
                    LaunchConfiguration("use_sim_time"), value_type=bool
                ),
                "tf_max_age_s": ParameterValue(
                    LaunchConfiguration("tf_max_age_s"), value_type=float
                ),
                "target_orientation_mode": "preserve_current_tcp",
                "maximum_preserved_book_orientation_error_deg": ParameterValue(
                    LaunchConfiguration(
                        "maximum_preserved_book_orientation_error_deg"
                    ),
                    value_type=float,
                ),
            },
        ],
    )
    return LaunchDescription(
        arguments
        + [
            LogInfo(
                msg=(
                    "Starting UNAPPROVED SPINE-MOUNT CANDIDATE in READ-ONLY "
                    "mode. No IK, planner, executor, trajectory, controller, "
                    "gripper, or robot-command node is launched."
                )
            ),
            node,
        ]
    )
