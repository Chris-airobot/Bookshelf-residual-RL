"""Calculate and visualize a calibrated pre-insertion target without motion."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution(
        [
            FindPackageShare("bookshelf_shadow_ros"),
            "config",
            "calibrated_preinsert_target.yaml",
        ]
    )
    arguments = [
        DeclareLaunchArgument(
            "target_config",
            default_value=default_config,
            description="Calibrated static slot, grasp, and target parameters.",
        ),
        DeclareLaunchArgument(
            "output_dir",
            default_value="/tmp/bookshelf_calibrated_target",
            description="Directory for calibrated_preinsert_target_report.json.",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Use /clock while comparing against rosbag TF.",
        ),
        DeclareLaunchArgument(
            "tf_max_age_s",
            default_value="0.50",
            description="Set to 0 for unrestricted offline bag TF age.",
        ),
        DeclareLaunchArgument(
            "target_orientation_mode",
            default_value="preserve_current_tcp",
            description=(
                "preserve_current_tcp latches the live TCP orientation; "
                "book_aligned retains the original geometric reference."
            ),
        ),
        DeclareLaunchArgument(
            "maximum_preserved_book_orientation_error_deg",
            default_value="15.0",
            description=(
                "Fail the preserved-orientation target when the resulting "
                "book orientation differs from the slot by more than this."
            ),
        ),
    ]
    node = Node(
        package="bookshelf_shadow_ros",
        executable="calibrated_preinsert_target",
        name="calibrated_preinsert_target",
        output="screen",
        parameters=[
            LaunchConfiguration("target_config"),
            {
                "output_dir": LaunchConfiguration("output_dir"),
                "use_sim_time": ParameterValue(
                    LaunchConfiguration("use_sim_time"), value_type=bool
                ),
                "tf_max_age_s": ParameterValue(
                    LaunchConfiguration("tf_max_age_s"), value_type=float
                ),
                "target_orientation_mode": LaunchConfiguration(
                    "target_orientation_mode"
                ),
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
                    "Starting READ-ONLY calibrated pre-insertion target "
                    "calculation. No IK, planner, executor, trajectory, "
                    "gripper, controller, or robot-command node is launched."
                )
            ),
            node,
        ]
    )
